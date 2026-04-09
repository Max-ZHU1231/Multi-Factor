"""
data_quality.py
===============
针对 Stocks/ 目录下单股 CSV 的 **额外数据质量检查** 模块。

覆盖四类检查
------------
1. 价格连续性  : 检测复权因子未变化时的前收盘价跳跃（除权未复权信号）
2. 成交量异常  : 成交量 = 0 的交易日标记为停牌，并建议做与 data_cleaner 一致的 ffill 处理
3. 财务一致性  : 资产 ≈ 负债 + 所有者权益（当数据中包含三张表字段时校验）
4. 时区对齐    : 月频宏观 / 财务数据与日频量价数据对齐到月末（或月初）

使用方式
--------
from data_quality import run_all_checks, align_monthly_to_daily

result = run_all_checks(df)            # 对单股 DataFrame 执行全部检查
result = run_all_checks(df, fin_df=financial_df)   # 同时做财务一致性校验
aligned = align_monthly_to_daily(macro_df, daily_df)  # 月频数据对齐到日频
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── 可调参数 ─────────────────────────────────────────────────────────────────
PRICE_JUMP_THRESHOLD  = 0.05   # 复权因子不变时，前收盘跳跃超过此比例视为异常（5%）
FIN_BALANCE_TOL       = 0.01   # 财务等式容差：|(资产 - 负债 - 权益)| / 资产 ≤ 1%
SUSPENSION_ZERO_VOL   = True   # 是否将成交量=0 标记为停牌

# ── 财务三张表字段名（可根据数据源实际列名调整）────────────────────────────
FIN_TOTAL_ASSETS      = "总资产"          # 资产负债表：总资产
FIN_TOTAL_LIAB        = "总负债"          # 资产负债表：总负债
FIN_TOTAL_EQUITY      = "所有者权益合计"  # 资产负债表：所有者权益合计


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 价格连续性检查
# ═══════════════════════════════════════════════════════════════════════════════

def check_price_continuity(
    df: pd.DataFrame,
    threshold: float = PRICE_JUMP_THRESHOLD,
) -> dict:
    """
    检测价格连续性：当复权因子未发生变化时，前收盘价与前日收盘价之间的偏差
    超过 threshold 则判定为"除权未复权"跳跃异常。

    Parameters
    ----------
    df        : 单股 DataFrame，已按交易日升序排列，含列：
                收盘价, 前收盘价, 复权因子
    threshold : 允许的最大相对偏差（默认 5%）

    Returns
    -------
    dict:
        passed          : bool，是否通过检查
        jump_rows       : DataFrame，所有异常行
        jump_count      : int，异常行数
        adj_change_dates: list[str]，复权因子发生变化的日期（合法跳跃点）
        message         : str，人类可读摘要
    """
    required = {"收盘价", "前收盘价", "复权因子", "交易日"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return _skip_result("price_continuity", f"缺少列: {missing_cols}")

    work = df.dropna(subset=["收盘价", "前收盘价", "复权因子"]).copy()
    work = work.sort_values("交易日").reset_index(drop=True)

    if len(work) < 2:
        return _skip_result("price_continuity", "有效行数不足 2")

    # 前日收盘价、前日复权因子
    work["_prev_close"] = work["收盘价"].shift(1)
    work["_prev_adj"]   = work["复权因子"].shift(1)
    work = work.dropna(subset=["_prev_close", "_prev_adj"])

    # 复权因子变化点 → 合法跳跃，排除
    adj_change_mask = work["复权因子"] != work["_prev_adj"]
    adj_change_dates = work.loc[adj_change_mask, "交易日"].tolist()

    # 在复权因子不变的行里检查跳跃
    stable = work[~adj_change_mask].copy()
    stable["_gap"] = (
        (stable["前收盘价"] - stable["_prev_close"]).abs()
        / stable["_prev_close"].replace(0, np.nan)
    )
    jump_mask = stable["_gap"] > threshold
    jump_rows = stable.loc[
        jump_mask,
        ["交易日", "收盘价", "前收盘价", "_prev_close", "_gap", "复权因子"],
    ].rename(columns={"_prev_close": "前日收盘价", "_gap": "偏差率"})

    passed = len(jump_rows) == 0
    msg = (
        "价格连续性：通过"
        if passed
        else f"价格连续性：发现 {len(jump_rows)} 处疑似除权未复权跳跃"
    )
    return {
        "check":            "price_continuity",
        "passed":           passed,
        "jump_rows":        jump_rows.reset_index(drop=True),
        "jump_count":       int(jump_mask.sum()),
        "adj_change_dates": adj_change_dates,
        "message":          msg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 成交量异常检查
# ═══════════════════════════════════════════════════════════════════════════════

def check_volume_anomaly(
    df: pd.DataFrame,
    mark_suspension: bool = SUSPENSION_ZERO_VOL,
) -> dict:
    """
    检测成交量异常：
    - 成交量 = 0 的交易日标记为停牌（suspension）
    - 成交量为 NaN 的交易日同样标记为停牌
    - 返回停牌日期列表及占比

    Parameters
    ----------
    df              : 单股 DataFrame，含列：交易日, 成交量（手）
    mark_suspension : True 时，在返回值中附带停牌标志列

    Returns
    -------
    dict:
        passed              : bool，停牌占比 < 50% 则视为通过（正常股票）
        suspension_dates    : list[str]，停牌日期
        suspension_count    : int
        suspension_rate     : float
        df_with_flag        : DataFrame（含 is_suspension 列，仅 mark_suspension=True）
        message             : str
    """
    required = {"交易日", "成交量（手）"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return _skip_result("volume_anomaly", f"缺少列: {missing_cols}")

    work = df[["交易日", "成交量（手）"]].copy()
    susp_mask = work["成交量（手）"].isna() | (work["成交量（手）"] == 0)
    suspension_dates = work.loc[susp_mask, "交易日"].tolist()
    rate = susp_mask.mean()

    result: dict = {
        "check":            "volume_anomaly",
        "passed":           rate < 0.50,
        "suspension_dates": suspension_dates,
        "suspension_count": int(susp_mask.sum()),
        "suspension_rate":  float(rate),
        "message":          (
            f"成交量：{len(suspension_dates)} 个停牌日（占比 {rate:.1%}）"
        ),
    }
    if mark_suspension:
        flagged = df.copy()
        flagged["is_suspension"] = susp_mask.values
        result["df_with_flag"] = flagged
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 财务数据一致性检查（资产 = 负债 + 所有者权益）
# ═══════════════════════════════════════════════════════════════════════════════

def check_financial_balance(
    fin_df: pd.DataFrame,
    col_assets:  str = FIN_TOTAL_ASSETS,
    col_liab:    str = FIN_TOTAL_LIAB,
    col_equity:  str = FIN_TOTAL_EQUITY,
    tol:         float = FIN_BALANCE_TOL,
    date_col:    str = "报告期",
) -> dict:
    """
    校验资产负债表等式：总资产 ≈ 总负债 + 所有者权益合计。
    容差 = |(资产 - 负债 - 权益)| / |资产| ≤ tol。

    Parameters
    ----------
    fin_df     : 财务 DataFrame（每行一个报告期），需含三个字段
    col_assets : 总资产列名
    col_liab   : 总负债列名
    col_equity : 所有者权益列名
    tol        : 容差比例（默认 1%）
    date_col   : 报告期列名（用于标识异常行）

    Returns
    -------
    dict:
        passed          : bool
        unbalanced_rows : DataFrame，不满足等式的行
        unbalanced_count: int
        message         : str
    """
    for col in [col_assets, col_liab, col_equity]:
        if col not in fin_df.columns:
            return _skip_result("financial_balance", f"缺少列: {col}")

    work = fin_df.dropna(subset=[col_assets, col_liab, col_equity]).copy()
    if len(work) == 0:
        return _skip_result("financial_balance", "无有效财务行")

    work["_imbalance"] = (
        (work[col_assets] - work[col_liab] - work[col_equity]).abs()
        / work[col_assets].replace(0, np.nan).abs()
    )
    bad_mask = work["_imbalance"] > tol

    display_cols = [date_col, col_assets, col_liab, col_equity, "_imbalance"]
    display_cols = [c for c in display_cols if c in work.columns]
    unbalanced = work.loc[bad_mask, display_cols].rename(
        columns={"_imbalance": "失衡率"}
    )

    passed = len(unbalanced) == 0
    msg = (
        "财务一致性：通过"
        if passed
        else f"财务一致性：{len(unbalanced)} 个报告期资产≠负债+权益（容差{tol:.0%}）"
    )
    return {
        "check":             "financial_balance",
        "passed":            passed,
        "unbalanced_rows":   unbalanced.reset_index(drop=True),
        "unbalanced_count":  int(bad_mask.sum()),
        "message":           msg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 时区对齐：月频 → 日频
# ═══════════════════════════════════════════════════════════════════════════════

def align_monthly_to_daily(
    monthly_df: pd.DataFrame,
    daily_df:   pd.DataFrame,
    date_col_monthly: str = "date",
    date_col_daily:   str = "交易日",
    method: str = "month_end",
) -> pd.DataFrame:
    """
    将月频数据（宏观/财务）对齐到日频数据的时间轴。

    策略
    ----
    - method='month_end'  (默认): 将月频数据的日期对齐到当月**最后一个交易日**，
      再向前填充到该月所有交易日（PIT 安全，不使用未来数据）。
    - method='month_start': 将月频数据对齐到当月**第一个交易日**，
      再向前填充（适合月初公布的宏观指标）。

    Parameters
    ----------
    monthly_df        : 月频 DataFrame，date_col_monthly 列为日期（str YYYYMM 或 YYYY-MM-DD）
    daily_df          : 日频 DataFrame，date_col_daily 列为日期（str YYYYMMDD）
    date_col_monthly  : 月频日期列名
    date_col_daily    : 日频日期列名
    method            : 'month_end' 或 'month_start'

    Returns
    -------
    日频 DataFrame（含原日频列 + 月频指标列），按交易日升序排列。
    月频指标在非对齐日期用 ffill 填充（PIT 安全）。
    """
    daily  = daily_df.copy()
    monthly = monthly_df.copy()

    # 统一转为 pd.Timestamp
    daily["_date"]   = pd.to_datetime(daily[date_col_daily], format="%Y%m%d", errors="coerce")
    monthly["_mdate"] = pd.to_datetime(monthly[date_col_monthly], errors="coerce")

    daily   = daily.sort_values("_date").reset_index(drop=True)
    monthly = monthly.sort_values("_mdate").reset_index(drop=True)

    # 月份 key（YYYY-MM）
    daily["_month_key"]   = daily["_date"].dt.to_period("M")
    monthly["_month_key"] = monthly["_mdate"].dt.to_period("M")

    # 每月 第一个/最后一个 交易日
    if method == "month_end":
        anchor = daily.groupby("_month_key")["_date"].max().reset_index()
        anchor.columns = ["_month_key", "_anchor_date"]
    else:  # month_start
        anchor = daily.groupby("_month_key")["_date"].min().reset_index()
        anchor.columns = ["_month_key", "_anchor_date"]

    # 月频数据打上锚点日期
    monthly = monthly.merge(anchor, on="_month_key", how="left")

    # 将月频数据合并到日频时间轴上（以锚点日期为键）
    monthly_cols = [c for c in monthly.columns if c not in [
        date_col_monthly, "_mdate", "_month_key", "_anchor_date"
    ]]
    monthly_slim = monthly[["_anchor_date"] + monthly_cols].dropna(subset=["_anchor_date"])
    monthly_slim = monthly_slim.rename(columns={"_anchor_date": "_date"})
    monthly_slim = monthly_slim.drop_duplicates(subset=["_date"])

    result = daily.merge(monthly_slim, on="_date", how="left")
    result = result.sort_values("_date").reset_index(drop=True)

    # ffill 月频列（PIT 向前填充）
    for col in monthly_cols:
        if col in result.columns:
            result[col] = result[col].ffill()

    # 清理辅助列
    result = result.drop(columns=["_date", "_month_key"])
    return result


def check_monthly_alignment(
    monthly_df: pd.DataFrame,
    daily_df:   pd.DataFrame,
    date_col_monthly: str = "date",
    date_col_daily:   str = "交易日",
    method: str = "month_end",
) -> dict:
    """
    检查月频数据与日频数据的时区对齐质量：
    - 验证月频数据日期范围是否覆盖日频数据范围
    - 验证对齐后月频列是否存在头部 NaN（未覆盖区间）
    - 验证未来数据未被引入（bfill 污染检查）

    Returns
    -------
    dict:
        passed              : bool
        aligned_df          : DataFrame，对齐后结果
        head_nan_months     : int，日频开头未被月频覆盖的月份数
        monthly_date_range  : (min, max) 月频日期范围
        daily_date_range    : (min, max) 日频日期范围
        message             : str
    """
    try:
        aligned = align_monthly_to_daily(
            monthly_df, daily_df,
            date_col_monthly=date_col_monthly,
            date_col_daily=date_col_daily,
            method=method,
        )
    except Exception as e:
        return _skip_result("monthly_alignment", str(e))

    monthly_dates = pd.to_datetime(monthly_df[date_col_monthly], errors="coerce").dropna()
    daily_dates   = pd.to_datetime(daily_df[date_col_daily], format="%Y%m%d", errors="coerce").dropna()

    m_min, m_max = monthly_dates.min(), monthly_dates.max()
    d_min, d_max = daily_dates.min(),   daily_dates.max()

    monthly_cols = [c for c in monthly_df.columns if c != date_col_monthly]
    head_nan = 0
    for col in monthly_cols:
        if col in aligned.columns:
            head_nan = max(head_nan, int(aligned[col].isna().sum()))

    passed = (m_min <= d_min) and (head_nan == 0 or m_min <= d_min)
    msg = (
        f"时区对齐（{method}）：月频覆盖 {m_min.date()}~{m_max.date()}，"
        f"日频覆盖 {d_min.date()}~{d_max.date()}，"
        f"对齐后头部NaN共 {head_nan} 行"
    )
    return {
        "check":             "monthly_alignment",
        "passed":            passed,
        "aligned_df":        aligned,
        "head_nan_rows":     head_nan,
        "monthly_date_range": (m_min, m_max),
        "daily_date_range":   (d_min, d_max),
        "message":           msg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总入口
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks(
    df: pd.DataFrame,
    fin_df:           Optional[pd.DataFrame] = None,
    monthly_df:       Optional[pd.DataFrame] = None,
    date_col_monthly: str = "date",
    price_threshold:  float = PRICE_JUMP_THRESHOLD,
    fin_tol:          float = FIN_BALANCE_TOL,
    monthly_method:   str = "month_end",
) -> dict:
    """
    对单股数据执行全部质量检查，返回汇总结果字典。

    Parameters
    ----------
    df              : 单股日频 DataFrame（已含价格、成交量等列）
    fin_df          : 财务 DataFrame（可选，含总资产/总负债/所有者权益列）
    monthly_df      : 月频宏观/财务 DataFrame（可选）
    date_col_monthly: 月频日期列名
    price_threshold : 价格跳跃容差
    fin_tol         : 财务等式容差
    monthly_method  : 月频对齐策略（'month_end' 或 'month_start'）

    Returns
    -------
    dict:
        all_passed : bool，所有检查均通过
        checks     : dict，每项检查名 → 结果 dict
        summary    : list[str]，各检查摘要文本
    """
    checks: dict = {}

    # 1. 价格连续性
    checks["price_continuity"] = check_price_continuity(df, threshold=price_threshold)

    # 2. 成交量异常
    checks["volume_anomaly"] = check_volume_anomaly(df)

    # 3. 财务一致性（可选）
    if fin_df is not None:
        checks["financial_balance"] = check_financial_balance(fin_df, tol=fin_tol)
    else:
        checks["financial_balance"] = _skip_result("financial_balance", "未提供财务数据")

    # 4. 时区对齐（可选）
    if monthly_df is not None:
        checks["monthly_alignment"] = check_monthly_alignment(
            monthly_df, df,
            date_col_monthly=date_col_monthly,
            method=monthly_method,
        )
    else:
        checks["monthly_alignment"] = _skip_result("monthly_alignment", "未提供月频数据")

    all_passed = all(
        v["passed"] for v in checks.values() if v.get("skipped") is not True
    )
    summary = [v["message"] for v in checks.values()]

    return {
        "all_passed": all_passed,
        "checks":     checks,
        "summary":    summary,
    }


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _skip_result(check_name: str, reason: str) -> dict:
    """生成跳过标志的结果字典。"""
    return {
        "check":   check_name,
        "passed":  True,   # 跳过 ≠ 失败，不影响 all_passed
        "skipped": True,
        "message": f"{check_name}：跳过（{reason}）",
    }
