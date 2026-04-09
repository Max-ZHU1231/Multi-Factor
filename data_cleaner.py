"""
data_cleaner.py
===============
针对 Stocks/ 目录下单股 CSV 的数据清洗模块。

清洗流程
--------
1. 极值处理  : MAD Winsorize（每列独立，3σ_MAD 阈值）
2. 缺失值处理:
   - 价格/成交量类列（停牌场景）: ffill，最多 5 个交易日
   - 估值类列（财报未披露）     : ffill 不限长度（PIT 原则，严禁用未来数据）
   - 随机缺失（缺失率 < 5%）    : 横截面均值填充（仅用于面板数据；单股模式跳过）
   - 大量缺失（缺失率 > 30%）   : 整列标记为无效（列名加入 INVALID_COLS 列表）
   - 新股（有效行数 < MIN_ROWS）: 整只股票剔除（返回 None）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── 可调参数 ─────────────────────────────────────────────────────────────────
MAD_THRESHOLD   = 3.0    # MAD Winsorize 阈值（中位数 ± k * MAD）
FFILL_PRICE_MAX = 5      # 停牌场景价格列向前填充最大天数
MIN_ROWS        = 60     # 新股最小有效行数（< 此值则剔除）
MISSING_RANDOM  = 0.05   # 随机缺失阈值（< 5% 则用横截面均值填充）
MISSING_HEAVY   = 0.30   # 大量缺失阈值（> 30% 则整列标记为无效）

# ── 列分组 ────────────────────────────────────────────────────────────────────
# 价格 / 量能列：停牌时向前填充（max 5 天）
PRICE_VOL_COLS = [
    "开盘价", "最高价", "最低价", "收盘价", "前收盘价",
    "涨跌额", "涨跌幅（%）",
    "成交量（手）", "成交额（千元）",
    "换手率（%）", "换手率（%，自由流通股）", "量比",
    "总市值（万元）", "流通市值（万元）",
    "当日涨停价", "当日跌停价",
]

# 估值 / 基本面列：使用最新已披露数据（ffill 不限长度，PIT 原则）
VALUATION_COLS = [
    "市盈率（亏损为空）", "市盈率（TTM，亏损为空）",
    "市净率", "市销率", "市销率（TTM）",
    "股息率（%）", "股息率（%，TTM）",
    "总股本（万股）", "流通股本（万股）", "自由流通股本（万）",
    "复权因子",
]

# 不做 Winsorize 的列（标识符、日期等）
NON_NUMERIC_COLS = ["股票代码", "股票名称", "交易日"]


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def mad_winsorize(series: pd.Series, k: float = MAD_THRESHOLD) -> pd.Series:
    """
    MAD Winsorize：将超出 [median - k*MAD, median + k*MAD] 的值截断到边界。
    MAD = median(|x - median(x)|)
    """
    s = series.dropna()
    if len(s) < 4:
        return series  # 样本太少，跳过

    med = s.median()
    mad = (s - med).abs().median()

    if mad == 0:
        return series  # 所有值相同，无需处理

    lower = med - k * mad
    upper = med + k * mad
    return series.clip(lower=lower, upper=upper)


def _try_winsorize(series: pd.Series, k: float = MAD_THRESHOLD):
    """
    合并"检查 + 执行"为一步（优化 4）：
    一次计算 median/MAD，直接返回裁剪结果和是否发生变化的布尔值。
    替代原来的 _has_outliers() + mad_winsorize() 两次调用（节省 ~2/3 中位数运算）。

    Returns
    -------
    (clipped_series, changed: bool)
    """
    s = series.dropna()
    if len(s) < 4:
        return series, False

    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return series, False

    lower  = med - k * mad
    upper  = med + k * mad
    clipped = series.clip(lower=lower, upper=upper)
    changed = not clipped.equals(series)
    return clipped, changed


def _has_outliers(series: pd.Series, k: float = MAD_THRESHOLD) -> bool:
    """检查 series 是否含有超出 MAD 边界的极值（清洗前调用）。"""
    s = series.dropna()
    if len(s) < 4:
        return False
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return False
    lower = med - k * mad
    upper = med + k * mad
    return bool(((s < lower) | (s > upper)).any())


def clean_stock_df(
    df: pd.DataFrame,
    symbol: str = "",
    min_rows: int = MIN_ROWS,
) -> Optional[pd.DataFrame]:
    """
    对单只股票的 DataFrame 执行完整清洗流程。

    Parameters
    ----------
    df      : 原始 DataFrame，含表头列（见 PRICE_VOL_COLS / VALUATION_COLS）
    symbol  : 股票代码，仅用于日志
    min_rows: 新股剔除阈值

    Returns
    -------
    清洗后的 DataFrame，若新股不足 min_rows 行则返回 None。
    附加属性：
        df.attrs['invalid_cols']   - 缺失率 > 30% 的列名列表
        df.attrs['winsorized_cols'] - 被 Winsorize 的列名列表
    """
    df = df.copy()

    # 1. 按交易日排序（升序，便于 ffill）
    if "交易日" in df.columns:
        df = df.sort_values("交易日").reset_index(drop=True)

    # 2. 新股剔除：有效行数不足
    if len(df) < min_rows:
        return None

    # 3. 确定数值列
    numeric_cols = [
        c for c in df.columns
        if c not in NON_NUMERIC_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    invalid_cols: list[str] = []
    winsorized_cols: list[str] = []

    for col in numeric_cols:
        miss_rate = df[col].isna().mean()

        # 4a. 估值列（VALUATION_COLS）先做 PIT ffill，再重新评估缺失率。
        #     因为财报披露频率低，未 ffill 前的缺失率会虚高，不能在此阶段
        #     直接以 MISSING_HEAVY 剔除整列。
        winsorized_this = False
        if col in VALUATION_COLS:
            # 先 Winsorize 已有非空值，避免极值污染 ffill（一次计算）
            df[col], winsorized_this = _try_winsorize(df[col])
            # PIT ffill（不限长度）
            df[col] = df[col].ffill()
            # ffill 后重算缺失率；若仍超 30%（列本身没有任何历史值），则无效
            miss_rate_post = df[col].isna().mean()
            if miss_rate_post > MISSING_HEAVY:
                invalid_cols.append(col)
                continue
            # ffill 后做第二次 Winsorize（一次计算）
            df[col], w2 = _try_winsorize(df[col])
            winsorized_this = winsorized_this or w2
            if winsorized_this:
                winsorized_cols.append(col)
            continue  # 跳过下面的通用流程

        # 4a'. 非估值列：大量缺失（> 30%）→ 标记为无效，跳过后续处理
        if miss_rate > MISSING_HEAVY:
            invalid_cols.append(col)
            continue

        # 4b. 第一次 MAD Winsorize（先于填充，避免极值污染填充结果）
        df[col], winsorized_this = _try_winsorize(df[col])

        # 4c. 缺失值填充
        if col in PRICE_VOL_COLS:
            # 停牌：向前填充，最多 5 个交易日
            df[col] = df[col].ffill(limit=FFILL_PRICE_MAX)
        else:
            # 其他列：若随机缺失 < 5%，用中位数填充（保守策略）
            if 0 < miss_rate < MISSING_RANDOM:
                df[col] = df[col].fillna(df[col].median())

        # 4d. 第二次 MAD Winsorize（填充后可能引入新极值，再清洗一次）
        df[col], w2 = _try_winsorize(df[col])
        winsorized_this = winsorized_this or w2

        if winsorized_this:
            winsorized_cols.append(col)

    df.attrs["invalid_cols"]    = invalid_cols
    df.attrs["winsorized_cols"] = winsorized_cols
    return df


def load_and_clean(csv_path: str | Path) -> Optional[pd.DataFrame]:
    """读取单只股票 CSV 并执行清洗，返回清洗后 DataFrame（新股返回 None）。"""
    df = pd.read_csv(csv_path, dtype={"股票代码": str, "交易日": str})
    return clean_stock_df(df, symbol=str(csv_path))


# ── 诊断工具（清洗前调用，用于检查是否还有脏数据）──────────────────────────

def diagnose_df(df: pd.DataFrame) -> dict:
    """
    对 DataFrame 进行诊断，返回：
    - missing_rates   : 各列缺失率
    - outlier_cols    : 仍含极值的列（MAD 检验）
    - invalid_cols    : 缺失率 > 30% 的列
    - total_rows      : 总行数
    """
    numeric_cols = [
        c for c in df.columns
        if c not in NON_NUMERIC_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    missing_rates = {c: float(df[c].isna().mean()) for c in numeric_cols}
    outlier_cols  = [c for c in numeric_cols if _has_outliers(df[c])]
    invalid_cols  = [c for c, r in missing_rates.items() if r > MISSING_HEAVY]

    return {
        "total_rows":    len(df),
        "missing_rates": missing_rates,
        "outlier_cols":  outlier_cols,
        "invalid_cols":  invalid_cols,
    }
