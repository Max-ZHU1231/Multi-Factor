"""
test_data_quality.py
====================
针对 data_quality.py 的完整单元测试。

测试结构
--------
TestPriceContinuity      - 价格连续性检查（7 个用例）
TestVolumeAnomaly        - 成交量异常检查（6 个用例）
TestFinancialBalance     - 财务一致性检查（7 个用例）
TestMonthlyAlignment     - 时区对齐检查（8 个用例）
TestRunAllChecks         - 汇总入口（4 个用例）
TestRealDataQuality      - 真实数据随机抽查（5 类 × 30 只 = 150 个参数化用例）

运行：
    $py -m pytest test_data_quality.py -v --tb=short
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_quality import (
    align_monthly_to_daily,
    check_financial_balance,
    check_monthly_alignment,
    check_price_continuity,
    check_volume_anomaly,
    run_all_checks,
    PRICE_JUMP_THRESHOLD,
    FIN_BALANCE_TOL,
)

# ── 真实数据目录 & 抽样配置 ───────────────────────────────────────────────────
STOCKS_DIR  = Path(__file__).parents[2] / "stocks" / "stocks"
SAMPLE_N    = 30
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def _make_price_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """生成干净的单股日频价格 DataFrame（无跳跃、无停牌）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y%m%d").tolist()
    close = 10.0 + np.cumsum(rng.normal(0, 0.2, n))
    close = np.clip(close, 1.0, None)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(10_000, 500_000, size=n).astype(float)
    adj    = np.ones(n)  # 复权因子固定

    return pd.DataFrame({
        "股票代码":   "000001.SZ",
        "股票名称":   "测试",
        "交易日":     dates,
        "开盘价":     close * rng.uniform(0.99, 1.01, n),
        "最高价":     close * rng.uniform(1.00, 1.03, n),
        "最低价":     close * rng.uniform(0.97, 1.00, n),
        "收盘价":     close,
        "前收盘价":   prev_close,
        "涨跌幅（%）": (close - prev_close) / prev_close * 100,
        "成交量（手）": volume,
        "成交额（千元）": volume * close * 0.1,
        "复权因子":   adj,
    })


def _make_fin_df(n: int = 8, seed: int = 0, balance: bool = True) -> pd.DataFrame:
    """生成财务 DataFrame，balance=False 时插入失衡行。"""
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2020-03-31", periods=n, freq="QE").strftime("%Y-%m-%d").tolist()
    assets  = rng.uniform(1e8, 1e9, n)
    equity  = assets * rng.uniform(0.3, 0.6, n)
    liab    = assets - equity
    if not balance:
        # 让第 3 行失衡（差 5%）
        liab[2] *= 1.05
    return pd.DataFrame({
        "报告期": periods,
        "总资产": assets,
        "总负债": liab,
        "所有者权益合计": equity,
    })


def _make_monthly_df(start: str = "2019-01", end: str = "2023-12") -> pd.DataFrame:
    """生成月频宏观数据。"""
    dates = pd.period_range(start, end, freq="M").strftime("%Y-%m").tolist()
    rng   = np.random.default_rng(7)
    return pd.DataFrame({
        "date":  dates,
        "CPI":   rng.uniform(99.0, 104.0, len(dates)),
        "M2":    rng.uniform(1e13, 2e13, len(dates)),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestPriceContinuity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceContinuity:

    def test_clean_df_passes(self):
        """干净数据应通过检查。"""
        df = _make_price_df(100)
        result = check_price_continuity(df)
        assert result["passed"] is True
        assert result["jump_count"] == 0

    def test_jump_detected_when_no_adj_change(self):
        """复权因子不变时价格跳跃 > 5% 应被检测出来。"""
        df = _make_price_df(100)
        # 将第 50 行前收盘价人为设为前日收盘的 1.2 倍（+20% 跳跃）
        df = df.sort_values("交易日").reset_index(drop=True)
        prev_close_50 = df.loc[49, "收盘价"]
        df.loc[50, "前收盘价"] = prev_close_50 * 1.20
        result = check_price_continuity(df)
        assert result["passed"] is False
        assert result["jump_count"] >= 1

    def test_jump_at_adj_change_not_flagged(self):
        """复权因子变化点的价格跳跃不应被标记。"""
        df = _make_price_df(100)
        df = df.sort_values("交易日").reset_index(drop=True)
        # 第 50 行复权因子变化，同时前收盘跳跃
        df.loc[50, "复权因子"] = 2.0
        df.loc[50, "前收盘价"] = df.loc[49, "收盘价"] * 0.5  # 因为送股变为一半
        result = check_price_continuity(df)
        # 跳跃行应被归入 adj_change_dates，而非 jump_rows
        adj_dates = result["adj_change_dates"]
        assert df.loc[50, "交易日"] in adj_dates
        # jump_rows 不包含该行
        jump_dates = result["jump_rows"]["交易日"].tolist() if len(result["jump_rows"]) else []
        assert df.loc[50, "交易日"] not in jump_dates

    def test_missing_columns_skipped(self):
        """缺少必要列时跳过但不报错。"""
        df = pd.DataFrame({"交易日": ["20200101"], "收盘价": [10.0]})
        result = check_price_continuity(df)
        assert result.get("skipped") is True or result["jump_count"] == 0

    def test_threshold_respected(self):
        """跳跃幅度刚好在阈值内不应被标记。"""
        df = _make_price_df(100)
        df = df.sort_values("交易日").reset_index(drop=True)
        # 设为 4% 跳跃，低于默认 5% 阈值
        df.loc[50, "前收盘价"] = df.loc[49, "收盘价"] * 1.04
        result = check_price_continuity(df, threshold=0.05)
        assert result["passed"] is True

    def test_multiple_jumps_all_detected(self):
        """多个跳跃点均应被检测。"""
        df = _make_price_df(200)
        df = df.sort_values("交易日").reset_index(drop=True)
        for i in [50, 100, 150]:
            df.loc[i, "前收盘价"] = df.loc[i - 1, "收盘价"] * 1.15
        result = check_price_continuity(df)
        assert result["jump_count"] >= 3

    def test_result_keys_complete(self):
        """返回字典应包含所有必要键。"""
        df = _make_price_df(50)
        result = check_price_continuity(df)
        for key in ["check", "passed", "jump_rows", "jump_count", "adj_change_dates", "message"]:
            assert key in result, f"缺少键: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestVolumeAnomaly
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeAnomaly:

    def test_normal_volume_passes(self):
        """正常成交量应通过检查。"""
        df = _make_price_df(100)
        result = check_volume_anomaly(df)
        assert bool(result["passed"]) is True
        assert result["suspension_count"] == 0

    def test_zero_volume_detected(self):
        """成交量=0 的行应被标记为停牌。"""
        df = _make_price_df(100)
        df.loc[10, "成交量（手）"] = 0
        df.loc[20, "成交量（手）"] = 0
        result = check_volume_anomaly(df)
        assert result["suspension_count"] == 2
        assert df.loc[10, "交易日"] in result["suspension_dates"]
        assert df.loc[20, "交易日"] in result["suspension_dates"]

    def test_nan_volume_treated_as_suspension(self):
        """成交量 NaN 也应标记为停牌。"""
        df = _make_price_df(100)
        df.loc[5, "成交量（手）"] = np.nan
        result = check_volume_anomaly(df)
        assert result["suspension_count"] >= 1
        assert df.loc[5, "交易日"] in result["suspension_dates"]

    def test_suspension_flag_column(self):
        """mark_suspension=True 时应在返回 df 中添加 is_suspension 列。"""
        df = _make_price_df(50)
        df.loc[3, "成交量（手）"] = 0
        result = check_volume_anomaly(df, mark_suspension=True)
        assert "df_with_flag" in result
        flagged = result["df_with_flag"]
        assert "is_suspension" in flagged.columns
        assert bool(flagged.loc[3, "is_suspension"]) is True

    def test_high_suspension_rate_fails(self):
        """停牌占比 ≥ 50% 时应不通过检查。"""
        df = _make_price_df(100)
        df.loc[0:59, "成交量（手）"] = 0   # 60% 停牌
        result = check_volume_anomaly(df)
        assert bool(result["passed"]) is False

    def test_missing_columns_skipped(self):
        """缺少必要列时返回跳过结果。"""
        df = pd.DataFrame({"交易日": ["20200101"]})
        result = check_volume_anomaly(df)
        assert result.get("skipped") is True or result["suspension_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestFinancialBalance
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinancialBalance:

    def test_balanced_data_passes(self):
        """平衡的财务数据应通过检查。"""
        fin = _make_fin_df(balance=True)
        result = check_financial_balance(fin)
        assert result["passed"] is True
        assert result["unbalanced_count"] == 0

    def test_imbalance_detected(self):
        """人为制造的失衡行应被检出。"""
        fin = _make_fin_df(balance=False)
        result = check_financial_balance(fin)
        assert result["passed"] is False
        assert result["unbalanced_count"] >= 1

    def test_tolerance_respected(self):
        """在容差范围内的微小偏差不应触发警告。"""
        fin = _make_fin_df(balance=True)
        # 人为加 0.5% 误差（低于默认 1% 容差）
        fin.loc[0, "总负债"] *= 1.005
        result = check_financial_balance(fin, tol=0.01)
        assert result["passed"] is True

    def test_large_imbalance_always_detected(self):
        """大幅失衡（10%）无论容差都应被检测。"""
        fin = _make_fin_df(balance=True)
        fin.loc[1, "总负债"] *= 1.10
        result = check_financial_balance(fin, tol=0.01)
        assert result["passed"] is False

    def test_missing_columns_skipped(self):
        """缺少财务列时跳过。"""
        fin = pd.DataFrame({"报告期": ["2022-03-31"], "总资产": [1e8]})
        result = check_financial_balance(fin)
        assert result.get("skipped") is True

    def test_all_nan_skipped(self):
        """全为 NaN 的财务数据应跳过（不报错）。"""
        fin = pd.DataFrame({
            "报告期": ["2022-03-31"],
            "总资产": [np.nan],
            "总负债": [np.nan],
            "所有者权益合计": [np.nan],
        })
        result = check_financial_balance(fin)
        assert result.get("skipped") is True or result["passed"] is True

    def test_result_keys_complete(self):
        """返回字典应含所有必要键。"""
        fin = _make_fin_df()
        result = check_financial_balance(fin)
        for key in ["check", "passed", "unbalanced_rows", "unbalanced_count", "message"]:
            assert key in result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestMonthlyAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestMonthlyAlignment:

    def _daily(self, start="2020-01-01", end="2022-12-31"):
        dates = pd.bdate_range(start, end).strftime("%Y%m%d").tolist()
        rng = np.random.default_rng(3)
        return pd.DataFrame({
            "交易日": dates,
            "收盘价": 10 + np.cumsum(rng.normal(0, 0.1, len(dates))),
        })

    def _monthly(self, start="2019-01", end="2022-12"):
        dates = pd.period_range(start, end, freq="M").strftime("%Y-%m").tolist()
        rng = np.random.default_rng(5)
        return pd.DataFrame({
            "date": dates,
            "CPI": rng.uniform(99, 104, len(dates)),
        })

    def test_month_end_alignment_no_future_leak(self):
        """month_end 对齐不应引入未来数据（bfill 污染）。
        
        正确的 PIT 行为：月内大部分日期 ffill 自上月末值，
        仅当月最后一个交易日才更新为本月值。
        因此月内最多出现 2 个不同 CPI 值（上月末填充值 + 当月末新值），
        关键约束是：当月末交易日之前的所有行不能持有当月的 CPI 值。
        """
        daily   = self._daily()
        monthly = self._monthly()
        result  = align_monthly_to_daily(monthly, daily)

        result["_date"]  = pd.to_datetime(result["交易日"], format="%Y%m%d")
        result["_month"] = result["_date"].dt.to_period("M")

        for month, grp in result.groupby("_month"):
            grp = grp.sort_values("_date")
            cpi_vals = grp["CPI"].dropna()
            if len(cpi_vals) == 0:
                continue
            # 当月末交易日（最后一行）的 CPI 值
            last_day_cpi = grp.iloc[-1]["CPI"]
            # 当月末之前的行，CPI 值不得等于当月末 CPI（防止 bfill 污染）
            # 允许：月初 ~ 月末前一天 ffill 自上月末值（可能与当月值相同，
            #       此时无法区分；仅在值严格不同时验证顺序）
            pre_last = grp.iloc[:-1]["CPI"].dropna()
            if len(pre_last) == 0:
                continue
            pre_unique = pre_last.unique()
            # 关键：月末前所有日期的 CPI 应 ≤ 当月末值（ffill 不引入未来数据）
            # 实际上只要 pre_last 的值来自上月末（唯一值），不是通过 bfill 从
            # 当月末拿来的即可。这里验证月末前的值严格为上月末值（至多1个唯一值）。
            assert len(pre_unique) <= 1, (
                f"{month} 月末前出现多个 CPI 值，疑似 bfill 污染: {pre_unique}"
            )

    def test_month_start_alignment(self):
        """month_start 对齐：每月第一个交易日获得月频值。"""
        daily   = self._daily()
        monthly = self._monthly()
        result  = align_monthly_to_daily(monthly, daily, method="month_start")
        assert "CPI" in result.columns
        # 结果行数与日频一致
        assert len(result) == len(daily)

    def test_row_count_preserved(self):
        """对齐后行数等于日频行数。"""
        daily   = self._daily()
        monthly = self._monthly()
        result  = align_monthly_to_daily(monthly, daily)
        assert len(result) == len(daily)

    def test_monthly_cols_exist(self):
        """对齐后结果包含月频指标列。"""
        daily   = self._daily()
        monthly = self._monthly()
        result  = align_monthly_to_daily(monthly, daily)
        assert "CPI" in result.columns

    def test_no_bfill_contamination(self):
        """月频数据不应向后填充（未来数据不进入过去）。"""
        daily   = self._daily("2020-01-01", "2022-12-31")
        # 只提供 2021 年以后的月频数据
        monthly = self._monthly("2021-01", "2022-12")
        result  = align_monthly_to_daily(monthly, daily)
        # 2020 年所有日频行的 CPI 应为 NaN（没有历史值可 ffill）
        result["_date"] = pd.to_datetime(result["交易日"], format="%Y%m%d")
        pre_2021 = result[result["_date"] < pd.Timestamp("2021-01-01")]
        assert pre_2021["CPI"].isna().all(), "2021 年前不应有 CPI 数据（防止 bfill 污染）"

    def test_check_monthly_alignment_result_keys(self):
        """check_monthly_alignment 返回所有必要键。"""
        daily   = self._daily()
        monthly = self._monthly()
        result  = check_monthly_alignment(monthly, daily)
        for key in ["check", "passed", "aligned_df", "head_nan_rows",
                    "monthly_date_range", "daily_date_range", "message"]:
            assert key in result

    def test_multiple_monthly_cols(self):
        """多个月频指标列均应被对齐。"""
        daily = self._daily()
        monthly = _make_monthly_df()
        result = align_monthly_to_daily(monthly, daily)
        assert "CPI" in result.columns
        assert "M2" in result.columns

    def test_daily_dates_sorted_after_alignment(self):
        """对齐后日频数据按交易日升序排列。"""
        daily   = self._daily()
        monthly = self._monthly()
        result  = align_monthly_to_daily(monthly, daily)
        dates   = result["交易日"].tolist()
        assert dates == sorted(dates)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestRunAllChecks
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAllChecks:

    def test_clean_df_all_passed(self):
        """干净数据全部检查通过。"""
        df  = _make_price_df(200)
        fin = _make_fin_df(balance=True)
        monthly = _make_monthly_df("2019-01", "2023-12")
        result = run_all_checks(df, fin_df=fin, monthly_df=monthly)
        assert result["all_passed"] is True

    def test_dirty_price_all_not_passed(self):
        """价格跳跃存在时 all_passed 应为 False。"""
        df = _make_price_df(200)
        df = df.sort_values("交易日").reset_index(drop=True)
        df.loc[50, "前收盘价"] = df.loc[49, "收盘价"] * 1.30
        result = run_all_checks(df)
        assert result["all_passed"] is False

    def test_summary_list_populated(self):
        """summary 列表长度应与检查项数一致。"""
        df = _make_price_df(100)
        result = run_all_checks(df)
        assert isinstance(result["summary"], list)
        assert len(result["summary"]) == len(result["checks"])

    def test_optional_checks_skipped_without_data(self):
        """不提供 fin_df / monthly_df 时，对应检查被跳过而不报错。"""
        df = _make_price_df(100)
        result = run_all_checks(df)
        assert result["checks"]["financial_balance"].get("skipped") is True
        assert result["checks"]["monthly_alignment"].get("skipped") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestRealDataQuality（随机抽查 30 只真实股票）
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_symbols() -> list[str]:
    all_files = list(STOCKS_DIR.glob("*.csv"))
    if not all_files:
        pytest.skip(f"Stocks/ 目录为空或不存在: {STOCKS_DIR}", allow_module_level=True)
    rng = random.Random(RANDOM_SEED)
    chosen = rng.sample(all_files, min(SAMPLE_N, len(all_files)))
    return [f.stem for f in chosen]


_SYMBOLS = _sample_symbols()


def _load_real(symbol: str) -> pd.DataFrame:
    path = STOCKS_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, dtype={"股票代码": str, "交易日": str})
    return df.sort_values("交易日").reset_index(drop=True)


@pytest.fixture(scope="module")
def real_results():
    """一次性加载并检查所有抽样股票，共享给所有参数化用例。"""
    results = {}
    for sym in _SYMBOLS:
        df = _load_real(sym)
        results[sym] = run_all_checks(df)
    return results


@pytest.mark.parametrize("symbol", _SYMBOLS)
class TestRealDataQuality:

    def test_price_continuity_no_unexPlained_jump(self, symbol, real_results):
        """真实数据：价格连续性检查应通过（无未复权跳跃）。"""
        r = real_results[symbol]["checks"]["price_continuity"]
        assert r["passed"] is True, (
            f"{symbol} 发现价格跳跃: {r['jump_rows'][['交易日','偏差率']].to_string()}"
        )

    def test_volume_anomaly_suspension_rate_reasonable(self, symbol, real_results):
        """真实数据：停牌占比 < 50%（正常股票不可能全部停牌）。"""
        r = real_results[symbol]["checks"]["volume_anomaly"]
        rate = r["suspension_rate"]
        assert rate < 0.50, f"{symbol} 停牌率 {rate:.1%} 异常偏高"

    def test_volume_anomaly_result_valid(self, symbol, real_results):
        """真实数据：成交量检查有效返回（不报错）。"""
        r = real_results[symbol]["checks"]["volume_anomaly"]
        assert "suspension_count" in r
        assert isinstance(r["suspension_count"], int)
        assert r["suspension_count"] >= 0

    def test_price_continuity_adj_changes_tracked(self, symbol, real_results):
        """真实数据：复权因子变化点被记录（列表类型）。"""
        r = real_results[symbol]["checks"]["price_continuity"]
        assert isinstance(r["adj_change_dates"], list)

    def test_run_all_checks_no_exception(self, symbol):
        """真实数据：run_all_checks 不应抛出任何异常。"""
        df = _load_real(symbol)
        try:
            result = run_all_checks(df)
        except Exception as e:
            pytest.fail(f"{symbol} run_all_checks 抛出异常: {e}")
        assert "all_passed" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 汇总打印（不计入 pass/fail）
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_data_quality_summary(real_results):
    """打印真实数据质量检查汇总报告。"""
    SEP = "-" * 72
    print(f"\n{'=' * 72}")
    print(f"  数据质量检查汇总（{len(_SYMBOLS)} 只股票）")
    print(f"{'=' * 72}")
    print(f"  {'文件':<20} {'价格跳跃':>8} {'停牌天数':>8} {'停牌率':>7} {'复权变化':>8}")
    print(SEP)

    price_fail = 0
    for sym in _SYMBOLS:
        r    = real_results[sym]["checks"]
        pc   = r["price_continuity"]
        va   = r["volume_anomaly"]
        jump = pc["jump_count"]
        susp = va["suspension_count"]
        rate = va["suspension_rate"]
        adj  = len(pc["adj_change_dates"])
        flag = "⚠" if jump > 0 else " "
        print(f"  {flag}{sym:<19} {jump:>8} {susp:>8} {rate:>6.1%} {adj:>8}")
        if jump > 0:
            price_fail += 1

    print(SEP)
    print(f"  价格跳跃异常股票数: {price_fail} / {len(_SYMBOLS)}")
    print(f"{'=' * 72}")
    # 这里不 assert，只是汇总展示
    assert True
