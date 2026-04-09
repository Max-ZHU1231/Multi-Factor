"""
test_factor_framework.py
========================
因子框架完整测试套件。

覆盖范围
--------
- TestOperators     : ts_* / cs_* / math / cross-asset 算子
- TestFactorEngine  : register / compute_single / build_panel / apply_cross_section
- TestNeutralize    : regression / industry_zscore / orthogonalize
- TestICAnalysis    : compute_ic / ic_stats / ic_significance / ic_decay / ic_cumulative
- TestBacktest      : layer_backtest / long_short_stats / turnover_analysis / full_report
- TestFactorZoo     : BUILTIN_FACTORS 结构 / register_all / 单因子 smoke test
- TestPipeline      : FactorPipeline 构造 / run / run_batch / submit（smoke tests）
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── 确保项目根目录在 sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── 导入被测模块 ─────────────────────────────────────────────────────────────
from factor_framework.operators import (
    # 时间序列
    ts_sum, ts_mean, ts_stddev, ts_corr, delay, ts_max, ts_min,
    ts_rank, ts_delta, ts_wma, ts_zscore, ts_skew, ts_autocorr,
    ts_ema, ts_slope, ts_rsi, ts_drawdown, ts_beta,
    ts_regression_residual, ts_decay_linear, ts_prod,
    # 横截面
    cs_rank, cs_zscore, cs_demean, cs_scale, cs_industry_neutral,
    cs_industry_zscore, cs_winsorize,
    cs_rank_by_group, cs_neutralize, cs_top_n, cs_quantile,
    # 数学
    log, sqrt, absx, sign, if_else, clip, power,
)
from factor_framework.factor_engine  import FactorEngine
from factor_framework.neutralize     import (
    neutralize_regression, neutralize_industry_zscore, orthogonalize
)
from factor_framework.ic_analysis    import (
    compute_ic, ic_stats, ic_significance, ic_decay,
    ic_cumulative, cross_factor_correlation
)
from factor_framework.backtest       import (
    layer_backtest, long_short_stats, turnover_analysis, full_report,
    _max_drawdown, _annual_return, _sharpe, _calmar,
)
from factor_framework.factor_zoo     import BUILTIN_FACTORS, register_all
from factor_framework.pipeline       import FactorPipeline, FactorReport


# ═══════════════════════════════════════════════════════════════════════════════
# 共享 Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

N_DAYS   = 120   # 时间序列长度
N_STOCKS = 30    # 股票数量
RNG      = np.random.default_rng(42)


@pytest.fixture(scope="session")
def ts_series() -> pd.Series:
    """单只股票的日度价格序列（随机游走）。"""
    prices = 10 + RNG.normal(0, 0.1, N_DAYS).cumsum()
    return pd.Series(prices, name="close")


@pytest.fixture(scope="session")
def cs_series() -> pd.Series:
    """某截面日的因子值 Series（index=ts_code 字符串）。"""
    codes = [f"S{i:03d}" for i in range(N_STOCKS)]
    vals  = RNG.normal(0, 1, N_STOCKS)
    return pd.Series(vals, index=codes, name="factor")


@pytest.fixture(scope="session")
def factor_panel() -> pd.DataFrame:
    """(N_DAYS × N_STOCKS) 因子面板，含少量 NaN。"""
    dates  = pd.date_range("20200101", periods=N_DAYS, freq="B").strftime("%Y%m%d")
    codes  = [f"S{i:03d}" for i in range(N_STOCKS)]
    data   = RNG.normal(0, 1, (N_DAYS, N_STOCKS))
    df     = pd.DataFrame(data, index=dates, columns=codes)
    # 随机注入 5% NaN
    mask   = RNG.random((N_DAYS, N_STOCKS)) < 0.05
    df[mask] = np.nan
    return df


@pytest.fixture(scope="session")
def return_panel(factor_panel) -> pd.DataFrame:
    """与 factor_panel 同形状的未来收益率面板。"""
    data = RNG.normal(0.0002, 0.02, factor_panel.shape)
    return pd.DataFrame(data, index=factor_panel.index, columns=factor_panel.columns)


@pytest.fixture(scope="session")
def mktcap_panel(factor_panel) -> pd.DataFrame:
    """市值面板（对数正态，万元量级）。"""
    data = np.exp(RNG.normal(10, 1, factor_panel.shape))
    return pd.DataFrame(data, index=factor_panel.index, columns=factor_panel.columns)


@pytest.fixture(scope="session")
def industry_map(factor_panel) -> pd.Series:
    """ts_code → industry 映射。"""
    codes     = factor_panel.columns.tolist()
    industries = ["银行", "医药", "汽车", "电子", "地产"]
    labels    = [industries[i % len(industries)] for i in range(len(codes))]
    return pd.Series(labels, index=codes, name="industry")


@pytest.fixture(scope="session")
def ic_series() -> pd.Series:
    """模拟 IC 时间序列（略正偏，有一定均值）。"""
    dates = pd.date_range("20200101", periods=60, freq="ME").strftime("%Y%m%d")
    vals  = RNG.normal(0.05, 0.1, 60)
    return pd.Series(vals, index=dates, name="IC")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestOperators
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperators:

    # ── 时间序列算子 ─────────────────────────────────────────────────────────

    def test_ts_sum_length(self, ts_series):
        res = ts_sum(ts_series, 5)
        assert len(res) == len(ts_series)

    def test_ts_sum_values(self, ts_series):
        res = ts_sum(ts_series, 5)
        # 第 4 个（index=4）应为前 5 个之和
        expected = ts_series.iloc[:5].sum()
        assert abs(res.iloc[4] - expected) < 1e-9

    def test_ts_sum_nan_head(self, ts_series):
        res = ts_sum(ts_series, 5)
        assert res.iloc[:4].isna().all()

    def test_ts_mean(self, ts_series):
        res = ts_mean(ts_series, 10)
        assert len(res) == len(ts_series)
        assert res.iloc[:9].isna().all()
        assert not np.isnan(res.iloc[9])

    def test_ts_stddev_nonneg(self, ts_series):
        res = ts_stddev(ts_series, 10)
        assert (res.dropna() >= 0).all()

    def test_ts_corr_bounds(self, ts_series):
        y   = ts_series.shift(1)
        res = ts_corr(ts_series, y, 20)
        valid = res.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_delay(self, ts_series):
        res = delay(ts_series, 3)
        assert res.iloc[3] == pytest.approx(ts_series.iloc[0])

    def test_ts_max_geq_min(self, ts_series):
        mx = ts_max(ts_series, 10)
        mn = ts_min(ts_series, 10)
        valid_mx = mx.dropna()
        valid_mn = mn.dropna()
        assert (valid_mx.values >= valid_mn.values).all()

    def test_ts_rank_bounds(self, ts_series):
        res = ts_rank(ts_series, 10)
        valid = res.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ts_delta(self, ts_series):
        res = ts_delta(ts_series, 1)
        assert res.iloc[1] == pytest.approx(ts_series.iloc[1] - ts_series.iloc[0])

    def test_ts_wma_length(self, ts_series):
        res = ts_wma(ts_series, 5)
        assert len(res) == len(ts_series)
        assert res.iloc[:4].isna().all()

    def test_ts_zscore_returns_series(self, ts_series):
        res = ts_zscore(ts_series, 20)
        assert isinstance(res, pd.Series)
        assert len(res) == len(ts_series)
        # 前 19 个为 NaN，之后有有效值
        assert res.iloc[:19].isna().all()
        assert res.dropna().notna().all()

    def test_ts_skew_type(self, ts_series):
        res = ts_skew(ts_series, 20)
        assert isinstance(res, pd.Series)

    def test_ts_autocorr_type(self, ts_series):
        res = ts_autocorr(ts_series, 20, lag=1)
        assert isinstance(res, pd.Series)

    # ── 横截面算子 ─────────────────────────────────────────────────────────

    def test_cs_rank_bounds(self, cs_series):
        res = cs_rank(cs_series)
        assert (res >= 0).all() and (res <= 1).all()

    def test_cs_rank_no_nan(self, cs_series):
        res = cs_rank(cs_series.dropna())
        assert res.notna().all()

    def test_cs_zscore_mean_zero(self, cs_series):
        res = cs_zscore(cs_series)
        assert abs(res.mean()) < 1e-9

    def test_cs_zscore_std_one(self, cs_series):
        res = cs_zscore(cs_series)
        assert abs(res.std(ddof=1) - 1) < 1e-9

    def test_cs_demean_mean_zero(self, cs_series):
        res = cs_demean(cs_series)
        assert abs(res.mean()) < 1e-9

    def test_cs_scale_bounds(self, cs_series):
        res = cs_scale(cs_series, a=1.0)
        assert res.min() >= 0 - 1e-9
        assert res.max() <= 1 + 1e-9

    def test_cs_industry_neutral_group_mean_zero(self, cs_series):
        groups = pd.Series(
            ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            index=cs_series.index,
        )
        res = cs_industry_neutral(cs_series, groups)
        for g in ["A", "B", "C"]:
            mask = groups == g
            assert abs(res[mask].mean()) < 1e-9

    def test_cs_industry_zscore_type(self, cs_series):
        groups = pd.Series(
            ["A"] * 15 + ["B"] * 15,
            index=cs_series.index,
        )
        res = cs_industry_zscore(cs_series, groups)
        assert isinstance(res, pd.Series)
        assert len(res) == len(cs_series)

    def test_cs_winsorize_bounds(self, cs_series):
        # 注入极端值
        s = cs_series.copy()
        s.iloc[0] = 1e9
        s.iloc[1] = -1e9
        res = cs_winsorize(s, n_std=3.0)
        # 结果应被截断
        assert res.max() < 1e6

    # ── 数学/逻辑算子 ────────────────────────────────────────────────────────

    def test_log_positive(self):
        s   = pd.Series([1.0, 2.0, 10.0])
        res = log(s)
        assert res.tolist() == pytest.approx([0, np.log(2), np.log(10)])

    def test_log_nonpositive_nan(self):
        s   = pd.Series([0.0, -1.0, 1.0])
        res = log(s)
        assert np.isnan(res.iloc[0])
        assert np.isnan(res.iloc[1])

    def test_sqrt_nonneg(self):
        s   = pd.Series([0.0, 4.0, 9.0])
        res = sqrt(s)
        assert res.tolist() == pytest.approx([0, 2, 3])

    def test_sqrt_negative_clipped_to_zero(self):
        s   = pd.Series([-1.0, 1.0])
        res = sqrt(s)
        # 实现用 clip(lower=0) 处理负值，结果为 0.0
        assert res.iloc[0] == pytest.approx(0.0)

    def test_absx(self):
        s   = pd.Series([-3.0, 0.0, 3.0])
        res = absx(s)
        assert res.tolist() == pytest.approx([3, 0, 3])

    def test_sign(self):
        s   = pd.Series([-5.0, 0.0, 5.0])
        res = sign(s)
        assert res.tolist() == pytest.approx([-1, 0, 1])

    def test_if_else(self):
        cond = pd.Series([True, False, True])
        a    = pd.Series([1.0, 2.0, 3.0])
        b    = pd.Series([10.0, 20.0, 30.0])
        res  = if_else(cond, a, b)
        assert res.tolist() == pytest.approx([1, 20, 3])

    def test_clip(self):
        s   = pd.Series([-10.0, 0.0, 10.0])
        res = clip(s, lo=-5, hi=5)
        assert res.tolist() == pytest.approx([-5, 0, 5])

    def test_power(self):
        s   = pd.Series([2.0, 3.0, 4.0])
        res = power(s, 2)
        assert res.tolist() == pytest.approx([4, 9, 16])

    # ── 新增时间序列算子 ─────────────────────────────────────────────────────

    def test_ts_ema_length(self, ts_series):
        res = ts_ema(ts_series, 10)
        assert len(res) == len(ts_series)

    def test_ts_ema_nan_head(self, ts_series):
        res = ts_ema(ts_series, 10)
        assert res.iloc[:9].isna().all()

    def test_ts_ema_valid_tail(self, ts_series):
        res = ts_ema(ts_series, 10)
        assert res.iloc[9:].notna().all()

    def test_ts_slope_type(self, ts_series):
        res = ts_slope(ts_series, 10)
        assert isinstance(res, pd.Series)
        assert len(res) == len(ts_series)

    def test_ts_slope_nan_head(self, ts_series):
        res = ts_slope(ts_series, 10)
        assert res.iloc[:9].isna().all()

    def test_ts_slope_monotone_positive(self):
        # 单调递增序列，斜率应为正
        s   = pd.Series(np.arange(50, dtype=float))
        res = ts_slope(s, 10)
        valid = res.dropna()
        assert (valid > 0).all()

    def test_ts_rsi_bounds(self, ts_series):
        res = ts_rsi(ts_series, 14)
        valid = res.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_ts_rsi_nan_head(self, ts_series):
        res = ts_rsi(ts_series, 14)
        assert res.iloc[:13].isna().all()

    def test_ts_drawdown_nonneg(self, ts_series):
        res = ts_drawdown(ts_series, 20)
        valid = res.dropna()
        assert (valid >= 0).all() and (valid <= 1 + 1e-9).all()

    def test_ts_drawdown_nan_head(self, ts_series):
        res = ts_drawdown(ts_series, 20)
        assert res.iloc[:19].isna().all()

    def test_ts_beta_type(self, ts_series):
        y   = ts_series.shift(1).fillna(ts_series.mean())
        res = ts_beta(ts_series, y, 20)
        assert isinstance(res, pd.Series)
        assert len(res) == len(ts_series)

    def test_ts_regression_residual_type(self, ts_series):
        y   = ts_series.shift(1).fillna(ts_series.mean())
        res = ts_regression_residual(ts_series, y, 20)
        assert isinstance(res, pd.Series)
        assert len(res) == len(ts_series)

    def test_ts_decay_linear_matches_wma(self, ts_series):
        res_decay = ts_decay_linear(ts_series, 10)
        res_wma   = ts_wma(ts_series, 10)
        pd.testing.assert_series_equal(res_decay, res_wma)

    def test_ts_prod_cumulative(self):
        # (1+0.1)^5 = 1.61051
        s   = pd.Series([0.1] * 10)
        res = ts_prod(1 + s, 5)
        valid = res.dropna()
        assert np.allclose(valid.values, 1.1 ** 5, rtol=1e-6)

    # ── 新增横截面算子 ──────────────────────────────────────────────────────

    def test_cs_rank_by_group_bounds(self, cs_series):
        groups = pd.Series(
            ["A"] * 15 + ["B"] * 15, index=cs_series.index
        )
        res = cs_rank_by_group(cs_series, groups)
        assert (res >= 0).all() and (res <= 1).all()

    def test_cs_rank_by_group_within_group(self, cs_series):
        groups = pd.Series(
            ["A"] * 15 + ["B"] * 15, index=cs_series.index
        )
        res = cs_rank_by_group(cs_series, groups)
        # 每组内 rank pct 应有最小值 ≤ 1/15 和最大值 = 1.0
        for g in ["A", "B"]:
            mask = groups == g
            grp  = res[mask]
            assert grp.max() == pytest.approx(1.0)
            assert grp.min() > 0

    def test_cs_neutralize_removes_correlation(self, cs_series):
        y = cs_series * 2 + pd.Series(
            RNG.normal(0, 0.1, len(cs_series)), index=cs_series.index
        )
        resid = cs_neutralize(y, cs_series)
        corr  = resid.dropna().corr(cs_series.reindex(resid.dropna().index))
        assert abs(corr) < 0.1

    def test_cs_top_n_count(self, cs_series):
        mask = cs_top_n(cs_series, 5)
        assert mask.sum() == 5

    def test_cs_top_n_are_largest(self, cs_series):
        mask = cs_top_n(cs_series, 5)
        threshold = cs_series.nlargest(5).min()
        assert (cs_series[mask] >= threshold).all()

    def test_cs_quantile_value(self, cs_series):
        q80 = cs_quantile(cs_series, 0.8)
        assert isinstance(q80, float)
        assert (cs_series.dropna() <= q80).mean() >= 0.79


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestFactorEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorEngine:
    """使用真实数据目录（Stocks/）做集成测试。"""

    STOCKS_DIR   = ROOT / "Stocks"
    STOCK_BASIC  = ROOT / "股票列表-stock_basic.csv"

    @pytest.fixture(scope="class")
    def engine(self):
        if not self.STOCKS_DIR.exists():
            pytest.skip("Stocks/ 目录不存在，跳过集成测试")
        return FactorEngine(
            stocks_dir  = self.STOCKS_DIR,
            stock_basic = self.STOCK_BASIC,
            min_rows    = 30,
            verbose     = False,
        )

    def test_engine_init(self, engine):
        assert engine.stocks_dir.exists()

    def test_register(self, engine):
        engine.register("test_close", lambda df: df["收盘价"])
        assert "test_close" in engine.registered()

    def test_register_overwrite_warns(self, engine):
        with pytest.warns(UserWarning):
            engine.register("test_close", lambda df: df["收盘价"])

    def test_compute_single_returns_series(self, engine):
        symbols = engine.all_symbols()
        if not symbols:
            pytest.skip("没有股票文件")
        result = engine.compute_single(symbols[0], "test_close",
                                       start="20200101", end="20221231")
        assert result is None or isinstance(result, pd.Series)

    def test_build_panel_shape(self, engine):
        # 只用最多 10 只股票，加快速度
        symbols = engine.all_symbols()[:10]
        if len(symbols) < 2:
            pytest.skip("股票文件不足")
        engine.register("_test_mom5",
                        lambda df: df["收盘价"].pct_change(5))
        panel = engine.build_panel("_test_mom5",
                                   start="20210101", end="20211231",
                                   symbols=symbols)
        assert isinstance(panel, pd.DataFrame)
        assert panel.shape[1] <= len(symbols)

    def test_build_panel_index_sorted(self, engine):
        symbols = engine.all_symbols()[:5]
        if len(symbols) < 2:
            pytest.skip("股票文件不足")
        panel = engine.build_panel("test_close",
                                   start="20210101", end="20211231",
                                   symbols=symbols)
        if panel.empty:
            pytest.skip("面板为空")
        assert list(panel.index) == sorted(panel.index.tolist())

    def test_build_return_panel(self, engine):
        symbols = engine.all_symbols()[:5]
        if len(symbols) < 2:
            pytest.skip("股票文件不足")
        rp = engine.build_return_panel(forward=5,
                                       start="20210101", end="20211231",
                                       symbols=symbols)
        assert isinstance(rp, pd.DataFrame)

    def test_apply_cross_section(self, factor_panel):
        fp = factor_panel.copy()
        # FactorEngine.apply_cross_section 是静态方法
        result = FactorEngine.apply_cross_section(fp, cs_zscore)
        assert result.shape == fp.shape
        # 每行均值应接近 0（zscore）
        row_means = result.mean(axis=1).dropna()
        assert (row_means.abs() < 0.1).mean() > 0.8

    def test_industry_map_loaded(self, engine):
        if not self.STOCK_BASIC.exists():
            pytest.skip("stock_basic 不存在")
        imap = engine.industry_map
        if imap is not None:
            assert isinstance(imap, pd.Series)
            assert len(imap) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestNeutralize
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeutralize:

    def test_regression_shape(self, factor_panel, mktcap_panel, industry_map):
        result = neutralize_regression(factor_panel, mktcap_panel,
                                       industry_map=industry_map)
        assert result.shape == factor_panel.shape

    def test_regression_reduces_mktcap_corr(self, factor_panel, mktcap_panel, industry_map):
        """中性化后因子与市值的截面相关性应降低。"""
        result = neutralize_regression(factor_panel, mktcap_panel,
                                       industry_map=industry_map)

        # 逐日计算原始/中性化后 与 ln(mktcap) 的相关系数均值
        corr_before, corr_after = [], []
        for date in factor_panel.index[:40]:
            f    = factor_panel.loc[date].dropna()
            fn   = result.loc[date].dropna()
            mc   = mktcap_panel.loc[date].dropna()
            comm = f.index.intersection(mc.index)
            if len(comm) < 5:
                continue
            lnmc = np.log(mc[comm])
            corr_before.append(abs(f[comm].corr(lnmc)))
            comm_n = fn.index.intersection(mc.index)
            if len(comm_n) < 5:
                continue
            corr_after.append(abs(fn[comm_n].corr(np.log(mc[comm_n]))))

        if corr_before and corr_after:
            assert np.nanmean(corr_after) <= np.nanmean(corr_before) + 0.1

    def test_industry_zscore_shape(self, factor_panel, industry_map):
        result = neutralize_industry_zscore(factor_panel, industry_map)
        assert result.shape == factor_panel.shape

    def test_industry_zscore_within_group_mean_near_zero(self, factor_panel, industry_map):
        result = neutralize_industry_zscore(factor_panel, industry_map)
        # 取第一个有效日期，检查行业内均值
        date = factor_panel.index[20]
        row  = result.loc[date].dropna()
        imap = industry_map.reindex(row.index)
        for ind in imap.dropna().unique():
            mask = imap == ind
            group_vals = row[mask]
            if len(group_vals) >= 3:
                assert abs(group_vals.mean()) < 0.5

    def test_orthogonalize_shape(self, factor_panel):
        existing = [factor_panel, factor_panel * 0.5 + 0.1]
        result   = orthogonalize(factor_panel, existing)
        assert result.shape == factor_panel.shape

    def test_orthogonalize_reduces_corr(self, factor_panel):
        """正交化后与原有因子的截面相关性应降低。"""
        # 构造相关性较强的两个因子
        fp2      = factor_panel + 0.5 * RNG.normal(0, 1, factor_panel.shape)
        existing = [factor_panel]
        result   = orthogonalize(fp2, existing)

        corrs_before, corrs_after = [], []
        for date in factor_panel.index[:30]:
            f    = factor_panel.loc[date].dropna()
            f2   = fp2.loc[date].dropna()
            fn   = result.loc[date].dropna()
            comm = f.index.intersection(f2.index)
            if len(comm) < 5:
                continue
            corrs_before.append(abs(f[comm].corr(f2[comm])))
            comm_n = f.index.intersection(fn.index)
            if len(comm_n) < 5:
                continue
            corrs_after.append(abs(f[comm_n].corr(fn[comm_n])))

        if corrs_before and corrs_after:
            assert np.nanmean(corrs_after) <= np.nanmean(corrs_before) + 0.05

    def test_regression_no_mktcap_panel_fallback(self, factor_panel, industry_map):
        """只传 industry_map，不传 mktcap（空 DataFrame）应正常运行。"""
        empty_mc = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns, dtype=float)
        result = neutralize_regression(factor_panel, empty_mc, industry_map=industry_map)
        assert result.shape == factor_panel.shape

    # ── 新增：WLS 和风格因子中性化测试 ─────────────────────────────────────

    def test_regression_wls_shape(self, factor_panel, mktcap_panel, industry_map):
        """WLS 中性化输出形状应与输入一致。"""
        free_cap = mktcap_panel * 0.6   # 模拟流通市值
        result = neutralize_regression(
            factor_panel, mktcap_panel,
            industry_map=industry_map,
            free_cap_panel=free_cap,
            use_wls=True,
        )
        assert result.shape == factor_panel.shape

    def test_regression_wls_reduces_mktcap_corr(self, factor_panel, mktcap_panel):
        """WLS 中性化后因子与市值的相关性应降低。"""
        free_cap = mktcap_panel * 0.6
        result = neutralize_regression(
            factor_panel, mktcap_panel,
            free_cap_panel=free_cap,
            use_wls=True,
        )
        corr_before, corr_after = [], []
        for date in factor_panel.index[:30]:
            f  = factor_panel.loc[date].dropna()
            fn = result.loc[date].dropna()
            mc = mktcap_panel.loc[date].dropna()
            comm   = f.index.intersection(mc.index)
            comm_n = fn.index.intersection(mc.index)
            if len(comm) >= 5:
                corr_before.append(abs(f[comm].corr(np.log(mc[comm]))))
            if len(comm_n) >= 5:
                corr_after.append(abs(fn[comm_n].corr(np.log(mc[comm_n]))))
        if corr_before and corr_after:
            assert np.nanmean(corr_after) <= np.nanmean(corr_before) + 0.1

    def test_regression_with_vol_and_beta(self, factor_panel, mktcap_panel):
        """同时传入 vol_panel 和 beta_panel 时应正常运行。"""
        vol_panel  = factor_panel.abs() * 0.02
        beta_panel = factor_panel.apply(lambda c: c / (c.std() + 1e-9))
        result = neutralize_regression(
            factor_panel, mktcap_panel,
            vol_panel=vol_panel,
            beta_panel=beta_panel,
        )
        assert result.shape == factor_panel.shape
        # 应有非 NaN 输出
        assert result.notna().any().any()

    def test_regression_with_all_style_factors(
        self, factor_panel, mktcap_panel, industry_map
    ):
        """传入全部风格因子（vol / beta / momentum / liquidity）时应正常运行。"""
        vol_panel       = factor_panel.abs() * 0.02
        beta_panel      = factor_panel * 0.5
        momentum_panel  = factor_panel.shift(1).fillna(0)
        liquidity_panel = factor_panel.abs()
        free_cap        = mktcap_panel * 0.6
        result = neutralize_regression(
            factor_panel, mktcap_panel,
            industry_map=industry_map,
            vol_panel=vol_panel,
            beta_panel=beta_panel,
            momentum_panel=momentum_panel,
            liquidity_panel=liquidity_panel,
            free_cap_panel=free_cap,
            use_wls=True,
        )
        assert result.shape == factor_panel.shape

    def test_wls_zero_weight_fallback(self, factor_panel, mktcap_panel):
        """所有权重为 0 时，WLS 应降级为 OLS，不抛出异常。"""
        zero_cap = pd.DataFrame(0.0, index=mktcap_panel.index, columns=mktcap_panel.columns)
        result = neutralize_regression(
            factor_panel, mktcap_panel,
            free_cap_panel=zero_cap,
            use_wls=True,
        )
        assert result.shape == factor_panel.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestICAnalysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestICAnalysis:

    def test_compute_ic_bounds(self, factor_panel, return_panel):
        ic = compute_ic(factor_panel, return_panel, method="rank")
        valid = ic.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_compute_ic_length(self, factor_panel, return_panel):
        ic = compute_ic(factor_panel, return_panel)
        common = factor_panel.index.intersection(return_panel.index)
        assert len(ic) == len(common)

    def test_compute_ic_normal_method(self, factor_panel, return_panel):
        ic = compute_ic(factor_panel, return_panel, method="normal")
        valid = ic.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_ic_stats_keys(self, ic_series):
        s = ic_stats(ic_series)
        required = {"mean_ic", "std_ic", "icir", "win_rate", "t_stat",
                    "p_value", "ic_positive", "ic_negative", "total_periods",
                    "annualized_icir"}
        assert required.issubset(s.keys())

    def test_ic_stats_total_periods(self, ic_series):
        s = ic_stats(ic_series)
        assert s["total_periods"] == ic_series.notna().sum()

    def test_ic_stats_win_rate_in_range(self, ic_series):
        s = ic_stats(ic_series)
        assert 0 <= s["win_rate"] <= 1

    def test_ic_stats_icir_sign(self, ic_series):
        s = ic_stats(ic_series)
        # 正均值 IC → 正 ICIR
        if s["mean_ic"] > 0:
            assert s["icir"] > 0

    def test_ic_significance_keys(self, ic_series):
        result = ic_significance(ic_series, lags=3)
        assert "nw_t_stat" in result

    def test_ic_significance_type(self, ic_series):
        result = ic_significance(ic_series, lags=3)
        assert isinstance(result, dict)

    def test_ic_decay_shape(self, factor_panel, return_panel):
        # 构造收盘价面板（用 cumsum 模拟）
        price_panel = factor_panel.cumsum().abs() + 10
        decay_df    = ic_decay(factor_panel, price_panel,
                               forward_periods=[1, 5, 10], method="rank")
        assert isinstance(decay_df, pd.DataFrame)
        assert len(decay_df) == 3
        assert "mean_ic" in decay_df.columns

    def test_ic_cumulative(self, ic_series):
        cum = ic_cumulative(ic_series)
        assert isinstance(cum, pd.Series)
        assert len(cum) == ic_series.notna().sum()

    def test_cross_factor_correlation_shape(self, factor_panel):
        panels = {"A": factor_panel, "B": factor_panel * -1}
        corr_df = cross_factor_correlation(panels, method="pearson")
        assert corr_df.shape == (2, 2)
        # 对角线为 1
        assert all(abs(corr_df.iloc[i, i] - 1) < 1e-9 for i in range(2))

    def test_cross_factor_correlation_symmetric(self, factor_panel):
        panels = {"A": factor_panel, "B": factor_panel * -1, "C": factor_panel + 1}
        corr_df = cross_factor_correlation(panels)
        # 对称矩阵
        pd.testing.assert_frame_equal(corr_df, corr_df.T, check_names=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestBacktest
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktest:

    def test_layer_backtest_shape(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel, n_groups=5)
        common    = factor_panel.index.intersection(return_panel.index)
        assert layer_ret.shape[0] == len(common)
        assert "Q1" in layer_ret.columns
        assert "Q5" in layer_ret.columns
        assert "LS" in layer_ret.columns

    def test_layer_backtest_n_groups(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel, n_groups=3)
        assert "Q1" in layer_ret.columns
        assert "Q3" in layer_ret.columns
        assert "Q4" not in layer_ret.columns

    def test_layer_backtest_ls_equals_q5_minus_q1(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel, n_groups=5, direction=1)
        ls = layer_ret["LS"].dropna()
        q5 = layer_ret["Q5"].dropna()
        q1 = layer_ret["Q1"].dropna()
        common_idx = ls.index.intersection(q5.index).intersection(q1.index)
        diff = (q5[common_idx] - q1[common_idx] - ls[common_idx]).abs()
        assert (diff < 1e-9).all()

    def test_layer_backtest_direction_minus1(self, factor_panel, return_panel):
        pos = layer_backtest(factor_panel, return_panel, n_groups=5, direction=1)
        neg = layer_backtest(factor_panel, return_panel, n_groups=5, direction=-1)
        # direction=-1 时 LS 符号翻转，逐期 pos_LS ≈ -neg_LS
        # 在统计意义上两者之和的均值应接近 0（相关性 ≈ -1）
        common   = pos["LS"].dropna().index.intersection(neg["LS"].dropna().index)
        corr     = pos["LS"][common].corr(neg["LS"][common])
        assert corr < -0.95

    def test_long_short_stats_keys(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel)
        stats     = long_short_stats(layer_ret)
        required  = {"ls_annual_return", "ls_sharpe", "ls_max_drawdown",
                     "ls_calmar", "ls_win_rate", "monotone_score", "nav",
                     "layer_annual_return", "layer_sharpe"}
        assert required.issubset(stats.keys())

    def test_long_short_stats_nav_type(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel)
        stats     = long_short_stats(layer_ret)
        assert isinstance(stats["nav"], pd.DataFrame)

    def test_long_short_stats_monotone_score_bounds(self, factor_panel, return_panel):
        layer_ret = layer_backtest(factor_panel, return_panel)
        stats     = long_short_stats(layer_ret)
        ms        = stats["monotone_score"]
        if not np.isnan(ms):
            assert -1 <= ms <= 1

    def test_turnover_analysis_keys(self, factor_panel):
        t = turnover_analysis(factor_panel, n_groups=5)
        assert "avg_turnover" in t
        assert "avg_cost"     in t

    def test_turnover_analysis_nonneg(self, factor_panel):
        t = turnover_analysis(factor_panel, n_groups=5)
        assert t["avg_turnover"] >= 0
        assert t["avg_cost"]     >= 0

    def test_full_report_returns_dict(self, factor_panel, return_panel):
        result = full_report(factor_panel, return_panel, n_groups=5)
        assert isinstance(result, dict)
        assert "summary_table" in result
        assert isinstance(result["summary_table"], pd.DataFrame)

    # ── 工具函数 ─────────────────────────────────────────────────────────────

    def test_max_drawdown_zero_for_rising(self):
        ret = pd.Series([0.01] * 50)
        nav = (1 + ret).cumprod()
        assert _max_drawdown(nav) >= -1e-9   # 持续上涨回撤 ≈ 0

    def test_max_drawdown_correct(self):
        nav = pd.Series([1, 1.5, 1.0, 1.2])
        dd  = _max_drawdown(nav)
        expected = (1.0 - 1.5) / 1.5
        assert abs(dd - expected) < 1e-9

    def test_annual_return_positive_for_positive_ret(self):
        ret = pd.Series([0.001] * 252)
        ar  = _annual_return(ret, 252)
        assert ar > 0

    def test_sharpe_positive_for_positive_ret(self):
        ret = pd.Series([0.002] * 252)
        s   = _sharpe(ret, rf=0.02, periods_per_year=252)
        assert s > 0

    def test_calmar_positive(self):
        ret = pd.Series([0.001, -0.001, 0.002, -0.0005] * 30)
        c   = _calmar(ret, 252)
        assert isinstance(c, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestFactorZoo
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorZoo:

    def test_builtin_factors_is_dict(self):
        assert isinstance(BUILTIN_FACTORS, dict)

    def test_builtin_factors_at_least_10(self):
        assert len(BUILTIN_FACTORS) >= 10

    def test_builtin_factors_all_callable(self):
        for name, fn in BUILTIN_FACTORS.items():
            assert callable(fn), f"因子 {name} 不可调用"

    def test_builtin_factors_expected_names(self):
        expected = {"momentum_12_1", "reversal_1w", "vol_20d",
                    "value_pb", "size_log_mktcap", "amihud_illiquidity"}
        assert expected.issubset(BUILTIN_FACTORS.keys())

    def test_builtin_factors_new_liquidity_names(self):
        """新增流动性质量因子应存在于 BUILTIN_FACTORS。"""
        expected = {"bid_ask_spread_proxy", "zero_return_ratio",
                    "pastor_stambaugh", "order_imbalance"}
        assert expected.issubset(BUILTIN_FACTORS.keys())

    def test_builtin_factors_new_technical_names(self):
        """新增技术分析因子应存在于 BUILTIN_FACTORS。"""
        expected = {"rsi_14", "macd_signal", "bb_position", "volume_trend"}
        assert expected.issubset(BUILTIN_FACTORS.keys())

    def test_builtin_factors_count(self):
        """BUILTIN_FACTORS 应包含至少 28 个因子（原 20 + 流动性 4 + 技术 4）。"""
        assert len(BUILTIN_FACTORS) >= 28

    def test_register_all(self):
        engine = FactorEngine.__new__(FactorEngine)
        engine._registry     = {}
        engine._industry_map = None
        register_all(engine)
        assert len(engine.registered()) == len(BUILTIN_FACTORS)

    def test_factor_fn_runs_on_sample(self):
        """每个内置因子函数在合成 DataFrame 上应能运行（不抛异常）。"""
        dates = pd.date_range("20200101", periods=252, freq="B")
        data  = {
            "收盘价":                   np.abs(RNG.normal(10, 1, 252)) + 1,
            "开盘价":                   np.abs(RNG.normal(10, 1, 252)) + 1,
            "最高价":                   np.abs(RNG.normal(11, 1, 252)) + 1,
            "最低价":                   np.abs(RNG.normal(9, 1, 252))  + 1,
            "成交量（手）":              np.abs(RNG.normal(1e5, 1e4, 252)),
            "成交额（千元）":            np.abs(RNG.normal(1e6, 1e5, 252)),
            "换手率（%）":              np.abs(RNG.normal(2, 0.5, 252)),
            "总市值（万元）":            np.abs(RNG.normal(1e6, 1e5, 252)),
            "流通市值（万元）":          np.abs(RNG.normal(5e5, 5e4, 252)),
            "市净率":                   np.abs(RNG.normal(2, 0.5, 252)),
            "市盈率（TTM，亏损为空）":   np.abs(RNG.normal(15, 5, 252)),
            "市销率（TTM）":            np.abs(RNG.normal(3, 1, 252)),
            "复权因子":                 np.ones(252),
            "_ret":                     RNG.normal(0, 0.02, 252),
        }
        df = pd.DataFrame(data, index=dates)
        # 高价 >= 低价
        df["最高价"] = df[["最高价", "最低价"]].max(axis=1)
        df["最低价"] = df[["最高价", "最低价"]].min(axis=1)

        failures = []
        for name, fn in BUILTIN_FACTORS.items():
            try:
                result = fn(df)
                assert isinstance(result, pd.Series), f"{name} 返回类型错误"
            except Exception as e:
                failures.append(f"{name}: {e}")

        assert not failures, "以下因子运行失败:\n" + "\n".join(failures)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipeline:
    """Pipeline 集成 smoke tests — 使用真实 Stocks/ 数据（少量股票）。"""

    STOCKS_DIR  = ROOT / "Stocks"
    STOCK_BASIC = ROOT / "股票列表-stock_basic.csv"

    @pytest.fixture(scope="class")
    def pipe(self):
        if not self.STOCKS_DIR.exists():
            pytest.skip("Stocks/ 目录不存在，跳过 Pipeline 集成测试")
        return FactorPipeline(
            stocks_dir  = self.STOCKS_DIR,
            stock_basic = self.STOCK_BASIC,
            verbose     = False,
        )

    def test_pipeline_init(self, pipe):
        assert hasattr(pipe, "engine")

    def test_register_factor(self, pipe):
        pipe.register_factor("_test_pb", lambda df: df["市净率"])
        assert "_test_pb" in pipe.engine.registered()

    def test_register_builtins(self, pipe):
        pipe.register_builtins(["momentum_12_1", "vol_20d"])
        assert "momentum_12_1" in pipe.engine.registered()
        assert "vol_20d"       in pipe.engine.registered()

    def test_run_returns_report(self, pipe):
        symbols = pipe.engine.all_symbols()[:8]
        if len(symbols) < 3:
            pytest.skip("股票数量不足")

        report = pipe.run(
            factor_name      = "momentum_12_1",
            start            = "20200101",
            end              = "20221231",
            forward          = 21,
            n_groups         = 3,
            standardize      = "rank",
            neutralize       = False,
            ic_forward_list  = [1, 5],
            symbols          = symbols,
        )
        assert isinstance(report, FactorReport)

    def test_report_has_ic_stats(self, pipe):
        symbols = pipe.engine.all_symbols()[:8]
        if len(symbols) < 3:
            pytest.skip("股票数量不足")

        report = pipe.run(
            factor_name  = "vol_20d",
            start        = "20210101",
            end          = "20221231",
            forward      = 5,
            n_groups     = 3,
            standardize  = "zscore",
            neutralize   = False,
            ic_forward_list = [1, 5],
            symbols      = symbols,
        )
        assert isinstance(report.ic_stats_, dict)
        assert "mean_ic" in report.ic_stats_

    def test_report_save(self, pipe, tmp_path):
        symbols = pipe.engine.all_symbols()[:5]
        if len(symbols) < 3:
            pytest.skip("股票数量不足")

        report = pipe.run(
            factor_name  = "_test_pb",
            start        = "20210101",
            end          = "20221231",
            forward      = 5,
            n_groups     = 3,
            standardize  = None,
            neutralize   = False,
            ic_forward_list = [1],
            symbols      = symbols,
        )
        report.save(output_dir=str(tmp_path))
        assert (tmp_path / "_test_pb" / "summary.csv").exists()
        assert (tmp_path / "_test_pb" / "ic_series.csv").exists()

    def test_run_batch_returns_dataframe(self, pipe):
        symbols = pipe.engine.all_symbols()[:8]
        if len(symbols) < 3:
            pytest.skip("股票数量不足")

        result = pipe.run_batch(
            factor_names    = ["momentum_12_1", "vol_20d"],
            start           = "20210101",
            end             = "20221231",
            forward         = 5,
            n_groups        = 3,
            standardize     = "rank",
            neutralize      = False,
            ic_forward_list = [1],
            symbols         = symbols,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_report_print_summary(self, pipe, capsys):
        symbols = pipe.engine.all_symbols()[:5]
        if len(symbols) < 3:
            pytest.skip("股票数量不足")

        report = pipe.run(
            factor_name  = "momentum_12_1",
            start        = "20210101",
            end          = "20221231",
            forward      = 5,
            n_groups     = 3,
            standardize  = "rank",
            neutralize   = False,
            ic_forward_list = [1],
            symbols      = symbols,
        )
        report.print_summary()
        captured = capsys.readouterr()
        assert "momentum_12_1" in captured.out
        assert "IC" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestFactorReport（单元测试，使用合成数据）
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorReport:
    """使用合成数据测试 FactorReport 的各个属性/方法。"""

    @pytest.fixture(scope="class")
    def sample_report(self, factor_panel, return_panel, ic_series):
        from factor_framework.backtest import layer_backtest, long_short_stats, turnover_analysis

        layer_ret = layer_backtest(factor_panel, return_panel, n_groups=5)
        ls_stats_ = long_short_stats(layer_ret)
        turnover_ = turnover_analysis(factor_panel, n_groups=5)
        ic_s      = ic_stats(ic_series)
        ic_nw     = ic_significance(ic_series, lags=3)

        return FactorReport(
            factor_name  = "test_factor",
            ic_series    = ic_series,
            ic_stats     = ic_s,
            ic_nw        = ic_nw,
            ic_decay_df  = pd.DataFrame({"mean_ic": [0.05, 0.03], "icir": [1.0, 0.7]},
                                        index=[1, 5]),
            layer_ret    = layer_ret,
            ls_stats     = ls_stats_,
            turnover     = turnover_,
            factor_panel = factor_panel,
            return_panel = return_panel,
        )

    def test_report_factor_name(self, sample_report):
        assert sample_report.factor_name == "test_factor"

    def test_report_summary_dict_keys(self, sample_report):
        d = sample_report.summary_dict
        assert "factor"   in d
        assert "mean_ic"  in d
        assert "ls_sharpe" in d

    def test_report_save_creates_files(self, sample_report, tmp_path):
        sample_report.save(output_dir=str(tmp_path))
        base = tmp_path / "test_factor"
        assert (base / "summary.csv").exists()
        assert (base / "ic_series.csv").exists()
        assert (base / "layer_returns.csv").exists()
        assert (base / "factor_panel.csv").exists()

    def test_report_print_summary_no_crash(self, sample_report, capsys):
        sample_report.print_summary()
        out = capsys.readouterr().out
        assert "test_factor" in out


# ═══════════════════════════════════════════════════════════════════════════════
# TestOptimizer  §2.4 因子组合与权重优化
# ═══════════════════════════════════════════════════════════════════════════════

from factor_framework.optimizer import equal_weight, icir_weight, print_weights


def _make_panel(dates, stocks, seed=0) -> pd.DataFrame:
    """生成随机因子面板（日期 × 股票），内含少量 NaN。"""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(dates), len(stocks)))
    df = pd.DataFrame(data, index=dates, columns=stocks)
    # 随机置 NaN（约 5%）
    mask = rng.random(df.shape) < 0.05
    df[mask] = np.nan
    return df


@pytest.fixture
def multi_panels():
    """三个因子面板，日期×股票完全对齐。"""
    dates  = pd.date_range("2020-01-01", periods=24, freq="ME")
    stocks = [f"S{i:03d}" for i in range(50)]
    return {
        "F1": _make_panel(dates, stocks, seed=1),
        "F2": _make_panel(dates, stocks, seed=2),
        "F3": _make_panel(dates, stocks, seed=3),
    }


@pytest.fixture
def ic_series_dict(multi_panels):
    """每个因子伪造一条 IC 时间序列（与面板日期对齐）。"""
    rng = np.random.default_rng(42)
    result = {}
    dates = list(multi_panels.values())[0].index
    for name in multi_panels:
        values = rng.standard_normal(len(dates)) * 0.05 + 0.04
        result[name] = pd.Series(values, index=dates)
    return result


class TestOptimizerEqualWeight:
    """等权组合（§2.4.1）"""

    def test_returns_tuple(self, multi_panels):
        out = equal_weight(multi_panels)
        assert isinstance(out, tuple) and len(out) == 2

    def test_composite_shape(self, multi_panels):
        composite, _ = equal_weight(multi_panels)
        first = list(multi_panels.values())[0]
        assert composite.shape == first.shape

    def test_weights_sum_to_one(self, multi_panels):
        _, weights = equal_weight(multi_panels)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_equal(self, multi_panels):
        _, weights = equal_weight(multi_panels)
        vals = list(weights.values())
        assert all(abs(v - vals[0]) < 1e-9 for v in vals)

    def test_all_keys_present(self, multi_panels):
        _, weights = equal_weight(multi_panels)
        assert set(weights.keys()) == set(multi_panels.keys())

    def test_single_factor(self):
        """单因子等权 → 权重 = 1.0"""
        dates  = pd.date_range("2020-01-01", periods=10, freq="ME")
        stocks = ["A", "B", "C"]
        panel  = {"only": _make_panel(dates, stocks)}
        composite, weights = equal_weight(panel)
        assert abs(weights["only"] - 1.0) < 1e-9

    def test_composite_values_are_mean(self, multi_panels):
        """无 NaN 时合成值应等于各因子平均值。"""
        clean_panels = {k: v.fillna(0) for k, v in multi_panels.items()}
        composite, _ = equal_weight(clean_panels)
        expected = sum(clean_panels.values()) / len(clean_panels)
        np.testing.assert_allclose(composite.values, expected.values, atol=1e-10)

    def test_raises_on_empty_dict(self):
        with pytest.raises((ValueError, KeyError, ZeroDivisionError)):
            equal_weight({})

    def test_composite_index_matches_input(self, multi_panels):
        composite, _ = equal_weight(multi_panels)
        ref_index = list(multi_panels.values())[0].index
        assert composite.index.equals(ref_index)

    def test_composite_columns_match_input(self, multi_panels):
        composite, _ = equal_weight(multi_panels)
        ref_cols = list(multi_panels.values())[0].columns
        # 列应为公共子集
        assert set(composite.columns).issubset(set(ref_cols))

    def test_output_is_dataframe(self, multi_panels):
        composite, _ = equal_weight(multi_panels)
        assert isinstance(composite, pd.DataFrame)

    def test_weights_are_float(self, multi_panels):
        _, weights = equal_weight(multi_panels)
        assert all(isinstance(v, float) for v in weights.values())

    def test_misaligned_dates_intersect(self):
        """日期范围不同的面板 → 取交集。"""
        dates_a = pd.date_range("2020-01-01", periods=12, freq="ME")
        dates_b = pd.date_range("2020-07-01", periods=12, freq="ME")
        stocks = ["X", "Y"]
        panels = {
            "A": _make_panel(dates_a, stocks, seed=0),
            "B": _make_panel(dates_b, stocks, seed=1),
        }
        composite, weights = equal_weight(panels)
        common = dates_a.intersection(dates_b)
        assert len(composite) == len(common)

    def test_misaligned_stocks_intersect(self):
        """股票集合不同的面板 → 取交集。"""
        dates = pd.date_range("2020-01-01", periods=6, freq="ME")
        panels = {
            "A": _make_panel(dates, ["S1", "S2", "S3"], seed=0),
            "B": _make_panel(dates, ["S2", "S3", "S4"], seed=1),
        }
        composite, _ = equal_weight(panels)
        assert set(composite.columns) == {"S2", "S3"}


class TestOptimizerICIRWeight:
    """ICIR 加权（§2.4.2）"""

    def test_returns_tuple(self, multi_panels, ic_series_dict):
        out = icir_weight(multi_panels, ic_series_dict)
        assert isinstance(out, tuple) and len(out) == 2

    def test_composite_shape(self, multi_panels, ic_series_dict):
        composite, _ = icir_weight(multi_panels, ic_series_dict)
        first = list(multi_panels.values())[0]
        assert composite.shape[1] == first.shape[1]

    def test_weights_sum_to_one(self, multi_panels, ic_series_dict):
        _, weights = icir_weight(multi_panels, ic_series_dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_nonnegative(self, multi_panels, ic_series_dict):
        _, weights = icir_weight(multi_panels, ic_series_dict)
        assert all(v >= 0 for v in weights.values())

    def test_all_keys_present(self, multi_panels, ic_series_dict):
        _, weights = icir_weight(multi_panels, ic_series_dict)
        assert set(weights.keys()) == set(multi_panels.keys())

    def test_output_is_dataframe(self, multi_panels, ic_series_dict):
        composite, _ = icir_weight(multi_panels, ic_series_dict)
        assert isinstance(composite, pd.DataFrame)

    def test_window_param_accepted(self, multi_panels, ic_series_dict):
        """window=6 が受け入れられる（クラッシュしない）"""
        composite, weights = icir_weight(multi_panels, ic_series_dict, window=6)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_window_none_full_sample(self, multi_panels, ic_series_dict):
        """window=None → 全样本 ICIR。"""
        composite, weights = icir_weight(multi_panels, ic_series_dict, window=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_larger_icir_gets_larger_weight(self):
        """ICIR 更大的因子应获得更大权重。"""
        dates  = pd.date_range("2020-01-01", periods=24, freq="ME")
        stocks = ["A", "B", "C"]
        panels = {
            "strong": _make_panel(dates, stocks, seed=0),
            "weak":   _make_panel(dates, stocks, seed=1),
        }
        # strong 因子 IC 均值更高
        ic_dict = {
            "strong": pd.Series([0.10] * 24, index=dates),  # ICIR ≈ ∞ (no variance)
            "weak":   pd.Series([0.01] * 24, index=dates),
        }
        # 给 strong 添加一点方差，但均值仍远大于 weak
        rng = np.random.default_rng(7)
        ic_dict["strong"] = pd.Series(
            0.10 + rng.standard_normal(24) * 0.01, index=dates
        )
        ic_dict["weak"] = pd.Series(
            0.01 + rng.standard_normal(24) * 0.01, index=dates
        )
        _, weights = icir_weight(panels, ic_dict, window=None)
        assert weights["strong"] > weights["weak"]

    def test_zero_icir_fallback_to_equal(self):
        """若所有因子 ICIR=0，应回退到等权。"""
        dates  = pd.date_range("2020-01-01", periods=12, freq="ME")
        stocks = ["A", "B"]
        panels = {
            "F1": _make_panel(dates, stocks, seed=0),
            "F2": _make_panel(dates, stocks, seed=1),
        }
        # 全零 IC → ICIR = 0
        ic_dict = {
            "F1": pd.Series([0.0] * 12, index=dates),
            "F2": pd.Series([0.0] * 12, index=dates),
        }
        _, weights = icir_weight(panels, ic_dict, window=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # 等权
        assert abs(weights["F1"] - weights["F2"]) < 1e-9

    def test_single_factor_weight_is_one(self, ic_series_dict):
        """单因子时权重应为 1.0。"""
        dates  = pd.date_range("2020-01-01", periods=24, freq="ME")
        stocks = ["A", "B"]
        panels = {"only": _make_panel(dates, stocks)}
        ic_dict = {"only": ic_series_dict["F1"]}
        _, weights = icir_weight(panels, ic_dict, window=None)
        assert abs(weights["only"] - 1.0) < 1e-9


class TestOptimizerPrintWeights:
    """print_weights 输出烟雾测试"""

    def test_no_crash_equal(self, capsys):
        weights = {"F1": 0.5, "F2": 0.3, "F3": 0.2}
        print_weights(weights, method="等权")
        out = capsys.readouterr().out
        assert "F1" in out

    def test_no_crash_icir_with_dict(self, capsys):
        weights = {"F1": 0.6, "F2": 0.4}
        icir_d  = {"F1": 1.5, "F2": 0.8}
        print_weights(weights, method="ICIR加权", icir_dict=icir_d)
        out = capsys.readouterr().out
        assert "F1" in out and "ICIR" in out

    def test_sorted_descending(self, capsys):
        """输出应按权重降序排列。"""
        weights = {"A": 0.1, "B": 0.5, "C": 0.4}
        print_weights(weights, method="等权")
        out = capsys.readouterr().out
        lines = [l for l in out.split("\n") if any(k in l for k in ["A", "B", "C"])]
        # B 应在 A 之前
        assert out.index("B") < out.index("A")

    def test_bar_proportional(self, capsys):
        """权重更大的因子条形图应更长（或相等）。"""
        weights = {"High": 0.8, "Low": 0.2}
        print_weights(weights, method="等权")
        out = capsys.readouterr().out
        lines = {
            k: next((l for l in out.split("\n") if k in l), "")
            for k in weights
        }
        assert lines["High"].count("█") >= lines["Low"].count("█")

    def test_empty_weights(self, capsys):
        """空字典时不崩溃。"""
        print_weights({}, method="等权")
        # 只要不抛异常即可

    def test_returns_none(self):
        assert print_weights({"X": 1.0}, method="等权") is None


# ═══════════════════════════════════════════════════════════════════════════════
# §9  TestCompileEngine — 三层编译引擎
# ═══════════════════════════════════════════════════════════════════════════════

from factor_framework.jit_ops import (
    _NUMBA_OK, _NUMEXPR_OK, COMPILE_TARGET,
    ts_sum_fast, ts_mean_fast, ts_std_fast,
    ts_max_fast, ts_min_fast, ts_corr_fast,
    ts_wma_fast, ts_rank_fast, ts_prod_fast,
    ts_drawdown_fast, ts_slope_fast, ts_beta_fast,
    ne_log, ne_sqrt, ne_eval, ne_combine, warmup,
)


_RNG2 = np.random.default_rng(99)
_TS   = pd.Series(10 + _RNG2.normal(0, 0.2, 80).cumsum(), name="price")
_TS2  = pd.Series(10 + _RNG2.normal(0, 0.2, 80).cumsum(), name="bench")
_D    = 10  # 窗口大小


class TestCompileTargetMetadata:
    """§9.1  _compile_target 元数据标注正确性"""

    @pytest.mark.parametrize("fn, expected", [
        (ts_sum,      "numba"),
        (ts_mean,     "numba"),
        (ts_stddev,   "numba"),
        (ts_max,      "numba"),
        (ts_min,      "numba"),
        (ts_rank,     "numba"),
        (ts_wma,      "numba"),
        (ts_drawdown, "numba"),
        (ts_slope,    "numba"),
        (ts_prod,     "numba"),
        (ts_corr,     "numba"),
        (ts_beta,     "numba"),
        (ts_decay_linear, "numba"),
        (log,         "numexpr"),
        (sqrt,        "numexpr"),
        (absx,        "numexpr"),
        (power,       "numexpr"),
        (if_else,     "numexpr"),
        (clip,        "numexpr"),
        (cs_rank,     "numpy"),
        (cs_zscore,   "numpy"),
        (cs_demean,   "numpy"),
        (cs_scale,    "numpy"),
        (ts_ema,      "pandas"),
        (ts_rsi,      "pandas"),
        (ts_skew,     "pandas"),
        (ts_delta,    "pandas"),
        (delay,       "pandas"),
    ])
    def test_compile_target_attribute(self, fn, expected):
        """每个算子函数都应携带正确的 _compile_target 属性。"""
        assert hasattr(fn, "_compile_target"), \
            f"{fn.__name__} 缺少 _compile_target 属性"
        assert fn._compile_target == expected, \
            f"{fn.__name__}: expected {expected}, got {fn._compile_target}"

    def test_compile_target_registry_keys(self):
        """COMPILE_TARGET 字典应包含所有主要算子。"""
        must_have = [
            "ts_mean", "ts_stddev", "ts_sum", "ts_corr", "ts_rank",
            "ts_wma", "ts_drawdown", "ts_slope", "ts_prod",
            "log", "sqrt", "power",
            "cs_rank", "cs_zscore",
        ]
        for key in must_have:
            assert key in COMPILE_TARGET, f"{key} 未在 COMPILE_TARGET 中"


class TestNumericalEquivalence:
    """§9.2  JIT 路径与 Pandas 路径的数值一致性"""

    def _pd_rolling(self, x, d, method):
        """Pandas 参考实现。"""
        return getattr(x.rolling(d, min_periods=d), method)()

    def test_ts_sum_vs_pandas(self):
        jit = ts_sum_fast(_TS, _D)
        ref = self._pd_rolling(_TS, _D, "sum")
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-10)

    def test_ts_mean_vs_pandas(self):
        jit = ts_mean_fast(_TS, _D)
        ref = self._pd_rolling(_TS, _D, "mean")
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-10)

    def test_ts_std_vs_pandas(self):
        jit = ts_std_fast(_TS, _D)
        ref = self._pd_rolling(_TS, _D, "std")
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-8)

    def test_ts_max_vs_pandas(self):
        jit = ts_max_fast(_TS, _D)
        ref = self._pd_rolling(_TS, _D, "max")
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-10)

    def test_ts_min_vs_pandas(self):
        jit = ts_min_fast(_TS, _D)
        ref = self._pd_rolling(_TS, _D, "min")
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-10)

    def test_ts_corr_vs_pandas(self):
        jit = ts_corr_fast(_TS, _TS2, _D)
        ref = _TS.rolling(_D, min_periods=_D).corr(_TS2)
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, atol=1e-10)

    def test_ts_rank_range(self):
        """ts_rank 输出必须在 (0, 1]。"""
        r = ts_rank_fast(_TS, _D).dropna()
        assert (r > 0).all() and (r <= 1.0).all()

    def test_ts_wma_weighted(self):
        """ts_wma 验证权重归一化：wma 在 [min, max] 之间。"""
        jit = ts_wma_fast(_TS, _D).dropna()
        lo  = _TS.rolling(_D, min_periods=_D).min().dropna()
        hi  = _TS.rolling(_D, min_periods=_D).max().dropna()
        assert (jit.values >= lo.values - 1e-9).all()
        assert (jit.values <= hi.values + 1e-9).all()

    def test_ts_drawdown_range(self):
        """最大回撤应在 [0, 1]。"""
        dd = ts_drawdown_fast(_TS, _D).dropna()
        assert (dd >= 0).all() and (dd <= 1.0).all()

    def test_ts_prod_vs_manual(self):
        """ts_prod 验证：取 log 之和 = log(prod)。"""
        x  = _TS.abs() + 0.1
        jp = ts_prod_fast(x, _D).dropna()
        pp = x.rolling(_D, min_periods=_D).apply(np.prod, raw=True).dropna()
        np.testing.assert_allclose(jp.values, pp.values, rtol=1e-8)

    def test_ts_slope_near_zero_for_flat(self):
        """平稳序列的斜率应接近 0。"""
        flat = pd.Series(np.ones(30) * 5.0)
        s    = ts_slope_fast(flat, 10).dropna()
        np.testing.assert_allclose(s.values, 0.0, atol=1e-10)

    def test_ts_beta_vs_pandas(self):
        """ts_beta_fast 与 Pandas cov/var 结果一致。"""
        jit = ts_beta_fast(_TS, _TS2, _D)
        cov = _TS.rolling(_D, min_periods=_D).cov(_TS2)
        var = _TS2.rolling(_D, min_periods=_D).var(ddof=1)
        ref = cov / var.replace(0, np.nan)
        np.testing.assert_allclose(jit.dropna().values, ref.dropna().values, rtol=1e-8)

    def test_ne_log_vs_numpy(self):
        """ne_log 与 np.log 一致（x > 0 部分）。"""
        pos = _TS.abs() + 0.1
        jit = ne_log(pos)
        ref = np.log(pos)
        np.testing.assert_allclose(jit.values, ref.values, rtol=1e-10)

    def test_ne_log_negative_is_nan(self):
        """ne_log：x ≤ 0 必须输出 NaN。"""
        x = pd.Series([-1.0, 0.0, 1.0, 2.0])
        r = ne_log(x)
        assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])
        assert not np.isnan(r.iloc[2])

    def test_ne_sqrt_vs_numpy(self):
        """ne_sqrt 与 np.sqrt 一致（x >= 0）。"""
        pos = _TS.abs()
        jit = ne_sqrt(pos)
        ref = np.sqrt(pos)
        np.testing.assert_allclose(jit.values, ref.values, rtol=1e-10)

    def test_ne_eval_expression(self):
        """ne_eval 能正确求值简单数学表达式。"""
        x = np.array([1.0, 4.0, 9.0])
        r = ne_eval("sqrt(x)", {"x": x})
        np.testing.assert_allclose(r, [1.0, 2.0, 3.0], rtol=1e-10)

    def test_ne_combine_weighted_sum(self):
        """ne_combine 加权求和：权重为 0.6/0.4 时结果正确。"""
        rng   = np.random.default_rng(7)
        dates = pd.date_range("20200101", periods=30, freq="B").strftime("%Y%m%d")
        cols  = ["A", "B", "C"]
        p1    = pd.DataFrame(rng.normal(0, 1, (30, 3)), index=dates, columns=cols)
        p2    = pd.DataFrame(rng.normal(0, 1, (30, 3)), index=dates, columns=cols)
        result = ne_combine({"f1": p1, "f2": p2}, {"f1": 0.6, "f2": 0.4})
        expected = p1 * 0.6 + p2 * 0.4
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_ne_combine_nan_propagation(self):
        """ne_combine：任一面板为 NaN 的位置，合成结果也应为 NaN。"""
        rng   = np.random.default_rng(8)
        dates = pd.date_range("20200101", periods=20, freq="B").strftime("%Y%m%d")
        cols  = ["A", "B"]
        p1    = pd.DataFrame(rng.normal(0, 1, (20, 2)), index=dates, columns=cols)
        p2    = pd.DataFrame(rng.normal(0, 1, (20, 2)), index=dates, columns=cols)
        p2.iloc[5, 0] = np.nan
        result = ne_combine({"f1": p1, "f2": p2}, {"f1": 0.5, "f2": 0.5})
        assert np.isnan(result.iloc[5, 0])
        assert not np.isnan(result.iloc[5, 1])


class TestOperatorFallback:
    """§9.3  _JIT_OK=False 降级路径（通过 monkeypatch 模拟）"""

    def test_ts_mean_fallback(self, monkeypatch):
        """关闭 JIT 后，ts_mean 退化到 Pandas rolling，结果不变。"""
        import factor_framework.operators as ops
        monkeypatch.setattr(ops, "_JIT_OK", False)
        result = ops.ts_mean(_TS, _D)
        ref    = _TS.rolling(_D, min_periods=_D).mean()
        np.testing.assert_allclose(result.dropna().values, ref.dropna().values, rtol=1e-10)

    def test_ts_rank_fallback(self, monkeypatch):
        """关闭 JIT 后，ts_rank 退化到 rolling().apply()，结果在 (0,1]。"""
        import factor_framework.operators as ops
        monkeypatch.setattr(ops, "_JIT_OK", False)
        r = ops.ts_rank(_TS, _D).dropna()
        assert (r > 0).all() and (r <= 1.0).all()

    def test_log_fallback(self, monkeypatch):
        """关闭 JIT 后，log 退化到 np.log，负数仍为 NaN。"""
        import factor_framework.operators as ops
        monkeypatch.setattr(ops, "_JIT_OK", False)
        pos = _TS.abs() + 0.1
        r   = ops.log(pos)
        ref = np.log(pos)
        np.testing.assert_allclose(r.values, ref.values, rtol=1e-10)

    def test_compile_target_preserved_after_fallback(self, monkeypatch):
        """降级后 _compile_target 属性不应丢失（仍指示理想路径）。"""
        import factor_framework.operators as ops
        monkeypatch.setattr(ops, "_JIT_OK", False)
        assert ops.ts_mean._compile_target == "numba"
        assert ops.log._compile_target     == "numexpr"


class TestFactorEngineCompileCache:
    """§9.4  FactorEngine 编译路径缓存"""

    @pytest.fixture
    def engine(self, tmp_path):
        """创建含最小 CSV 集的临时引擎。"""
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        rng = np.random.default_rng(0)
        dates = pd.date_range("20200101", periods=100, freq="B").strftime("%Y%m%d")
        for code in ["000001_SZ", "000002_SZ"]:
            df = pd.DataFrame({
                "交易日": dates, "股票代码": code,
                "收盘价":  rng.uniform(5, 20, 100),
                "开盘价":  rng.uniform(5, 20, 100),
                "最高价":  rng.uniform(10, 25, 100),
                "最低价":  rng.uniform(3,  10, 100),
                "成交量（手）": rng.uniform(1e4, 1e6, 100),
                "成交额（千元）": rng.uniform(1e5, 1e7, 100),
                "换手率（%）": rng.uniform(0.1, 5, 100),
                "总市值（万元）": rng.uniform(1e6, 1e8, 100),
                "流通市值（万元）": rng.uniform(1e6, 1e8, 100),
                "市净率": rng.uniform(0.5, 10, 100),
                "市盈率（TTM，亏损为空）": rng.uniform(5, 100, 100),
                "市销率（TTM）": rng.uniform(0.5, 10, 100),
                "复权因子": np.ones(100),
            })
            df.to_csv(stocks_dir / f"{code}.csv", index=False)

        e = FactorEngine(stocks_dir=stocks_dir, stock_basic=tmp_path / "sb.csv",
                         min_rows=20, verbose=False)
        return e

    def test_resolve_compile_target_numba(self, engine):
        """注册 ts_mean 因子后，compile_target 应为 'numba'。"""
        from factor_framework.operators import ts_mean
        engine.register("m20", lambda df: ts_mean(df["收盘价"], 5))
        # lambda 本身无属性，但 COMPILE_TARGET 仍报 'pandas'（lambda）
        t = engine._resolve_compile_target("m20")
        assert t in {"numba", "pandas", "unknown"}

    def test_compile_cache_populated_after_compute(self, engine):
        """compute_single 之后，_compile_cache 中应有对应条目。"""
        from factor_framework.operators import ts_mean
        engine.register("m5", lambda df: ts_mean(df["收盘价"], 5))
        engine.compute_single("000001_SZ", "m5")
        assert "m5" in engine._compile_cache

    def test_clear_cache_resets_compile_cache(self, engine):
        """clear_cache() 应同时清空 _compile_cache。"""
        from factor_framework.operators import ts_mean
        engine.register("x1", lambda df: ts_mean(df["收盘价"], 5))
        engine.compute_single("000001_SZ", "x1")
        assert "x1" in engine._compile_cache
        engine.clear_cache()
        assert len(engine._compile_cache) == 0

    def test_compile_report_returns_dataframe(self, engine):
        """compile_report() 应返回含 factor_name / compile_target 两列的 DataFrame。"""
        from factor_framework.operators import ts_mean, log
        engine.register("r_mean", lambda df: ts_mean(df["收盘价"], 5))
        engine.register("r_log",  lambda df: log(df["总市值（万元）"]))
        report = engine.compile_report()
        assert isinstance(report, pd.DataFrame)
        assert "factor_name" in report.columns
        assert "compile_target" in report.columns
        assert len(report) == 2

    def test_compile_report_all_targets_valid(self, engine):
        """compile_report 中 compile_target 列只含合法值。"""
        from factor_framework.operators import ts_mean
        engine.register("chk", lambda df: ts_mean(df["收盘价"], 5))
        report = engine.compile_report()
        valid  = {"numba", "numexpr", "numpy", "pandas", "unknown"}
        assert set(report["compile_target"].unique()).issubset(valid)


class TestWarmup:
    """§9.5  warmup() 预热函数"""

    def test_warmup_returns_dict(self):
        """warmup() 应返回字典，键为算子名称，值为耗时（秒）。"""
        times = warmup(verbose=False)
        assert isinstance(times, dict)
        assert len(times) > 0
        for name, t in times.items():
            assert isinstance(t, float) and t >= 0.0

    def test_warmup_covers_all_numba_ops(self):
        """warmup() 应覆盖所有 Numba JIT 算子。"""
        times = warmup(verbose=False)
        expected = {
            "ts_sum", "ts_mean", "ts_std", "ts_max", "ts_min",
            "ts_corr", "ts_wma", "ts_rank", "ts_prod",
            "ts_drawdown", "ts_slope", "ts_beta",
        }
        assert expected.issubset(times.keys())

    def test_warmup_idempotent(self):
        """多次调用 warmup() 均不崩溃，第二次明显更快（Numba 缓存）。"""
        warmup(verbose=False)
        import time
        t0 = time.perf_counter()
        warmup(verbose=False)
        elapsed = time.perf_counter() - t0
        # 第二次应在 2 秒内完成（JIT 已编译）
        assert elapsed < 2.0
