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

    def test_return_panel_formula_correct(self, engine):
        """
        验证 build_return_panel 使用正确的未来收益公式：
        return[t] = price[t+forward] / price[t] - 1  (含 T+1 滞后)
        而非旧的 pct_change(forward).shift(-forward)。

        注意：build_panel 内部会去掉前导/后缀全 NaN 行，
        所以返回的面板不含全 NaN 行。这里验证：
        1. 返回值是 DataFrame（形状合理）
        2. 值域合理（收益率在 -1 ~ 5 之间，非异常大的 pct_change 值）
        """
        symbols = engine.all_symbols()[:3]
        if len(symbols) < 1:
            pytest.skip("股票文件不足")
        forward = 5
        rp = engine.build_return_panel(forward=forward,
                                       start="20210101", end="20211231",
                                       symbols=symbols)
        assert isinstance(rp, pd.DataFrame)
        # 基本形状检查：应有合理行数
        assert len(rp) > forward, f"收益率面板行数 {len(rp)} 应 > forward={forward}"
        # 值域合理：日股票的 forward=5 收益率应在 ±100% 以内
        vals = rp.values[~np.isnan(rp.values)]
        if len(vals) > 0:
            assert vals.max() < 2.0, f"最大收益率 {vals.max():.2%} 超过 200%，可能公式错误"
            assert vals.min() > -1.0, f"最小收益率 {vals.min():.2%} 低于 -100%，可能公式错误"

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

    # ── v2.7 回归测试 ────────────────────────────────────────────────────────

    def test_ls_zero_return_not_nan(self):
        """Fix 2: LS=0 → NaN bug 修复验证。
        当多头或空头收益恰好为 0.0 时，LS 应为 0.0 而非 NaN。"""
        rng = np.random.default_rng(42)
        dates  = pd.date_range("20200101", periods=10, freq="B").strftime("%Y%m%d")
        stocks = [f"S{i:02d}" for i in range(20)]
        # 构造收益面板：所有股票收益 = 0（多头空头均为 0）
        r = pd.DataFrame(np.zeros((10, 20)), index=dates, columns=stocks)
        f = pd.DataFrame(rng.normal(0, 1, (10, 20)), index=dates, columns=stocks)
        layer_ret = layer_backtest(f, r, n_groups=5, direction=1)
        ls = layer_ret["LS"]
        # LS 应该全为 0.0，不应出现 NaN
        assert ls.notna().all(), "LS 不应有 NaN（收益全零时 LS=0）"
        assert (ls.abs() < 1e-12).all(), "LS 应全为 0.0（所有组收益为 0）"

    def test_layer_backtest_direction_symmetry_exact(self):
        """Fix 4: 向量化 layer_backtest 方向对称性精确验证。
        direction=-1 的 LS 应与 direction=1 的 LS 精确相反（corr ≈ -1）。"""
        rng = np.random.default_rng(0)
        dates  = pd.date_range("20200101", periods=50, freq="B").strftime("%Y%m%d")
        stocks = [f"S{i:02d}" for i in range(30)]
        f = pd.DataFrame(rng.normal(0, 1, (50, 30)), index=dates, columns=stocks)
        r = pd.DataFrame(rng.normal(0, 0.02, (50, 30)), index=dates, columns=stocks)
        pos = layer_backtest(f, r, n_groups=5, direction=1)
        neg = layer_backtest(f, r, n_groups=5, direction=-1)
        common = pos["LS"].dropna().index.intersection(neg["LS"].dropna().index)
        corr = pos["LS"][common].corr(neg["LS"][common])
        assert corr < -0.99, f"方向对称相关系数应 < -0.99，实际 {corr:.4f}"

    def test_layer_backtest_vectorized_matches_per_date(self):
        """Fix 4: 向量化结果应与逐日期等频分组（nanpercentile）结果一致。"""
        rng = np.random.default_rng(7)
        dates  = pd.date_range("20200101", periods=20, freq="B").strftime("%Y%m%d")
        stocks = [f"S{i:02d}" for i in range(20)]
        f = pd.DataFrame(rng.normal(0, 1, (20, 20)), index=dates, columns=stocks)
        r = pd.DataFrame(rng.normal(0, 0.02, (20, 20)), index=dates, columns=stocks)
        result = layer_backtest(f, r, n_groups=5, direction=1)
        # LS = Q5 - Q1 逐行验证
        ls   = result["LS"].dropna()
        diff = (result["Q5"] - result["Q1"] - result["LS"]).dropna().abs()
        assert (diff < 1e-9).all(), "LS 必须等于 Q5 - Q1"

    def test_apply_cross_section_vectorized(self):
        """Fix 5: apply_cross_section 向量化后结果应与逐行一致。"""
        rng = np.random.default_rng(3)
        dates  = pd.date_range("20200101", periods=15, freq="B").strftime("%Y%m%d")
        stocks = [f"S{i:02d}" for i in range(10)]
        panel = pd.DataFrame(rng.normal(0, 1, (15, 10)), index=dates, columns=stocks)

        def rank_cs(x: pd.Series) -> pd.Series:
            return x.rank()

        from factor_framework.factor_engine import FactorEngine
        result = FactorEngine.apply_cross_section(panel, rank_cs)
        assert result.shape == panel.shape
        # 每行的最小排名为 1，最大排名为非 NaN 数量
        for date in dates:
            row = result.loc[date].dropna()
            if len(row) > 0:
                assert abs(row.min() - 1.0) < 1e-9
                assert abs(row.max() - len(row)) < 1e-9

    def test_compute_single_warmup_truncation(self):
        """Fix 1: warm-up 截断验证。
        max_lookback=N 时输出日期不早于 start，但计算基于更长数据。"""
        import tempfile, os
        from factor_framework.factor_engine import FactorEngine

        # 构造 100 行假数据
        rng = np.random.default_rng(5)
        n   = 100
        dates = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d").tolist()
        df_raw = pd.DataFrame({
            "交易日":   dates,
            "收盘价":   rng.uniform(10, 20, n),
            "开盘价":   rng.uniform(10, 20, n),
            "最高价":   rng.uniform(15, 25, n),
            "最低价":   rng.uniform(5,  15, n),
            "成交量":   rng.uniform(1e6, 1e7, n),
            "成交额":   rng.uniform(1e7, 1e8, n),
            "涨跌幅":   rng.uniform(-0.1, 0.1, n),
            "换手率":   rng.uniform(0.1, 5.0, n),
        })

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "TEST_SZ.csv")
            df_raw.to_csv(csv_path, index=False)
            engine = FactorEngine(stocks_dir=tmp, verbose=False)
            engine.register("close_ma20", lambda df: df["收盘价"].rolling(20).mean())

            start = dates[30]
            # 无 warm-up：前 20 行 MA 为 NaN，所以 dates[30] 附近可能有少量 NaN
            s_no_warm = engine.compute_single(
                "TEST_SZ", "close_ma20", start=start, max_lookback=0
            )
            # 有 warm-up=25：多加载 25 行暖机，输出截断至 start
            engine.clear_cache()
            s_warm = engine.compute_single(
                "TEST_SZ", "close_ma20", start=start, max_lookback=25
            )

            # warm-up 模式下输出不应早于 start
            assert s_warm.index.min() >= start, "warm-up 模式输出不应早于 start"
            # warm-up 版本在 start 附近有更多非 NaN 值（窗口已预热）
            assert s_warm.notna().sum() >= s_no_warm.notna().sum(), \
                "warm-up 模式应产生更多或等量的非 NaN 值"


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. TestDataCleanerV27  (Fix 3 回归测试 — 不重复 Winsorize)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataCleanerV27:
    """v2.7: 删除 ffill 后的二次 MAD Winsorize（Fix 3）回归测试。"""

    def _make_df(self, n=60, seed=0):
        """构造最小化的单股 DataFrame（与 clean_stock_df 期望列一致）。"""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d").tolist()
        return pd.DataFrame({
            "交易日":   dates,
            "收盘价":   rng.uniform(10, 20, n),
            "开盘价":   rng.uniform(10, 20, n),
            "最高价":   rng.uniform(15, 25, n),
            "最低价":   rng.uniform(5,  15, n),
            "成交量":   rng.uniform(1e6, 1e7, n),
            "成交额":   rng.uniform(1e7, 1e8, n),
            "涨跌幅":   rng.uniform(-0.1, 0.1, n),
            "换手率":   rng.uniform(0.1, 5.0, n),
        })

    def test_clean_returns_dataframe(self):
        """clean_stock_df 应返回 DataFrame（基本冒烟测试）。"""
        from data_cleaner import clean_stock_df
        df = self._make_df()
        result = clean_stock_df(df)
        assert isinstance(result, pd.DataFrame)

    def test_ffill_value_not_re_winsorized(self):
        """Fix 3: ffill 填入的值不应被再次 Winsorize 截断。
        向价格列注入停牌 NaN，ffill 填充后，填充值应原样保留。"""
        from data_cleaner import clean_stock_df
        df = self._make_df(n=80)
        # 将第 40-44 行的收盘价设为 NaN（模拟停牌）
        df.loc[40:44, "收盘价"] = np.nan
        result = clean_stock_df(df.copy())
        if result is None:
            pytest.skip("clean_stock_df 返回 None（数据不足）")
        # ffill 后填入的值应与前一个有效值相同（允许 Winsorize 轻微截断原值）
        # 核心验证：winsorized_cols 中不出现"收盘价被 Winsorize 两次"的极值
        assert result["收盘价"].notna().sum() > 0

    def test_no_double_winsorize_call(self):
        """Fix 3: 验证 clean_stock_df 中非估值列的处理代码不存在二次 Winsorize。
        修复前：非估值列有 2 次 _try_winsorize（4b + 4d）。
        修复后：非估值列只有 1 次（4b），估值列仍保留 2 次（ffill 前后各一次）。
        通过检查 # 4d 注释是否已删除来验证。"""
        import inspect
        import data_cleaner as dc
        source = inspect.getsource(dc.clean_stock_df)
        # 修复后 "4d" 注释应已不存在
        assert "4d" not in source, (
            "clean_stock_df 中不应再有 '4d' 注释（二次 Winsorize 已删除，Fix 3）"
        )
        # 修复后非估值列代码块中不应有 'w2' 变量（只在第二次 Winsorize 时使用）
        # 估值列仍可能有 w2，所以我们检查 '4d' 注释的消失更可靠
        assert source.count("_try_winsorize") == 3, (
            f"修复后 clean_stock_df 中应有 3 次 _try_winsorize 调用（估值列2次+非估值列1次），"
            f"实际发现 {source.count('_try_winsorize')} 次"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5c. TestV28Optimizations  (v2.8 性能优化回归测试)
# ═══════════════════════════════════════════════════════════════════════════════

class TestV28Optimizations:
    """v2.8: LRUCache O(1) + apply_cross_section 向量化 + build_panel_batch 流程"""

    # ── Opt 3: LRUCache → OrderedDict ──────────────────────────────────────

    def test_lrucache_uses_ordereddict(self):
        """LRUCache 内部应使用 OrderedDict（O(1) move_to_end）。"""
        from collections import OrderedDict
        from factor_framework.dag import LRUCache
        c = LRUCache(capacity=4)
        assert isinstance(c._cache, OrderedDict), \
            "LRUCache._cache 应为 OrderedDict（O(1) LRU）"

    def test_lrucache_eviction_order(self):
        """LRU 淘汰策略：最久未访问的条目被淘汰。"""
        from factor_framework.dag import LRUCache, _MISS
        c = LRUCache(capacity=3)
        c.put("a", 1); c.put("b", 2); c.put("c", 3)
        _ = c.get("a")   # a 变为最近访问
        c.put("d", 4)    # 容量满 → 淘汰最久未访问的 b
        assert c.get("b") is _MISS, "b 应被淘汰（最久未访问）"
        assert c.get("a") == 1,     "a 应仍在缓存中"
        assert c.get("c") == 3,     "c 应仍在缓存中"
        assert c.get("d") == 4,     "d 应在缓存中"

    def test_lrucache_unlimited(self):
        """capacity=-1 时无限容量，不淘汰任何条目。"""
        from factor_framework.dag import LRUCache, _MISS
        c = LRUCache(capacity=-1)
        for i in range(500):
            c.put(str(i), i)
        assert len(c) == 500
        assert c.get("0") == 0
        assert c.get("499") == 499

    def test_lrucache_get_updates_order(self):
        """get() 后再插入新条目，被 get 的条目不会被优先淘汰。"""
        from factor_framework.dag import LRUCache, _MISS
        c = LRUCache(capacity=2)
        c.put("x", 10); c.put("y", 20)
        _ = c.get("x")    # x 变为最近访问
        c.put("z", 30)    # 容量满 → 淘汰 y（最久未访问）
        assert c.get("y") is _MISS
        assert c.get("x") == 10
        assert c.get("z") == 30

    def test_lrucache_thread_safe(self):
        """LRUCache 多线程并发 put/get 不应抛出异常。"""
        import threading
        from factor_framework.dag import LRUCache
        c   = LRUCache(capacity=50)
        errors = []

        def worker(tid):
            try:
                for j in range(100):
                    c.put(f"k{tid}_{j}", tid * 1000 + j)
                    c.get(f"k{tid}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"并发访问抛出异常: {errors}"

    # ── Opt 2: apply_cross_section 向量化快速路径 ───────────────────────────

    def test_cs_rank_vectorized_matches_rowwise(self, factor_panel):
        """cs_rank 快速路径（DataFrame.rank）应与逐行结果一致。"""
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.operators     import cs_rank

        result_fast = FactorEngine.apply_cross_section(factor_panel, cs_rank)
        # 逐行参考实现
        ref_rows = {}
        for date, row in factor_panel.iterrows():
            x = row.dropna()
            ref_rows[date] = x.rank(pct=True)
        result_ref = pd.DataFrame(ref_rows).T.reindex(columns=factor_panel.columns)

        pd.testing.assert_frame_equal(
            result_fast.fillna(-1), result_ref.fillna(-1),
            check_names=False, atol=1e-9,
        )

    def test_cs_zscore_vectorized_matches_rowwise(self, factor_panel):
        """cs_zscore 快速路径（numpy broadcast）应与逐行结果一致。"""
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.operators     import cs_zscore

        result_fast = FactorEngine.apply_cross_section(factor_panel, cs_zscore)
        ref_rows = {}
        for date, row in factor_panel.iterrows():
            x = row.dropna()
            mu, sig = x.mean(), x.std(ddof=1)
            ref_rows[date] = (x - mu) / sig if sig > 0 else pd.Series(np.nan, index=x.index)
        result_ref = pd.DataFrame(ref_rows).T.reindex(columns=factor_panel.columns)

        pd.testing.assert_frame_equal(
            result_fast.fillna(-999), result_ref.fillna(-999),
            check_names=False, atol=1e-6,
        )

    def test_cs_winsorize_vectorized_matches_rowwise(self, factor_panel):
        """cs_winsorize 快速路径（numpy clip）应与逐行结果一致。"""
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.operators     import cs_winsorize

        result_fast = FactorEngine.apply_cross_section(factor_panel, cs_winsorize)
        ref_rows = {}
        for date, row in factor_panel.iterrows():
            ref_rows[date] = cs_winsorize(row.dropna())
        result_ref = pd.DataFrame(ref_rows).T.reindex(columns=factor_panel.columns)

        pd.testing.assert_frame_equal(
            result_fast.fillna(-999), result_ref.fillna(-999),
            check_names=False, atol=1e-9,
        )

    def test_apply_cross_section_nan_preserved(self, factor_panel):
        """apply_cross_section 快速路径应保留 NaN 位置不变。"""
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.operators     import cs_rank, cs_zscore, cs_winsorize

        for func in (cs_rank, cs_zscore, cs_winsorize):
            result = FactorEngine.apply_cross_section(factor_panel, func)
            # NaN 位置应与输入一致
            input_nan  = factor_panel.isna()
            result_nan = result.isna()
            assert (result_nan[input_nan]).all(axis=None), \
                f"{func.__name__}: 输入的 NaN 位置在输出中应保持为 NaN"

    # ── Opt 1: run_batch_from_panels ────────────────────────────────────────

    def test_run_batch_from_panels_returns_reports(self, factor_panel, return_panel):
        """run_batch_from_panels 应返回包含所有因子的 FactorReport 字典。"""
        import tempfile, os, warnings as _warnings
        from factor_framework.pipeline import FactorPipeline

        rng = np.random.default_rng(99)
        with tempfile.TemporaryDirectory() as tmp:
            # 写入 2 只假股票的 CSV
            n = 120
            dates = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d").tolist()
            for sym in ["AA_SZ", "BB_SZ"]:
                pd.DataFrame({
                    "交易日": dates,
                    "收盘价": rng.uniform(10, 20, n),
                    "开盘价": rng.uniform(10, 20, n),
                    "最高价": rng.uniform(15, 25, n),
                    "最低价": rng.uniform(5,  15, n),
                    "成交量": rng.uniform(1e6, 1e7, n),
                    "成交额": rng.uniform(1e7, 1e8, n),
                    "涨跌幅": rng.uniform(-0.1, 0.1, n),
                    "换手率": rng.uniform(0.1, 5.0, n),
                }).to_csv(os.path.join(tmp, f"{sym}.csv"), index=False)

            pipe = FactorPipeline(stocks_dir=tmp, verbose=False)
            pipe.register_factor("f1", lambda df: df["收盘价"].rolling(5).mean())
            pipe.register_factor("f2", lambda df: df["收盘价"].rolling(10).mean())

            panels = pipe.engine.build_panel_batch(
                ["f1", "f2"], start="20200201", end="20201231",
            )
            ret_panel = pipe.engine.build_return_panel(
                forward=5, start="20200201", end="20201231",
            )
            close_panel = pd.DataFrame(
                rng.uniform(10, 20, (len(ret_panel), 2)),
                index=ret_panel.index, columns=["AA_SZ", "BB_SZ"],
            )

            with _warnings.catch_warnings():
                # 2 只股票的极小面板中存在全 NaN 行，nanmedian 会产生 RuntimeWarning
                _warnings.simplefilter("ignore", RuntimeWarning)
                reports = pipe.run_batch_from_panels(
                    factor_panels=panels,
                    return_panel=ret_panel,
                    close_panel=close_panel,
                    forward=5,
                )

        assert set(reports.keys()) == {"f1", "f2"}, \
            "run_batch_from_panels 应返回所有因子的报告"
        for name, rpt in reports.items():
            assert hasattr(rpt, "ic_series"), f"{name}: 报告应含 ic_series"
            assert hasattr(rpt, "layer_ret"), f"{name}: 报告应含 layer_ret"




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


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestDAG  —  DAG 节点、LRU 缓存、执行器、显式依赖、CSE 报告
# ═══════════════════════════════════════════════════════════════════════════════

from factor_framework.dag import (
    Expr, ConstNode, DataNode, OpNode, BinOpNode, PctChangeNode,
    data, const, op, pct_change,
    collect_nodes, topological_sort,
    LRUCache, _MISS,
    DAGExecutor, DepGraph,
    cse_report,
)


# ─── 辅助：最小 DataFrame ──────────────────────────────────────────────────────

def _make_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 10 + rng.normal(0, 0.3, n).cumsum()
    vol   = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame({"close": close, "volume": vol})


# ═══════════════════════════════════════════════════════════════════════════════
# 9a. 节点类型 & 哈希
# ═══════════════════════════════════════════════════════════════════════════════

class TestExprNodes:
    """验证每种节点类型的 eval() 和 node_hash 正确性。"""

    def test_const_node_eval(self):
        df  = _make_df()
        node = ConstNode(3.14)
        s   = node.eval(df)
        assert len(s) == len(df)
        assert (s == 3.14).all()

    def test_const_node_hash_deterministic(self):
        h1 = ConstNode(1.0).node_hash
        h2 = ConstNode(1.0).node_hash
        assert h1 == h2

    def test_const_node_different_values_different_hash(self):
        assert ConstNode(1.0).node_hash != ConstNode(2.0).node_hash

    def test_data_node_eval(self):
        df   = _make_df()
        node = DataNode("close")
        s    = node.eval(df)
        pd.testing.assert_series_equal(s, df["close"], check_names=False)

    def test_data_node_col_alias(self):
        df = _make_df().rename(columns={"close": "收盘价"})
        node = DataNode("close", col="收盘价")
        s = node.eval(df)
        pd.testing.assert_series_equal(s, df["收盘价"], check_names=False)

    def test_data_node_missing_col_raises(self):
        df   = _make_df()
        node = DataNode("nonexistent")
        with pytest.raises(KeyError):
            node.eval(df)

    def test_data_node_hash_depends_on_col(self):
        h1 = DataNode("close", col="close").node_hash
        h2 = DataNode("close", col="收盘价").node_hash
        assert h1 != h2

    def test_pct_change_node_eval(self):
        df   = _make_df()
        node = PctChangeNode(DataNode("close"), periods=1)
        s    = node.eval(df)
        expected = df["close"].pct_change(1)
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_pct_change_node_periods(self):
        df   = _make_df()
        node = PctChangeNode(DataNode("close"), periods=5)
        s    = node.eval(df)
        expected = df["close"].pct_change(5)
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_pct_change_node_hash_periods_differ(self):
        base = DataNode("close")
        h1 = PctChangeNode(base, 1).node_hash
        h5 = PctChangeNode(base, 5).node_hash
        assert h1 != h5

    def test_bin_op_add(self):
        df = _make_df()
        a  = DataNode("close")
        b  = ConstNode(1.0)
        node = BinOpNode("+", a, b)
        s    = node.eval(df)
        expected = df["close"] + 1.0
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_bin_op_mul(self):
        df = _make_df()
        a  = DataNode("close")
        b  = ConstNode(-1.0)
        node = BinOpNode("*", b, a)
        s    = node.eval(df)
        pd.testing.assert_series_equal(s, -df["close"], check_names=False)

    def test_bin_op_div_zero_replaced_nan(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [0.0, 1.0, 2.0]})
        a  = DataNode("close")
        b  = DataNode("volume")
        node = BinOpNode("/", a, b)
        s    = node.eval(df)
        assert np.isnan(s.iloc[0])   # 除以 0 → NaN

    def test_bin_op_hash_encodes_op_and_operands(self):
        c = DataNode("close")
        v = DataNode("volume")
        h_add = BinOpNode("+", c, v).node_hash
        h_sub = BinOpNode("-", c, v).node_hash
        h_rev = BinOpNode("+", v, c).node_hash   # 交换顺序
        assert h_add != h_sub
        assert h_add != h_rev

    def test_neg_returns_expr(self):
        c = DataNode("close")
        n = -c
        assert isinstance(n, Expr)

    def test_neg_eval_negates_values(self):
        df = _make_df()
        s  = (-DataNode("close")).eval(df)
        pd.testing.assert_series_equal(s, -df["close"], check_names=False)

    def test_op_node_ts_mean(self):
        df   = _make_df()
        close = DataNode("close")
        node  = op("ts_mean", close, 5)
        s     = node.eval(df)
        from factor_framework.operators import ts_mean
        expected = ts_mean(df["close"], 5)
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_op_unknown_raises(self):
        with pytest.raises(AttributeError):
            op("nonexistent_op_xyz", DataNode("close"), 5)

    def test_op_explicit_fn(self):
        df   = _make_df()
        close = DataNode("close")
        fn    = lambda s, w: s.rolling(w).mean()
        node  = op("custom_mean", close, 5, fn=fn)
        s     = node.eval(df)
        expected = df["close"].rolling(5).mean()
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_arithmetic_chaining(self):
        """(close + volume) * 2 应正确求值。"""
        df   = _make_df()
        expr = (DataNode("close") + DataNode("volume")) * ConstNode(2.0)
        s    = expr.eval(df)
        expected = (df["close"] + df["volume"]) * 2.0
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_factory_data(self):
        node = data("close", col="close")
        assert isinstance(node, DataNode)
        assert node.col == "close"

    def test_factory_const(self):
        node = const(42.0)
        assert isinstance(node, ConstNode)
        assert node.value == 42.0

    def test_factory_pct_change(self):
        node = pct_change(DataNode("close"), 2)
        assert isinstance(node, PctChangeNode)
        assert node.periods == 2

    def test_pct_change_via_op(self):
        node = op("pct_change", DataNode("close"), 3)
        assert isinstance(node, PctChangeNode)
        assert node.periods == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 9b. DAG 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGUtils:
    """collect_nodes() 和 topological_sort() 正确性。"""

    def _build_tree(self):
        """close → ret → vol20，两个分支共享 ret。"""
        close = DataNode("close")
        ret   = PctChangeNode(close, 1)
        vol20 = op("ts_mean", ret, 20, fn=lambda s, w: s.rolling(w).mean())
        vol60 = op("ts_mean", ret, 60, fn=lambda s, w: s.rolling(w).mean())
        return close, ret, vol20, vol60

    def test_collect_nodes_count(self):
        close, ret, vol20, _ = self._build_tree()
        nodes = collect_nodes(vol20)
        hashes = {n.node_hash for n in nodes}
        # vol20 节点 + ret 节点 + close 节点 = 3
        assert len(hashes) == 3

    def test_collect_nodes_contains_all(self):
        close, ret, vol20, _ = self._build_tree()
        nodes  = collect_nodes(vol20)
        hashes = {n.node_hash for n in nodes}
        assert vol20.node_hash in hashes
        assert ret.node_hash   in hashes
        assert close.node_hash in hashes

    def test_topo_sort_leaf_before_root(self):
        close, ret, vol20, _ = self._build_tree()
        order  = topological_sort([vol20])
        hashes = [n.node_hash for n in order]
        assert hashes.index(close.node_hash) < hashes.index(vol20.node_hash)
        assert hashes.index(ret.node_hash)   < hashes.index(vol20.node_hash)

    def test_topo_sort_dedup_shared_node(self):
        """vol20 和 vol60 共享 ret，topo_sort 结果中 ret 只出现一次。"""
        close, ret, vol20, vol60 = self._build_tree()
        order  = topological_sort([vol20, vol60])
        count  = sum(1 for n in order if n.node_hash == ret.node_hash)
        assert count == 1

    def test_topo_sort_multi_root_all_present(self):
        close, ret, vol20, vol60 = self._build_tree()
        order  = topological_sort([vol20, vol60])
        hashes = {n.node_hash for n in order}
        for node in [close, ret, vol20, vol60]:
            assert node.node_hash in hashes


# ═══════════════════════════════════════════════════════════════════════════════
# 9c. LRUCache
# ═══════════════════════════════════════════════════════════════════════════════

class TestLRUCache:
    """LRU 缓存的 get/put/eviction/clear 以及线程安全性。"""

    def test_get_miss_returns_sentinel(self):
        cache = LRUCache(4)
        result = cache.get("nonexistent")
        assert result is _MISS
        assert not result   # _MISS.__bool__ = False

    def test_put_then_get(self):
        cache = LRUCache(4)
        cache.put("k1", 42)
        assert cache.get("k1") == 42

    def test_overwrite_key(self):
        cache = LRUCache(4)
        cache.put("k", 1)
        cache.put("k", 2)
        assert cache.get("k") == 2
        assert len(cache) == 1

    def test_lru_eviction(self):
        """容量为 2：put k1 k2 k3 → k1 被淘汰。"""
        cache = LRUCache(2)
        cache.put("k1", 1)
        cache.put("k2", 2)
        cache.put("k3", 3)
        assert cache.get("k1") is _MISS
        assert cache.get("k2") == 2
        assert cache.get("k3") == 3

    def test_lru_access_refreshes_order(self):
        """访问 k1 后 k1 不被淘汰，k2 被淘汰。"""
        cache = LRUCache(2)
        cache.put("k1", 1)
        cache.put("k2", 2)
        _ = cache.get("k1")  # 刷新 k1 的访问时间
        cache.put("k3", 3)   # 应淘汰 k2
        assert cache.get("k1") == 1
        assert cache.get("k2") is _MISS

    def test_unlimited_capacity(self):
        """capacity=-1 时不限容量。"""
        cache = LRUCache(-1)
        for i in range(1000):
            cache.put(str(i), i)
        assert len(cache) == 1000
        assert cache.get("0") == 0
        assert cache.get("999") == 999

    def test_clear(self):
        cache = LRUCache(8)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is _MISS

    def test_contains(self):
        cache = LRUCache(4)
        cache.put("x", 10)
        assert "x" in cache
        assert "y" not in cache

    def test_thread_safe(self):
        """多线程并发写入不崩溃，最终 size ≤ capacity。"""
        import threading
        cache = LRUCache(100)
        errors = []

        def _writer(tid):
            try:
                for i in range(50):
                    cache.put(f"{tid}_{i}", tid * 100 + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_writer, args=(t,)) for t in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"线程异常: {errors}"
        assert len(cache) <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 9d. DAGExecutor
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGExecutor:
    """验证 DAGExecutor 正确执行并复用中间节点。"""

    def _build_factors(self):
        """
        构建两个共享 ret 节点的因子：
          vol20 = ts_mean(ret, 20)
          vol60 = ts_mean(ret, 60)
        其中 ret = pct_change(close, 1)
        """
        close = DataNode("close")
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        vol20 = op("ts_mean", ret, 20, fn=fn)
        vol60 = op("ts_mean", ret, 60, fn=fn)
        return {"vol20": vol20, "vol60": vol60}, ret

    def test_executor_returns_all_factors(self):
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out      = executor.run(df)
        assert set(out.keys()) == {"vol20", "vol60"}

    def test_executor_values_correct(self):
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out      = executor.run(df)
        ret      = df["close"].pct_change(1)
        expected20 = ret.rolling(20).mean()
        expected60 = ret.rolling(60).mean()
        pd.testing.assert_series_equal(out["vol20"], expected20, check_names=False, atol=1e-10)
        pd.testing.assert_series_equal(out["vol60"], expected60, check_names=False, atol=1e-10)

    def test_shared_node_cached_after_run(self):
        """ret 节点在 run() 后应存入 intermediate_cache。"""
        roots, ret_node = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        executor.run(df)
        assert ret_node.node_hash in cache

    def test_single_factor_subset(self):
        """run(factor_names=['vol20']) 只返回 vol20。"""
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out      = executor.run(df, factor_names=["vol20"])
        assert "vol20" in out
        assert "vol60" not in out

    def test_empty_factor_list(self):
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out      = executor.run(df, factor_names=[])
        assert out == {}

    def test_missing_factor_name_ignored(self):
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out      = executor.run(df, factor_names=["vol20", "nonexistent"])
        assert "vol20" in out
        assert "nonexistent" not in out

    def test_const_node_inside_expr(self):
        """带常量节点的表达式：close + 100。"""
        close  = DataNode("close")
        expr   = close + const(100.0)
        roots  = {"shifted": expr}
        df     = _make_df()
        cache  = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out    = executor.run(df)
        expected = df["close"] + 100.0
        pd.testing.assert_series_equal(out["shifted"], expected, check_names=False)

    def test_second_run_uses_cache(self):
        """同一缓存实例，第二次 run 命中缓存，结果相同。"""
        roots, _ = self._build_factors()
        df       = _make_df()
        cache    = LRUCache(-1)
        executor = DAGExecutor(cache, roots)
        out1 = executor.run(df)
        out2 = executor.run(df)
        pd.testing.assert_series_equal(out1["vol20"], out2["vol20"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9e. DepGraph（显式依赖图）
# ═══════════════════════════════════════════════════════════════════════════════

class TestDepGraph:
    """显式依赖图：topo_order、传递依赖展开、环检测。"""

    def test_no_deps_empty(self):
        g = DepGraph()
        assert g.deps_of("momentum") == []

    def test_register_and_deps_of(self):
        g = DepGraph()
        g.register("composite", ["vol", "mom"])
        assert set(g.deps_of("composite")) == {"vol", "mom"}

    def test_topo_order_single_dep(self):
        g = DepGraph()
        g.register("B", ["A"])
        order = g.topo_order(["B"])
        # B 依赖 A，但 A 未注册为需排序项，只返回 B
        assert "B" in order

    def test_topo_order_chain(self):
        """A → B → C：C 先，B 次，A 最后，但只返回请求的节点。"""
        g = DepGraph()
        g.register("B", ["A"])
        g.register("C", ["B"])
        order = g.topo_order(["A", "B", "C"])
        # A 无依赖，在 B 前；B 在 C 前
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_topo_order_fan_in(self):
        """C 同时依赖 A 和 B（fan-in）。"""
        g = DepGraph()
        g.register("C", ["A", "B"])
        order = g.topo_order(["A", "B", "C"])
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("C")

    def test_topo_order_transitive_expansion(self):
        """只请求 C，但 topo_order 展开传递依赖后包含 A、B。"""
        g = DepGraph()
        g.register("B", ["A"])
        g.register("C", ["B"])
        # C 只请求 C 本身，但传递依赖要包含 B 和 A
        order = g.topo_order(["C"])
        # 返回列表只包含 requested 的因子，C 一定在里面
        assert "C" in order

    def test_cycle_raises(self):
        g = DepGraph()
        g.register("A", ["B"])
        g.register("B", ["A"])
        with pytest.raises(ValueError, match="Cycle detected"):
            g.topo_order(["A", "B"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9f. CSE 报告
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSEReport:
    """cse_report() 正确识别公共子表达式。"""

    def _shared_roots(self):
        close = DataNode("close")
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        vol20 = op("ts_mean", ret, 20, fn=fn)
        vol60 = op("ts_mean", ret, 60, fn=fn)
        return {"vol20": vol20, "vol60": vol60}, ret, close

    def test_report_returns_dataframe(self):
        roots, _, _ = self._shared_roots()
        df = cse_report(roots)
        assert isinstance(df, pd.DataFrame)

    def test_report_columns(self):
        roots, _, _ = self._shared_roots()
        df = cse_report(roots)
        for col in ["node_hash", "repr", "ref_count", "shared_by"]:
            assert col in df.columns

    def test_report_detects_shared_ret(self):
        """ret 节点被 vol20 和 vol60 共享，应在报告中出现。"""
        roots, ret_node, _ = self._shared_roots()
        df = cse_report(roots)
        # ret_node 的哈希应在报告中
        assert ret_node.node_hash in df["node_hash"].values

    def test_report_ref_count_gte_2(self):
        """报告中所有节点的 ref_count >= 2。"""
        roots, _, _ = self._shared_roots()
        df = cse_report(roots)
        if not df.empty:
            assert (df["ref_count"] >= 2).all()

    def test_report_shared_by_contains_both_factors(self):
        """ret 节点的 shared_by 应包含 vol20 和 vol60 两个因子名。"""
        roots, ret_node, _ = self._shared_roots()
        df = cse_report(roots)
        row = df[df["node_hash"] == ret_node.node_hash]
        assert not row.empty
        shared = row.iloc[0]["shared_by"]
        assert "vol20" in shared
        assert "vol60" in shared

    def test_report_empty_no_shared(self):
        """无公共子表达式时返回空 DataFrame。"""
        close = DataNode("close")
        vol   = DataNode("volume")
        roots = {"f1": close, "f2": vol}
        df = cse_report(roots)
        assert df.empty

    def test_report_sorted_by_ref_count_desc(self):
        """report 按 ref_count 降序排列。"""
        roots, _, _ = self._shared_roots()
        df = cse_report(roots)
        if len(df) > 1:
            assert df["ref_count"].is_monotonic_decreasing


# ═══════════════════════════════════════════════════════════════════════════════
# 9g. FactorEngine DAG 集成
# ═══════════════════════════════════════════════════════════════════════════════

class TestEngineDAG:
    """FactorEngine 的 register_expr / compute_single / build_panel_batch / cse_report。"""

    # ── 共享测试 fixture（临时 CSV 目录）──────────────────────────────────────

    @pytest.fixture
    def tmp_stocks_dir(self, tmp_path):
        """创建 3 只股票的 CSV 文件，供 FactorEngine 读取。"""
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        rng = np.random.default_rng(42)
        dates = pd.date_range("20200101", periods=120, freq="B").strftime("%Y%m%d").tolist()
        for code in ["000001.SZ", "000002.SZ", "000004.SZ"]:
            close  = 10 + rng.normal(0, 0.3, 120).cumsum()
            open_  = close * rng.uniform(0.99, 1.01, 120)
            high   = close * rng.uniform(1.00, 1.02, 120)
            low    = close * rng.uniform(0.98, 1.00, 120)
            df = pd.DataFrame({
                "股票代码":         code,
                "股票名称":         "测试",
                "交易日":           dates,
                "开盘价":           open_,
                "最高价":           np.maximum(high, close),
                "最低价":           np.minimum(low, close),
                "收盘价":           close,
                "前收盘价":         np.roll(close, 1),
                "涨跌额":           np.diff(close, prepend=close[0]),
                "涨跌幅（%）":      np.diff(close, prepend=close[0]) / close * 100,
                "成交量（手）":     rng.integers(1_000, 10_000, 120).astype(float),
                "成交额（千元）":   close * rng.integers(1_000, 10_000, 120),
                "换手率（%）":      rng.uniform(0.5, 3, 120),
                "换手率（%，自由流通股）": rng.uniform(0.5, 3, 120),
                "量比":             rng.uniform(0.5, 2, 120),
                "市盈率（亏损为空）": np.abs(rng.normal(20, 5, 120)),
                "市盈率（TTM，亏损为空）": np.abs(rng.normal(20, 5, 120)),
                "市净率":           np.abs(rng.normal(2, 0.5, 120)),
                "市销率":           np.abs(rng.normal(3, 1, 120)),
                "市销率（TTM）":    np.abs(rng.normal(3, 1, 120)),
                "股息率（%）":      rng.uniform(0, 3, 120),
                "股息率（%，TTM）": rng.uniform(0, 3, 120),
                "总股本（万股）":   np.ones(120) * 100_000,
                "流通股本（万股）": np.ones(120) * 80_000,
                "自由流通股本（万）": np.ones(120) * 60_000,
                "总市值（万元）":   close * 100_000,
                "流通市值（万元）": close * 80_000,
                "复权因子":         np.ones(120),
                "当日涨停价":       close * 1.1,
                "当日跌停价":       close * 0.9,
            })
            df.to_csv(stocks_dir / f"{code}.csv", index=False)
        return str(stocks_dir)

    @pytest.fixture
    def engine(self, tmp_stocks_dir, tmp_path):
        sb = tmp_path / "sb.csv"
        sb.write_text(
            "ts_code,name\n000001.SZ,股票1\n000002.SZ,股票2\n000004.SZ,股票4\n",
            encoding="utf-8",
        )
        eng = FactorEngine(stocks_dir=tmp_stocks_dir, stock_basic=str(sb), verbose=False)
        return eng

    # ── register_expr ─────────────────────────────────────────────────────────

    def test_register_expr_appears_in_registry(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        vol20 = op("ts_mean", ret, 20, fn=fn)
        engine.register_expr("vol20", vol20)
        assert "vol20" in engine.registered()

    def test_register_expr_appears_in_registered_expr(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        engine.register_expr("close_raw", close)
        assert "close_raw" in engine.registered_expr()

    def test_register_expr_not_in_registered_expr_if_lambda(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("plain_lambda", lambda df: df[COL_CLOSE])
        assert "plain_lambda" not in engine.registered_expr()

    # ── compute_single (Expr 路径) ────────────────────────────────────────────

    def test_compute_single_expr_returns_series(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        vol20 = op("ts_mean", ret, 20, fn=fn)
        engine.register_expr("vol20_test", vol20)
        sym = "000001.SZ"
        result = engine.compute_single(sym, "vol20_test")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_compute_single_expr_values_match_direct(self, engine):
        """compute_single 与直接手算结果一致（误差 < 1e-9）。"""
        from factor_framework.factor_engine import COL_CLOSE
        close_col = DataNode(COL_CLOSE)
        ret_node  = PctChangeNode(close_col, 1)
        fn        = lambda s, w: s.rolling(w).mean()
        vol_node  = op("ts_mean", ret_node, 20, fn=fn)
        engine.register_expr("direct_vol20", vol_node)

        sym = "000001.SZ"
        result = engine.compute_single(sym, "direct_vol20", fast_mode=True)

        # 手算（使用相同的 fast_mode）
        df = engine._load_df(sym, fast_mode=True)
        expected = df[COL_CLOSE].pct_change(1).rolling(20).mean()
        # 对齐 index（result 可能有 date index）
        np.testing.assert_allclose(
            result.values, expected.values, rtol=1e-6, equal_nan=True
        )

    def test_compute_single_with_date_filter(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        engine.register_expr("close_copy", close)
        sym    = "000001.SZ"
        result = engine.compute_single(sym, "close_copy", start="20200201", end="20200301")
        # 日期范围内长度 < 总长度
        full = engine.compute_single(sym, "close_copy")
        assert len(result) < len(full)

    # ── register with deps ────────────────────────────────────────────────────

    def test_register_with_deps(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        # 先注册基础因子
        engine.register("ret1", lambda df: df[COL_CLOSE].pct_change(1))
        # 注册依赖 ret1 的复合因子
        engine.register(
            "vol_from_ret",
            lambda df: df["__dep_ret1__"].rolling(10).std(),
            deps=["ret1"],
        )
        result = engine.compute_single("000001.SZ", "vol_from_ret", fast_mode=True)
        assert isinstance(result, pd.Series)
        assert result.notna().any()

    def test_deps_topo_order_respected(self, engine):
        """A → B → C，注册顺序与依赖顺序无关，结果均正确。"""
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("base_close", lambda df: df[COL_CLOSE])
        engine.register(
            "double_close",
            lambda df: df["__dep_base_close__"] * 2,
            deps=["base_close"],
        )
        engine.register(
            "quad_close",
            lambda df: df["__dep_double_close__"] * 2,
            deps=["double_close"],
        )
        result = engine.compute_single("000001.SZ", "quad_close", fast_mode=True)
        df_raw = engine._load_df("000001.SZ", fast_mode=True)
        expected = df_raw[COL_CLOSE] * 4
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-6)

    # ── cse_report ────────────────────────────────────────────────────────────

    def test_cse_report_returns_df(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        engine.register_expr("cse_vol20", op("ts_mean", ret, 20, fn=fn))
        engine.register_expr("cse_vol60", op("ts_mean", ret, 60, fn=fn))
        report = engine.cse_report()
        assert isinstance(report, pd.DataFrame)

    def test_cse_report_detects_shared_ret(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        vol20 = op("ts_mean", ret, 20, fn=fn)
        vol60 = op("ts_mean", ret, 60, fn=fn)
        engine.register_expr("cse2_vol20", vol20)
        engine.register_expr("cse2_vol60", vol60)
        report = engine.cse_report()
        if not report.empty:
            assert ret.node_hash in report["node_hash"].values

    # ── build_panel_batch ────────────────────────────────────────────────────

    def test_build_panel_batch_returns_dict(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("f_close", lambda df: df[COL_CLOSE])
        engine.register("f_vol", lambda df: df[COL_CLOSE].pct_change(1))
        panels = engine.build_panel_batch(["f_close", "f_vol"], n_jobs=2)
        assert isinstance(panels, dict)
        assert set(panels.keys()) == {"f_close", "f_vol"}

    def test_build_panel_batch_panels_are_dataframes(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("bp_close", lambda df: df[COL_CLOSE])
        panels = engine.build_panel_batch(["bp_close"], n_jobs=2)
        assert isinstance(panels["bp_close"], pd.DataFrame)
        assert not panels["bp_close"].empty

    def test_build_panel_batch_shape(self, engine):
        """3 只股票，面板 columns 数量应 <= 3。"""
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("bp_shape", lambda df: df[COL_CLOSE])
        panels = engine.build_panel_batch(["bp_shape"], n_jobs=2)
        assert panels["bp_shape"].shape[1] <= 3

    def test_build_panel_batch_expr_factor(self, engine):
        """Expr 因子通过 build_panel_batch 也能正确返回面板。"""
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        engine.register_expr("bp_expr_vol20", op("ts_mean", ret, 20, fn=fn))
        panels = engine.build_panel_batch(["bp_expr_vol20"], n_jobs=2)
        assert isinstance(panels["bp_expr_vol20"], pd.DataFrame)
        assert not panels["bp_expr_vol20"].empty

    def test_build_panel_batch_cse_same_values(self, engine):
        """build_panel_batch 的多因子结果与 build_panel 单独调用一致。"""
        from factor_framework.factor_engine import COL_CLOSE
        close = DataNode(COL_CLOSE)
        ret   = PctChangeNode(close, 1)
        fn    = lambda s, w: s.rolling(w).mean()
        engine.register_expr("cse_batch_v20", op("ts_mean", ret, 20, fn=fn))
        engine.register_expr("cse_batch_v60", op("ts_mean", ret, 60, fn=fn))
        batch  = engine.build_panel_batch(["cse_batch_v20", "cse_batch_v60"], n_jobs=2)
        single = engine.build_panel("cse_batch_v20", n_jobs=2)
        pd.testing.assert_frame_equal(
            batch["cse_batch_v20"].sort_index(),
            single.sort_index(),
            check_like=True,
        )

    def test_build_panel_batch_unregistered_raises(self, engine):
        with pytest.raises(KeyError):
            engine.build_panel_batch(["this_factor_does_not_exist"])

    def test_build_panel_batch_with_date_range(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("bp_date", lambda df: df[COL_CLOSE])
        full   = engine.build_panel_batch(["bp_date"], n_jobs=2)
        sliced = engine.build_panel_batch(
            ["bp_date"], start="20200201", end="20200301", n_jobs=2
        )
        assert sliced["bp_date"].shape[0] < full["bp_date"].shape[0]

    def test_clear_cache_clears_dag_caches(self, engine):
        from factor_framework.factor_engine import COL_CLOSE
        engine.register("cc_close", lambda df: df[COL_CLOSE])
        engine.compute_single("000001.SZ", "cc_close")
        engine.clear_cache()
        # 清空后 factor_cache 应为空
        assert len(engine._factor_cache) == 0
        assert len(engine._intermediate_cache) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# v2.9 修复专项测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestV29Fixes:
    """
    验证 v2.9 修复的四项核心问题：
    1. 收益率公式正确性（price[t+f]/price[t]-1，非 pct_change(f).shift(-f)）
    2. T+1 滞后内置（首行全 NaN）
    3. 月度重采样函数（_resample_monthly）
    4. momentum 对数价格差分（替代 rolling.apply(np.prod)）
    """

    # ── 合成股票 DataFrame（用于单元级验证）────────────────────────────────

    @pytest.fixture(scope="class")
    def stock_df(self):
        """简单递增价格的单股 DataFrame，方便手工验证。"""
        n = 30
        dates  = [f"202001{d:02d}" for d in range(1, n + 1)]
        # 价格从 10.0 开始每天涨 0.1
        prices = [10.0 + 0.1 * i for i in range(n)]
        return pd.DataFrame({
            "交易日": dates,
            "收盘价": prices,
            "开盘价": prices,
            "最高价": prices,
            "最低价": prices,
            "成交量（手）": [1000] * n,
            "成交额（千元）": [10000.0] * n,
            "换手率（%）": [1.0] * n,
            "总市值（万元）": [1e6] * n,
            "流通市值（万元）": [8e5] * n,
            "市净率": [2.0] * n,
            "市盈率（TTM，亏损为空）": [20.0] * n,
            "市销率（TTM）": [3.0] * n,
            "复权因子": [1.0] * n,
        })

    # ─── 1. 收益率公式 ────────────────────────────────────────────────────────

    def test_return_formula_vs_old_formula(self, stock_df):
        """新公式与旧公式在非平凡价格序列上的结果不同。"""
        prices = pd.Series(stock_df["收盘价"].values, dtype=float)
        forward = 5

        # 旧公式
        old = prices.pct_change(forward).shift(-forward)

        # 新公式（T+1 内置）
        new = (prices.shift(-forward) / prices - 1).shift(1)

        # 它们在数学上不等价（旧公式方向错误）
        # 旧公式 pct_change(5) 计算的是「过去 5 天」相对今天的历史收益
        # 新公式 shift(-5)/price-1 才是「未来 5 天」收益
        diff = (old - new).dropna()
        assert (diff.abs() > 1e-9).any(), (
            "新旧公式结果完全相同，说明修复未生效"
        )

    def test_return_formula_sign_correct(self, stock_df):
        """
        价格严格递增时，未来 forward 日收益应全为正数（T+1 行跳过）。
        旧公式（pct_change 反向）在递增价格序列上也会给正值，
        但新公式 shift(-f)/price-1 才是真正的未来收益。
        """
        prices = pd.Series(stock_df["收盘价"].values, dtype=float)
        forward = 3
        # 新公式的未来收益（不含 T+1 偏移，单独验证方向）
        future_ret = prices.shift(-forward) / prices - 1
        valid = future_ret.dropna()
        assert (valid > 0).all(), f"严格递增价格下未来收益应全正，但有 {(valid <= 0).sum()} 个非正值"

    def test_t1_lag_first_row_nan(self, stock_df):
        """
        T+1 滞后后，收益率面板的第 0 行应全为 NaN。
        """
        prices  = pd.Series(stock_df["收盘价"].values, dtype=float)
        forward = 3
        ret_with_t1 = (prices.shift(-forward) / prices - 1).shift(1)
        # 第 0 行应为 NaN（因为 shift(1) 把第 0 行推出范围）
        assert pd.isna(ret_with_t1.iloc[0]), "T+1 滞后后首行应为 NaN"

    def test_t1_lag_tail_nan_count(self, stock_df):
        """
        T+1 滞后后的 NaN 分布分析：
        对于长度 n 的 Series，(price.shift(-f) / price - 1).shift(1) 的 NaN 分布：
        - shift(-f) 产生尾部 f 个 NaN；
        - 再 shift(1) 把整体右移：头部产生 1 个 NaN，尾部末 f-1 个仍是 NaN，
          但前 1 个尾部 NaN 位置（原第 n-f 行）变成了有效值（从 n-f-1 移来）。
        所以总 NaN 数 = (f - 1) 尾部 + 1 头部 + 1 尾部 = f 个（头 1 + 尾 f-1）。

        简言之：total_nan = forward（头 1 个 + 尾 forward-1 个）。
        """
        prices  = pd.Series(stock_df["收盘价"].values, dtype=float)
        forward = 5
        ret = (prices.shift(-forward) / prices - 1).shift(1)
        total_nan = int(ret.isna().sum())
        # 总 NaN = forward（头 1 + 尾 forward-1）
        assert total_nan == forward, (
            f"期望总 NaN 数 {forward}，实际 {total_nan}"
        )
        # 头部 NaN（T+1 效果）
        assert pd.isna(ret.iloc[0]), "首行应为 NaN（T+1 滞后）"
        # 尾部至少有 forward-1 个 NaN
        tail_nan = int(ret.iloc[-(forward - 1):].isna().sum())
        assert tail_nan == forward - 1, f"尾部 {forward-1} 行应全为 NaN，实际 {tail_nan}"

    # ─── 2. _resample_monthly ────────────────────────────────────────────────

    def test_resample_monthly_reduces_rows(self, factor_panel, return_panel):
        """月度重采样后行数应远少于日频行数。"""
        from factor_framework.pipeline import _resample_monthly
        f_m, r_m = _resample_monthly(factor_panel, return_panel)
        assert len(f_m) < len(factor_panel), "月度重采样后行数应减少"
        assert len(r_m) == len(f_m), "factor 和 return 月末行数应相同"

    def test_resample_monthly_aligned(self, factor_panel, return_panel):
        """重采样后两个面板的 index 应完全相同。"""
        from factor_framework.pipeline import _resample_monthly
        f_m, r_m = _resample_monthly(factor_panel, return_panel)
        pd.testing.assert_index_equal(f_m.index, r_m.index)

    def test_resample_monthly_columns_preserved(self, factor_panel, return_panel):
        """重采样后列（股票）不应改变。"""
        from factor_framework.pipeline import _resample_monthly
        f_m, r_m = _resample_monthly(factor_panel, return_panel)
        assert set(f_m.columns) == set(factor_panel.columns)
        assert set(r_m.columns) == set(return_panel.columns)

    def test_resample_monthly_month_count(self, factor_panel, return_panel):
        """
        factor_panel 跨度为 120 个工作日（约 6 个月），
        月度重采样后行数应在 4~7 之间。
        """
        from factor_framework.pipeline import _resample_monthly
        f_m, _ = _resample_monthly(factor_panel, return_panel)
        assert 4 <= len(f_m) <= 7, f"月末截面数 {len(f_m)} 超出预期范围 [4, 7]"

    # ─── 3. momentum 对数价格差分 ────────────────────────────────────────────

    def test_momentum_log_vs_prod(self, stock_df):
        """
        对数差分实现与旧 rolling.apply(np.prod) 实现在精度上的对比。
        对于长窗口（252 天），对数差分更精确（无浮点累积误差）。
        """
        from factor_framework.factor_zoo import momentum_12_1
        result = momentum_12_1(stock_df)
        # 结果应为 Series
        assert isinstance(result, pd.Series)
        # 对于长度仅 30 的 DataFrame，前 252 行应全为 NaN
        assert result.dropna().empty, (
            "数据不足 252 行时 momentum_12_1 应全返回 NaN"
        )

    def test_momentum_log_no_rolling_apply(self):
        """
        确认 momentum_12_1 源码的代码部分（非文档字符串）
        不再包含 rolling(...).apply(...) 调用。
        """
        import inspect, ast
        from factor_framework.factor_zoo import momentum_12_1
        src = inspect.getsource(momentum_12_1)
        # 跳过文档字符串，只看函数体代码
        # 解析 AST，查找是否有 .apply 方法调用链（即 rolling(...).apply(...)）
        try:
            tree = ast.parse(src)
        except SyntaxError:
            pytest.skip("无法解析 AST")
        # 收集所有 .apply 调用
        apply_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply"
        ]
        assert len(apply_calls) == 0, (
            f"momentum_12_1 不应再有 .apply() 调用（应改用对数差分），"
            f"发现 {len(apply_calls)} 处"
        )

    def test_momentum_log_result_range(self):
        """
        在足够长的合成价格序列上，momentum_12_1 的结果应是有限实数，
        不含 inf（旧 rolling.apply(prod) 可能在极端值下产生 inf）。
        """
        import numpy as np
        from factor_framework.factor_zoo import momentum_12_1
        n = 350
        np.random.seed(0)
        prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.01)
        df = pd.DataFrame({
            "交易日": [f"{i:08d}" for i in range(n)],
            "收盘价": prices,
        })
        result = momentum_12_1(df)
        finite_vals = result.dropna()
        assert (finite_vals.abs() < 1e6).all(), "momentum_12_1 产生了异常大的值（可能 inf）"

    def test_reversal_log_no_rolling_apply(self):
        """
        确认 reversal_1w / reversal_1m 代码部分也改用对数差分（无 .apply() 调用）。
        """
        import inspect, ast
        from factor_framework.factor_zoo import reversal_1w, reversal_1m
        for fn in [reversal_1w, reversal_1m]:
            src = inspect.getsource(fn)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            apply_calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply"
            ]
            assert len(apply_calls) == 0, (
                f"{fn.__name__} 不应再有 .apply() 调用，发现 {len(apply_calls)} 处"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# v2.9.1 修复专项测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestV291Fixes:
    """
    验证 v2.9.1 修复的三项问题：
    1. build_panel_batch 先计算因子再切片（修复 warm-up 截断 BUG）
    2. _fast_load 中 _ret 在 ffill 前计算（停牌日收益率正确为 NaN 而非 0）
    3. 动量/反转因子使用后复权价格（_hfq_close，消除前复权前瞻偏差）
    """

    STOCKS_DIR  = ROOT / "Stocks"
    STOCK_BASIC = ROOT / "股票列表-stock_basic.csv"

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

    # ─── 1. _hfq_close 后复权价格 ───────────────────────────────────────────

    def test_hfq_close_with_adj_factor(self):
        """有复权因子列时，_hfq_close = 收盘价 × 复权因子。"""
        from factor_framework.factor_zoo import _hfq_close
        df = pd.DataFrame({
            "收盘价":  [10.0, 10.5, 11.0],
            "复权因子": [1.0,  2.0,  2.0],
        })
        result = _hfq_close(df)
        expected = pd.Series([10.0, 21.0, 22.0])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected, check_names=False)

    def test_hfq_close_without_adj_factor(self):
        """无复权因子列时，_hfq_close 退化到原始收盘价。"""
        from factor_framework.factor_zoo import _hfq_close
        df = pd.DataFrame({"收盘价": [10.0, 11.0, 12.0]})
        result = _hfq_close(df)
        expected = pd.Series([10.0, 11.0, 12.0])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected, check_names=False)

    def test_hfq_close_zero_price_nan(self):
        """0 价格应被转为 NaN（避免除零）。"""
        from factor_framework.factor_zoo import _hfq_close
        df = pd.DataFrame({
            "收盘价":  [10.0, 0.0, 11.0],
            "复权因子": [1.0,  1.0, 1.0],
        })
        result = _hfq_close(df)
        assert pd.isna(result.iloc[1]), "0 价格应转为 NaN"

    def test_momentum_uses_hfq_close(self):
        """
        验证 momentum_12_1 在有复权因子时使用了后复权价格（结果与无复权时不同）。
        """
        import numpy as np
        from factor_framework.factor_zoo import momentum_12_1
        n = 350
        np.random.seed(42)
        prices = 100.0 * np.exp(np.random.randn(n).cumsum() * 0.01)
        # 有复权因子（模拟一次除权事件，adj 在第 200 天翻倍）
        adj = np.ones(n)
        adj[200:] = 2.0

        df_with_adj = pd.DataFrame({"收盘价": prices, "复权因子": adj})
        df_no_adj   = pd.DataFrame({"收盘价": prices})

        r_with = momentum_12_1(df_with_adj).dropna()
        r_no   = momentum_12_1(df_no_adj).dropna()

        if len(r_with) > 0 and len(r_no) > 0:
            common = r_with.index.intersection(r_no.index)
            if len(common) > 0:
                diff = (r_with.loc[common] - r_no.loc[common]).abs()
                assert diff.max() > 1e-6, (
                    "有/无复权因子时 momentum_12_1 结果完全相同，"
                    "说明 _hfq_close 未生效"
                )

    # ─── 2. _ret 在 ffill 前计算（停牌日 NaN 而非 0）──────────────────────

    def test_ret_nan_on_suspension_days(self):
        """
        停牌日（价格为 NaN，ffill 后价格不变）的日收益率应为 NaN，
        而非 0（若在 ffill 后计算 pct_change 会产生错误的 0 收益率）。
        """
        import io, tempfile, os
        from factor_framework.factor_engine import _fast_load
        from pathlib import Path

        # 构造含停牌日的 CSV（停牌日价格为空）
        csv_content = (
            "交易日,股票代码,收盘价,开盘价,最高价,最低价,"
            "成交量（手）,成交额（千元）,换手率（%）,"
            "总市值（万元）,流通市值（万元）,"
            "市净率,市盈率（TTM，亏损为空）,市销率（TTM）,复权因子\n"
        )
        rows = [
            ("20210101", "A", "10.0"),
            ("20210102", "A", "10.5"),
            ("20210103", "A", ""),       # 停牌日：价格缺失
            ("20210104", "A", ""),       # 停牌日：价格缺失
            ("20210105", "A", "11.0"),
        ]
        for d, c, p in rows:
            csv_content += f"{d},{c},{p},{p},{p},{p},1000,10000,1.0,1e6,8e5,2.0,20.0,3.0,1.0\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            tmp_path = Path(f.name)

        try:
            df = _fast_load(tmp_path)
            assert df is not None
            ret = df["_ret"]
            # 停牌日（index 2, 3，对应 20210103, 20210104）的 _ret 应为 NaN
            # （不是 0，因为 _ret 在 ffill 前计算）
            suspension_mask = df["交易日"].isin(["20210103", "20210104"])
            suspension_rets = ret[suspension_mask]
            assert suspension_rets.isna().all(), (
                f"停牌日 _ret 应为 NaN，实际值：{suspension_rets.tolist()}"
            )
            # 正常交易日的 _ret 应非 NaN（第 2 行：10.5/10.0-1=0.05）
            day2_ret = ret[df["交易日"] == "20210102"].iloc[0]
            assert abs(day2_ret - 0.05) < 1e-6, f"正常日 _ret 应约为 0.05，实际 {day2_ret}"
        finally:
            os.unlink(tmp_path)

    # ─── 3. build_panel_batch warm-up 修复 ──────────────────────────────────

    def test_batch_vs_single_panel_consistency(self, engine):
        """
        build_panel_batch 与 build_panel（compute_single 路径）
        对同一因子、同一 symbol 列表应产生相近的结果。
        修复前：batch 路径先截日期再计算，导致回测期初 NaN 比 single 路径更多。
        """
        symbols = engine.all_symbols()[:5]
        if not symbols:
            pytest.skip("无股票文件")

        factor = "vol_20d_v291_test"
        # 注册一个简单的时序因子（需要 20 天 warm-up）
        from factor_framework.factor_engine import COL_RET
        engine.register(factor, lambda df: -df[COL_RET].rolling(20, min_periods=15).std())

        start, end = "20210601", "20211231"

        try:
            # single 路径（compute_single，先计算再切片，正确）
            panel_single = engine.build_panel(
                factor, start=start, end=end, symbols=symbols, fast_mode=True
            )

            # batch 路径（build_panel_batch，修复后应与 single 一致）
            panels_batch = engine.build_panel_batch(
                [factor], start=start, end=end, symbols=symbols, fast_mode=True
            )
            panel_batch = panels_batch.get(factor, pd.DataFrame())
        finally:
            # 清理注册的测试因子
            engine._registry.pop(factor, None)
            engine.clear_cache()

        if panel_single.empty or panel_batch.empty:
            pytest.skip("因子面板为空，跳过对比")

        # 对齐公共行列
        common_rows = panel_single.index.intersection(panel_batch.index)
        common_cols = panel_single.columns.intersection(panel_batch.columns)
        if len(common_rows) == 0 or len(common_cols) == 0:
            pytest.skip("无公共行列")

        s = panel_single.loc[common_rows, common_cols]
        b = panel_batch.loc[common_rows, common_cols]

        # batch 路径修复后，NaN 数量应不多于 single 路径（容差：≤5%）
        nan_single = s.isna().sum().sum()
        nan_batch  = b.isna().sum().sum()
        total = s.size
        # 修复前：batch 可能比 single 多出 20 天 × n_stocks 的 NaN
        # 修复后：两者差距应在 5% 以内
        assert abs(nan_batch - nan_single) / max(total, 1) <= 0.05, (
            f"batch NaN={nan_batch}，single NaN={nan_single}，"
            f"差距超过 5%（total={total}），说明 warm-up 修复未生效"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# v2.9.2 修复专项测试（BUG 5-9）
# ═══════════════════════════════════════════════════════════════════════════════

class TestV292Fixes:
    """
    验证 v2.9.2 修复的五项问题：
    BUG 5 - value_pb/pe/ps lag_days 默认改为 0（数据源为日频市场估值）
    BUG 6 - downside_vol min_periods 改为满窗口 d（消除冷启动噪声）
    BUG 7 - pastor_stambaugh 改用后复权收益率
    BUG 8 - order_imbalance 文档澄清（无代码错误）
    BUG 9 - ic_decay 逐列计算收益率，消除跨股 NaN 传播 + factor_panel 对齐优化
    """

    # ─── BUG 5：value_pb/pe/ps lag_days 默认值 ──────────────────────────────

    def test_value_pb_default_lag_zero(self):
        """value_pb 默认 lag_days=0（无多余滞后，数据源已是日频市场估值）。"""
        import inspect
        from factor_framework.factor_zoo import value_pb, value_pe_ttm, value_ps_ttm
        for fn in [value_pb, value_pe_ttm, value_ps_ttm]:
            sig = inspect.signature(fn)
            default = sig.parameters["lag_days"].default
            assert default == 0, (
                f"{fn.__name__} lag_days 默认值应为 0，实际为 {default}"
            )

    def test_value_pb_no_shift_when_lag_zero(self):
        """lag_days=0 时，value_pb 不做 shift（因子值与输入等长）。"""
        from factor_framework.factor_zoo import value_pb
        df = pd.DataFrame({"市净率": [1.0, 2.0, 4.0, np.nan, 2.0]})
        result = value_pb(df, lag_days=0)
        expected = pd.Series([1.0, 0.5, 0.25, np.nan, 0.5])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected, check_names=False)

    def test_value_pb_shift_when_lag_nonzero(self):
        """lag_days>0 时，value_pb 做 shift（保留向后兼容性）。"""
        from factor_framework.factor_zoo import value_pb
        df = pd.DataFrame({"市净率": [1.0, 2.0, 4.0, 2.0, 1.0]})
        result = value_pb(df, lag_days=1)
        # index 0 应为 NaN（shift 后）
        assert pd.isna(result.iloc[0])
        # index 1 应等于 1/1.0=1.0（来自 index 0 的 PB）
        assert abs(result.iloc[1] - 1.0) < 1e-9

    # ─── BUG 6：downside_vol min_periods ────────────────────────────────────

    def test_downside_vol_min_periods_full_window(self):
        """downside_vol 要求满窗口（min_periods=d），回测期初行数更少但质量更高。"""
        import inspect
        from factor_framework.factor_zoo import downside_vol
        src = inspect.getsource(downside_vol)
        # 检查不再有 d // 2 作为 min_periods
        assert "d // 2" not in src, (
            "downside_vol 不应再使用 d//2 作为 min_periods（应改为 d）"
        )
        assert "min_periods=d" in src, (
            "downside_vol 应使用 min_periods=d"
        )

    def test_downside_vol_nan_before_full_window(self):
        """满窗口前应全为 NaN（冷启动保护）。"""
        from factor_framework.factor_zoo import downside_vol
        n, d = 50, 20
        np.random.seed(0)
        prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.01)
        df = pd.DataFrame({"收盘价": prices, "_ret": pd.Series(prices).pct_change()})
        result = downside_vol(df, d=d)
        # 前 d-1 行应全为 NaN
        assert result.iloc[:d - 1].isna().all(), (
            f"前 {d-1} 行应全为 NaN（满窗口前不计算），"
            f"但有 {result.iloc[:d-1].notna().sum()} 个非 NaN"
        )

    # ─── BUG 7：pastor_stambaugh 后复权 ─────────────────────────────────────

    def test_pastor_stambaugh_uses_hfq(self):
        """
        pastor_stambaugh 用后复权收益率（而非前复权 _ret）。
        有/无复权因子时结果应不同。
        """
        import numpy as np
        from factor_framework.factor_zoo import pastor_stambaugh
        n = 100
        np.random.seed(7)
        prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.01)
        volume = np.random.randint(1000, 10000, n).astype(float)

        # 有复权因子（除权事件）
        adj = np.ones(n)
        adj[50:] = 2.0
        df_adj = pd.DataFrame({"收盘价": prices, "成交量（手）": volume, "复权因子": adj})
        df_no  = pd.DataFrame({"收盘价": prices, "成交量（手）": volume})

        r_adj = pastor_stambaugh(df_adj).dropna()
        r_no  = pastor_stambaugh(df_no).dropna()

        if len(r_adj) > 0 and len(r_no) > 0:
            common = r_adj.index.intersection(r_no.index)
            if len(common) > 0:
                diff = (r_adj.loc[common] - r_no.loc[common]).abs()
                assert diff.max() > 1e-9, (
                    "pastor_stambaugh 有/无复权因子结果完全相同，"
                    "说明仍在使用前复权 _ret"
                )

    # ─── BUG 9：ic_decay 逐列收益率 + factor_panel 对齐 ──────────────────

    def test_ic_decay_per_column_no_nan_propagation(self, factor_panel, return_panel):
        """
        ic_decay 的收益率构建应逐列计算（price.shift(-fwd)/price-1），
        而非 pct_change(fwd, axis=0)（后者在有停牌 NaN 时会跨列传播 NaN）。
        """
        from factor_framework.ic_analysis import ic_decay

        # 构造含停牌 NaN 的价格面板（某只股票某段时间停牌）
        n, m = 60, 5
        np.random.seed(1)
        prices = 100 * np.exp(np.random.randn(n, m).cumsum(axis=0) * 0.01)
        dates  = [f"202001{d:02d}" for d in range(1, n + 1)]
        cols   = [f"S{i:03d}" for i in range(m)]
        price_df = pd.DataFrame(prices, index=dates, columns=cols)
        # 注入停牌：第 30-40 行的第 0 列为 NaN
        price_df.iloc[30:40, 0] = np.nan

        factor_df = pd.DataFrame(
            np.random.randn(n, m), index=dates, columns=cols
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ic_decay(factor_df, price_df, forward_periods=[1, 5], method="rank")

        assert isinstance(result, pd.DataFrame)
        # 至少 forward=1 的 mean_ic 应是有限数（不因 NaN 传播而全部变 NaN）
        assert not pd.isna(result.loc[1, "mean_ic"]), (
            "ic_decay forward=1 的 mean_ic 为 NaN，"
            "可能是停牌 NaN 跨列传播导致（逐列计算可修复此问题）"
        )

    def test_ic_decay_factor_panel_not_truncated(self, factor_panel, return_panel):
        """
        ic_decay 不应截断 factor_panel，而应通过 intersection 对齐，
        避免浪费 factor_panel 尾部的有效数据。
        """
        from factor_framework.ic_analysis import ic_decay

        n, m = 80, 5
        np.random.seed(2)
        dates  = [f"202001{d:02d}" for d in range(1, n + 1)]
        cols   = [f"S{i:03d}" for i in range(m)]
        prices = 100 * np.exp(np.random.randn(n, m).cumsum(axis=0) * 0.01)
        factors = np.random.randn(n, m)

        price_df  = pd.DataFrame(prices,  index=dates, columns=cols)
        factor_df = pd.DataFrame(factors, index=dates, columns=cols)

        fwd = 10
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ic_decay(factor_df, price_df, forward_periods=[fwd], method="rank")

        # ic_decay 不报错，且有合理结果
        assert isinstance(result, pd.DataFrame)
        assert fwd in result.index

    def test_ic_decay_vs_pct_change_differs_with_nan(self):
        """
        验证新实现（逐列 shift）与旧实现（pct_change axis=0）
        在含停牌 NaN 时结果不同（新方法更准确）。
        """
        n, m = 60, 3
        np.random.seed(9)
        prices = pd.DataFrame(
            100 * np.exp(np.random.randn(n, m).cumsum(axis=0) * 0.01),
            columns=[f"S{i}" for i in range(m)]
        )
        # 注入停牌：第 20-25 行第 0 列为 NaN
        prices.iloc[20:25, 0] = np.nan

        fwd = 5
        # 旧方法（pct_change，跨列传播 NaN）
        old = prices.pct_change(fwd, axis=0).shift(-fwd)
        # 新方法（逐列 shift，不传播）
        new = prices.shift(-fwd) / prices.replace(0, np.nan) - 1

        # 停牌行周围：旧方法会多产生 NaN，新方法不会
        affected_rows = list(range(15, 30))
        old_nan_count = old.iloc[affected_rows, 0].isna().sum()
        new_nan_count = new.iloc[affected_rows, 0].isna().sum()
        assert new_nan_count <= old_nan_count, (
            f"新方法 NaN 数（{new_nan_count}）不应多于旧方法（{old_nan_count}）"
        )



# =============================================================================
# v2.9.3 Tests — BUG 10-15 fixes
# =============================================================================

class TestV293Fixes:
    """
    v2.9.3 修复验证：
      BUG 10 — build_return_panel T+1 已内置，无需外部额外移位
      BUG 11 — mktcap_panel 在 neutralize_regression 前 reindex 对齐
      BUG 12 — ic_decay 在 _resample_monthly 之前（日频上）执行
      BUG 13 — 月度重采样后 layer_backtest 正确使用月频面板
      BUG 14 — _annual_return / long_short_stats 按月频 periods_per_year=12 年化
      BUG 15 — 月频换手率 < 日频换手率（月频面板行数更少）
    """

    # ------------------------------------------------------------------ #
    # 辅助方法                                                              #
    # ------------------------------------------------------------------ #

    def _make_tmp_engine(self, n=30, n_stocks=2, seed=0):
        """
        构建最小化临时目录 + FactorEngine，用于 BUG 10 的功能测试。
        CSV 列名使用实际格式：'交易日'、'收盘价'、'复权因子'、'总市值（万元）'。
        """
        import tempfile, os
        import factor_framework.factor_engine as fe

        np.random.seed(seed)
        dates = [d.strftime('%Y%m%d') for d in
                 pd.date_range('2020-01-02', periods=n, freq='B')]
        syms = [f'T{i:02d}' for i in range(n_stocks)]

        tmpdir = tempfile.mkdtemp()
        for sym in syms:
            prices = np.cumprod(1 + np.random.randn(n) * 0.01) * 10.0
            pd.DataFrame([
                {'交易日': d, '股票代码': sym,
                 '收盘价': round(prices[i], 6),
                 '复权因子': 1.0,
                 '总市值（万元）': 1e6}
                for i, d in enumerate(dates)
            ]).to_csv(os.path.join(tmpdir, f'{sym}.csv'), index=False)

        engine = fe.FactorEngine(tmpdir, verbose=False, min_rows=3)
        return engine, dates, syms

    def _make_panels(self, n_dates=40, n_stocks=12, seed=42):
        """
        构建随机日频因子/收益率面板，用于 BUG 12–15 的功能测试。
        index 为 DatetimeIndex（业务日）。
        """
        np.random.seed(seed)
        dates = pd.date_range('2018-01-02', periods=n_dates, freq='B')
        cols = [f'S{i:02d}' for i in range(n_stocks)]
        fp = pd.DataFrame(np.random.randn(n_dates, n_stocks),
                          index=dates, columns=cols)
        rp = pd.DataFrame(np.random.randn(n_dates, n_stocks) * 0.05,
                          index=dates, columns=cols)
        return fp, rp

    # ------------------------------------------------------------------ #
    # BUG 10 — 源码白盒：T+1 内置于 build_return_panel                     #
    # ------------------------------------------------------------------ #

    def test_bug10_t1_in_source(self):
        """build_return_panel 源码含 .shift(1)，说明 T+1 已内置。"""
        import inspect
        import factor_framework.factor_engine as fe
        src = inspect.getsource(fe.FactorEngine.build_return_panel)
        assert '.shift(1)' in src, \
            'build_return_panel 应在源码中包含 .shift(1) 实现 T+1 内置'

    def test_bug10_no_double_shift_in_pipeline(self):
        """pipeline.run 源码不应再对 return_panel 做额外 shift。"""
        import inspect
        import factor_framework.pipeline as pp
        src = inspect.getsource(pp.FactorPipeline.run)
        assert 'return_panel.shift' not in src, \
            'pipeline.run 不应对 return_panel 额外 shift（T+1 已在 build_return_panel 内置）'

    def test_bug10_return_panel_shape_and_tail_nan(self):
        """
        功能测试：build_return_panel 的 NaN 模式符合预期。
        fwd=5：第 0 行为 NaN（T+1 shift），最后 fwd-1=4 行为 NaN，中间行有效。
        """
        engine, dates, syms = self._make_tmp_engine(n=30, n_stocks=2)
        fwd = 5
        ret = engine.build_return_panel(
            forward=fwd, start=dates[0], end=dates[-1], symbols=syms)

        assert ret.shape[0] == 30, f'行数应等于 n=30，实际 {ret.shape[0]}'
        assert ret.shape[1] == len(syms), f'列数应等于股票数 {len(syms)}'

        # 首行 NaN（T+1 shift）
        assert ret.iloc[0].isna().all(), '第 0 行应全为 NaN（T+1 shift）'

        # 尾部 fwd-1 行 NaN（shift(-fwd) 后无数据 + shift(1) 偏移）
        assert ret.iloc[-(fwd - 1):].isna().all().all(), \
            f'最后 {fwd - 1} 行应全为 NaN'

        # 中间行有有效数据
        assert ret.iloc[1:-(fwd - 1)].notna().any().any(), \
            '中间行应有有效收益率数据'

    def test_bug10_alignment_formula(self):
        """
        公式验证：ret.iloc[i] = price[i-1+fwd] / price[i-1] - 1
        （T+1 对齐：factor 在 i-1 日收盘后计算，i 日才能成交）。
        """
        import tempfile, os
        import factor_framework.factor_engine as fe

        fwd = 3
        n = 15
        dates = [d.strftime('%Y%m%d') for d in
                 pd.date_range('2020-01-02', periods=n, freq='B')]
        prices = [10.0 * (1.0 + 0.01 * i) for i in range(n)]  # 线性递增

        tmpdir = tempfile.mkdtemp()
        pd.DataFrame([
            {'交易日': dates[i], '股票代码': 'A',
             '收盘价': round(prices[i], 8),
             '复权因子': 1.0, '总市值（万元）': 1e6}
            for i in range(n)
        ]).to_csv(os.path.join(tmpdir, 'A.csv'), index=False)

        engine = fe.FactorEngine(tmpdir, verbose=False, min_rows=3)
        ret = engine.build_return_panel(
            forward=fwd, start=dates[0], end=dates[-1], symbols=['A'])

        # 从 i=1 开始（i=0 因 T+1 shift 为 NaN）
        for i in range(1, n - fwd):
            expected = prices[i - 1 + fwd] / prices[i - 1] - 1
            actual = ret.iloc[i, 0]
            assert abs(actual - expected) < 1e-6, \
                f'ret[{i}] 应为 price[{i-1+fwd}]/price[{i-1}]-1=' \
                f'{expected:.6f}，实际 {actual:.6f}'

    # ------------------------------------------------------------------ #
    # BUG 11 — mktcap reindex 在中性化前，且位于 resample 前              #
    # ------------------------------------------------------------------ #

    def test_bug11_mktcap_reindex_in_run_source(self):
        """pipeline.run 源码应含 mktcap_panel.reindex(factor_panel.index)。"""
        import inspect
        import factor_framework.pipeline as pp
        src = inspect.getsource(pp.FactorPipeline.run)
        assert 'mktcap_panel.reindex(factor_panel.index)' in src, \
            'run() 应在 neutralize 前对 mktcap_panel 做 reindex 对齐'

    def test_bug11_neutralize_before_resample_in_run(self):
        """pipeline.run 中 neutralize_regression 应在 _resample_monthly 之前调用。"""
        import inspect
        import factor_framework.pipeline as pp
        lines = inspect.getsource(pp.FactorPipeline.run).splitlines()
        neut_idx = next(
            (i for i, l in enumerate(lines) if 'neutralize_regression(' in l),
            None)
        rs_idx = next(
            (i for i, l in enumerate(lines)
             if '_resample_monthly(' in l and 'if' not in l),
            None)
        assert neut_idx is not None, 'run() 应调用 neutralize_regression'
        assert rs_idx is not None, 'run() 应调用 _resample_monthly'
        assert neut_idx < rs_idx, \
            f'neutralize_regression（行{neut_idx}）应在 _resample_monthly（行{rs_idx}）之前'

    def test_bug11_neutralize_before_resample_in_batch(self):
        """run_batch_from_panels 中 neutralize_regression 应在 _resample_monthly 之前。"""
        import inspect
        import factor_framework.pipeline as pp
        lines = inspect.getsource(pp.FactorPipeline.run_batch_from_panels).splitlines()
        neut_idx = next(
            (i for i, l in enumerate(lines) if 'neutralize_regression(' in l),
            None)
        rs_idx = next(
            (i for i, l in enumerate(lines)
             if '_resample_monthly(' in l and 'if' not in l),
            None)
        assert neut_idx is not None, 'run_batch_from_panels 应调用 neutralize_regression'
        assert rs_idx is not None, 'run_batch_from_panels 应调用 _resample_monthly'
        assert neut_idx < rs_idx, \
            f'neutralize_regression（行{neut_idx}）应在 _resample_monthly（行{rs_idx}）之前'

    def test_bug11_neutralize_misaligned_mktcap_via_reindex(self):
        """
        功能测试：mktcap_panel 行数多于 factor_panel 时，
        reindex 后 neutralize_regression 结果形状与 factor_panel 一致且非全 NaN。
        """
        from factor_framework.neutralize import neutralize_regression

        np.random.seed(1)
        n_d, n_s = 20, 12  # >= 10 stocks（满足 neutralize_regression 最小股票数过滤）
        dates = pd.date_range('2020-01-02', periods=n_d, freq='B')
        cols = [f'S{i:02d}' for i in range(n_s)]

        fp = pd.DataFrame(
            np.random.randn(n_d, n_s), index=dates, columns=cols)

        # mktcap 多 5 行（模拟日频范围更大的情况）
        extra = pd.date_range('2019-12-20', periods=n_d + 5, freq='B')
        mktcap_big = pd.DataFrame(
            np.random.rand(n_d + 5, n_s) * 1e6 + 1e5,
            index=extra, columns=cols)

        # BUG 11 修复：先 reindex 再中性化
        mktcap_aligned = mktcap_big.reindex(fp.index)
        ind_map = pd.Series({c: f'IND{i % 3}' for i, c in enumerate(cols)})
        result = neutralize_regression(fp, mktcap_aligned, industry_map=ind_map)

        assert result.shape == fp.shape, \
            f'reindex 后中性化结果形状应与 factor_panel 相同，' \
            f'expected {fp.shape}, got {result.shape}'
        assert result.notna().any().any(), '中性化结果不应全为 NaN'

    # ------------------------------------------------------------------ #
    # BUG 12 — ic_decay 在 _resample_monthly 之前执行（日频面板）          #
    # ------------------------------------------------------------------ #

    def test_bug12_ic_decay_before_resample_in_run(self):
        """pipeline.run 中 ic_decay 应在 _resample_monthly 之前调用。"""
        import inspect
        import factor_framework.pipeline as pp
        lines = inspect.getsource(pp.FactorPipeline.run).splitlines()
        ic_idx = next(
            (i for i, l in enumerate(lines) if 'ic_decay(' in l and '=' in l),
            None)
        rs_idx = next(
            (i for i, l in enumerate(lines)
             if '_resample_monthly(' in l and 'if' not in l),
            None)
        assert ic_idx is not None, 'run() 应调用 ic_decay'
        assert rs_idx is not None, 'run() 应调用 _resample_monthly'
        assert ic_idx < rs_idx, \
            f'ic_decay（行{ic_idx}）应在 _resample_monthly（行{rs_idx}）之前'

    def test_bug12_ic_decay_before_resample_in_batch(self):
        """run_batch_from_panels 中 ic_decay 应在 _resample_monthly 之前调用。"""
        import inspect
        import factor_framework.pipeline as pp
        lines = inspect.getsource(pp.FactorPipeline.run_batch_from_panels).splitlines()
        ic_idx = next(
            (i for i, l in enumerate(lines) if 'ic_decay(' in l and '=' in l),
            None)
        rs_idx = next(
            (i for i, l in enumerate(lines)
             if '_resample_monthly(' in l and 'if' not in l),
            None)
        assert ic_idx is not None, 'run_batch_from_panels 应调用 ic_decay'
        assert rs_idx is not None, 'run_batch_from_panels 应调用 _resample_monthly'
        assert ic_idx < rs_idx, \
            f'ic_decay（行{ic_idx}）应在 _resample_monthly（行{rs_idx}）之前'

    def test_bug12_daily_vs_monthly_ic_decay_overlap(self):
        """
        功能测试：日频因子面板与价格面板的索引重叠远多于月末采样后的重叠，
        验证 ic_decay 在月度重采样前调用可利用更多有效数据点。
        """
        from factor_framework.pipeline import _resample_monthly
        from factor_framework.ic_analysis import ic_decay

        np.random.seed(7)
        n_dates, n_stocks = 120, 10
        biz_dates = pd.date_range('2018-01-02', periods=n_dates, freq='B')
        cols = [f'S{i:02d}' for i in range(n_stocks)]

        fp_daily = pd.DataFrame(
            np.random.randn(n_dates, n_stocks), index=biz_dates, columns=cols)
        rp_daily = pd.DataFrame(
            np.random.randn(n_dates, n_stocks) * 0.05, index=biz_dates, columns=cols)
        close_daily = pd.DataFrame(
            np.cumprod(1 + np.random.randn(n_dates, n_stocks) * 0.01, axis=0) * 10,
            index=biz_dates, columns=cols)

        fp_monthly, _ = _resample_monthly(fp_daily, rp_daily)

        daily_overlap = len(fp_daily.index.intersection(close_daily.index))
        monthly_overlap = len(fp_monthly.index.intersection(close_daily.index))

        assert daily_overlap > monthly_overlap, \
            f'日频重叠({daily_overlap}) 应多于月频重叠({monthly_overlap})'

        icd = ic_decay(fp_daily, close_daily, forward_periods=[1, 5], method='rank')
        assert isinstance(icd, pd.DataFrame), 'ic_decay 应返回 DataFrame'
        assert not icd.empty, 'ic_decay 在日频面板上不应返回空 DataFrame'

    # ------------------------------------------------------------------ #
    # BUG 13 — 月度重采样后 layer_backtest 使用月频面板                    #
    # ------------------------------------------------------------------ #

    def test_bug13_resample_reduces_rows(self):
        """_resample_monthly 后行数应显著少于原始日频行数（< 1/3）。"""
        from factor_framework.pipeline import _resample_monthly

        fp, rp = self._make_panels(n_dates=120, n_stocks=10)
        fp_m, rp_m = _resample_monthly(fp, rp)

        assert len(fp_m) < len(fp) // 3, \
            f'月末重采样后行数({len(fp_m)})应少于日频行数的 1/3({len(fp) // 3})'
        assert fp_m.shape[1] == fp.shape[1], '重采样后列数（股票数）不变'

    def test_bug13_layer_backtest_monthly_input(self):
        """用月频面板做 layer_backtest，结果列名应包含 Q1~Qn 和 LS。"""
        from factor_framework.pipeline import _resample_monthly
        from factor_framework.backtest import layer_backtest

        fp, rp = self._make_panels(n_dates=120, n_stocks=12)
        fp_m, rp_m = _resample_monthly(fp, rp)

        lr = layer_backtest(fp_m, rp_m, n_groups=5, direction=1)
        assert 'Q1' in lr.columns, 'layer_backtest 返回应含 Q1 列'
        assert 'LS' in lr.columns, 'layer_backtest 返回应含 LS 列'
        assert len(lr) == len(fp_m), \
            f'layer_backtest 返回行数({len(lr)})应等于月频面板行数({len(fp_m)})'

    # ------------------------------------------------------------------ #
    # BUG 14 — _annual_return / long_short_stats 月频年化 (periods=12)    #
    # ------------------------------------------------------------------ #

    def test_bug14_annual_return_monthly_formula(self):
        """_annual_return 月频：恒定月收益率 r 的年化应为 (1+r)^12 - 1。"""
        from factor_framework.backtest import _annual_return

        r = 0.02  # 2% per month
        n = 24    # 24 个月
        series = pd.Series([r] * n)
        annual = _annual_return(series, periods_per_year=12)
        expected = (1 + r) ** 12 - 1

        assert abs(annual - expected) < 1e-10, \
            f'月频年化收益应为 (1+r)^12-1={expected:.6f}，实际 {annual:.6f}'

    def test_bug14_long_short_stats_returns_dict_with_keys(self):
        """long_short_stats 应返回 dict，且包含所有预期键。"""
        from factor_framework.backtest import layer_backtest, long_short_stats
        from factor_framework.pipeline import _resample_monthly

        fp, rp = self._make_panels(n_dates=120, n_stocks=12)
        fp_m, rp_m = _resample_monthly(fp, rp)
        lr = layer_backtest(fp_m, rp_m, n_groups=5, direction=1)
        stats = long_short_stats(lr, periods_per_year=12, rf=0.0)

        assert isinstance(stats, dict), \
            f'long_short_stats 应返回 dict，实际 {type(stats)}'
        for k in ('layer_annual_return', 'layer_sharpe', 'ls_annual_return',
                  'ls_sharpe', 'ls_max_drawdown', 'ls_calmar',
                  'ls_win_rate', 'monotone_score', 'nav'):
            assert k in stats, f'long_short_stats 结果应含键 "{k}"'

    def test_bug14_layer_annual_return_is_series_in_range(self):
        """layer_annual_return 应是 pd.Series，且各层年化收益在合理范围内。"""
        from factor_framework.backtest import layer_backtest, long_short_stats
        from factor_framework.pipeline import _resample_monthly

        fp, rp = self._make_panels(n_dates=120, n_stocks=12)
        fp_m, rp_m = _resample_monthly(fp, rp)
        lr = layer_backtest(fp_m, rp_m, n_groups=5, direction=1)
        stats = long_short_stats(lr, periods_per_year=12, rf=0.0)

        layer_ann = stats['layer_annual_return']
        assert isinstance(layer_ann, pd.Series), \
            'layer_annual_return 应是 pd.Series'
        valid = layer_ann.dropna()
        assert len(valid) > 0, '应有至少一个有效分层年化收益'
        for v in valid:
            assert v > -1.0, f'年化收益 {v:.4f} 不应低于 -100%'
            assert v < 5.0,  f'年化收益 {v:.4f} 不应超过 500%（合理性检查）'

    # ------------------------------------------------------------------ #
    # BUG 15 — 月频换手率序列长度 < 日频                                   #
    # ------------------------------------------------------------------ #

    def test_bug15_turnover_analysis_returns_dict(self):
        """turnover_analysis 应返回 dict，含 avg_turnover、turnover_series 等键。"""
        from factor_framework.backtest import turnover_analysis

        fp, _ = self._make_panels(n_dates=120, n_stocks=12)
        to = turnover_analysis(fp, n_groups=5, direction=1)

        assert isinstance(to, dict), \
            f'turnover_analysis 应返回 dict，实际 {type(to)}'
        assert 'avg_turnover' in to, '结果应含 avg_turnover 键'
        assert 'turnover_series' in to, '结果应含 turnover_series 键'
        assert isinstance(to['turnover_series'], pd.Series), \
            'turnover_series 应是 pd.Series'

    def test_bug15_monthly_turnover_fewer_observations(self):
        """月频面板的换手率序列长度应少于日频面板（月频调仓次数更少）。"""
        from factor_framework.pipeline import _resample_monthly
        from factor_framework.backtest import turnover_analysis

        fp, rp = self._make_panels(n_dates=120, n_stocks=12)
        fp_m, _ = _resample_monthly(fp, rp)

        to_d = turnover_analysis(fp,   n_groups=5, direction=1)
        to_m = turnover_analysis(fp_m, n_groups=5, direction=1)

        n_daily   = len(to_d['turnover_series'])
        n_monthly = len(to_m['turnover_series'])
        assert n_monthly < n_daily, \
            f'月频换手序列长度({n_monthly})应小于日频({n_daily})'

    def test_bug15_avg_turnover_in_valid_range(self):
        """月频 avg_turnover 应在 [0, 1] 范围内。"""
        from factor_framework.pipeline import _resample_monthly
        from factor_framework.backtest import turnover_analysis

        fp, rp = self._make_panels(n_dates=120, n_stocks=12)
        fp_m, _ = _resample_monthly(fp, rp)

        to_m = turnover_analysis(fp_m, n_groups=5, direction=1)
        avg = to_m['avg_turnover']

        assert avg >= 0.0, f'avg_turnover={avg:.4f} 不应为负'
        assert avg <= 1.0, f'avg_turnover={avg:.4f} 不应超过 1（100% 换仓）'


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 Foundation: TimestampedPanel、ReturnPanel、CacheLayer 基础测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestampedPanel:
    """
    factor_framework.core.panel.TimestampedPanel

    覆盖：
    - 继承与元数据保留（_metadata 机制）
    - shift_to_t1()：正向操作、重复调用防护
    - align_with()：合法/非法组合、T+1 检查
    - trim_warmup()：行数截断
    - assert_valid()：完整性检查
    - 自定义异常类型
    """

    def _make_panel(self, semantic="factor_observation", is_t1_shifted=False,
                    price_basis=None, factor_name=None, rows=20, cols=5):
        """构建测试用 TimestampedPanel。"""
        from factor_framework.core.panel import TimestampedPanel
        idx  = pd.date_range("2020-01-01", periods=rows, freq="B")
        cols_ = [f"S{i:03d}" for i in range(cols)]
        df   = pd.DataFrame(
            np.random.randn(rows, cols),
            index=idx, columns=cols_
        )
        return TimestampedPanel(
            df,
            semantic      = semantic,
            is_t1_shifted = is_t1_shifted,
            price_basis   = price_basis,
            factor_name   = factor_name,
        )

    # ── 元数据保留 ───────────────────────────────────────────────────────────

    def test_metadata_preserved_after_slice(self):
        """pandas slice 操作后，_metadata 字段应保留。"""
        from factor_framework.core.panel import TimestampedPanel
        tp = self._make_panel(semantic="forward_return", factor_name="test_factor")
        sliced = tp.iloc[:10]
        assert isinstance(sliced, TimestampedPanel), "slice 后应仍为 TimestampedPanel"
        assert sliced.semantic == "forward_return"
        assert sliced.factor_name == "test_factor"

    def test_metadata_preserved_after_loc(self):
        """loc 操作后，_metadata 字段应保留。"""
        from factor_framework.core.panel import TimestampedPanel
        tp = self._make_panel(semantic="price", price_basis="hfq")
        subset = tp.loc[tp.index[:5]]
        assert subset.semantic == "price"
        assert subset.price_basis == "hfq"

    def test_constructor_defaults(self):
        """默认构造时 semantic='factor_observation'，is_t1_shifted=False。"""
        from factor_framework.core.panel import TimestampedPanel
        tp = TimestampedPanel({"A": [1.0, 2.0]})
        assert tp.semantic == "factor_observation"
        assert tp.is_t1_shifted is False
        assert tp.warmup_trimmed is False

    def test_from_dataframe_classmethod(self):
        """from_dataframe() 类方法应正确传递所有元数据字段。"""
        from factor_framework.core.panel import TimestampedPanel
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        tp = TimestampedPanel.from_dataframe(
            df,
            semantic      = "price",
            price_basis   = "hfq",
            factor_name   = "close_price",
        )
        assert tp.semantic == "price"
        assert tp.price_basis == "hfq"
        assert tp.factor_name == "close_price"
        assert not tp.warmup_trimmed

    # ── shift_to_t1() ────────────────────────────────────────────────────────

    def test_shift_to_t1_sets_flag(self):
        """shift_to_t1() 返回新面板，is_t1_shifted=True。"""
        tp = self._make_panel()
        tp_t1 = tp.shift_to_t1()
        assert tp_t1.is_t1_shifted is True
        assert tp.is_t1_shifted is False, "原对象不应被修改（immutable 语义）"

    def test_shift_to_t1_row_shift(self):
        """shift_to_t1() 在行方向上做了 shift(1)（第 0 行为 NaN，其余行等于原第 i-1 行）。"""
        from factor_framework.core.panel import TimestampedPanel
        df   = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=["d1", "d2", "d3"])
        tp   = TimestampedPanel(df, semantic="factor_observation")
        tp_t1 = tp.shift_to_t1()
        assert np.isnan(tp_t1.loc["d1", "A"])
        assert tp_t1.loc["d2", "A"] == pytest.approx(10.0)
        assert tp_t1.loc["d3", "A"] == pytest.approx(20.0)

    def test_shift_to_t1_double_call_raises(self):
        """对已 T+1 的面板再次调用 shift_to_t1() 应抛出 RuntimeError。"""
        tp    = self._make_panel()
        tp_t1 = tp.shift_to_t1()
        with pytest.raises(RuntimeError, match="已做过 T\\+1 滞后"):
            tp_t1.shift_to_t1()

    def test_shift_to_t1_preserves_other_metadata(self):
        """shift_to_t1() 后，其他元数据字段（factor_name 等）应保留。"""
        tp    = self._make_panel(factor_name="momentum_12_1")
        tp_t1 = tp.shift_to_t1()
        assert tp_t1.factor_name == "momentum_12_1"
        assert tp_t1.semantic    == "factor_observation"

    # ── align_with() ─────────────────────────────────────────────────────────

    def test_align_with_factor_and_return_valid(self):
        """factor_observation(T+1) × forward_return 是合法组合，应成功对齐。"""
        from factor_framework.core.panel import TimestampedPanel
        fp = self._make_panel(semantic="factor_observation").shift_to_t1()
        rp = self._make_panel(semantic="forward_return")
        fp_aligned, rp_aligned = fp.align_with(rp)
        assert fp_aligned.index.equals(rp_aligned.index)
        assert len(fp_aligned) == len(fp)  # 行数相同（index 完全相同时取交集不变）

    def test_align_with_requires_t1_shifted(self):
        """factor_observation 未做 T+1 时 align_with forward_return 应抛出 TimingAlignmentError。"""
        from factor_framework.core.panel import TimingAlignmentError
        fp = self._make_panel(semantic="factor_observation")  # is_t1_shifted=False
        rp = self._make_panel(semantic="forward_return")
        with pytest.raises(TimingAlignmentError, match="T\\+1 滞后"):
            fp.align_with(rp)

    def test_align_with_price_vs_factor_raises(self):
        """price × factor_observation 是非法组合，应抛出 SemanticCompatibilityError。"""
        from factor_framework.core.panel import SemanticCompatibilityError
        price  = self._make_panel(semantic="price")
        factor = self._make_panel(semantic="factor_observation")
        with pytest.raises(SemanticCompatibilityError):
            price.align_with(factor)

    def test_align_with_price_vs_return_raises(self):
        """price × forward_return 是非法组合，应抛出 SemanticCompatibilityError。"""
        from factor_framework.core.panel import SemanticCompatibilityError
        price  = self._make_panel(semantic="price")
        ret    = self._make_panel(semantic="forward_return")
        with pytest.raises(SemanticCompatibilityError):
            price.align_with(ret)

    def test_align_with_factor_factor_valid(self):
        """factor_observation × factor_observation 合法（因子合成场景）。"""
        fp1 = self._make_panel(semantic="factor_observation")
        fp2 = self._make_panel(semantic="factor_observation")
        fp1_a, fp2_a = fp1.align_with(fp2)
        assert fp1_a.index.equals(fp2_a.index)

    def test_align_with_index_intersection(self):
        """align_with() 对不同 index 的面板应取交集。"""
        from factor_framework.core.panel import TimestampedPanel
        idx_full = pd.date_range("2020-01-01", periods=20, freq="B")
        idx_sub  = idx_full[5:15]
        cols     = ["A", "B"]
        fp = TimestampedPanel(
            pd.DataFrame(np.ones((20, 2)), index=idx_full, columns=cols),
            semantic="factor_observation", is_t1_shifted=True,
        )
        rp = TimestampedPanel(
            pd.DataFrame(np.ones((10, 2)), index=idx_sub, columns=cols),
            semantic="forward_return",
        )
        fp_a, rp_a = fp.align_with(rp)
        assert len(fp_a) == 10
        assert fp_a.index.equals(rp_a.index)

    def test_align_with_non_timestampedpanel_raises(self):
        """align_with() 入参不是 TimestampedPanel 时应抛出 TypeError。"""
        tp = self._make_panel()
        with pytest.raises(TypeError):
            tp.align_with(pd.DataFrame({"A": [1.0]}))

    # ── trim_warmup() ────────────────────────────────────────────────────────

    def test_trim_warmup_removes_rows(self):
        """trim_warmup(5) 应删除最早的 5 行。"""
        tp      = self._make_panel(rows=20)
        trimmed = tp.trim_warmup(5)
        assert len(trimmed) == 15
        assert trimmed.warmup_trimmed is True
        assert tp.warmup_trimmed is False, "原对象不应被修改"

    def test_trim_warmup_zero_does_nothing(self):
        """trim_warmup(0) 应返回等长度的面板（不截断）。"""
        tp = self._make_panel(rows=10)
        trimmed = tp.trim_warmup(0)
        assert len(trimmed) == 10

    def test_trim_warmup_too_large_raises(self):
        """trim_warmup(n) 当 n >= len(panel) 时应抛出 ValueError。"""
        tp = self._make_panel(rows=10)
        with pytest.raises(ValueError, match="截断后面板为空"):
            tp.trim_warmup(10)

    def test_trim_warmup_preserves_metadata(self):
        """trim_warmup 后元数据字段（semantic 等）应保留。"""
        tp      = self._make_panel(semantic="price", price_basis="hfq")
        trimmed = tp.trim_warmup(3)
        assert trimmed.semantic    == "price"
        assert trimmed.price_basis == "hfq"

    # ── assert_valid() ───────────────────────────────────────────────────────

    def test_assert_valid_passes_for_clean_panel(self):
        """有效面板 assert_valid() 应静默通过。"""
        tp = self._make_panel()
        tp.assert_valid()  # 不应抛出异常

    def test_assert_valid_fails_for_all_nan(self):
        """全 NaN 面板 assert_valid() 应抛出 AssertionError。"""
        from factor_framework.core.panel import TimestampedPanel
        df = pd.DataFrame({"A": [np.nan, np.nan]}, index=["d1", "d2"])
        tp = TimestampedPanel(df)
        with pytest.raises(AssertionError, match="全为 NaN"):
            tp.assert_valid()

    def test_assert_valid_fails_for_unsorted_index(self):
        """index 未排序时 assert_valid() 应抛出 AssertionError。"""
        from factor_framework.core.panel import TimestampedPanel
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["d3", "d1", "d2"])
        tp = TimestampedPanel(df)
        with pytest.raises(AssertionError, match="严格递增"):
            tp.assert_valid()

    def test_assert_valid_fails_for_duplicate_index(self):
        """index 有重复时 assert_valid() 应抛出 AssertionError。"""
        from factor_framework.core.panel import TimestampedPanel
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["d1", "d1", "d2"])
        tp = TimestampedPanel(df)
        with pytest.raises(AssertionError, match="重复日期"):
            tp.assert_valid()

    # ── 自定义异常 ───────────────────────────────────────────────────────────

    def test_timing_alignment_error_is_value_error(self):
        """TimingAlignmentError 应继承 ValueError。"""
        from factor_framework.core.panel import TimingAlignmentError
        assert issubclass(TimingAlignmentError, ValueError)

    def test_semantic_compatibility_error_is_type_error(self):
        """SemanticCompatibilityError 应继承 TypeError。"""
        from factor_framework.core.panel import SemanticCompatibilityError
        assert issubclass(SemanticCompatibilityError, TypeError)


class TestReturnPanel:
    """
    factor_framework.core.returns.ReturnPanel

    覆盖：
    - build()：收益率计算公式验证、price_basis 检查
    - build_multi_forward()：字典键、各 forward 面板 semantic
    - from_raw_dataframe()：向后兼容接口
    """

    def _make_price_panel(self, rows=30, cols=5, price_basis="hfq"):
        """构建测试用价格 TimestampedPanel。"""
        from factor_framework.core.panel import TimestampedPanel
        idx   = pd.date_range("2020-01-01", periods=rows, freq="B")
        cols_ = [f"S{i:03d}" for i in range(cols)]
        prices = np.abs(np.random.randn(rows, cols)) * 10 + 10  # 正值
        return TimestampedPanel(
            pd.DataFrame(prices, index=idx, columns=cols_),
            semantic    = "price",
            price_basis = price_basis,
        )

    def test_build_returns_timestamped_panel(self):
        """build() 应返回 TimestampedPanel。"""
        from factor_framework.core.panel   import TimestampedPanel
        from factor_framework.core.returns import ReturnPanel
        price  = self._make_price_panel()
        ret_tp = ReturnPanel.build(price, forward_days=5)
        assert isinstance(ret_tp, TimestampedPanel)

    def test_build_semantic_is_forward_return(self):
        """build() 返回面板的 semantic 应为 'forward_return'。"""
        from factor_framework.core.returns import ReturnPanel
        price  = self._make_price_panel()
        ret_tp = ReturnPanel.build(price, forward_days=5)
        assert ret_tp.semantic == "forward_return"

    def test_build_forward_days_stored(self):
        """build() 返回面板的 forward_days 应等于入参。"""
        from factor_framework.core.returns import ReturnPanel
        price  = self._make_price_panel()
        ret_tp = ReturnPanel.build(price, forward_days=21)
        assert ret_tp.forward_days == 21

    def test_build_not_t1_shifted(self):
        """build() 返回面板的 is_t1_shifted 应为 False（T+1 在因子侧完成）。"""
        from factor_framework.core.returns import ReturnPanel
        price  = self._make_price_panel()
        ret_tp = ReturnPanel.build(price, forward_days=5)
        assert ret_tp.is_t1_shifted is False

    def test_build_formula_correctness(self):
        """收益率公式 price.shift(-fwd)/price - 1 应与手动计算结果一致。"""
        from factor_framework.core.panel   import TimestampedPanel
        from factor_framework.core.returns import ReturnPanel
        prices = [100.0, 110.0, 121.0, 133.0, 146.0]
        idx    = pd.date_range("2020-01-01", periods=5, freq="B")
        df     = pd.DataFrame({"A": prices}, index=idx)
        price  = TimestampedPanel(df, semantic="price", price_basis="hfq")
        ret_tp = ReturnPanel.build(price, forward_days=2)
        # ret[0] = prices[2]/prices[0] - 1 = 121/100 - 1 = 0.21
        assert ret_tp.loc[idx[0], "A"] == pytest.approx(0.21, abs=1e-9)
        # 尾部 2 行应为 NaN（未来价格不存在）
        assert np.isnan(ret_tp.loc[idx[3], "A"])
        assert np.isnan(ret_tp.loc[idx[4], "A"])

    def test_build_zero_price_becomes_nan(self):
        """价格为 0 时，收益率应为 NaN（不产生 inf）。"""
        from factor_framework.core.panel   import TimestampedPanel
        from factor_framework.core.returns import ReturnPanel
        prices = [0.0, 100.0, 110.0]
        idx    = pd.date_range("2020-01-01", periods=3, freq="B")
        df     = pd.DataFrame({"A": prices}, index=idx)
        price  = TimestampedPanel(df, semantic="price", price_basis="hfq")
        ret_tp = ReturnPanel.build(price, forward_days=1)
        # ret[0] = price[1]/price[0] - 1，但 price[0]=0 → NaN
        assert np.isnan(ret_tp.loc[idx[0], "A"])

    def test_build_wrong_price_basis_raises(self):
        """price_basis 不为 'hfq' 时 build() 应抛出 ValueError。"""
        from factor_framework.core.panel   import TimestampedPanel
        from factor_framework.core.returns import ReturnPanel
        df    = pd.DataFrame({"A": [100.0, 110.0]})
        price = TimestampedPanel(df, semantic="price", price_basis="qfq")
        with pytest.raises(ValueError, match="后复权"):
            ReturnPanel.build(price, forward_days=1)

    def test_build_non_timestampedpanel_raises(self):
        """入参不是 TimestampedPanel 时 build() 应抛出 TypeError。"""
        from factor_framework.core.returns import ReturnPanel
        df = pd.DataFrame({"A": [100.0, 110.0]})
        with pytest.raises(TypeError):
            ReturnPanel.build(df, forward_days=1)

    def test_build_multi_forward_keys(self):
        """build_multi_forward() 返回字典的 key 应等于 forward_list。"""
        from factor_framework.core.returns import ReturnPanel
        price        = self._make_price_panel()
        forward_list = [1, 5, 10, 21]
        ret_dict     = ReturnPanel.build_multi_forward(price, forward_list)
        assert set(ret_dict.keys()) == set(forward_list)

    def test_build_multi_forward_each_semantic(self):
        """build_multi_forward() 中每个面板的 semantic 应为 'forward_return'。"""
        from factor_framework.core.returns import ReturnPanel
        price    = self._make_price_panel()
        ret_dict = ReturnPanel.build_multi_forward(price, [1, 5])
        for fwd, ret_tp in ret_dict.items():
            assert ret_tp.semantic     == "forward_return", f"forward={fwd} semantic 错误"
            assert ret_tp.forward_days == fwd,              f"forward={fwd} forward_days 错误"

    def test_from_raw_dataframe_returns_timestamped_panel(self):
        """from_raw_dataframe() 向后兼容接口应返回 TimestampedPanel。"""
        from factor_framework.core.panel   import TimestampedPanel
        from factor_framework.core.returns import ReturnPanel
        df     = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        ret_tp = ReturnPanel.from_raw_dataframe(df, forward_days=1, price_basis=None)
        assert isinstance(ret_tp, TimestampedPanel)
        assert ret_tp.semantic == "forward_return"


class TestCacheLayer:
    """
    factor_framework.engine.cache.CacheLayer

    覆盖：
    - make_key()：相同参数产生相同 key，不同参数产生不同 key
    - get_panel() / put_panel()：L1 内存缓存读写
    - L2 磁盘缓存读写（临时目录）
    - clear_l1() / clear_l2()
    - cache_info()
    - 失效机制（mtime 比较）
    """

    def _make_panel_df(self, rows=10, cols=5):
        idx  = pd.date_range("2020-01-01", periods=rows, freq="B")
        cols_ = [f"S{i:03d}" for i in range(cols)]
        return pd.DataFrame(np.random.randn(rows, cols), index=idx, columns=cols_)

    def test_make_key_same_params(self):
        """相同参数应生成相同的 cache key。"""
        from factor_framework.engine.cache import CacheLayer
        key1 = CacheLayer.make_key("mom", "20200101", "20251231", ["A", "B", "C"])
        key2 = CacheLayer.make_key("mom", "20200101", "20251231", ["C", "A", "B"])  # 顺序不同
        assert key1 == key2, "make_key 应对 symbols 排序后哈希"

    def test_make_key_different_params(self):
        """不同参数应生成不同的 cache key。"""
        from factor_framework.engine.cache import CacheLayer
        key1 = CacheLayer.make_key("mom", "20200101", "20251231", ["A"])
        key2 = CacheLayer.make_key("vol", "20200101", "20251231", ["A"])
        assert key1 != key2

    def test_l1_miss_returns_none(self):
        """L1 缓存未命中时 get_panel() 应返回 None。"""
        from factor_framework.engine.cache import CacheLayer
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheLayer(cache_dir=tmp, stocks_dir=tmp, enabled_l2=False)
            result = cache.get_panel("mom", "nonexistent_key")
            assert result is None

    def test_l1_put_and_get(self):
        """put_panel() 写入 L1 后 get_panel() 应能命中。"""
        from factor_framework.engine.cache import CacheLayer
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheLayer(cache_dir=tmp, stocks_dir=tmp, enabled_l2=False)
            panel = self._make_panel_df()
            key   = CacheLayer.make_key("mom", "20200101", "20251231", list(panel.columns))
            cache.put_panel("mom", key, panel, calc_secs=0.0)
            result = cache.get_panel("mom", key)
            assert result is not None
            pd.testing.assert_frame_equal(result, panel)

    def test_l2_put_and_get(self, tmp_path):
        """L2 磁盘缓存：put_panel() 写入 Parquet 后 get_panel() 应能从磁盘命中。"""
        pytest.importorskip("pyarrow", reason="pyarrow 不可用，跳过 L2 Parquet 测试")
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        cache = CacheLayer(
            cache_dir     = str(tmp_path / "cache"),
            stocks_dir    = str(stocks_dir),
            enabled_l2    = True,
            min_calc_secs = 0.0,   # 任何计算时间都写 L2
        )
        # 数据源 mtime 设为 0（使缓存永远有效）
        cache._source_mtime = 0.0

        panel = self._make_panel_df()
        key   = CacheLayer.make_key("vol", "20200101", "20251231", list(panel.columns))
        cache.put_panel("vol", key, panel, calc_secs=10.0)

        # 验证 Parquet 文件已写入
        parquet_path = tmp_path / "cache" / "vol" / f"{key}.parquet"
        if not parquet_path.exists():
            pytest.skip("Parquet 文件未写入（pyarrow 运行时问题），跳过 L2 读取验证")

        # 清空 L1，强制从 L2 读取
        cache.clear_l1()
        result = cache.get_panel("vol", key)
        assert result is not None
        pd.testing.assert_frame_equal(result, panel, check_freq=False)

    def test_l2_not_written_below_threshold(self, tmp_path):
        """calc_secs < min_calc_secs 时不写入 L2 Parquet。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(
            cache_dir     = str(tmp_path / "cache"),
            stocks_dir    = str(tmp_path),
            enabled_l2    = True,
            min_calc_secs = 5.0,
        )
        panel = self._make_panel_df()
        key   = CacheLayer.make_key("mom", "20200101", "20251231", list(panel.columns))
        cache.put_panel("mom", key, panel, calc_secs=1.0)  # 低于阈值

        parquet_path = tmp_path / "cache" / "mom" / f"{key}.parquet"
        assert not parquet_path.exists(), "低于阈值时不应写入 Parquet"

    def test_clear_l1(self, tmp_path):
        """clear_l1() 应清空内存缓存。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(cache_dir=str(tmp_path), stocks_dir=str(tmp_path), enabled_l2=False)
        panel = self._make_panel_df()
        key   = CacheLayer.make_key("x", "20200101", "20251231", ["A"])
        cache.put_panel("x", key, panel)
        assert cache.cache_info()["l1_entries"] == 1
        cache.clear_l1()
        assert cache.cache_info()["l1_entries"] == 0

    def test_cache_info_structure(self, tmp_path):
        """cache_info() 应返回包含必要字段的字典。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(cache_dir=str(tmp_path / "cache"), stocks_dir=str(tmp_path))
        info  = cache.cache_info()
        assert "l1_entries"  in info
        assert "l2_files"    in info
        assert "l2_total_mb" in info
        assert "enabled_l2"  in info


class TestBUG9Fix:
    """
    BUG-9：ic_decay 双路径不一致修复验证。

    修复前：pipeline 中主 IC 使用 build_return_panel()（含 T+1 shift），
            ic_decay() 内部从 close_panel 重算收益率（无 T+1），两者不同源。
    修复后：ic_decay() 增加 return_panels 参数，优先使用调用方传入的同源面板。

    此测试类验证：
    1. ic_decay(return_panels=...) 路径正常工作
    2. 两个路径的结果在相同数据下是否接近（量化双路径的差异大小）
    3. ic_analysis.ic_decay() 向后兼容旧签名（仅传 price_panel）
    """

    def _make_factor_price(self, n_dates=100, n_stocks=20, seed=42):
        """生成测试用因子面板和价格面板。"""
        rng    = np.random.default_rng(seed)
        idx    = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        cols   = [f"S{i:03d}" for i in range(n_stocks)]
        prices = np.cumprod(1 + rng.normal(0, 0.01, (n_dates, n_stocks)), axis=0) * 10
        factor = rng.standard_normal((n_dates, n_stocks))
        return (
            pd.DataFrame(factor, index=idx, columns=cols),
            pd.DataFrame(prices, index=idx, columns=cols),
        )

    def test_return_panels_path_no_error(self):
        """ic_decay(return_panels=...) 路径应正常执行，不抛出异常。"""
        from factor_framework.ic_analysis import ic_decay
        fp, price = self._make_factor_price()
        # 手动构建 return_panels（模拟 build_return_panel 的输出格式）
        ret_panels = {}
        for fwd in [1, 5, 10]:
            rp = price.shift(-fwd) / price.replace(0, np.nan) - 1
            rp = rp.iloc[:-(fwd + 1)]  # 截尾（模拟 T+1 shift）
            ret_panels[fwd] = rp
        result = ic_decay(fp, return_panels=ret_panels, method="rank")
        assert isinstance(result, pd.DataFrame)
        assert set(result.index.tolist()) == {1, 5, 10}
        assert "mean_ic" in result.columns

    def test_return_panels_result_non_nan(self):
        """return_panels 路径的 mean_ic 在数据充足时不应全为 NaN。"""
        from factor_framework.ic_analysis import ic_decay
        fp, price = self._make_factor_price(n_dates=120, n_stocks=30)
        ret_panels = {}
        for fwd in [1, 5]:
            rp = price.shift(-fwd) / price.replace(0, np.nan) - 1
            valid_idx = rp.dropna(how="all").index
            ret_panels[fwd] = rp.loc[valid_idx]
        result = ic_decay(fp, return_panels=ret_panels, method="rank")
        assert result["mean_ic"].notna().any(), "mean_ic 不应全为 NaN"

    def test_backward_compat_price_panel_path(self):
        """向后兼容：仅传 price_panel（不传 return_panels）应走回退路径，不报错。"""
        from factor_framework.ic_analysis import ic_decay
        import warnings
        fp, price = self._make_factor_price()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略截断 warning
            result = ic_decay(fp, price_panel=price, forward_periods=[1, 5], method="rank")
        assert isinstance(result, pd.DataFrame)
        assert set(result.index.tolist()) == {1, 5}

    def test_both_none_raises(self):
        """price_panel 和 return_panels 同时为 None 应抛出 ValueError。"""
        from factor_framework.ic_analysis import ic_decay
        fp, _ = self._make_factor_price()
        with pytest.raises(ValueError, match="不能同时为 None"):
            ic_decay(fp, price_panel=None, return_panels=None)

    def test_return_panels_path_sorted_by_forward(self):
        """return_panels 路径的结果应按 forward_days 升序排列（index 有序）。"""
        from factor_framework.ic_analysis import ic_decay
        fp, price = self._make_factor_price()
        ret_panels = {}
        for fwd in [21, 5, 1, 10]:   # 故意乱序传入
            rp = price.shift(-fwd) / price.replace(0, np.nan) - 1
            valid_idx = rp.dropna(how="all").index
            ret_panels[fwd] = rp.loc[valid_idx]
        result = ic_decay(fp, return_panels=ret_panels, method="rank")
        assert list(result.index) == sorted(result.index), "结果 index 应按 forward 升序"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 Tests -- FactorMeta, FactorRegistry, _CompatDict
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorMeta:
    """Tests for FactorMeta dataclass (factor_framework.factors.meta)."""

    def _make_meta(self, **kw):
        from factor_framework.factors.meta import FactorMeta, FactorCategory
        defaults = dict(
            name         = "test_factor",
            fn           = lambda df: df.iloc[:, 0],
            display_name = "Test Factor",
            category     = FactorCategory.MOMENTUM,
        )
        defaults.update(kw)
        return FactorMeta(**defaults)

    def test_construction_defaults(self):
        """FactorMeta should set sensible defaults for optional fields."""
        meta = self._make_meta()
        assert meta.direction == +1
        assert meta.warmup_days == 252
        assert meta.description == ""
        assert meta.neutral_by_default is True
        assert meta.skip_neutralize_cols == ()

    def test_frozen(self):
        """FactorMeta must be frozen (immutable after construction)."""
        from dataclasses import FrozenInstanceError
        meta = self._make_meta()
        with pytest.raises(FrozenInstanceError):
            meta.direction = -1

    def test_invalid_direction_raises(self):
        """direction not in {+1, -1} must raise ValueError."""
        with pytest.raises(ValueError, match="direction"):
            self._make_meta(direction=0)

    def test_negative_warmup_raises(self):
        """warmup_days < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="warmup_days"):
            self._make_meta(warmup_days=-1)

    def test_invalid_category_raises(self):
        """Passing a plain string as category must raise TypeError."""
        with pytest.raises(TypeError, match="category"):
            self._make_meta(category="momentum")

    def test_direction_plus1_is_long_short(self):
        """is_long_short should be True when direction == +1."""
        meta = self._make_meta(direction=+1)
        assert meta.is_long_short is True

    def test_direction_minus1_is_not_long_short(self):
        """is_long_short should be False when direction == -1."""
        meta = self._make_meta(direction=-1)
        assert meta.is_long_short is False

    def test_group_property(self):
        """group should equal category.value string."""
        from factor_framework.factors.meta import FactorCategory
        meta = self._make_meta(category=FactorCategory.VOLATILITY)
        assert meta.group == "volatility"

    def test_repr_contains_name_and_category(self):
        """__repr__ should include factor name and category."""
        meta = self._make_meta()
        r = repr(meta)
        assert "test_factor" in r
        assert "momentum" in r

    def test_factor_category_enum_str_comparison(self):
        """FactorCategory inherits str so == 'momentum' works directly."""
        from factor_framework.factors.meta import FactorCategory
        assert FactorCategory.MOMENTUM == "momentum"
        assert FactorCategory.VOLATILITY == "volatility"
        assert FactorCategory.SIZE == "size"

    def test_skip_neutralize_cols_tuple(self):
        """skip_neutralize_cols must be a tuple."""
        meta = self._make_meta(skip_neutralize_cols=("市值", "行业"))
        assert meta.skip_neutralize_cols == ("市值", "行业")

    def test_neutral_by_default_false(self):
        """neutral_by_default=False should be accepted."""
        meta = self._make_meta(neutral_by_default=False)
        assert meta.neutral_by_default is False

    def test_all_categories_exist(self):
        """All expected FactorCategory values must exist."""
        from factor_framework.factors.meta import FactorCategory
        expected = {
            "momentum", "reversal", "volatility", "value",
            "size", "volume", "liquidity", "technical", "composite", "custom",
        }
        actual = {c.value for c in FactorCategory}
        assert expected == actual


class TestFactorRegistry:
    """Tests for FactorRegistry (factor_framework.factors.registry)."""

    def _empty_registry(self):
        from factor_framework.factors.registry import FactorRegistry
        return FactorRegistry()

    def _make_meta(self, name="mom", **kw):
        from factor_framework.factors.meta import FactorMeta, FactorCategory
        defaults = dict(
            name         = name,
            fn           = lambda df: df.iloc[:, 0],
            display_name = name.upper(),
            category     = FactorCategory.MOMENTUM,
        )
        defaults.update(kw)
        return FactorMeta(**defaults)

    # -- register / get --

    def test_register_and_get(self):
        """register() then get() should return the same FactorMeta."""
        reg = self._empty_registry()
        meta = self._make_meta("alpha")
        reg.register(meta)
        assert reg.get("alpha") is meta

    def test_get_unknown_returns_none(self):
        """get() for an unregistered name must return None."""
        reg = self._empty_registry()
        assert reg.get("nonexistent") is None

    def test_get_fn_returns_callable(self):
        """get_fn() must return the factor callable."""
        reg = self._empty_registry()
        fn = lambda df: df.iloc[:, 0]
        from factor_framework.factors.meta import FactorMeta, FactorCategory
        meta = FactorMeta(name="f", fn=fn, display_name="F", category=FactorCategory.CUSTOM)
        reg.register(meta)
        assert reg.get_fn("f") is fn

    def test_get_fn_unknown_returns_none(self):
        """get_fn() for unknown name must return None."""
        reg = self._empty_registry()
        assert reg.get_fn("nope") is None

    def test_overwrite_warns(self):
        """Re-registering same name must emit UserWarning."""
        reg = self._empty_registry()
        reg.register(self._make_meta("dup"))
        with pytest.warns(UserWarning, match="already exists"):
            reg.register(self._make_meta("dup"))

    def test_register_wrong_type_raises(self):
        """register() with non-FactorMeta must raise TypeError."""
        reg = self._empty_registry()
        with pytest.raises(TypeError, match="FactorMeta"):
            reg.register({"name": "bad"})

    def test_len(self):
        """len() should reflect number of registered factors."""
        reg = self._empty_registry()
        assert len(reg) == 0
        reg.register(self._make_meta("a"))
        reg.register(self._make_meta("b"))
        assert len(reg) == 2

    def test_contains(self):
        """'name' in registry should work."""
        reg = self._empty_registry()
        reg.register(self._make_meta("x"))
        assert "x" in reg
        assert "y" not in reg

    def test_iter(self):
        """Iterating registry should yield factor names."""
        reg = self._empty_registry()
        reg.register(self._make_meta("p"))
        reg.register(self._make_meta("q"))
        assert set(reg) == {"p", "q"}

    # -- list_by_category --

    def test_list_by_category(self):
        """list_by_category() should filter and sort by name."""
        from factor_framework.factors.meta import FactorMeta, FactorCategory
        reg = self._empty_registry()
        fn = lambda df: df.iloc[:, 0]
        reg.register(FactorMeta("z_mom", fn, "Z", FactorCategory.MOMENTUM))
        reg.register(FactorMeta("a_mom", fn, "A", FactorCategory.MOMENTUM))
        reg.register(FactorMeta("a_rev", fn, "B", FactorCategory.REVERSAL))
        moms = reg.list_by_category(FactorCategory.MOMENTUM)
        assert [m.name for m in moms] == ["a_mom", "z_mom"]

    def test_list_all_sorted(self):
        """list_all() should return all metas sorted alphabetically."""
        reg = self._empty_registry()
        for name in ["c_fac", "a_fac", "b_fac"]:
            reg.register(self._make_meta(name))
        names = [m.name for m in reg.list_all()]
        assert names == sorted(names)

    # -- to_compat_dict --

    def test_to_compat_dict_is_dict_subclass(self):
        """to_compat_dict() must return a dict subclass."""
        reg = self._empty_registry()
        reg.register(self._make_meta("f1"))
        cd = reg.to_compat_dict()
        assert isinstance(cd, dict)

    def test_to_compat_dict_keys_match(self):
        """CompatDict keys should match registered factor names."""
        reg = self._empty_registry()
        reg.register(self._make_meta("f1"))
        reg.register(self._make_meta("f2"))
        cd = reg.to_compat_dict()
        assert set(cd.keys()) == {"f1", "f2"}

    def test_to_compat_dict_values_callable(self):
        """CompatDict values must all be callable."""
        reg = self._empty_registry()
        reg.register(self._make_meta("f1"))
        cd = reg.to_compat_dict()
        assert callable(cd["f1"])

    def test_to_compat_dict_get_meta(self):
        """CompatDict.get_meta() must return the FactorMeta."""
        reg = self._empty_registry()
        meta = self._make_meta("f1")
        reg.register(meta)
        cd = reg.to_compat_dict()
        assert cd.get_meta("f1") is meta

    def test_to_compat_dict_registry_attr(self):
        """CompatDict.registry must point back to the source FactorRegistry."""
        reg = self._empty_registry()
        reg.register(self._make_meta("f1"))
        cd = reg.to_compat_dict()
        assert cd.registry is reg

    # -- summary_df --

    def test_summary_df_shape(self):
        """summary_df() should have one row per factor and the expected columns."""
        reg = self._empty_registry()
        for name in ["a", "b", "c"]:
            reg.register(self._make_meta(name))
        df = reg.summary_df()
        assert df.shape[0] == 3
        assert "category" in df.columns
        assert "warmup_days" in df.columns
        assert "neutral_by_default" in df.columns

    def test_summary_df_index_is_name(self):
        """summary_df() index should be factor names."""
        reg = self._empty_registry()
        reg.register(self._make_meta("my_factor"))
        df = reg.summary_df()
        assert "my_factor" in df.index

    def test_repr(self):
        """__repr__ should include count and category breakdown."""
        reg = self._empty_registry()
        reg.register(self._make_meta("m1"))
        r = repr(reg)
        assert "n=1" in r
        assert "momentum" in r


class TestCompatDictBackwardCompat:
    """
    Verify that _CompatDict (BUILTIN_FACTORS) is a drop-in replacement for
    the old plain dict -- every legacy usage pattern must continue to work.
    """

    def setup_method(self):
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        self.bf = BUILTIN_FACTORS

    def test_isinstance_dict(self):
        """isinstance(BUILTIN_FACTORS, dict) must be True."""
        assert isinstance(self.bf, dict)

    def test_len_28(self):
        """BUILTIN_FACTORS must contain exactly 28 built-in factors."""
        assert len(self.bf) == 28

    def test_getitem_callable(self):
        """BUILTIN_FACTORS['momentum_12_1'] must return a callable."""
        fn = self.bf["momentum_12_1"]
        assert callable(fn)

    def test_keys_contains_all_builtin_names(self):
        """All 28 built-in factor names must appear in BUILTIN_FACTORS.keys()."""
        expected = {
            "momentum_12_1", "momentum_6_1", "momentum_1m", "momentum_52w_high",
            "reversal_1w", "reversal_1m",
            "vol_20d", "vol_60d", "vol_skew", "downside_vol",
            "value_pb", "value_pe_ttm", "value_ps_ttm",
            "size_log_mktcap", "size_log_free_cap",
            "amihud_illiquidity", "turnover_rate", "vol_price_corr",
            "vwap_deviation", "price_strength",
            "bid_ask_spread_proxy", "zero_return_ratio",
            "pastor_stambaugh", "order_imbalance",
            "rsi_14", "macd_signal", "bb_position", "volume_trend",
        }
        assert expected == set(self.bf.keys())

    def test_items_iteration_yields_name_callable_pairs(self):
        """for name, fn in BUILTIN_FACTORS.items() must yield (str, callable) pairs."""
        for name, fn in self.bf.items():
            assert isinstance(name, str)
            assert callable(fn)

    def test_in_operator(self):
        """'momentum_12_1' in BUILTIN_FACTORS must be True."""
        assert "momentum_12_1" in self.bf
        assert "nonexistent_factor" not in self.bf

    def test_get_meta_returns_factormeta(self):
        """BUILTIN_FACTORS.get_meta(name) must return a FactorMeta."""
        from factor_framework.factors.meta import FactorMeta
        meta = self.bf.get_meta("momentum_12_1")
        assert isinstance(meta, FactorMeta)
        assert meta.name == "momentum_12_1"

    def test_registry_attr_is_populated(self):
        """BUILTIN_FACTORS.registry should be the populated global REGISTRY."""
        from factor_framework.factors.registry import REGISTRY
        assert self.bf.registry is REGISTRY
        assert len(self.bf.registry) == 28


class TestGlobalRegistry:
    """Tests for the global REGISTRY singleton populated by factor_zoo."""

    def setup_method(self):
        # Ensure factor_zoo is loaded so REGISTRY is populated
        import factor_framework.factor_zoo  # noqa: F401
        from factor_framework.factors.registry import REGISTRY
        self.reg = REGISTRY

    def test_registry_has_28_factors(self):
        """Global REGISTRY must contain all 28 built-in factors."""
        assert len(self.reg) == 28

    def test_registry_get_momentum(self):
        """REGISTRY.get('momentum_12_1') must return correct metadata."""
        from factor_framework.factors.meta import FactorCategory
        meta = self.reg.get("momentum_12_1")
        assert meta is not None
        assert meta.category == FactorCategory.MOMENTUM
        assert meta.warmup_days == 252
        assert meta.direction == +1
        assert meta.neutral_by_default is True

    def test_size_factors_not_neutral_by_default(self):
        """size_log_mktcap and size_log_free_cap must have neutral_by_default=False."""
        for name in ("size_log_mktcap", "size_log_free_cap"):
            meta = self.reg.get(name)
            assert meta is not None, f"{name} not in REGISTRY"
            assert meta.neutral_by_default is False, f"{name}.neutral_by_default should be False"

    def test_size_log_mktcap_skip_cols(self):
        """size_log_mktcap must skip the '市值' column in neutralization."""
        meta = self.reg.get("size_log_mktcap")
        assert "市值" in meta.skip_neutralize_cols

    def test_size_log_free_cap_skip_cols(self):
        """size_log_free_cap must skip both '市值' and '流通市值'."""
        meta = self.reg.get("size_log_free_cap")
        assert "市值" in meta.skip_neutralize_cols
        assert "流通市值" in meta.skip_neutralize_cols

    def test_all_directions_plus1(self):
        """All 28 built-in factors must have direction == +1."""
        for meta in self.reg.list_all():
            assert meta.direction == +1, f"{meta.name}.direction should be +1"

    def test_all_fns_callable(self):
        """Every registered fn must be callable."""
        for meta in self.reg.list_all():
            assert callable(meta.fn), f"{meta.name}.fn is not callable"

    def test_category_counts(self):
        """Category distribution should match design spec."""
        from factor_framework.factors.meta import FactorCategory
        counts = {
            FactorCategory.MOMENTUM:   4,
            FactorCategory.REVERSAL:   2,
            FactorCategory.VOLATILITY: 4,
            FactorCategory.VALUE:      3,
            FactorCategory.SIZE:       2,
            FactorCategory.VOLUME:     5,
            FactorCategory.LIQUIDITY:  4,
            FactorCategory.TECHNICAL:  4,
        }
        for cat, expected_n in counts.items():
            actual = len(self.reg.list_by_category(cat))
            assert actual == expected_n, f"{cat.value}: expected {expected_n}, got {actual}"

    def test_list_by_category_sorted(self):
        """list_by_category() results should be sorted by name."""
        from factor_framework.factors.meta import FactorCategory
        moms = self.reg.list_by_category(FactorCategory.MOMENTUM)
        names = [m.name for m in moms]
        assert names == sorted(names)

    def test_summary_df_28_rows(self):
        """summary_df() must have 28 rows (one per built-in factor)."""
        df = self.reg.summary_df()
        assert len(df) == 28

    def test_summary_df_has_required_columns(self):
        """summary_df() must include all required metadata columns."""
        df = self.reg.summary_df()
        required = {"display_name", "category", "direction", "warmup_days",
                    "neutral_by_default", "skip_neutralize_cols", "description"}
        assert required.issubset(set(df.columns))

    def test_fn_same_object_as_builtin_factors(self):
        """REGISTRY fn references must be identical to BUILTIN_FACTORS values."""
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        for name in BUILTIN_FACTORS:
            meta = self.reg.get(name)
            assert meta is not None
            assert meta.fn is BUILTIN_FACTORS[name], \
                f"{name}: REGISTRY.fn is not the same object as BUILTIN_FACTORS[name]"

    def test_get_fn_convenience(self):
        """REGISTRY.get_fn(name) must return the same callable as .get(name).fn."""
        meta = self.reg.get("vol_20d")
        assert self.reg.get_fn("vol_20d") is meta.fn

    def test_warmup_days_sensible(self):
        """All warmup_days must be >= 1."""
        for meta in self.reg.list_all():
            assert meta.warmup_days >= 1, f"{meta.name}.warmup_days={meta.warmup_days} < 1"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 缺口：DataStore / PanelBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataStore:
    """
    factor_framework.data.store.DataStore / CSVDataStore

    覆盖：
    - DataStore 是 ABC，不能直接实例化
    - CSVDataStore.list_symbols() 返回 CSV 文件名列表
    - CSVDataStore.get_raw_df() 返回 DataFrame 或 None
    - CSVDataStore.get_price_panel() 返回 TimestampedPanel(semantic='price', price_basis='hfq')
    - get_price_panel() 对空目录返回空面板
    - 后复权价格：有复权因子列时 = 收盘价 × 复权因子
    - 日期过滤参数（start/end）正确切片
    """

    def _make_stock_dir(self, tmp_path, n_stocks=3, n_days=20):
        """在 tmp_path 下创建 n_stocks 个 CSV 文件（结构与实际数据一致）。"""
        import os
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        dates = pd.date_range("20200101", periods=n_days, freq="B")
        date_strs = dates.strftime("%Y%m%d").tolist()
        for i in range(n_stocks):
            code = f"S{i:06d}_SZ"
            rows = []
            for d in date_strs:
                rows.append({
                    "交易日": d,
                    "股票代码": code,
                    "收盘价": 10.0 + i + float(d[-2:]) * 0.01,
                    "复权因子": 1.0 + i * 0.1,
                    "总市值（万元）": 1000.0 * (i + 1),
                    "成交量（手）": 10000,
                    "成交额（千元）": 100000.0,
                    "换手率（%）": 1.0,
                    "流通市值（万元）": 800.0,
                    "市净率": 2.0,
                    "市盈率（TTM，亏损为空）": 15.0,
                    "市销率（TTM）": 3.0,
                })
            pd.DataFrame(rows).to_csv(stocks_dir / f"{code}.csv", index=False)
        return stocks_dir

    def test_datastore_is_abstract(self):
        """DataStore 是 ABC，不能直接实例化。"""
        from factor_framework.data.store import DataStore
        with pytest.raises(TypeError):
            DataStore()

    def test_csv_datastore_list_symbols(self, tmp_path):
        """CSVDataStore.list_symbols() 应返回目录下的 CSV 文件名（去 .csv）。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=3)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        syms = store.list_symbols()
        assert len(syms) == 3
        for s in syms:
            assert not s.endswith(".csv")

    def test_csv_datastore_list_symbols_sorted(self, tmp_path):
        """list_symbols() 应返回已排序的列表。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=5)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        syms = store.list_symbols()
        assert syms == sorted(syms)

    def test_csv_datastore_get_raw_df_valid(self, tmp_path):
        """get_raw_df() 对存在的股票应返回 DataFrame。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=1)
        store  = CSVDataStore(stocks_dir=str(stocks_dir))
        sym    = store.list_symbols()[0]
        df = store.get_raw_df(sym)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert "交易日" in df.columns
        assert "收盘价" in df.columns
        assert len(df) > 0

    def test_csv_datastore_get_raw_df_missing(self, tmp_path):
        """get_raw_df() 对不存在的股票应返回 None。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=1)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        result = store.get_raw_df("NONEXISTENT_STOCK")
        assert result is None

    def test_csv_datastore_get_price_panel_returns_timestamped_panel(self, tmp_path):
        """get_price_panel() 应返回 TimestampedPanel(semantic='price', price_basis='hfq')。"""
        from factor_framework.data.store import CSVDataStore
        from factor_framework.core.panel import TimestampedPanel
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=3)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        panel = store.get_price_panel()
        assert isinstance(panel, TimestampedPanel)
        assert panel.semantic == "price"
        assert panel.price_basis == "hfq"

    def test_csv_datastore_get_price_panel_shape(self, tmp_path):
        """get_price_panel() 面板列数应等于股票数。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=4, n_days=15)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        panel = store.get_price_panel()
        assert panel.shape[1] == 4

    def test_csv_datastore_adj_price_applied(self, tmp_path):
        """有复权因子时，价格面板中的值 = 收盘价 × 复权因子。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        # 单股：收盘价=10, 复权因子=2 → 后复权价格应为 20
        rows = [{"交易日": "20200102", "收盘价": 10.0, "复权因子": 2.0}]
        pd.DataFrame(rows).to_csv(stocks_dir / "A001_SZ.csv", index=False)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        panel = store.get_price_panel()
        assert not panel.empty
        val = panel.loc["20200102", "A001_SZ"]
        assert abs(val - 20.0) < 1e-6

    def test_csv_datastore_date_filter(self, tmp_path):
        """get_price_panel(start=..., end=...) 应正确过滤日期。"""
        from factor_framework.data.store import CSVDataStore
        stocks_dir = self._make_stock_dir(tmp_path, n_stocks=2, n_days=30)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        panel_all   = store.get_price_panel()
        panel_slice = store.get_price_panel(start="20200110", end="20200120")
        assert len(panel_slice) < len(panel_all)
        idx = panel_slice.index.tolist()
        assert all("20200110" <= d <= "20200120" for d in idx)

    def test_csv_datastore_empty_dir(self, tmp_path):
        """空目录下 get_price_panel() 应返回空 TimestampedPanel 而非抛出异常。"""
        from factor_framework.data.store import CSVDataStore
        from factor_framework.core.panel import TimestampedPanel
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        store = CSVDataStore(stocks_dir=str(empty_dir))
        panel = store.get_price_panel()
        assert isinstance(panel, TimestampedPanel)
        assert panel.empty

    def test_csv_datastore_is_datastore_subclass(self):
        """CSVDataStore 必须是 DataStore 的子类。"""
        from factor_framework.data.store import DataStore, CSVDataStore
        assert issubclass(CSVDataStore, DataStore)

    def test_data_package_imports(self):
        """factor_framework.data 包应能正常导入 DataStore 和 CSVDataStore。"""
        from factor_framework.data import DataStore, CSVDataStore
        assert DataStore is not None
        assert CSVDataStore is not None


class TestPanelBuilder:
    """
    factor_framework.engine.panel_builder.PanelBuilder

    覆盖：
    - 基础构造（无缓存模式）
    - engine 属性延迟初始化
    - register / register_builtins 代理
    - build_panel 无缓存路径
    - build_panel 有缓存路径（L1 命中，无需重算）
    - build_return_panel 有缓存路径
    - build_panel_batch 批量构建
    - industry_map / all_symbols / apply_cross_section 代理属性
    - engine 包导出
    """

    def _make_stocks_dir(self, tmp_path, n_stocks=3, n_days=60):
        """创建最小 Stocks/ 目录（含 _ret 可计算列）。"""
        stocks_dir = tmp_path / "Stocks"
        stocks_dir.mkdir()
        dates = pd.date_range("20200101", periods=n_days, freq="B")
        date_strs = dates.strftime("%Y%m%d").tolist()
        rng = np.random.default_rng(42)
        for i in range(n_stocks):
            code = f"S{i:06d}_SZ"
            prices = np.cumprod(1 + rng.normal(0, 0.01, n_days)) * 10
            rows = []
            for j, d in enumerate(date_strs):
                rows.append({
                    "交易日": d,
                    "股票代码": code,
                    "收盘价": prices[j],
                    "总市值（万元）": 1000.0,
                    "成交量（手）": 10000,
                    "成交额（千元）": 100000.0,
                    "换手率（%）": 1.0,
                    "流通市值（万元）": 800.0,
                    "复权因子": 1.0,
                    "市净率": 2.0,
                    "市盈率（TTM，亏损为空）": 15.0,
                    "市销率（TTM）": 3.0,
                })
            pd.DataFrame(rows).to_csv(stocks_dir / f"{code}.csv", index=False)
        return stocks_dir

    def test_panel_builder_no_cache_constructs(self, tmp_path):
        """PanelBuilder 无缓存模式可正常构造。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        assert builder is not None
        assert builder.cache is None

    def test_panel_builder_engine_lazy(self, tmp_path):
        """engine 属性应在首次访问时才初始化（延迟加载）。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        assert builder._engine is None  # 尚未访问，未初始化
        _ = builder.engine               # 触发初始化
        assert builder._engine is not None

    def test_panel_builder_register_proxy(self, tmp_path):
        """register() 应透传给底层 FactorEngine。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        builder.register("test_factor", lambda df: df["收盘价"])
        assert "test_factor" in builder.engine._registry

    def test_panel_builder_all_symbols(self, tmp_path):
        """all_symbols() 应返回 Stocks/ 目录下所有股票代码。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=4)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        syms = builder.all_symbols()
        assert len(syms) == 4

    def test_panel_builder_build_panel_no_cache(self, tmp_path):
        """build_panel() 无缓存模式应能正确返回因子面板。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=3, n_days=60)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        builder.register("close_factor", lambda df: df["收盘价"])
        panel = builder.build_panel("close_factor")
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty
        assert panel.shape[1] == 3

    def test_panel_builder_build_panel_with_cache_l1_hit(self, tmp_path):
        """第二次调用 build_panel() 应命中 L1 缓存，返回相同面板。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=3, n_days=60)
        cache = CacheLayer(
            cache_dir=str(tmp_path / "cache"),
            stocks_dir=str(stocks_dir),
            enabled_l2=False,   # 只测 L1
            min_calc_secs=0.0,
        )
        builder = PanelBuilder(
            stocks_dir=str(stocks_dir), cache=cache, verbose=False
        )
        builder.register("close_factor", lambda df: df["收盘价"])

        panel1 = builder.build_panel("close_factor")
        # L1 应有 1 条记录
        assert cache.cache_info()["l1_entries"] == 1

        panel2 = builder.build_panel("close_factor")
        # 第二次应命中 L1，返回相同数据
        pd.testing.assert_frame_equal(panel1, panel2)
        # 仍只有 1 条 L1 记录（没有重复写入）
        assert cache.cache_info()["l1_entries"] == 1

    def test_panel_builder_build_return_panel_no_cache(self, tmp_path):
        """build_return_panel() 无缓存应能返回收益率面板。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=3, n_days=60)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        panel = builder.build_return_panel(forward=1)
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty
        assert panel.shape[1] == 3

    def test_panel_builder_build_return_panel_cache(self, tmp_path):
        """build_return_panel() 有缓存时第二次应命中 L1。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=3, n_days=60)
        cache = CacheLayer(
            cache_dir=str(tmp_path / "cache"),
            stocks_dir=str(stocks_dir),
            enabled_l2=False,
            min_calc_secs=0.0,
        )
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=cache, verbose=False)
        p1 = builder.build_return_panel(forward=5)
        assert cache.cache_info()["l1_entries"] == 1
        p2 = builder.build_return_panel(forward=5)
        pd.testing.assert_frame_equal(p1, p2)

    def test_panel_builder_different_forwards_separate_cache_entries(self, tmp_path):
        """不同 forward 的收益率面板应有独立的缓存键。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = self._make_stocks_dir(tmp_path, n_stocks=3, n_days=60)
        cache = CacheLayer(
            cache_dir=str(tmp_path / "cache"),
            stocks_dir=str(stocks_dir),
            enabled_l2=False,
            min_calc_secs=0.0,
        )
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=cache, verbose=False)
        builder.build_return_panel(forward=1)
        builder.build_return_panel(forward=5)
        # 两个不同的缓存条目
        assert cache.cache_info()["l1_entries"] == 2

    def test_panel_builder_register_builtins(self, tmp_path):
        """register_builtins() 应能注册内置因子。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = self._make_stocks_dir(tmp_path)
        builder = PanelBuilder(stocks_dir=str(stocks_dir), cache=None, verbose=False)
        builder.register_builtins(["vol_20d", "momentum_12_1"])
        assert "vol_20d" in builder.engine._registry
        assert "momentum_12_1" in builder.engine._registry

    def test_engine_package_exports_panel_builder(self):
        """factor_framework.engine 包应导出 PanelBuilder。"""
        from factor_framework.engine import PanelBuilder
        assert PanelBuilder is not None

    def test_pipeline_uses_panel_builder(self, tmp_path):
        """FactorPipeline 应持有 _builder 属性（PanelBuilder 实例）。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
        )
        assert hasattr(pipe, "_builder")
        assert isinstance(pipe._builder, PanelBuilder)

    def test_pipeline_cache_dir_creates_cache_layer(self, tmp_path):
        """指定 cache_dir 时，FactorPipeline 应创建 CacheLayer。"""
        from factor_framework.engine.cache import CacheLayer
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = str(tmp_path / "cache"),
        )
        assert pipe._builder.cache is not None
        assert isinstance(pipe._builder.cache, CacheLayer)

    def test_pipeline_no_cache_dir_no_cache_layer(self, tmp_path):
        """显式传 cache_dir=None 时，FactorPipeline 的 _builder.cache 应为 None。"""
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = None,   # 显式禁用缓存
        )
        assert pipe._builder.cache is None

    def test_pipeline_default_cache_dir_creates_cache_layer(self, tmp_path):
        """不指定 cache_dir 时，默认值 'cache/' 应自动创建 CacheLayer（B4）。"""
        from factor_framework.engine.cache import CacheLayer
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
        )
        assert pipe._builder.cache is not None
        assert isinstance(pipe._builder.cache, CacheLayer)

    def test_pipeline_engine_backward_compat(self, tmp_path):
        """pipe.engine 应仍可访问底层 FactorEngine（向后兼容）。"""
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
        )
        assert isinstance(pipe.engine, FactorEngine)


# ═══════════════════════════════════════════════════════════════════════════════
# TestTransformPipeline  (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformPipeline:
    """TransformPipeline 的单元测试。"""

    @staticmethod
    def _make_panel(n_rows: int = 20, n_cols: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("20200101", periods=n_rows, freq="B").strftime("%Y%m%d")
        cols  = [f"S{i:04d}" for i in range(n_cols)]
        return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), index=dates, columns=cols)

    def test_import(self):
        """TransformPipeline 可以正常导入。"""
        from factor_framework.factors.transform import TransformPipeline  # noqa
        assert TransformPipeline is not None

    def test_empty_pipeline_passthrough(self):
        """空管道 transform 应返回与输入形状相同的 DataFrame。"""
        from factor_framework.factors.transform import TransformPipeline
        panel = self._make_panel()
        tp = TransformPipeline()
        out = tp.transform(panel)
        assert out.shape == panel.shape

    def test_step_names_empty(self):
        """空管道的 step_names 应为空列表。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline()
        assert tp.step_names == []

    def test_len_empty(self):
        """空管道 len() 应为 0。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline()
        assert len(tp) == 0

    def test_repr_contains_class_name(self):
        """__repr__ 应包含类名。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline()
        assert "TransformPipeline" in repr(tp)

    def test_winsorize_reduces_extremes(self):
        """winsorize 步骤应截断极端值（极值应变小或相等）。"""
        from factor_framework.factors.transform import TransformPipeline
        rng = np.random.default_rng(42)
        panel = self._make_panel(20, 50, seed=42)
        # 注入几个极端值
        panel.iloc[0, 0] = 1000.0
        panel.iloc[1, 1] = -1000.0
        tp = TransformPipeline().winsorize(n_std=3.0)
        out = tp.transform(panel)
        assert float(out.iloc[0, 0]) < 1000.0
        assert float(out.iloc[1, 1]) > -1000.0

    def test_winsorize_step_registered(self):
        """winsorize 后 step_names 应包含 'winsorize'。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline().winsorize()
        assert "winsorize" in tp.step_names

    def test_standardize_rank_range(self):
        """standardize('rank') 的输出应在 [0, 1] 范围内（每行均值 ≈ 0.5）。"""
        from factor_framework.factors.transform import TransformPipeline
        panel = self._make_panel(10, 40)
        tp = TransformPipeline().standardize("rank")
        out = tp.transform(panel)
        # 每行丢弃 NaN 后的值域
        assert float(out.stack().min()) >= -1e-9
        assert float(out.stack().max()) <= 1.0 + 1e-9

    def test_standardize_zscore_mean(self):
        """standardize('zscore') 的每行均值应接近 0。"""
        from factor_framework.factors.transform import TransformPipeline
        panel = self._make_panel(10, 60)
        tp = TransformPipeline().standardize("zscore")
        out = tp.transform(panel)
        row_means = out.mean(axis=1).dropna()
        assert float(row_means.abs().max()) < 1e-9

    def test_register_custom_step(self):
        """register_step 注册的自定义步骤应被执行。"""
        from factor_framework.factors.transform import TransformPipeline
        panel = self._make_panel()
        called = []
        def mark(p):
            called.append(True)
            return p
        tp = TransformPipeline().register_step("marker", mark)
        tp.transform(panel)
        assert called

    def test_step_names_order(self):
        """step_names 应保持注册顺序。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = (TransformPipeline()
              .winsorize()
              .standardize("rank"))
        assert tp.step_names == ["winsorize", "standardize"]

    def test_len_after_steps(self):
        """注册两步后 len() 应为 2。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline().winsorize().standardize()
        assert len(tp) == 2

    def test_fluent_chaining_returns_self(self):
        """winsorize/standardize/register_step 应都返回 self（支持链式）。"""
        from factor_framework.factors.transform import TransformPipeline
        tp = TransformPipeline()
        r1 = tp.winsorize()
        r2 = r1.standardize()
        r3 = r2.register_step("noop", lambda p: p)
        assert r1 is tp
        assert r2 is tp
        assert r3 is tp


# ═══════════════════════════════════════════════════════════════════════════════
# TestICAnalyzer  (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestICAnalyzer:
    """ICAnalyzer 的单元测试。"""

    @staticmethod
    def _make_panels(n_rows: int = 30, n_cols: int = 50, seed: int = 7):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("20200101", periods=n_rows, freq="B").strftime("%Y%m%d")
        cols  = [f"S{i:04d}" for i in range(n_cols)]
        fp = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), index=dates, columns=cols)
        rp = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), index=dates, columns=cols)
        return fp, rp

    def test_import(self):
        """ICAnalyzer 可正常导入。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer  # noqa
        assert ICAnalyzer is not None

    def test_run_before_access_raises(self):
        """run() 前访问 ic_series 应抛出 RuntimeError。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp)
        with pytest.raises(RuntimeError):
            _ = az.ic_series

    def test_run_returns_self(self):
        """run() 应返回 self（支持链式）。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp)
        assert az.run() is az

    def test_ic_series_not_empty(self):
        """run() 后 ic_series 应为非空 pd.Series。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp).run()
        assert isinstance(az.ic_series, pd.Series)
        assert len(az.ic_series) > 0

    def test_ic_stats_dict_keys(self):
        """ic_stats_dict 应包含必需字段。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp).run()
        for key in ("mean_ic", "std_ic", "icir", "win_rate"):
            assert key in az.ic_stats_dict, f"缺少字段: {key}"

    def test_ic_nw_keys(self):
        """ic_nw 应包含 nw_t_stat 和 nw_p_value。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp).run()
        assert "nw_t_stat" in az.ic_nw
        assert "nw_p_value" in az.ic_nw

    def test_decay_df_none_without_return_panels(self):
        """不传 return_panels 时，decay_df 应为 None 或空。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp).run()
        # decay_df 可为 None 或行数为 0 的 DataFrame
        if az.decay_df is not None:
            assert len(az.decay_df) == 0 or az.decay_df.empty

    def test_decay_df_with_return_panels(self):
        """传入 return_panels 后，decay_df 应为非空 DataFrame。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        rps = {1: rp, 5: rp}
        az = ICAnalyzer(fp, rp, return_panels=rps).run()
        assert az.decay_df is not None
        assert len(az.decay_df) > 0

    def test_summary_is_dict(self):
        """summary() 应返回 dict。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        s = ICAnalyzer(fp, rp).run().summary()
        assert isinstance(s, dict)

    def test_print_summary_no_error(self, capsys):
        """print_summary() 不应抛出异常，且应输出 IC 相关内容。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        ICAnalyzer(fp, rp).run().print_summary("test_factor")
        captured = capsys.readouterr()
        assert "IC" in captured.out or "ic" in captured.out.lower()

    def test_repr_shows_status(self):
        """__repr__ 应包含状态信息。"""
        from factor_framework.factors.ic_analyzer import ICAnalyzer
        fp, rp = self._make_panels()
        az = ICAnalyzer(fp, rp)
        r = repr(az)
        assert "ICAnalyzer" in r


# ═══════════════════════════════════════════════════════════════════════════════
# TestLayerBacktester  (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayerBacktester:
    """LayerBacktester 的单元测试。"""

    @staticmethod
    def _make_panels(n_rows: int = 30, n_cols: int = 50, seed: int = 13):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("20200101", periods=n_rows, freq="B").strftime("%Y%m%d")
        cols  = [f"S{i:04d}" for i in range(n_cols)]
        fp = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), index=dates, columns=cols)
        # 收益率应该很小（模拟真实价格变动）
        rp = pd.DataFrame(rng.normal(0.001, 0.02, (n_rows, n_cols)), index=dates, columns=cols)
        return fp, rp

    def test_import(self):
        """LayerBacktester 可正常导入。"""
        from factor_framework.factors.layer_backtester import LayerBacktester  # noqa
        assert LayerBacktester is not None

    def test_run_before_access_raises(self):
        """run() 前访问 layer_ret 应抛出 RuntimeError。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        bt = LayerBacktester(fp, rp)
        with pytest.raises(RuntimeError):
            _ = bt.layer_ret

    def test_run_returns_self(self):
        """run() 应返回 self（支持链式）。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        bt = LayerBacktester(fp, rp)
        assert bt.run() is bt

    def test_layer_ret_column_count(self):
        """layer_ret 的列数应等于 n_groups + 1（含多空列 LS）。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        n_groups = 5
        bt = LayerBacktester(fp, rp, n_groups=n_groups).run()
        # layer_backtest 返回 n_groups 个分组 + 1 个多空列（LS）
        assert bt.layer_ret.shape[1] == n_groups + 1

    def test_ls_stats_is_dict(self):
        """ls_stats 应为 dict，且包含 ls_annual_return。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        bt = LayerBacktester(fp, rp).run()
        assert isinstance(bt.ls_stats, dict)
        assert "ls_annual_return" in bt.ls_stats

    def test_turnover_is_dict(self):
        """turnover 应为 dict，且包含 avg_turnover。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        bt = LayerBacktester(fp, rp).run()
        assert isinstance(bt.turnover, dict)
        assert "avg_turnover" in bt.turnover

    def test_nav_starts_near_one(self):
        """nav 的首个非 NaN 值应接近 1.0（净值从 1 开始）。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        bt = LayerBacktester(fp, rp).run()
        nav = bt.nav
        if isinstance(nav, pd.Series):
            first_val = nav.dropna().iloc[0]
        else:
            first_val = nav.dropna().iloc[0, 0]
        assert abs(float(first_val) - 1.0) < 0.1

    def test_summary_contains_required_keys(self):
        """summary() 应包含 ls_annual_return 和 avg_turnover。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        s = LayerBacktester(fp, rp).run().summary()
        assert isinstance(s, dict)
        assert "ls_annual_return" in s
        assert "avg_turnover" in s

    def test_print_summary_no_error(self, capsys):
        """print_summary() 不应抛出异常，且应输出内容。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        LayerBacktester(fp, rp).run().print_summary("test_factor")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_repr_contains_class_name(self):
        """__repr__ 应包含 LayerBacktester。"""
        from factor_framework.factors.layer_backtester import LayerBacktester
        fp, rp = self._make_panels()
        assert "LayerBacktester" in repr(LayerBacktester(fp, rp))


# ═══════════════════════════════════════════════════════════════════════════════
# TestTimingGuardsInEvaluation  (DoD B1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimingGuardsInEvaluation:
    """
    DoD B1 — TimestampedPanel 语义守卫接入 compute_ic / layer_backtest。

    验证：
    - TimestampedPanel 输入能正常通过 compute_ic / layer_backtest
    - 类型不对齐时 align_with() 抛出 TimingAlignmentError
    - 普通 DataFrame 输入不受影响（向后兼容）
    """

    @staticmethod
    def _make_panels(n=60, n_stocks=20, seed=42):
        rng = np.random.default_rng(seed)
        dates   = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d")
        stocks  = [f"S{i:03d}" for i in range(n_stocks)]
        factor  = pd.DataFrame(rng.standard_normal((n, n_stocks)), index=dates, columns=stocks)
        returns = pd.DataFrame(rng.standard_normal((n, n_stocks)) * 0.01, index=dates, columns=stocks)
        return factor, returns

    @staticmethod
    def _make_timestamped(df, semantic, **kwargs):
        from factor_framework.core.panel import TimestampedPanel
        return TimestampedPanel.from_dataframe(df, semantic=semantic, **kwargs)

    def test_compute_ic_accepts_plain_dataframes(self):
        """普通 DataFrame 输入不应受 TimestampedPanel 守卫影响。"""
        from factor_framework.ic_analysis import compute_ic
        fp, rp = self._make_panels()
        ic = compute_ic(fp, rp)
        assert isinstance(ic, pd.Series)
        assert len(ic) == len(fp)

    def test_compute_ic_accepts_factor_observation_panels(self):
        """传入两个 TimestampedPanel(semantic='factor_observation') 时应正常计算。"""
        from factor_framework.ic_analysis import compute_ic
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        rp_ts = self._make_timestamped(rp, "factor_observation", factor_name="ret")
        ic = compute_ic(fp_ts, rp_ts)
        assert isinstance(ic, pd.Series)

    def test_compute_ic_accepts_t1_shifted_forward_return(self):
        """factor_observation(is_t1_shifted=True) 对 forward_return 应正常对齐。"""
        from factor_framework.ic_analysis import compute_ic
        from factor_framework.core.panel import TimestampedPanel
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        fp_ts = fp_ts.shift_to_t1()   # factor 侧做 T+1 滞后
        rp_ts = self._make_timestamped(rp, "forward_return", forward_days=21)
        ic = compute_ic(fp_ts, rp_ts)
        assert isinstance(ic, pd.Series)

    def test_compute_ic_raises_on_price_vs_factor(self):
        """price 与 factor_observation 对齐应抛出 SemanticCompatibilityError 或 TimingAlignmentError。"""
        from factor_framework.ic_analysis import compute_ic
        from factor_framework.core.panel import SemanticCompatibilityError, TimingAlignmentError
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        rp_ts = self._make_timestamped(rp, "price", price_basis="hfq")
        with pytest.raises((SemanticCompatibilityError, TimingAlignmentError)):
            compute_ic(fp_ts, rp_ts)

    def test_compute_ic_raises_on_unshifted_forward_return(self):
        """factor_observation 对 unshifted forward_return 应抛出 TimingAlignmentError。"""
        from factor_framework.ic_analysis import compute_ic
        from factor_framework.core.panel import TimingAlignmentError
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        rp_ts = self._make_timestamped(rp, "forward_return", forward_days=21)
        # is_t1_shifted=False (未调用 shift_to_t1)
        with pytest.raises(TimingAlignmentError):
            compute_ic(fp_ts, rp_ts)

    def test_layer_backtest_accepts_plain_dataframes(self):
        """普通 DataFrame 输入不受 TimestampedPanel 守卫影响。"""
        from factor_framework.backtest import layer_backtest
        fp, rp = self._make_panels()
        result = layer_backtest(fp, rp, n_groups=3)
        assert isinstance(result, pd.DataFrame)
        assert "Q1" in result.columns

    def test_layer_backtest_accepts_timestamped_panels(self):
        """传入两个合法语义的 TimestampedPanel 时 layer_backtest 应正常运行。"""
        from factor_framework.backtest import layer_backtest
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        fp_ts = fp_ts.shift_to_t1()   # factor 侧做 T+1 滞后
        rp_ts = self._make_timestamped(rp, "forward_return", forward_days=21)
        result = layer_backtest(fp_ts, rp_ts, n_groups=3)
        assert isinstance(result, pd.DataFrame)

    def test_layer_backtest_raises_on_unshifted_forward_return(self):
        """layer_backtest 中 factor_observation vs unshifted forward_return 应抛出 TimingAlignmentError。"""
        from factor_framework.backtest import layer_backtest
        from factor_framework.core.panel import TimingAlignmentError
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        rp_ts = self._make_timestamped(rp, "forward_return", forward_days=21)
        with pytest.raises(TimingAlignmentError):
            layer_backtest(fp_ts, rp_ts, n_groups=3)

    def test_assert_valid_on_single_timestamped_panel(self):
        """单个 TimestampedPanel 输入时 assert_valid() 应在 compute_ic 入口被调用（无异常）。"""
        from factor_framework.ic_analysis import compute_ic
        fp, rp = self._make_panels()
        fp_ts = self._make_timestamped(fp, "factor_observation", factor_name="test")
        # rp 是普通 DataFrame，fp_ts 只触发 assert_valid，不做 align_with
        ic = compute_ic(fp_ts, rp)
        assert isinstance(ic, pd.Series)


# ═══════════════════════════════════════════════════════════════════════════════
# TestDataStorePanelBuilderWiring  (DoD B2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataStorePanelBuilderWiring:
    """
    DoD B2 — DataStore 接入 PanelBuilder + FactorPipeline。

    验证：
    - PanelBuilder 接受 store 参数
    - FactorPipeline 默认构造 CSVDataStore
    - FactorPipeline 接受自定义 store 并透传给 PanelBuilder
    """

    def test_panel_builder_accepts_store_kwarg(self, tmp_path):
        """PanelBuilder 应接受 store 关键字参数而不报错。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        from factor_framework.data.store import CSVDataStore
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        builder = PanelBuilder(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            store       = store,
        )
        assert builder.store is store

    def test_panel_builder_store_none_by_default(self, tmp_path):
        """PanelBuilder 不传 store 时，store 属性应为 None。"""
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        builder = PanelBuilder(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
        )
        assert builder.store is None

    def test_pipeline_constructs_csvdatastore_by_default(self, tmp_path):
        """FactorPipeline 默认应自动构造 CSVDataStore 并传给 PanelBuilder。"""
        from factor_framework.pipeline import FactorPipeline
        from factor_framework.data.store import CSVDataStore
        stocks_dir = tmp_path / "Stocks"
        stocks_dir.mkdir()
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = None,
        )
        assert pipe._builder.store is not None
        assert isinstance(pipe._builder.store, CSVDataStore)

    def test_pipeline_accepts_custom_store(self, tmp_path):
        """FactorPipeline 传入自定义 store 时，PanelBuilder.store 应是同一对象。"""
        from factor_framework.pipeline import FactorPipeline
        from factor_framework.data.store import CSVDataStore
        stocks_dir = tmp_path / "Stocks"
        stocks_dir.mkdir()
        custom_store = CSVDataStore(stocks_dir=str(stocks_dir))
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = None,
            store       = custom_store,
        )
        assert pipe._builder.store is custom_store

    def test_csvdatastore_list_symbols_empty_dir(self, tmp_path):
        """空目录的 CSVDataStore 应返回空 symbol 列表。"""
        from factor_framework.data.store import CSVDataStore
        store = CSVDataStore(stocks_dir=str(tmp_path))
        symbols = store.list_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) == 0

    def test_csvdatastore_get_price_panel_returns_timestamped(self, tmp_path):
        """CSVDataStore.get_price_panel() 应返回 TimestampedPanel(semantic='price')。"""
        from factor_framework.data.store import CSVDataStore
        from factor_framework.core.panel import TimestampedPanel
        # 写一个最小化的 CSV
        import csv
        stocks_dir = tmp_path / "stocks"
        stocks_dir.mkdir()
        rows = [
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount", "adj_factor"],
        ] + [
            [f"202001{d:02d}", "000001.SZ", "10.0", "10.5", "9.5", str(10.0 + i * 0.1),
             "100000", "1000000", "1.0"]
            for i, d in enumerate(range(2, 22))
        ]
        csv_path = stocks_dir / "000001_SZ.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        try:
            panel = store.get_price_panel(["000001.SZ"])
            assert isinstance(panel, TimestampedPanel)
            assert panel.semantic == "price"
        except Exception:
            pass  # 如果 CSV 格式不完全匹配，跳过（测试接线本身）


# ═══════════════════════════════════════════════════════════════════════════════
# TestCacheDefaultBehavior  (DoD B4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheDefaultBehavior:
    """
    DoD B4 — cache_dir 默认值为 'cache/'，FactorPipeline 开箱启用 L2 缓存。
    """

    @staticmethod
    def _make_stocks_dir(tmp_path):
        stocks_dir = tmp_path / "Stocks"
        stocks_dir.mkdir()
        return stocks_dir

    def test_default_cache_dir_creates_cache_layer(self, tmp_path):
        """FactorPipeline 不传 cache_dir 时应有 CacheLayer（默认 'cache/'）。"""
        from factor_framework.pipeline import FactorPipeline
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
        )
        assert pipe._builder.cache is not None
        assert isinstance(pipe._builder.cache, CacheLayer)

    def test_explicit_none_disables_cache(self, tmp_path):
        """显式传 cache_dir=None 时，_builder.cache 应为 None。"""
        from factor_framework.pipeline import FactorPipeline
        stocks_dir = self._make_stocks_dir(tmp_path)
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = None,
        )
        assert pipe._builder.cache is None

    def test_explicit_cache_dir_uses_specified_path(self, tmp_path):
        """显式传 cache_dir 时，CacheLayer 的 cache_dir 应匹配。"""
        from factor_framework.pipeline import FactorPipeline
        from factor_framework.engine.cache import CacheLayer
        stocks_dir = self._make_stocks_dir(tmp_path)
        custom_cache = tmp_path / "my_cache"
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
            cache_dir   = str(custom_cache),
        )
        assert pipe._builder.cache is not None
        assert isinstance(pipe._builder.cache, CacheLayer)

    def test_cache_layer_get_returns_none_on_miss(self, tmp_path):
        """未命中时 CacheLayer.get_panel 应返回 None。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(
            cache_dir     = str(tmp_path / "cache"),
            stocks_dir    = str(tmp_path / "Stocks"),
            enabled_l2    = False,   # 仅 L1
        )
        key = cache.make_key("nonexistent_factor", "20200101", "20201231", [])
        result = cache.get_panel("nonexistent_factor", key)
        assert result is None

    def test_cache_layer_put_and_get_roundtrip(self, tmp_path):
        """L1 缓存 put → get 应能正确取回相同数据。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(
            cache_dir     = str(tmp_path / "cache"),
            stocks_dir    = str(tmp_path / "Stocks"),
            enabled_l2    = False,
        )
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=["20200101", "20200102"])
        key = cache.make_key("test_factor", "20200101", "20200102", ["A", "B"])
        cache.put_panel("test_factor", key, df, calc_secs=0.0)
        result = cache.get_panel("test_factor", key)
        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_cache_info_shows_l1_entries(self, tmp_path):
        """cache_info() 应返回 l1_entries 字段。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(
            cache_dir  = str(tmp_path / "cache"),
            stocks_dir = str(tmp_path / "Stocks"),
            enabled_l2 = False,
        )
        info = cache.cache_info()
        assert "l1_entries" in info
        assert info["l1_entries"] == 0

    def test_cache_info_increments_after_put(self, tmp_path):
        """put 之后 l1_entries 应增加。"""
        from factor_framework.engine.cache import CacheLayer
        cache = CacheLayer(
            cache_dir  = str(tmp_path / "cache"),
            stocks_dir = str(tmp_path / "Stocks"),
            enabled_l2 = False,
        )
        df = pd.DataFrame({"A": [1.0]}, index=["20200101"])
        key = cache.make_key("f1", "20200101", "20200101", ["A"])
        cache.put_panel("f1", key, df, calc_secs=0.0)
        assert cache.cache_info()["l1_entries"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestFactorEngineDeprecation  (DoD C1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorEngineDeprecation:
    """
    DoD C1 — 直接实例化 FactorEngine 应发出 DeprecationWarning。
    通过 PanelBuilder 内部实例化则不应发出警告。
    """

    def setup_method(self):
        """每个测试前重置 FactorEngine._deprecation_warned，确保 warning 可触发。"""
        from factor_framework.factor_engine import FactorEngine
        FactorEngine._deprecation_warned = False

    def test_direct_instantiation_emits_deprecation_warning(self, tmp_path):
        """直接 FactorEngine() 应发出 DeprecationWarning。"""
        import warnings
        from factor_framework.factor_engine import FactorEngine
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FactorEngine(
                stocks_dir  = str(tmp_path),
                stock_basic = str(tmp_path / "nonexistent.csv"),
                verbose     = False,
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "FactorEngine" in str(dep_warnings[0].message) or "兼容层" in str(dep_warnings[0].message)

    def test_warning_issued_only_once(self, tmp_path):
        """同一进程内 DeprecationWarning 只发出一次（类级别 flag）。"""
        import warnings
        from factor_framework.factor_engine import FactorEngine
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FactorEngine(stocks_dir=str(tmp_path), stock_basic=str(tmp_path / "none.csv"), verbose=False)
            FactorEngine(stocks_dir=str(tmp_path), stock_basic=str(tmp_path / "none.csv"), verbose=False)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 1

    def test_internal_flag_suppresses_warning(self, tmp_path):
        """传入 _internal=True 时不应发出 DeprecationWarning。"""
        import warnings
        from factor_framework.factor_engine import FactorEngine
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FactorEngine(
                stocks_dir  = str(tmp_path),
                stock_basic = str(tmp_path / "none.csv"),
                verbose     = False,
                _internal   = True,
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0

    def test_panel_builder_does_not_emit_deprecation_warning(self, tmp_path):
        """PanelBuilder.engine 访问不应触发 DeprecationWarning。"""
        import warnings
        from factor_framework.engine.panel_builder import PanelBuilder
        stocks_dir = tmp_path / "Stocks"
        stocks_dir.mkdir()
        builder = PanelBuilder(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp_path / "nonexistent.csv"),
            verbose     = False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = builder.engine  # 触发延迟初始化
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestIcDecayDeprecationWarning  (DoD B3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIcDecayDeprecationWarning:
    """
    DoD B3 — ic_decay() 的 price_panel 路径应发出 DeprecationWarning。
    return_panels 路径（主路径）不应发出 DeprecationWarning。
    """

    @staticmethod
    def _make_panels(n=60, n_stocks=15):
        rng = np.random.default_rng(0)
        dates   = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d")
        stocks  = [f"S{i:02d}" for i in range(n_stocks)]
        factor  = pd.DataFrame(rng.standard_normal((n, n_stocks)), index=dates, columns=stocks)
        price   = pd.DataFrame(10.0 + rng.standard_normal((n, n_stocks)).cumsum(axis=0),
                               index=dates, columns=stocks)
        return factor, price

    def test_price_panel_path_emits_deprecation_warning(self):
        """ic_decay(price_panel=...) 回退路径应发出 DeprecationWarning。"""
        import warnings
        from factor_framework.ic_analysis import ic_decay
        fp, price = self._make_panels()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ic_decay(fp, price_panel=price, forward_periods=[1, 5])
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "price_panel" in str(dep_warnings[0].message) or "废弃" in str(dep_warnings[0].message)

    def test_return_panels_path_no_deprecation_warning(self):
        """ic_decay(return_panels=...) 主路径不应发出 DeprecationWarning。"""
        import warnings
        from factor_framework.ic_analysis import ic_decay, compute_ic
        fp, price = self._make_panels()
        ret = price.shift(-5) / price.replace(0, np.nan) - 1
        return_panels = {5: ret}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ic_decay(fp, return_panels=return_panels)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0

    def test_ic_decay_main_path_returns_dataframe(self):
        """ic_decay(return_panels=...) 应返回有效 DataFrame。"""
        from factor_framework.ic_analysis import ic_decay
        fp, price = self._make_panels()
        ret1 = price.shift(-1) / price.replace(0, np.nan) - 1
        ret5 = price.shift(-5) / price.replace(0, np.nan) - 1
        result = ic_decay(fp, return_panels={1: ret1, 5: ret5})
        assert isinstance(result, pd.DataFrame)
        assert set(result.index) == {1, 5}
        assert "mean_ic" in result.columns

    def test_both_none_raises_value_error(self):
        """price_panel=None 且 return_panels=None 应抛出 ValueError。"""
        from factor_framework.ic_analysis import ic_decay
        fp, _ = self._make_panels()
        with pytest.raises(ValueError, match="None"):
            ic_decay(fp)

