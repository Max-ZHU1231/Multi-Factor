"""
tests/unit/test_ic_decay_diagnostics.py
========================================
IC 衰减异常诊断框架（6 模块）的单元测试。

测试策略
--------
使用合成数据（已知真相），验证每个模块的输出结构、判断逻辑和边界情况。

合成数据设计
-----------
- "无泄露"数据：因子 t 日计算，收益严格从 t+1 开始
- "有泄露"数据：因子 t 日 = t 日收益的一部分（必然 IC lag=0 > lag=1）
- "累计放大"数据：因子只预测短期，但长 forward IC 因窗口累积而虚高
- "真实中期"数据：因子持续预测 60 天的增量收益
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from factor_framework.analytics.ic_decay_diagnostics import (
    ICDecayDiagnostics,
    DiagnosticReport,
    DiagnosticResult,
    _compute_ic_series,
    _ic_stats_dict,
    _nw_t,
    _build_forward_ret,
    _shift_factor,
    _resample_monthly_last,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: 合成数据
# ═══════════════════════════════════════════════════════════════════════════════

N_DATES  = 500   # 约2年日频（月度换仓≈24期）
N_STOCKS = 80
RNG      = np.random.default_rng(42)

def _make_dates(n: int) -> pd.DatetimeIndex:
    """生成连续工作日日期序列。"""
    return pd.bdate_range("2020-01-02", periods=n)

def _date_strs(n: int) -> list:
    return _make_dates(n).strftime("%Y%m%d").tolist()

def _make_price_panel(n_dates: int = N_DATES, n_stocks: int = N_STOCKS,
                      seed: int = 42) -> pd.DataFrame:
    """模拟股票价格（几何布朗运动，确保恒正）。"""
    rng    = np.random.default_rng(seed)
    daily  = rng.normal(0.0005, 0.02, size=(n_dates, n_stocks))
    prices = 100 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(prices,
                        index=_date_strs(n_dates),
                        columns=[f"S{i:03d}" for i in range(n_stocks)])


@pytest.fixture(scope="module")
def base_price_panel():
    return _make_price_panel()


@pytest.fixture(scope="module")
def clean_factor_panel(base_price_panel):
    """无泄露因子：纯噪声 + 少量未来收益信号（与 t+1 起始收益相关）。"""
    n_dates, n_stocks = base_price_panel.shape
    rng = np.random.default_rng(99)
    # 构造：因子 = 3日后收益的弱信号 + 噪声
    p = base_price_panel.values
    ret3 = np.full_like(p, np.nan)
    ret3[:-3] = p[3:] / p[:-3] - 1
    noise = rng.normal(0, 1, size=(n_dates, n_stocks))
    signal = np.where(np.isnan(ret3), 0, ret3)
    factor_vals = 0.3 * signal + 0.7 * noise
    return pd.DataFrame(factor_vals,
                        index=base_price_panel.index,
                        columns=base_price_panel.columns)


@pytest.fixture(scope="module")
def lookahead_factor_panel(base_price_panel):
    """含前瞻偏差因子：因子直接使用当日收益（极端泄露）。"""
    p = base_price_panel.values
    ret1 = np.full_like(p, np.nan)
    ret1[:-1] = p[1:] / p[:-1] - 1
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.3, size=p.shape)
    factor_vals = 0.8 * np.where(np.isnan(ret1), 0, ret1) + 0.2 * noise
    return pd.DataFrame(factor_vals,
                        index=base_price_panel.index,
                        columns=base_price_panel.columns)


@pytest.fixture(scope="module")
def industry_map(base_price_panel):
    """简单行业映射（5个行业）。"""
    stocks = base_price_panel.columns
    inds   = [f"IND{i % 5}" for i in range(len(stocks))]
    return pd.Series(inds, index=stocks, name="industry")


@pytest.fixture(scope="module")
def mktcap_panel(base_price_panel):
    """简单市值面板（与价格同步）。"""
    return base_price_panel * 1e6   # 万元（模拟）


@pytest.fixture(scope="module")
def diag_clean(clean_factor_panel, base_price_panel, industry_map, mktcap_panel):
    """无泄露因子的诊断对象。"""
    return ICDecayDiagnostics(
        factor_panel = clean_factor_panel,
        price_panel  = base_price_panel,
        forward_list = [1, 5, 10, 21, 60],
        industry_map = industry_map,
        mktcap_panel = mktcap_panel,
        ic_method    = "rank",
        factor_name  = "clean_factor",
    )


@pytest.fixture(scope="module")
def diag_lookahead(lookahead_factor_panel, base_price_panel):
    """含前瞻偏差因子的诊断对象。"""
    return ICDecayDiagnostics(
        factor_panel = lookahead_factor_panel,
        price_panel  = base_price_panel,
        forward_list = [1, 5, 10, 21, 60],
        ic_method    = "rank",
        factor_name  = "lookahead_factor",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestInternalUtils:

    def test_compute_ic_series_shape(self, clean_factor_panel, base_price_panel):
        """IC 系列长度与公共日期数一致。"""
        ret = _build_forward_ret(base_price_panel, 5)
        ic  = _compute_ic_series(clean_factor_panel, ret, method="rank")
        assert isinstance(ic, pd.Series)
        common = clean_factor_panel.index.intersection(ret.index)
        assert len(ic) == len(common)

    def test_compute_ic_series_in_minus1_1(self, clean_factor_panel, base_price_panel):
        """IC 值在 [-1, 1]。"""
        ret = _build_forward_ret(base_price_panel, 5)
        ic  = _compute_ic_series(clean_factor_panel, ret)
        assert ic.dropna().between(-1, 1).all()

    def test_ic_stats_dict_keys(self, clean_factor_panel, base_price_panel):
        """ic_stats_dict 包含所有必要字段。"""
        ret = _build_forward_ret(base_price_panel, 5)
        ic  = _compute_ic_series(clean_factor_panel, ret)
        st  = _ic_stats_dict(ic)
        for key in ["mean_ic", "std_ic", "icir", "win_rate", "t_stat", "n"]:
            assert key in st

    def test_ic_stats_dict_empty(self):
        """空 IC 系列返回 NaN。"""
        empty = pd.Series([], dtype=float)
        st    = _ic_stats_dict(empty)
        assert st["n"] == 0
        assert np.isnan(st["mean_ic"])

    def test_nw_t_returns_finite(self, clean_factor_panel, base_price_panel):
        """Newey-West t 统计量返回有限值。"""
        ret = _build_forward_ret(base_price_panel, 5)
        ic  = _compute_ic_series(clean_factor_panel, ret)
        nw  = _nw_t(ic.dropna())
        assert np.isfinite(nw)

    def test_build_forward_ret_shape(self, base_price_panel):
        """build_forward_ret 返回与价格面板相同 shape。"""
        ret = _build_forward_ret(base_price_panel, 10)
        assert ret.shape == base_price_panel.shape

    def test_build_forward_ret_tail_nan(self, base_price_panel):
        """forward=10 时，末尾10行应为 NaN。"""
        ret = _build_forward_ret(base_price_panel, 10)
        assert ret.iloc[-10:].isna().all().all()

    def test_shift_factor_lag0(self, clean_factor_panel):
        """lag=0 时不移位（原地返回）。"""
        shifted = _shift_factor(clean_factor_panel, 0)
        pd.testing.assert_frame_equal(shifted, clean_factor_panel)

    def test_shift_factor_lag1(self, clean_factor_panel):
        """lag=1 时第一行全为 NaN。"""
        shifted = _shift_factor(clean_factor_panel, 1)
        assert shifted.iloc[0].isna().all()

    def test_resample_monthly_last_reduces_rows(self, clean_factor_panel):
        """月末重采样后行数 < 原始行数（日频 → 月频）。"""
        monthly = _resample_monthly_last(clean_factor_panel)
        assert len(monthly) < len(clean_factor_panel)
        assert len(monthly) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# ICDecayDiagnostics 构造
# ═══════════════════════════════════════════════════════════════════════════════

class TestICDecayDiagnosticsInit:

    def test_init_stores_forward_list_sorted(self, clean_factor_panel, base_price_panel):
        """forward_list 存储时应排序。"""
        diag = ICDecayDiagnostics(
            factor_panel = clean_factor_panel,
            price_panel  = base_price_panel,
            forward_list = [60, 1, 21, 5, 10],
        )
        assert diag.forward_list == [1, 5, 10, 21, 60]

    def test_init_prebuilds_ret_panels(self, diag_clean):
        """构造时预构建所有 forward 的收益率面板。"""
        assert set(diag_clean._ret_panels.keys()) == {1, 5, 10, 21, 60}
        for fwd, rp in diag_clean._ret_panels.items():
            assert isinstance(rp, pd.DataFrame)
            assert not rp.empty

    def test_init_without_optional_params(self, clean_factor_panel, base_price_panel):
        """不提供 industry_map 和 mktcap_panel 也能正常构造。"""
        diag = ICDecayDiagnostics(
            factor_panel = clean_factor_panel,
            price_panel  = base_price_panel,
            forward_list = [1, 21],
        )
        assert diag.industry_map is None
        assert diag.mktcap_panel is None


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1 — 时间对齐与前瞻偏差检查
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule1TimeAlignment:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module1_time_alignment()
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 1

    def test_evidence_is_dataframe(self, diag_clean):
        result = diag_clean.module1_time_alignment()
        assert isinstance(result.evidence, pd.DataFrame)

    def test_evidence_has_lag_index(self, diag_clean):
        result = diag_clean.module1_time_alignment(lag_list=[0, 1, 2])
        # index = (forward, lag) MultiIndex
        assert "lag" in result.evidence.index.names

    def test_clean_factor_passes(self, diag_clean):
        """无前瞻偏差因子应通过 Module 1。"""
        result = diag_clean.module1_time_alignment()
        # 无泄露因子：lag=0 和 lag=1 的 IC 差值不应 >0.01
        assert result.passed is True
        assert result.risk_level == "LOW"

    def test_lookahead_factor_fails(self, diag_lookahead):
        """含前瞻偏差因子：lag=0 IC >> lag=1 IC，应 FAIL。"""
        result = diag_lookahead.module1_time_alignment()
        # 因子直接包含当日收益，lag=0 IC 应远高于 lag=1
        assert result.passed is False
        assert result.risk_level == "HIGH"

    def test_ic_lag0_gt_lag1_for_lookahead(self, diag_lookahead):
        """含泄露因子：IC(lag=0) > IC(lag=1)。"""
        result = diag_lookahead.module1_time_alignment(lag_list=[0, 1, 2])
        ev = result.evidence
        # 取第一个 forward 的数据
        fwd0 = diag_lookahead.forward_list[0]
        ic_lag0 = ev.loc[(fwd0, 0), "mean_ic"]
        ic_lag1 = ev.loc[(fwd0, 1), "mean_ic"]
        assert ic_lag0 > ic_lag1

    def test_custom_lag_list(self, diag_clean):
        """自定义 lag 列表（只测 lag=0 和 lag=1）。"""
        result = diag_clean.module1_time_alignment(lag_list=[0, 1])
        ev = result.evidence
        assert 0 in ev.index.get_level_values("lag")
        assert 1 in ev.index.get_level_values("lag")
        assert 2 not in ev.index.get_level_values("lag")

    def test_evidence_columns_present(self, diag_clean):
        result = diag_clean.module1_time_alignment()
        for col in ["mean_ic", "icir", "t_stat", "n"]:
            assert col in result.evidence.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Module 2 — 收益定义与累计窗口拆解
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule2IncrementalIC:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 2

    def test_evidence_has_both_tables(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        assert isinstance(result.evidence, dict)
        assert "cumul_ic" in result.evidence
        assert "incr_ic" in result.evidence

    def test_cumul_ic_has_all_forwards(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        cumul_df = result.evidence["cumul_ic"]
        assert isinstance(cumul_df, pd.DataFrame)
        for fwd in [1, 5, 10, 21, 60]:
            assert fwd in cumul_df.index

    def test_incr_ic_has_k_index(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        incr_df = result.evidence["incr_ic"]
        assert isinstance(incr_df, pd.Series)
        assert 1 in incr_df.index

    def test_cumul_ic_columns(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        cumul_df = result.evidence["cumul_ic"]
        for col in ["cumul_ic", "cumul_icir", "t_stat", "nw_t_stat"]:
            assert col in cumul_df.columns

    def test_passed_is_bool_or_none(self, diag_clean):
        result = diag_clean.module2_incremental_ic()
        assert result.passed in (True, False, None)

    def test_incr_ic_values_in_range(self, diag_clean):
        """增量 IC 也应在 [-1, 1] 范围内。"""
        result = diag_clean.module2_incremental_ic()
        incr_df = result.evidence["incr_ic"]
        valid = incr_df.dropna()
        assert (valid.abs() <= 1.0).all()

    def test_nw_t_stat_in_cumul(self, diag_clean):
        """Newey-West t 统计量被填入 cumul_ic 表格。"""
        result = diag_clean.module2_incremental_ic()
        nw_vals = result.evidence["cumul_ic"]["nw_t_stat"].dropna()
        assert len(nw_vals) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Module 3 — 市场/风格暴露剥离
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule3ExposureStrip:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module3_exposure_strip()
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 3

    def test_evidence_is_dataframe(self, diag_clean):
        result = diag_clean.module3_exposure_strip()
        assert isinstance(result.evidence, pd.DataFrame)

    def test_evidence_has_all_forward_rows(self, diag_clean):
        result = diag_clean.module3_exposure_strip()
        ev = result.evidence
        for fwd in [1, 5, 10, 21, 60]:
            assert fwd in ev.index

    def test_evidence_has_all_ic_versions(self, diag_clean):
        result = diag_clean.module3_exposure_strip()
        ev = result.evidence
        expected_cols = [
            "ic_raw", "ic_mkt_excess",
            "ic_ind_excess", "ic_mktcap_neut_factor", "ic_dual_neut_factor"
        ]
        for col in expected_cols:
            assert col in ev.columns

    def test_mkt_excess_ic_finite(self, diag_clean):
        """市场超额 IC 应有有限值（始终可计算）。"""
        result = diag_clean.module3_exposure_strip()
        ev = result.evidence
        mkt_ic = ev["ic_mkt_excess"].dropna()
        assert len(mkt_ic) == len(diag_clean.forward_list)
        assert np.all(np.isfinite(mkt_ic))

    def test_without_industry_map_nan_ind_col(self, clean_factor_panel, base_price_panel):
        """不提供 industry_map 时，行业超额 IC 列全为 NaN。"""
        diag = ICDecayDiagnostics(
            factor_panel = clean_factor_panel,
            price_panel  = base_price_panel,
            forward_list = [1, 21],
        )
        result = diag.module3_exposure_strip()
        ev = result.evidence
        assert ev["ic_ind_excess"].isna().all()

    def test_passed_and_risk_level_set(self, diag_clean):
        result = diag_clean.module3_exposure_strip()
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════════
# Module 4 — 样本偏差检查
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule4SampleBias:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module4_sample_bias()
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 4

    def test_evidence_is_dataframe(self, diag_clean):
        result = diag_clean.module4_sample_bias()
        assert isinstance(result.evidence, pd.DataFrame)

    def test_evidence_has_all_columns(self, diag_clean):
        result = diag_clean.module4_sample_bias()
        ev = result.evidence
        for col in ["n_dates", "n_stocks", "factor_coverage",
                    "ret_coverage", "joint_coverage", "n_monthly_periods"]:
            assert col in ev.columns

    def test_coverage_in_0_1(self, diag_clean):
        result = diag_clean.module4_sample_bias()
        ev = result.evidence
        assert ev["joint_coverage"].between(0, 1).all()
        assert ev["factor_coverage"].between(0, 1).all()

    def test_long_fwd_coverage_lte_short(self, diag_clean):
        """长 forward 覆盖率 <= 短 forward 覆盖率（尾部 NaN 更多）。"""
        result = diag_clean.module4_sample_bias()
        ev = result.evidence
        cov1  = ev.loc[1, "ret_coverage"]
        cov60 = ev.loc[60, "ret_coverage"]
        # 理论上 forward=60 尾部更多 NaN，覆盖率更低
        assert cov60 <= cov1 + 1e-6   # 允许浮点误差

    def test_monthly_periods_positive(self, diag_clean):
        result = diag_clean.module4_sample_bias()
        ev = result.evidence
        assert (ev["n_monthly_periods"] > 0).all()

    def test_pass_on_clean_data(self, diag_clean):
        """合成数据无存活偏差，应 PASS 或 N/A（差距 < 5%）。"""
        result = diag_clean.module4_sample_bias()
        # 合成数据（GBM，无停牌退市）覆盖率差距极小
        assert result.passed in (True, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Module 5 — 因子属性验证（时效性/半衰期）
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule5FactorHalflife:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 5

    def test_evidence_has_required_keys(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        ev = result.evidence
        assert "autocorr_by_lag_month" in ev
        assert "halflife_months_estimate" in ev
        assert "incr_ic_by_k" in ev
        assert "incr_halflife_days" in ev

    def test_autocorr_dataframe_shape(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        ac_df  = result.evidence["autocorr_by_lag_month"]
        assert isinstance(ac_df, pd.DataFrame)
        assert "autocorr" in ac_df.columns
        assert len(ac_df) >= 1

    def test_autocorr_lag1_in_minus1_1(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        ac_df  = result.evidence["autocorr_by_lag_month"]
        ac1    = ac_df.loc[1, "autocorr"] if 1 in ac_df.index else np.nan
        assert -1 <= ac1 <= 1

    def test_incr_ic_series_has_k1(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        incr   = result.evidence["incr_ic_by_k"]
        assert 1 in incr.index

    def test_halflife_estimate_numeric_or_na(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        hl     = result.evidence["halflife_months_estimate"]
        assert isinstance(hl, (float, int, str))  # float 或 "N/A"

    def test_risk_level_set(self, diag_clean):
        result = diag_clean.module5_factor_halflife()
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════════
# Module 6 — 稳健性复核
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule6Robustness:

    def test_returns_diagnostic_result(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        assert isinstance(result, DiagnosticResult)
        assert result.module_id == 6

    def test_evidence_has_three_tables(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        ev = result.evidence
        assert "split_period_ic" in ev
        assert "winsor_sensitivity" in ev
        assert "regime_ic" in ev

    def test_split_period_has_n_rows(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        split_df = result.evidence["split_period_ic"]
        assert len(split_df) == 3

    def test_split_period_has_ic_columns(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        split_df = result.evidence["split_period_ic"]
        for fwd in [1, 5, 10, 21, 60]:
            assert f"ic_fwd{fwd}" in split_df.columns

    def test_winsor_sensitivity_has_three_rows(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        winsor_df = result.evidence["winsor_sensitivity"]
        assert len(winsor_df) == 3
        assert set(winsor_df["label"]) == {"strict", "standard", "loose"}

    def test_winsor_mean_ic_in_range(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        winsor_df = result.evidence["winsor_sensitivity"]
        ic_vals = winsor_df["mean_ic"].dropna()
        if len(ic_vals) > 0:
            assert (ic_vals.abs() <= 1.0).all()

    def test_regime_ic_has_three_regimes(self, diag_clean):
        result = diag_clean.module6_robustness(n_splits=3)
        regime_df = result.evidence["regime_ic"]
        assert set(regime_df["regime"]) >= {"bull", "bear", "flat"}

    def test_n_splits_2(self, diag_clean):
        """n_splits=2 时子期表有2行。"""
        result = diag_clean.module6_robustness(n_splits=2)
        split_df = result.evidence["split_period_ic"]
        assert len(split_df) == 2

    def test_passed_and_risk_level_set(self, diag_clean):
        result = diag_clean.module6_robustness()
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════════
# DiagnosticResult 容器
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticResult:

    def _make_result(self, passed, risk_level="LOW"):
        return DiagnosticResult(
            module_id   = 1,
            module_name = "测试模块",
            passed      = passed,
            evidence    = {"key": 1.0},
            conclusion  = "测试结论",
            risk_level  = risk_level,
        )

    def test_status_str_pass(self):
        r = self._make_result(True)
        assert r.status_str() == "[PASS]"

    def test_status_str_fail(self):
        r = self._make_result(False)
        assert r.status_str() == "[FAIL]"

    def test_status_str_none(self):
        r = self._make_result(None)
        assert r.status_str() == "[N/A ]"

    def test_to_dict_keys(self):
        r = self._make_result(True, "LOW")
        d = r.to_dict()
        for key in ["module_id", "module_name", "passed", "risk_level", "conclusion", "evidence"]:
            assert key in d

    def test_to_dict_passed_value(self):
        r = self._make_result(False, "HIGH")
        d = r.to_dict()
        assert d["passed"] is False
        assert d["risk_level"] == "HIGH"

    def test_to_dict_with_dataframe_evidence(self, diag_clean):
        """evidence 为 DataFrame 时，to_dict 能正确序列化。"""
        result = diag_clean.module1_time_alignment()
        d = result.to_dict()
        assert isinstance(d["evidence"], dict)  # DataFrame → dict(orient="index")


# ═══════════════════════════════════════════════════════════════════════════════
# DiagnosticReport 容器
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticReport:

    @pytest.fixture
    def sample_results(self):
        return [
            DiagnosticResult(1, "M1", True,  {"a": 1}, "M1 pass", "LOW"),
            DiagnosticResult(2, "M2", False, {"b": 2}, "M2 fail", "HIGH"),
            DiagnosticResult(3, "M3", None,  {"c": 3}, "M3 n/a",  "UNKNOWN"),
        ]

    def test_report_stores_results(self, sample_results):
        rep = DiagnosticReport(sample_results, factor_name="test")
        assert len(rep.results) == 3

    def test_to_dict_structure(self, sample_results):
        rep = DiagnosticReport(sample_results, factor_name="test")
        d   = rep.to_dict()
        assert d["factor_name"] == "test"
        assert len(d["modules"]) == 3

    def test_print_full_no_error(self, sample_results, capsys):
        """print_full() 不应抛出异常。"""
        rep = DiagnosticReport(sample_results, factor_name="test")
        rep.print_full()
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "M1" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# run_all 集成测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAll:

    def test_run_all_returns_report(self, diag_clean):
        report = diag_clean.run_all(verbose=False)
        assert isinstance(report, DiagnosticReport)

    def test_run_all_has_six_modules(self, diag_clean):
        report = diag_clean.run_all(verbose=False)
        assert len(report.results) == 6
        module_ids = [r.module_id for r in report.results]
        assert module_ids == [1, 2, 3, 4, 5, 6]

    def test_run_all_subset_modules(self, diag_clean):
        """只跑指定模块。"""
        report = diag_clean.run_all(run_modules=[1, 4], verbose=False)
        assert len(report.results) == 2
        assert {r.module_id for r in report.results} == {1, 4}

    def test_run_all_to_dict(self, diag_clean):
        report = diag_clean.run_all(run_modules=[1, 2], verbose=False)
        d = report.to_dict()
        assert d["factor_name"] == "clean_factor"
        assert len(d["modules"]) == 2

    def test_run_all_no_crashes_with_minimal_data(self, base_price_panel):
        """极小数据集（50 行 × 10 股）不崩溃。"""
        small_price = base_price_panel.iloc[:50, :10]
        small_factor = pd.DataFrame(
            np.random.randn(50, 10),
            index   = small_price.index,
            columns = small_price.columns,
        )
        diag = ICDecayDiagnostics(
            factor_panel = small_factor,
            price_panel  = small_price,
            forward_list = [1, 5, 21],
            ic_method    = "rank",
        )
        # 只跑快速模块，避免超时
        report = diag.run_all(run_modules=[1, 4], verbose=False)
        assert len(report.results) == 2

    def test_run_all_lookahead_m1_fails(self, diag_lookahead):
        """含泄露因子：Module 1 应 FAIL。"""
        report = diag_lookahead.run_all(run_modules=[1], verbose=False)
        m1 = report.results[0]
        assert m1.module_id == 1
        assert m1.passed is False

    def test_run_all_all_results_have_conclusion(self, diag_clean):
        """每个模块的 conclusion 非空字符串。"""
        report = diag_clean.run_all(run_modules=[1, 2, 3, 4], verbose=False)
        for r in report.results:
            assert isinstance(r.conclusion, str)
            assert len(r.conclusion) > 0

    def test_run_all_print_full_no_error(self, diag_clean, capsys):
        """run_all 后调用 print_full() 不崩溃。"""
        report = diag_clean.run_all(run_modules=[1, 4], verbose=False)
        report.print_full()
        captured = capsys.readouterr()
        assert "clean_factor" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# 边界与异常情况
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_nan_factor_does_not_crash(self, base_price_panel):
        """全 NaN 因子面板不抛出异常，返回 passed=None 或 passed=False。"""
        all_nan = pd.DataFrame(
            np.nan,
            index   = base_price_panel.index,
            columns = base_price_panel.columns,
        )
        diag = ICDecayDiagnostics(
            factor_panel = all_nan,
            price_panel  = base_price_panel,
            forward_list = [1, 21],
        )
        result = diag.module1_time_alignment()
        assert isinstance(result, DiagnosticResult)

    def test_single_forward(self, clean_factor_panel, base_price_panel):
        """forward_list 只有一个元素也能运行。"""
        diag = ICDecayDiagnostics(
            factor_panel = clean_factor_panel,
            price_panel  = base_price_panel,
            forward_list = [21],
        )
        result = diag.module4_sample_bias()
        assert isinstance(result, DiagnosticResult)

    def test_module_error_does_not_propagate_in_run_all(self, clean_factor_panel, base_price_panel):
        """单模块异常时 run_all 继续执行其他模块，不整体崩溃。"""
        diag = ICDecayDiagnostics(
            factor_panel = clean_factor_panel,
            price_panel  = base_price_panel,
            forward_list = [1, 21],
        )
        # 故意破坏 mktcap_panel 类型（传错误类型到 _neutralize_mktcap）
        diag.mktcap_panel = "bad_type"   # 会在 Module 3 报错
        diag.industry_map = pd.Series(
            ["IND0"] * len(base_price_panel.columns),
            index=base_price_panel.columns,
        )
        report = diag.run_all(run_modules=[3], verbose=False)
        assert len(report.results) == 1
        # Module 3 应返回 passed=None 并记录错误，而非整体崩溃
        m3 = report.results[0]
        assert m3.module_id == 3
