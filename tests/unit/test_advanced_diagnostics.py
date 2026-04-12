from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from factor_framework.analytics.advanced_diagnostics import run_advanced_diagnostics


def _mock_report() -> SimpleNamespace:
    idx = pd.date_range("2022-01-31", periods=24, freq="ME").strftime("%Y%m%d")
    cols = [f"S{i:03d}" for i in range(30)]
    rng = np.random.default_rng(42)

    factor_panel = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    return_panel = pd.DataFrame(rng.normal(0.01, 0.05, size=(len(idx), len(cols))), index=idx, columns=cols)
    peer_factor_panels = {
        "value_pb": pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols),
        "momentum_12_1": pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols),
    }

    layer_ret = pd.DataFrame({
        "Q1": rng.normal(0.002, 0.02, len(idx)),
        "Q2": rng.normal(0.004, 0.02, len(idx)),
        "Q3": rng.normal(0.006, 0.02, len(idx)),
        "Q4": rng.normal(0.008, 0.02, len(idx)),
        "Q5": rng.normal(0.010, 0.02, len(idx)),
    }, index=idx)
    layer_ret["LS"] = layer_ret["Q5"] - layer_ret["Q1"]
    pit_rows = []
    for dt in idx:
        for s in cols[:10]:
            pit_rows.append({"trade_date": dt, "index_code": "000300.SH", "ts_code": s})
        for s in cols[10:20]:
            pit_rows.append({"trade_date": dt, "index_code": "000905.SH", "ts_code": s})
        for s in cols[20:30]:
            pit_rows.append({"trade_date": dt, "index_code": "000852.SH", "ts_code": s})
    index_membership_pit = pd.DataFrame(pit_rows)

    return SimpleNamespace(
        factor_name="dummy_factor",
        factor_panel=factor_panel,
        return_panel=return_panel,
        peer_factor_panels=peer_factor_panels,
        layer_ret=layer_ret,
        index_membership_pit=index_membership_pit,
        ls_stats={"ls_annual_return": 0.12, "ls_sharpe": 1.1, "ls_max_drawdown": -0.2},
        turnover={"avg_turnover": 0.25},
    )


def test_advanced_diagnostics_output_contract(tmp_path):
    report = _mock_report()
    out_dir = tmp_path / "advanced_diagnostics"

    adv = run_advanced_diagnostics(
        report,
        output_dir=out_dir,
        n_groups=5,
        direction=1,
        periods_per_year=12,
    )

    assert (out_dir / "advanced_summary.json").exists()
    assert (out_dir / "timing.csv").exists()
    assert (out_dir / "fm_summary.csv").exists()
    assert (out_dir / "alpha_models.csv").exists()
    assert (out_dir / "orthogonal_ic.csv").exists()
    assert (out_dir / "factor_corr_matrix.csv").exists()
    assert (out_dir / "factor_corr_matrix_wide.csv").exists()
    assert (out_dir / "size_bucket_report.csv").exists()
    assert (out_dir / "rolling_oos.csv").exists()
    assert (out_dir / "regime_stability.csv").exists()
    assert (out_dir / "param_sensitivity.csv").exists()
    assert (out_dir / "monotonicity_tests.csv").exists()
    assert (out_dir / "turnover_boundary.csv").exists()
    assert (out_dir / "cost_scenarios.csv").exists()
    assert (out_dir / "long_short_intensity.csv").exists()
    assert len(adv.module_status) == 15
    assert adv.module_status["monotonicity_tests.csv"] == "implemented"
    assert adv.module_status["fm_summary.csv"] == "implemented"

    fm = pd.read_csv(out_dir / "fm_summary.csv")
    expected_cols = {
        "fm_lambda_mean", "fm_t", "fm_nw_t", "fm_p", "fm_lambda_std",
        "fm_nw_p", "n_periods", "valid_periods", "total_periods",
        "mean_cs_nobs", "mean_cs_r2", "controls_spec",
        "nw_lags", "is_placeholder",
    }
    assert expected_cols.issubset(set(fm.columns))
    assert bool(fm.loc[0, "is_placeholder"]) is False

    with open(out_dir / "advanced_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert bool(summary["is_placeholder"]) is False
    assert "alpha_models.csv" in summary["implemented_modules"]
    assert "orthogonal_ic.csv" in summary["implemented_modules"]
    assert "factor_corr_matrix.csv" in summary["implemented_modules"]
    assert "size_bucket_report.csv" in summary["implemented_modules"]
    assert "rolling_oos.csv" in summary["implemented_modules"]
    assert "regime_stability.csv" in summary["implemented_modules"]
    assert "param_sensitivity.csv" in summary["implemented_modules"]
    assert "vol_20d" in summary["unavailable_peer_factors"]
    assert "universe_membership" in summary
    assert "buckets" in summary["universe_membership"]

    alpha = pd.read_csv(out_dir / "alpha_models.csv")
    alpha_expected = {
        "model", "alpha", "alpha_t", "alpha_nw_t", "alpha_p", "alpha_nw_p",
        "beta_mkt", "beta_smb", "beta_hml", "r2", "n_obs", "nw_lags", "model_note", "is_placeholder",
    }
    assert alpha_expected.issubset(set(alpha.columns))
    assert set(alpha["model"]) == {"CAPM", "FF3"}
    assert (alpha["is_placeholder"] == False).all()
    ff3_note = alpha.loc[alpha["model"] == "FF3", "model_note"].iloc[0]
    assert isinstance(ff3_note, str)

    orth = pd.read_csv(out_dir / "orthogonal_ic.csv")
    orth_expected = {
        "raw_ic", "resid_ic", "delta_ic",
        "raw_t", "resid_t", "raw_nw_t", "resid_nw_t",
        "control_factor", "n_periods", "is_placeholder",
    }
    assert orth_expected.issubset(set(orth.columns))
    assert bool(orth.loc[0, "is_placeholder"]) is False

    corr = pd.read_csv(out_dir / "factor_corr_matrix.csv")
    corr_expected = {
        "peer_factor", "mean_cs_corr", "median_cs_corr", "std_cs_corr",
        "t_stat", "nw_t_stat", "n_periods", "mean_cs_nobs", "min_cs_nobs",
        "high_corr_flag", "is_placeholder",
    }
    assert corr_expected.issubset(set(corr.columns))
    assert (corr["is_placeholder"] == False).all()

    corr_wide = pd.read_csv(out_dir / "factor_corr_matrix_wide.csv")
    assert {"target_factor", "as_of", "sample_period"}.issubset(set(corr_wide.columns))

    size_bucket = pd.read_csv(out_dir / "size_bucket_report.csv")
    size_expected = {
        "bucket_type", "ic_mean", "ls_ann", "ls_sharpe", "ls_mdd",
        "n_periods", "coverage_ratio", "coverage_ratio_mean", "coverage_ratio_min",
        "membership_source", "point_in_time_flag", "as_of", "sample_period", "is_placeholder",
    }
    assert size_expected.issubset(set(size_bucket.columns))
    assert (size_bucket["point_in_time_flag"] == True).all()
    assert (size_bucket["membership_source"] == "pit:index_membership").all()

    rolling = pd.read_csv(out_dir / "rolling_oos.csv")
    rolling_expected = {
        "oos_ic_mean", "is_ic_mean", "oos_is_ratio", "oos_minus_is",
        "oos_over_abs_is", "degradation", "oos_t", "oos_nw_t",
        "window_train_len", "window_test_len", "step", "n_windows", "status", "status_msg",
        "is_placeholder",
    }
    assert rolling_expected.issubset(set(rolling.columns))
    assert rolling.loc[0, "status"] in {"ok", "insufficient_windows"}

    regime = pd.read_csv(out_dir / "regime_stability.csv")
    regime_expected = {
        "regime", "ls_mean", "ls_sharpe", "ls_mdd", "ic_mean",
        "regime_method", "q1", "q2", "n_periods", "is_placeholder",
    }
    assert regime_expected.issubset(set(regime.columns))

    sensitivity = pd.read_csv(out_dir / "param_sensitivity.csv")
    sensitivity_expected = {
        "param_name", "param_value", "ic_mean", "sharpe",
        "delta_ic", "delta_sharpe", "n_periods", "is_placeholder",
    }
    assert sensitivity_expected.issubset(set(sensitivity.columns))
    assert "base" in set(sensitivity["param_name"])


def test_fama_macbeth_boundary_insufficient_cross_section(tmp_path):
    idx = ["20220131", "20220228", "20220331"]
    cols = ["S001", "S002"]
    report = SimpleNamespace(
        factor_name="tiny_factor",
        factor_panel=pd.DataFrame([[1, 2], [2, 3], [3, 4]], index=idx, columns=cols),
        return_panel=pd.DataFrame([[0.01, 0.02], [0.03, 0.01], [0.0, 0.02]], index=idx, columns=cols),
        layer_ret=pd.DataFrame({"Q1": [0.01, 0.01, 0.01], "Q5": [0.02, 0.02, 0.02], "LS": [0.01, 0.01, 0.01]}, index=idx),
        ls_stats={"ls_annual_return": 0.1, "ls_sharpe": 1.0, "ls_max_drawdown": -0.1},
        turnover={"avg_turnover": 0.2},
    )
    out_dir = tmp_path / "advanced_diagnostics_small"
    run_advanced_diagnostics(report, output_dir=out_dir, n_groups=5, direction=1, periods_per_year=12)
    fm = pd.read_csv(out_dir / "fm_summary.csv")
    assert "fm_lambda_mean" in fm.columns
    # Too few observations for cross-sectional FM; should degrade gracefully.
    assert np.isnan(fm.loc[0, "fm_lambda_mean"])
    alpha = pd.read_csv(out_dir / "alpha_models.csv")
    assert set(alpha["model"]) == {"CAPM", "FF3"}

