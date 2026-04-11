"""
tests/regression/test_factor_baselines.py
==========================================
v4.0 Regression baseline: IC/Sharpe snapshot tests.

Baseline source: artifacts/batch_results/factor_screening_summary.csv
                 (run 2026-04-11, commit 33de1cb)
Baseline status: baseline_v1_provisional

Configuration locked at:
  start=20200101, end=20251231, forward=21, n_groups=5
  periods_per_year=12, rf=0.02, cost_per_side=0.002
  standardize=rank, winsorize=True, neutralize=False
  resample_monthly=True

TOLERANCE: ±0.02 absolute for Mean IC, ±0.15 absolute for L/S Sharpe.
These are intentionally loose — tighten after baseline_v1 is confirmed stable.

NOTE: These tests read from the saved artifacts CSV. They do NOT re-run the
full pipeline (that belongs in tests/slow/). A missing artifacts file causes
the tests to be skipped (not fail), so CI remains green on fresh checkouts.
"""
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_CSV = ROOT / "artifacts" / "batch_results" / "factor_screening_summary.csv"

# ── Tolerances ───────────────────────────────────────────────────────────────
IC_TOL    = 0.02   # absolute Mean IC tolerance
SHARPE_TOL = 0.15  # absolute L/S Sharpe tolerance

# ── Provisional baseline (captured 2026-04-11, commit 33de1cb) ───────────────
# Only "excellent" (★★★) and "good" (★★) factors are pinned here.
# Factors graded ★ or below are excluded from regression gates.
BASELINE = {
    # factor_name: (mean_ic, ls_sharpe)
    "vwap_deviation":       (+0.117, 4.67),
    "price_strength":       (+0.063, 3.31),
    "amihud_illiquidity":   (+0.086, 1.70),
    "bid_ask_spread_proxy": (+0.093, None),   # Sharpe not pinned — volatile
    "vol_60d":              (+0.087, None),
    "vol_20d":              (+0.086, None),
    "size_log_mktcap":      (+0.072, None),
    "value_pb":             (+0.067, None),
    "value_ps_ttm":         (+0.050, None),
    "vol_price_corr":       (+0.038, None),
}


@pytest.fixture(scope="module")
def summary_df():
    if not BASELINE_CSV.exists():
        pytest.skip(f"Baseline CSV not found: {BASELINE_CSV} — skipping regression tests")
    df = pd.read_csv(BASELINE_CSV, index_col=0)
    return df


@pytest.mark.parametrize("factor_name,expected", [
    (k, v) for k, v in BASELINE.items()
])
def test_mean_ic_within_tolerance(summary_df, factor_name, expected):
    """Mean IC must not drift more than IC_TOL from the provisional baseline."""
    expected_ic, _ = expected
    if factor_name not in summary_df.index:
        pytest.skip(f"{factor_name} not found in summary CSV")
    actual_ic = float(summary_df.loc[factor_name, "mean_ic"])
    assert abs(actual_ic - expected_ic) <= IC_TOL, (
        f"{factor_name}: Mean IC drifted from baseline {expected_ic:+.4f} "
        f"to {actual_ic:+.4f} (tolerance ±{IC_TOL})"
    )


@pytest.mark.parametrize("factor_name,expected", [
    (k, v) for k, v in BASELINE.items() if v[1] is not None
])
def test_ls_sharpe_within_tolerance(summary_df, factor_name, expected):
    """L/S Sharpe must not drift more than SHARPE_TOL from provisional baseline."""
    _, expected_sharpe = expected
    sharpe_col = "ls_sharpe"
    if factor_name not in summary_df.index or sharpe_col not in summary_df.columns:
        pytest.skip(f"{factor_name}/{sharpe_col} not found in summary CSV")
    actual_sharpe = float(summary_df.loc[factor_name, sharpe_col])
    assert abs(actual_sharpe - expected_sharpe) <= SHARPE_TOL, (
        f"{factor_name}: L/S Sharpe drifted from baseline {expected_sharpe:.2f} "
        f"to {actual_sharpe:.2f} (tolerance ±{SHARPE_TOL})"
    )
