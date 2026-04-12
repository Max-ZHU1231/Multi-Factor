"""
tests/integration/test_universe_lookahead.py
=============================================
Anti-look-ahead bias tests for the dynamic universe feature.

Critical requirements (from design doc §8.4):
  1. Snapshot must NOT use data after effective_date
  2. lag=0 must warn / require explicit opt-in
  3. Artificially inflated future mktcap must NOT cause early inclusion
  4. filter_panel must not expose future-universe membership
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from universes.builder import DynamicUniverseBuilder
from universes.membership import UniverseMembership


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_stock_csv(path: Path, rows: list[dict]) -> None:
    """Write a stock CSV with the required Chinese column names."""
    df = pd.DataFrame(rows)
    # Ensure required columns present
    for col, default in [("交易日", None), ("总市值（万元）", 0.0)]:
        if col not in df.columns:
            df[col] = default
    df.to_csv(path, index=False)


def _make_env(tmp_path, stocks: dict[str, list[dict]], cal_dates: list[str]):
    """
    Create a minimal test environment.

    Parameters
    ----------
    stocks     : { "000001.SZ": [{"交易日": "...", "总市值（万元）": ...}, ...] }
    cal_dates  : list of 'YYYYMMDD' strings (all treated as trading days)
    """
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir()
    for stem, rows in stocks.items():
        _write_stock_csv(stocks_dir / f"{stem}.csv", rows)

    cal_df = pd.DataFrame({
        "cal_date": cal_dates,
        "is_open":  ["1"] * len(cal_dates),
    })
    cal_path = tmp_path / "cal.csv"
    cal_df.to_csv(cal_path, index=False)

    return stocks_dir, cal_path


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: snapshot only uses decision_date data (not effective_date data)
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_uses_only_decision_date_data(tmp_path):
    """
    decision_date=20220630, effective_date=20220701 (lag=1).
    Stock A has mktcap=100 on 20220630 but 9999 on 20220701.
    Stock B has mktcap=200 on 20220630.

    Snapshot must rank B > A (using decision_date 20220630 data only).
    The future inflated value of A on 20220701 must NOT affect ranking.
    """
    stocks = {
        "000001.SZ": [
            {"交易日": "20220630", "总市值（万元）": 100.0},
            {"交易日": "20220701", "总市值（万元）": 9999.0},  # future — must be ignored
        ],
        "000002.SZ": [
            {"交易日": "20220630", "总市值（万元）": 200.0},
        ],
    }
    cal_dates = ["20220629", "20220630", "20220701", "20220704"]
    stocks_dir, cal_path = _make_env(tmp_path, stocks, cal_dates)

    builder = DynamicUniverseBuilder(
        stocks_dir=stocks_dir, trade_cal=cal_path, top_n=1
    )
    snap = builder.build_snapshot("20220630")

    # Only top-1: B (mktcap=200) must win, not A
    assert len(snap) == 1
    assert snap["symbol"].iloc[0] == "000002.SZ", (
        "000001.SZ should NOT rank above 000002.SZ — future mktcap leaked into snapshot!"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: lag=0 without explicit flag raises
# ─────────────────────────────────────────────────────────────────────────────

def test_lag_zero_raises_without_explicit_flag():
    with pytest.raises(ValueError, match="未来函数"):
        DynamicUniverseBuilder(effective_lag_days=0)


def test_lag_zero_with_explicit_flag_does_not_raise():
    b = DynamicUniverseBuilder(effective_lag_days=0, allow_lag_zero=True)
    assert b.effective_lag_days == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: membership.get_symbols never exposes future rebalance
# ─────────────────────────────────────────────────────────────────────────────

def test_membership_no_look_ahead():
    """
    Two rebalances:
      eff_date_1 = 20220701: {A, B}
      eff_date_2 = 20230101: {A, C}   (B dropped, C added)

    Querying on 20221231 (one day before eff_date_2) must return {A, B}, NOT {A, C}.
    """
    snapshots = pd.DataFrame([
        {"effective_date": "20220701", "decision_date": "20220630",
         "symbol": "000001.SZ", "rank": 1},
        {"effective_date": "20220701", "decision_date": "20220630",
         "symbol": "000002.SZ", "rank": 2},
        {"effective_date": "20230101", "decision_date": "20221230",
         "symbol": "000001.SZ", "rank": 1},
        {"effective_date": "20230101", "decision_date": "20221230",
         "symbol": "000003.SZ", "rank": 2},
    ])
    mem = UniverseMembership(snapshots)

    syms_before = mem.get_symbols("20221231")
    assert set(syms_before) == {"000001.SZ", "000002.SZ"}, (
        "On 20221231, universe should still be {A,B} — future rebalance must not leak!"
    )

    syms_after = mem.get_symbols("20230101")
    assert set(syms_after) == {"000001.SZ", "000003.SZ"}


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: filter_panel does not expose future membership
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_panel_no_future_membership():
    """
    Panel has a column for 000003.SZ which is only added in the 2023 rebalance.
    For rows dated in 2022, that column must be NaN (not visible).
    """
    snapshots = pd.DataFrame([
        {"effective_date": "20220701", "symbol": "000001.SZ", "rank": 1},
        {"effective_date": "20220701", "symbol": "000002.SZ", "rank": 2},
        {"effective_date": "20230101", "symbol": "000001.SZ", "rank": 1},
        {"effective_date": "20230101", "symbol": "000003.SZ", "rank": 2},
    ])
    mem = UniverseMembership(snapshots)

    panel = pd.DataFrame(
        {
            "000001.SZ": [1.0, 2.0, 3.0],
            "000002.SZ": [1.0, 2.0, 3.0],
            "000003.SZ": [1.0, 2.0, 3.0],
        },
        index=["20220701", "20221231", "20230101"],
    )
    filtered = mem.filter_panel(panel)

    # 2022 rows: 000003 not yet in universe → must be NaN
    assert pd.isna(filtered.loc["20220701",  "000003.SZ"]), "Future member must be NaN in 2022!"
    assert pd.isna(filtered.loc["20221231",  "000003.SZ"]), "Future member must be NaN in 2022!"
    # 2023 row: 000003 is now in universe → must have value
    assert not pd.isna(filtered.loc["20230101", "000003.SZ"])
    # 2023 row: 000002 dropped → must be NaN
    assert pd.isna(filtered.loc["20230101", "000002.SZ"])


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: artificially inflated future mktcap does not cause early inclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_artificially_inflated_future_mktcap_not_included_early(tmp_path):
    """
    Stock C has tiny mktcap on decision_date but enormous mktcap the day after.
    It must NOT appear in the snapshot built on decision_date.
    """
    stocks = {
        "000001.SZ": [{"交易日": "20220630", "总市值（万元）": 500.0}],
        "000002.SZ": [{"交易日": "20220630", "总市值（万元）": 400.0}],
        "000003.SZ": [  # C: tiny on decision_date, huge on next day
            {"交易日": "20220630", "总市值（万元）": 1.0},    # decision_date
            {"交易日": "20220701", "总市值（万元）": 99999.0}, # future — must not matter
        ],
    }
    cal_dates = ["20220629", "20220630", "20220701", "20220704"]
    stocks_dir, cal_path = _make_env(tmp_path, stocks, cal_dates)

    builder = DynamicUniverseBuilder(
        stocks_dir=stocks_dir, trade_cal=cal_path, top_n=2
    )
    snap = builder.build_snapshot("20220630")

    # Top-2 by decision_date mktcap: 000001, 000002 — NOT 000003
    included = set(snap["symbol"].values)
    assert "000003.SZ" not in included, (
        "000003.SZ has tiny mktcap on decision_date — "
        "its future value must NOT cause early inclusion!"
    )
    assert included == {"000001.SZ", "000002.SZ"}
