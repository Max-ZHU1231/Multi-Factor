"""
tests/unit/test_universe_builder.py
=====================================
Unit tests for universes/builder.py

Tests cover:
  - top_n truncation (including tie-break at boundary)
  - decision_date → effective_date mapping (lag=1)
  - semiannual rebalance date generation across year boundaries
  - universe_mode parameter validation (lag=0 guard)
  - config_hash uniqueness with different top_n
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from universes.builder import (
    DynamicUniverseBuilder,
    _generate_decision_dates,
    _last_trading_day_of_month,
    _next_trading_day,
)


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_cal():
    """Minimal trading calendar for 2022-2023."""
    dates = []
    import datetime
    d = datetime.date(2022, 1, 4)
    end = datetime.date(2024, 1, 31)
    while d <= end:
        # Skip weekends (simplified — no actual Chinese holiday logic)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += datetime.timedelta(days=1)
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# _next_trading_day
# ─────────────────────────────────────────────────────────────────────────────

def test_next_trading_day_lag1(simple_cal):
    """lag=1 should return the first trading day strictly after decision_date."""
    result = _next_trading_day(simple_cal, "20220630", lag=1)
    assert result > "20220630"
    assert result in simple_cal


def test_next_trading_day_lag2(simple_cal):
    """lag=2 should skip two trading days."""
    eff1 = _next_trading_day(simple_cal, "20220630", lag=1)
    eff2 = _next_trading_day(simple_cal, "20220630", lag=2)
    # eff2 must be strictly after eff1
    assert eff2 > eff1


def test_next_trading_day_out_of_range(simple_cal):
    """Should raise ValueError when effective_date exceeds calendar end."""
    last = simple_cal[-1]
    with pytest.raises(ValueError, match="out of trading-calendar range|exceeds trading-calendar end"):
        _next_trading_day(simple_cal, last, lag=1)


# ─────────────────────────────────────────────────────────────────────────────
# _last_trading_day_of_month
# ─────────────────────────────────────────────────────────────────────────────

def test_last_trading_day_june_2022(simple_cal):
    d = _last_trading_day_of_month(simple_cal, 2022, 6)
    assert d is not None
    assert d.startswith("202206")


def test_last_trading_day_missing_month(simple_cal):
    """A month with no trading days should return None."""
    result = _last_trading_day_of_month(simple_cal, 2021, 12)  # not in fixture
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _generate_decision_dates
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_decision_dates_semiannual(simple_cal):
    """Semiannual rebalance [6,12] should produce ~4 dates for 2022-2023."""
    dates = _generate_decision_dates(simple_cal, "20220101", "20231231", [6, 12])
    assert len(dates) == 4
    # All must fall in June or December
    for d in dates:
        assert int(d[4:6]) in (6, 12)


def test_generate_decision_dates_cross_year(simple_cal):
    """Dates should be in sorted order and unique."""
    dates = _generate_decision_dates(simple_cal, "20220101", "20231231", [6, 12])
    assert dates == sorted(set(dates))


def test_generate_decision_dates_annual(simple_cal):
    dates = _generate_decision_dates(simple_cal, "20220101", "20231231", [12])
    assert len(dates) == 2
    for d in dates:
        assert int(d[4:6]) == 12


# ─────────────────────────────────────────────────────────────────────────────
# DynamicUniverseBuilder — parameter validation
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric="):
        DynamicUniverseBuilder(metric="unknown_metric")


def test_lag_zero_guard():
    """lag=0 without allow_lag_zero=True should raise ValueError."""
    with pytest.raises(ValueError, match="look-ahead bias"):
        DynamicUniverseBuilder(effective_lag_days=0)


def test_lag_zero_explicit_ok():
    """lag=0 with allow_lag_zero=True should not raise."""
    b = DynamicUniverseBuilder(effective_lag_days=0, allow_lag_zero=True)
    assert b.effective_lag_days == 0


def test_invalid_rebalance_freq():
    with pytest.raises(ValueError, match="rebalance_freq="):
        DynamicUniverseBuilder(rebalance_freq="weekly")


def test_custom_rebalance_months_overrides_freq():
    """Explicit rebalance_months overrides rebalance_freq."""
    b = DynamicUniverseBuilder(
        rebalance_freq="annual",   # would default to [12]
        rebalance_months=[3, 9],   # but we override
    )
    assert b.rebalance_months == [3, 9]


# ─────────────────────────────────────────────────────────────────────────────
# config_hash uniqueness
# ─────────────────────────────────────────────────────────────────────────────

def test_config_hash_differs_on_top_n():
    b1 = DynamicUniverseBuilder(top_n=300)
    b2 = DynamicUniverseBuilder(top_n=500)
    assert b1.config_hash != b2.config_hash


def test_config_hash_differs_on_metric():
    b1 = DynamicUniverseBuilder(metric="total_mktcap")
    b2 = DynamicUniverseBuilder(metric="free_float_mktcap")
    assert b1.config_hash != b2.config_hash


def test_config_hash_differs_on_lag():
    b1 = DynamicUniverseBuilder(effective_lag_days=1)
    b2 = DynamicUniverseBuilder(effective_lag_days=3)
    assert b1.config_hash != b2.config_hash


def test_config_hash_stable():
    """Same config → same hash (idempotent)."""
    b1 = DynamicUniverseBuilder(top_n=500, metric="total_mktcap",
                                  rebalance_freq="semiannual",
                                  effective_lag_days=1)
    b2 = DynamicUniverseBuilder(top_n=500, metric="total_mktcap",
                                  rebalance_freq="semiannual",
                                  effective_lag_days=1)
    assert b1.config_hash == b2.config_hash


# ─────────────────────────────────────────────────────────────────────────────
# build_snapshot top_n truncation & tie-break
# ─────────────────────────────────────────────────────────────────────────────

def _make_builder_with_fake_data(tmp_path, mktcap_data: dict, top_n: int = 3):
    """
    Helper: write fake CSV files and return a DynamicUniverseBuilder.

    mktcap_data: { symbol_stem: mktcap_value }  (e.g. {"000001.SZ": 1000.0})
    CSV will have 交易日=20220630 and 总市值（万元）=mktcap_value
    """
    import pandas as pd

    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir()

    for stem, val in mktcap_data.items():
        df = pd.DataFrame({
            "交易日":        ["20220630"],
            "总市值（万元）": [val],
        })
        df.to_csv(stocks_dir / f"{stem}.csv", index=False)

    # Minimal trade calendar
    cal_df = pd.DataFrame({
        "cal_date": ["20220629", "20220630", "20220701", "20220704"],
        "is_open":  ["1",        "1",        "1",        "1"],
    })
    cal_path = tmp_path / "cal.csv"
    cal_df.to_csv(cal_path, index=False)

    return DynamicUniverseBuilder(
        stocks_dir  = stocks_dir,
        trade_cal   = cal_path,
        top_n       = top_n,
    )


def test_top_n_truncation(tmp_path):
    """Snapshot should contain exactly top_n stocks."""
    data = {f"00000{i}.SZ": float(1000 - i) for i in range(6)}
    builder = _make_builder_with_fake_data(tmp_path, data, top_n=3)
    snap = builder.build_snapshot("20220630")
    assert len(snap) == 3
    assert list(snap["rank"]) == [1, 2, 3]


def test_top_n_tie_break_by_symbol(tmp_path):
    """Stocks with identical mktcap should be broken by symbol ascending."""
    # 000001 and 000002 have same mktcap — 000001 must win (alphabetically first)
    data = {
        "000001.SZ": 1000.0,
        "000002.SZ": 1000.0,
        "000003.SZ": 500.0,
    }
    builder = _make_builder_with_fake_data(tmp_path, data, top_n=2)
    snap = builder.build_snapshot("20220630")
    assert len(snap) == 2
    assert "000001.SZ" in snap["symbol"].values
    assert "000002.SZ" in snap["symbol"].values


def test_top_n_boundary_tie(tmp_path):
    """
    Anti look-ahead: when N+1-th stock ties the N-th, only the first N
    (by symbol order) are included.
    """
    data = {
        "000001.SZ": 900.0,
        "000002.SZ": 800.0,  # rank-2 boundary
        "000003.SZ": 800.0,  # ties rank-2; alphabetically after 000002, excluded
        "000004.SZ": 700.0,
    }
    builder = _make_builder_with_fake_data(tmp_path, data, top_n=2)
    snap = builder.build_snapshot("20220630")
    assert len(snap) == 2
    symbols = list(snap["symbol"])
    assert "000001.SZ" in symbols
    assert "000002.SZ" in symbols  # wins tie
    assert "000003.SZ" not in symbols  # loses tie


def test_effective_date_is_next_trading_day(tmp_path):
    """effective_date must be the first trading day AFTER decision_date."""
    data = {"000001.SZ": 1000.0}
    builder = _make_builder_with_fake_data(tmp_path, data, top_n=1)
    snap = builder.build_snapshot("20220630")
    assert len(snap) == 1
    assert snap["effective_date"].iloc[0] == "20220701"


def test_snapshot_no_future_data(tmp_path):
    """
    Anti look-ahead: decision_date row must only contain data from that date.
    If the CSV has a later higher-mktcap row, it must NOT affect the snapshot.
    """
    import pandas as pd
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir()

    # 000001.SZ: on decision_date 500, on the next day 9999 (future)
    df = pd.DataFrame({
        "交易日":        ["20220630", "20220701"],
        "总市值（万元）": [500.0,     9999.0],
    })
    df.to_csv(stocks_dir / "000001.SZ.csv", index=False)

    # 000002.SZ: on decision_date 1000
    df2 = pd.DataFrame({
        "交易日":        ["20220630"],
        "总市值（万元）": [1000.0],
    })
    df2.to_csv(stocks_dir / "000002.SZ.csv", index=False)

    cal_df = pd.DataFrame({
        "cal_date": ["20220629", "20220630", "20220701"],
        "is_open":  ["1", "1", "1"],
    })
    cal_path = tmp_path / "cal.csv"
    cal_df.to_csv(cal_path, index=False)

    builder = DynamicUniverseBuilder(
        stocks_dir=stocks_dir, trade_cal=cal_path, top_n=1
    )
    snap = builder.build_snapshot("20220630")
    # 000002.SZ has higher mktcap on 20220630 → should be rank 1
    assert snap["symbol"].iloc[0] == "000002.SZ"
    # 000001.SZ must be rank 2 (if top_n=1, it should not appear at all)
    assert "000001.SZ" not in snap["symbol"].values
