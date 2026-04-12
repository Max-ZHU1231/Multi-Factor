"""
tests/unit/test_universe_membership.py
========================================
Unit tests for universes/membership.py

Tests cover:
  - get_symbols() returns correct members per date
  - dates before earliest effective_date return None (strict=False) or raise (strict=True)
  - is_member() correctness
  - filter_panel() zero look-ahead: pool-out columns become NaN
  - rebalance_dates() and summary() correctness
  - build_date_symbol_map() batch correctness
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from universes.membership import UniverseMembership


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_snapshots():
    """
    Two rebalance points:
      effective_date=20220701: [A, B, C]
      effective_date=20230101: [A, B, D]  (C dropped, D added)
    """
    return pd.DataFrame([
        {"effective_date": "20220701", "decision_date": "20220630",
         "symbol": "000001.SZ", "rank": 1, "mktcap_value": 1000.0},
        {"effective_date": "20220701", "decision_date": "20220630",
         "symbol": "000002.SZ", "rank": 2, "mktcap_value": 900.0},
        {"effective_date": "20220701", "decision_date": "20220630",
         "symbol": "000003.SZ", "rank": 3, "mktcap_value": 800.0},
        # 2nd rebalance: 000003 dropped, 000004 added
        {"effective_date": "20230101", "decision_date": "20221230",
         "symbol": "000001.SZ", "rank": 1, "mktcap_value": 1100.0},
        {"effective_date": "20230101", "decision_date": "20221230",
         "symbol": "000002.SZ", "rank": 2, "mktcap_value": 950.0},
        {"effective_date": "20230101", "decision_date": "20221230",
         "symbol": "000004.SZ", "rank": 3, "mktcap_value": 820.0},
    ])


@pytest.fixture
def mem(simple_snapshots):
    return UniverseMembership(simple_snapshots)


# ─────────────────────────────────────────────────────────────────────────────
# get_symbols
# ─────────────────────────────────────────────────────────────────────────────

def test_get_symbols_before_first_effective(mem):
    """Date before all effective_dates should return None (strict=False)."""
    result = mem.get_symbols("20220101")
    assert result is None


def test_get_symbols_strict_raises(simple_snapshots):
    mem_strict = UniverseMembership(simple_snapshots, strict=True)
    with pytest.raises(ValueError, match="earlier than the first snapshot"):
        mem_strict.get_symbols("20220101")


def test_get_symbols_on_effective_date(mem):
    """Date == effective_date should use that snapshot."""
    syms = mem.get_symbols("20220701")
    assert set(syms) == {"000001.SZ", "000002.SZ", "000003.SZ"}


def test_get_symbols_between_rebalances(mem):
    """Date between two effective_dates should use the earlier snapshot."""
    syms = mem.get_symbols("20220901")
    assert set(syms) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert "000004.SZ" not in syms


def test_get_symbols_after_second_rebalance(mem):
    syms = mem.get_symbols("20230601")
    assert set(syms) == {"000001.SZ", "000002.SZ", "000004.SZ"}
    assert "000003.SZ" not in syms


def test_get_symbols_on_second_effective_date(mem):
    syms = mem.get_symbols("20230101")
    assert set(syms) == {"000001.SZ", "000002.SZ", "000004.SZ"}


# ─────────────────────────────────────────────────────────────────────────────
# is_member
# ─────────────────────────────────────────────────────────────────────────────

def test_is_member_true(mem):
    assert mem.is_member("000001.SZ", "20220701") is True


def test_is_member_false_dropped(mem):
    """000003 drops after 20230101."""
    assert mem.is_member("000003.SZ", "20230601") is False


def test_is_member_before_first_effective(mem):
    assert mem.is_member("000001.SZ", "20220101") is False


# ─────────────────────────────────────────────────────────────────────────────
# filter_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_panel_zeros_out_of_universe(mem):
    """
    filter_panel should NaN columns not in the universe for each row's date.
    """
    panel = pd.DataFrame(
        {
            "000001.SZ": [1.0, 2.0],
            "000003.SZ": [3.0, 4.0],
            "000004.SZ": [5.0, 6.0],
        },
        index=["20220701", "20230101"],  # first row: eff1, second: eff2
    )
    filtered = mem.filter_panel(panel)

    # 20220701: universe = {000001, 000002, 000003}
    #   → 000004 should be NaN
    assert pd.isna(filtered.loc["20220701", "000004.SZ"])
    assert filtered.loc["20220701", "000001.SZ"] == 1.0
    assert filtered.loc["20220701", "000003.SZ"] == 3.0

    # 20230101: universe = {000001, 000002, 000004}
    #   → 000003 should be NaN
    assert pd.isna(filtered.loc["20230101", "000003.SZ"])
    assert filtered.loc["20230101", "000001.SZ"] == 2.0
    assert filtered.loc["20230101", "000004.SZ"] == 6.0


def test_filter_panel_before_first_effective(mem):
    """Rows before earliest effective_date should be all NaN."""
    panel = pd.DataFrame(
        {"000001.SZ": [1.0], "000002.SZ": [2.0]},
        index=["20220101"],
    )
    filtered = mem.filter_panel(panel)
    assert filtered.isna().all(axis=None)


# ─────────────────────────────────────────────────────────────────────────────
# rebalance_dates / get_schedule / summary
# ─────────────────────────────────────────────────────────────────────────────

def test_rebalance_dates(mem):
    assert mem.rebalance_dates() == ["20220701", "20230101"]


def test_get_schedule_structure(mem):
    schedule = mem.get_schedule()
    assert set(schedule.keys()) == {"20220701", "20230101"}
    assert set(schedule["20220701"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}


def test_summary_shape(mem):
    df = mem.summary()
    assert len(df) == 2
    assert "effective_date" in df.columns
    assert "n_symbols" in df.columns
    assert list(df["n_symbols"]) == [3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# build_date_symbol_map
# ─────────────────────────────────────────────────────────────────────────────

def test_build_date_symbol_map(mem):
    dates = ["20220101", "20220801", "20230201"]
    mapping = mem.build_date_symbol_map(dates)
    assert mapping["20220101"] is None  # before first effective
    assert set(mapping["20220801"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert set(mapping["20230201"]) == {"000001.SZ", "000002.SZ", "000004.SZ"}


# ─────────────────────────────────────────────────────────────────────────────
# missing columns guard
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_symbol_column_raises():
    bad_df = pd.DataFrame({"effective_date": ["20220701"]})
    with pytest.raises(ValueError, match="missing required columns"):
        UniverseMembership(bad_df)


def test_missing_effective_date_column_raises():
    bad_df = pd.DataFrame({"symbol": ["000001.SZ"]})
    with pytest.raises(ValueError, match="missing required columns"):
        UniverseMembership(bad_df)


# ─────────────────────────────────────────────────────────────────────────────
# repr
# ─────────────────────────────────────────────────────────────────────────────

def test_repr(mem):
    r = repr(mem)
    assert "UniverseMembership" in r
    assert "20220701" in r
