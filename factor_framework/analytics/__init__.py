"""
factor_framework.analytics
===========================
Analytics layer (v4.0): IC calculation, decay analysis, significance testing,
and structured wrappers for IC analysis and layer backtesting.

Public API
----------
IC core functions (from analytics.ic_analysis):
    compute_ic          — per-period IC Series
    ic_stats            — Mean IC / Std / ICIR / win-rate / t-stat
    ic_decay            — multi-forward decay DataFrame
    ic_cumulative       — cumulative IC curve
    ic_significance     — Newey-West t-test
    cross_factor_correlation — pairwise factor IC correlation
    incremental_ic      — orthogonalised incremental IC

Structured wrappers:
    ICAnalyzer          — single-object IC analysis result container
    LayerBacktester     — single-object layer backtest result container
"""

from factor_framework.analytics.ic_analysis import (
    compute_ic,
    ic_stats,
    ic_decay,
    ic_cumulative,
    ic_significance,
    cross_factor_correlation,
    residual_ic,
)
from factor_framework.analytics.ic_analyzer import ICAnalyzer
from factor_framework.analytics.layer_backtester import LayerBacktester

__all__ = [
    # ic_analysis
    "compute_ic",
    "ic_stats",
    "ic_decay",
    "ic_cumulative",
    "ic_significance",
    "cross_factor_correlation",
    "residual_ic",
    # wrappers
    "ICAnalyzer",
    "LayerBacktester",
]
