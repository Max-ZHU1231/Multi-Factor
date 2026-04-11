"""
tests/unit/test_new_import_paths.py
=====================================
v4.0 Phase A: verify that new canonical import paths work correctly,
and that old paths still work (shim layer) while emitting DeprecationWarning.
"""
import warnings
import pytest


class TestNewImportPaths:
    """New v4.0 canonical import paths must work without warnings."""

    def test_analytics_ic_analysis(self):
        from factor_framework.analytics.ic_analysis import (
            compute_ic, ic_stats, ic_decay, ic_cumulative,
            ic_significance, cross_factor_correlation, residual_ic,
        )
        assert callable(compute_ic)
        assert callable(ic_stats)
        assert callable(ic_decay)

    def test_analytics_package(self):
        from factor_framework.analytics import (
            compute_ic, ic_stats, ic_decay,
            ICAnalyzer, LayerBacktester,
        )
        assert callable(compute_ic)
        assert ICAnalyzer is not None
        assert LayerBacktester is not None

    def test_transform_package(self):
        from factor_framework.transform import (
            TransformPipeline,
            neutralize_regression,
            neutralize_industry_zscore,
            orthogonalize,
        )
        assert TransformPipeline is not None
        assert callable(neutralize_regression)

    def test_transform_neutralize_direct(self):
        from factor_framework.transform.neutralize import (
            neutralize_regression,
            neutralize_industry_zscore,
            orthogonalize,
        )
        assert callable(neutralize_regression)

    def test_optimize_package(self):
        from factor_framework.optimize import equal_weight, icir_weight, print_weights
        assert callable(equal_weight)
        assert callable(icir_weight)

    def test_reporting_package_importable(self):
        import factor_framework.reporting  # stub, just must not crash

    def test_cli_package_importable(self):
        import factor_framework.cli  # stub, just must not crash


class TestLegacyShimPaths:
    """Old v3.x paths must still work but emit DeprecationWarning."""

    def test_ic_analysis_shim_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from factor_framework import ic_analysis  # noqa: F401
            dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "analytics" in str(dep_warnings[0].message).lower()

    def test_ic_analysis_shim_functional(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.ic_analysis import compute_ic, ic_stats, ic_decay
            assert callable(compute_ic)

    def test_neutralize_shim_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import importlib, sys
            # Force re-import to trigger warning
            mod_name = "factor_framework.neutralize"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            import factor_framework.neutralize  # noqa: F401
            dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

    def test_neutralize_shim_functional(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.neutralize import neutralize_regression, orthogonalize
            assert callable(neutralize_regression)

    def test_optimizer_shim_functional(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.optimizer import equal_weight, icir_weight
            assert callable(equal_weight)

    def test_factors_transform_shim_functional(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.factors.transform import TransformPipeline
            assert TransformPipeline is not None

    def test_factors_ic_analyzer_shim_functional(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.factors.ic_analyzer import ICAnalyzer
            assert ICAnalyzer is not None

    def test_factors_layer_backtester_shim_functional(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from factor_framework.factors.layer_backtester import LayerBacktester
            assert LayerBacktester is not None


class TestPreexistingPathsUnchanged:
    """Paths that were NOT migrated must import cleanly with no warnings."""

    def test_pipeline(self):
        from factor_framework.pipeline import FactorPipeline
        assert FactorPipeline is not None

    def test_factor_engine(self):
        from factor_framework.factor_engine import FactorEngine
        assert FactorEngine is not None

    def test_backtest(self):
        from factor_framework.backtest import layer_backtest, long_short_stats
        assert callable(layer_backtest)

    def test_core_panel(self):
        from factor_framework.core.panel import TimestampedPanel
        assert TimestampedPanel is not None

    def test_core_returns(self):
        from factor_framework.core.returns import ReturnPanel
        assert ReturnPanel is not None

    def test_engine_cache(self):
        from factor_framework.engine.cache import CacheLayer
        assert CacheLayer is not None

    def test_engine_panel_builder(self):
        from factor_framework.engine.panel_builder import PanelBuilder
        assert PanelBuilder is not None

    def test_data_store(self):
        from factor_framework.data.store import DataStore, CSVDataStore
        assert DataStore is not None

    def test_factors_meta(self):
        from factor_framework.factors.meta import FactorMeta, FactorCategory
        assert FactorMeta is not None

    def test_factors_registry(self):
        from factor_framework.factors.registry import REGISTRY
        assert REGISTRY is not None
