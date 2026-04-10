"""
factor_framework.factors
========================
Phase 3: factor metadata layer.

Sub-modules
-----------
meta.py      : FactorMeta, FactorCategory
registry.py  : FactorRegistry, _CompatDict, REGISTRY

Usage
-----
    from factor_framework.factors.meta import FactorMeta, FactorCategory
    from factor_framework.factors.registry import REGISTRY, FactorRegistry

    # After factor_zoo has been imported, REGISTRY is fully populated:
    meta = REGISTRY.get("momentum_12_1")

Note: This __init__.py intentionally imports nothing at package-init time.
Doing so would create a circular initialisation cycle because factor_zoo.py
imports factor_framework.factors.meta/registry during its own module-level
execution, which would re-trigger this __init__ before it finishes.
"""
# No imports here -- see docstring above.


