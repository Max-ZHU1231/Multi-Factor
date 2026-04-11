"""
factor_framework.optimize
==========================
Optimize layer (v4.0): factor composite weighting and portfolio construction.

Public API
----------
    equal_weight    — equal-weight composite factor
    icir_weight     — ICIR-weighted composite factor
    print_weights   — formatted weight display utility
"""

from factor_framework.optimize.optimizer import equal_weight, icir_weight, print_weights

__all__ = [
    "equal_weight",
    "icir_weight",
    "print_weights",
]
