"""
optimizer.py  [v4.0 COMPATIBILITY SHIM]
==========================================
This module has moved to factor_framework.optimize (v4.0).
The old path will be removed in v4.2. Please update your imports:

    Old: from factor_framework.optimizer import equal_weight, icir_weight
    New: from factor_framework.optimize import equal_weight, icir_weight
"""
import warnings as _warnings
_warnings.warn(
    "factor_framework.optimizer has moved to factor_framework.optimize. "
    "The old path will be removed in v4.2.",
    DeprecationWarning,
    stacklevel=2,
)
from factor_framework.optimize.optimizer import *  # noqa: F401,F403
from factor_framework.optimize.optimizer import equal_weight, icir_weight, print_weights