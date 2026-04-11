"""
factor_framework.transform
===========================
Transform layer (v4.0): cross-sectional transformations composable as a pipeline.

Public API
----------
    TransformPipeline   — composable winsorize / neutralize / standardize pipeline
    neutralize_regression       — OLS/WLS factor neutralization
    neutralize_industry_zscore  — industry z-score neutralization
    orthogonalize               — factor orthogonalization (residual IC helper)
"""

from factor_framework.transform.transform import TransformPipeline
from factor_framework.transform.neutralize import (
    neutralize_regression,
    neutralize_industry_zscore,
    orthogonalize,
)

__all__ = [
    "TransformPipeline",
    "neutralize_regression",
    "neutralize_industry_zscore",
    "orthogonalize",
]
