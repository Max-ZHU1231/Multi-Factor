"""
factor_framework.factors.momentum
===================================
动量 / 反转类内置因子函数。

函数签名统一：(df: pd.DataFrame) -> pd.Series
df 为单只股票日频 DataFrame（已清洗，按交易日升序）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_framework.operators import ts_max, ts_mean
from factor_framework.factor_zoo import _hfq_close   # 复用后复权辅助函数


# ── re-export from factor_zoo so both import paths work ─────────────────────
from factor_framework.factor_zoo import (   # noqa: F401
    momentum_12_1,
    momentum_6_1,
    momentum_1m,
    momentum_52w_high,
    reversal_1w,
    reversal_1m,
)

__all__ = [
    "momentum_12_1",
    "momentum_6_1",
    "momentum_1m",
    "momentum_52w_high",
    "reversal_1w",
    "reversal_1m",
]
