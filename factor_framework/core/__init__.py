"""
factor_framework.core
=====================
阶段二数据结构基础层。

模块
----
panel.py    : TimestampedPanel —— 带语义标注的因子/收益率面板（继承 pd.DataFrame）
returns.py  : ReturnPanel      —— 唯一收益率构建入口
"""

from factor_framework.core.panel import (
    TimestampedPanel,
    TimingAlignmentError,
    SemanticCompatibilityError,
)
from factor_framework.core.returns import ReturnPanel

__all__ = [
    "TimestampedPanel",
    "TimingAlignmentError",
    "SemanticCompatibilityError",
    "ReturnPanel",
]
