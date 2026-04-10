"""
factor_framework.engine
========================
引擎子包：计算调度、面板构建、缓存层。

模块
----
cache.py         : CacheLayer     —— L1 内存 LRU + L2 Parquet 磁盘缓存
panel_builder.py : PanelBuilder   —— 带缓存的因子面板构建器（CacheLayer + FactorEngine 协调层）
"""

from factor_framework.engine.cache import CacheLayer
from factor_framework.engine.panel_builder import PanelBuilder

__all__ = ["CacheLayer", "PanelBuilder"]
