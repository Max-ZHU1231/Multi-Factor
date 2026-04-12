"""
universes — 股票池管理模块。

快速使用
--------
from universes import UniverseLoader, DynamicUniverseBuilder, UniverseMembership

# 静态股票池（沪深300等）
syms = UniverseLoader.load("hs300")   # 返回 ts_code 列表
syms = UniverseLoader.load("all")     # 全部股票，返回 None

# 动态市值前N股票池
builder = DynamicUniverseBuilder(top_n=500, metric="total_mktcap")
snapshots = builder.build(start="20200101", end="20251231")
mem = UniverseMembership.from_builder(snapshots)
syms_today = mem.get_symbols("20230615")
"""

from universes.loader import UniverseLoader
from universes.builder import DynamicUniverseBuilder
from universes.membership import UniverseMembership, build_membership_from_config

__all__ = [
    "UniverseLoader",
    "DynamicUniverseBuilder",
    "UniverseMembership",
    "build_membership_from_config",
]
