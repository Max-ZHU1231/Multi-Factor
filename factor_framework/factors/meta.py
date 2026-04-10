"""
factor_framework.factors.meta
==============================
FactorMeta —— 单因子的完整元数据描述（v3.0 规范 §2.1）。

设计原则
--------
- 纯 dataclass，无业务逻辑，只存元数据。
- 所有字段均有合理默认值，最小化创建成本。
- 构造后冻结（frozen=True），防止运行时意外修改。
- direction 语义：因子函数内部已处理负号，direction 统一为 +1
  （v3.0 决策 §6："所有内置因子函数内部已处理负号，direction 字段统一为 +1"）。
  Size 因子（size_log_mktcap、size_log_free_cap）例外：已在函数内取负，
  direction = +1 表示"返回值越大→预期收益越高"。

字段说明
--------
name               : 因子唯一标识（与注册键相同）
display_name       : 中文展示名（用于报告/图表）
category           : 因子分类枚举（用于分组分析）
direction          : +1（统一）
warmup_days        : 计算该因子所需的最少历史天数（用于 trim_warmup）
description        : 简要说明（自由文本）
neutral_by_default : 是否默认参与市值 + 行业中性化。
                     True  → 大多数因子（默认值）
                     False → size_log_mktcap、size_log_free_cap
                     （v3.0 决策 §8：规模因子不做中性化，本身即暴露规模）
skip_neutralize_cols : 中性化回归时跳过的列名列表（如市值因子本身已是市值）
                       （v3.0 决策 §3：支持细粒度排除）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 因子分类枚举
# ═══════════════════════════════════════════════════════════════════════════════

class FactorCategory(str, Enum):
    """
    因子大类分类。

    继承 str 使其在字符串比较和 JSON 序列化时行为自然
    （``meta.category == "momentum"`` 直接成立）。
    """
    MOMENTUM    = "momentum"       # 动量因子
    REVERSAL    = "reversal"       # 反转因子
    VOLATILITY  = "volatility"     # 波动率因子
    VALUE       = "value"          # 估值因子
    SIZE        = "size"           # 规模因子
    VOLUME      = "volume"         # 量价因子
    LIQUIDITY   = "liquidity"      # 流动性质量因子
    TECHNICAL   = "technical"      # 技术分析因子
    COMPOSITE   = "composite"      # 合成因子（多因子加权）
    CUSTOM      = "custom"         # 用户自定义


# ═══════════════════════════════════════════════════════════════════════════════
# FactorMeta dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FactorMeta:
    """
    单个因子的完整元数据描述。

    Parameters
    ----------
    name : str
        因子唯一键（与 FactorRegistry / BUILTIN_FACTORS 中的注册名相同）。
    fn : Callable[[pd.DataFrame], pd.Series]
        因子计算函数，签名为 (df: pd.DataFrame) -> pd.Series。
    display_name : str
        中文展示名，用于报告标题和图表标签。
    category : FactorCategory
        因子大类分类。
    direction : int
        方向标记，统一为 +1（v3.0 决策：函数内部已处理符号）。
    warmup_days : int
        计算该因子所需的最少历史天数。用于 trim_warmup() 自动截断热身期。
    description : str
        因子的简要说明（可多行）。
    neutral_by_default : bool
        是否默认参与市值 + 行业中性化。
        True（默认）→ 大多数因子。
        False        → size_log_mktcap、size_log_free_cap（规模因子不做中性化）。
    skip_neutralize_cols : tuple[str, ...]
        中性化回归时在设计矩阵中跳过的解释变量列名。
        例如 size_log_mktcap 已是对数市值本身，再用市值中性化无意义。
        空元组 = 不跳过任何列（默认）。

    Notes
    -----
    frozen=True：构造后字段不可修改，防止运行时意外改写元数据。
    fn 是 Callable，理论上不可哈希，但 frozen dataclass 的哈希方法仅
    对所有字段执行 hash()；因此 fn 必须可哈希（普通函数默认可哈希）。
    """
    name                 : str
    fn                   : Callable[[pd.DataFrame], pd.Series]
    display_name         : str
    category             : FactorCategory
    direction            : int                  = +1
    warmup_days          : int                  = 252
    description          : str                  = ""
    neutral_by_default   : bool                 = True
    skip_neutralize_cols : tuple                = ()

    def __post_init__(self) -> None:
        # 验证 direction 取值
        if self.direction not in (+1, -1):
            raise ValueError(
                f"FactorMeta.direction 必须为 +1 或 -1，收到: {self.direction!r}"
            )
        # 验证 warmup_days 非负
        if self.warmup_days < 0:
            raise ValueError(
                f"FactorMeta.warmup_days 必须 >= 0，收到: {self.warmup_days!r}"
            )
        # 验证 category 类型
        if not isinstance(self.category, FactorCategory):
            raise TypeError(
                f"FactorMeta.category 必须为 FactorCategory 枚举，收到: {self.category!r}"
            )

    # ── 便捷属性 ────────────────────────────────────────────────────────────

    @property
    def is_long_short(self) -> bool:
        """direction == +1 时因子值越大 → 预期收益越高（做多）。
        由于所有内置因子 direction 均为 +1，此属性始终为 True。
        保留供未来扩展（如 direction=-1 的遗留因子）。
        """
        return self.direction == +1

    @property
    def group(self) -> str:
        """返回分类名称字符串（与 category.value 相同），方便分组聚合。"""
        return self.category.value

    def __repr__(self) -> str:
        return (
            f"FactorMeta(name={self.name!r}, category={self.category.value!r}, "
            f"direction={self.direction:+d}, warmup={self.warmup_days}d, "
            f"neutral={self.neutral_by_default})"
        )
