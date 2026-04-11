"""
factor_framework.factors.meta
==============================
FactorMeta —— 单因子的完整元数据描述（v4.0 Phase E1 扩展）。

设计原则
--------
- 纯 dataclass，无业务逻辑，只存元数据。
- 所有字段均有合理默认值，最小化创建成本。
- 构造后冻结（frozen=True），防止运行时意外修改。
- direction 语义：因子函数内部已处理负号，direction 统一为 +1
  （v3.0 决策 §6："所有内置因子函数内部已处理负号，direction 字段统一为 +1"）。
  Size 因子（size_log_mktcap、size_log_free_cap）例外：已在函数内取负，
  direction = +1 表示"返回值越大→预期收益越高"。

Phase E1 新增字段（v4.0）
--------------------------
inputs            : 因子依赖的原始数据列名列表（用于文档 + audit）
output_semantic   : 输出值语义描述（如 "higher=stronger_momentum"）
forward_safe      : 是否无前瞻偏差（True = 已验证，False = 有已知偏差）
version           : 因子实现版本字符串（如 "2.9.1"）
tags              : 自由标签列表（如 ["hfq", "log_return"]）
status            : 因子状态（"active"/"experimental"/"deprecated"）

字段说明（原有字段）
--------------------
name               : 因子唯一标识（与注册键相同）
display_name       : 中文展示名（用于报告/图表）
category           : 因子分类枚举（用于分组分析）
direction          : +1（统一）
warmup_days        : 计算该因子所需的最少历史天数（用于 trim_warmup）
description        : 简要说明（自由文本）
neutral_by_default : 是否默认参与市值 + 行业中性化。
                     True  → 大多数因子（默认值）
                     False → size_log_mktcap、size_log_free_cap
skip_neutralize_cols : 中性化回归时跳过的列名列表（如市值因子本身已是市值）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

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
# FactorStatus 枚举
# ═══════════════════════════════════════════════════════════════════════════════

class FactorStatus(str, Enum):
    """
    因子生命周期状态（Phase E1 新增）。

    ACTIVE       : 已验证、推荐使用
    EXPERIMENTAL : 正在测试，不保证稳定性
    DEPRECATED   : 已弃用，将在未来版本移除
    """
    ACTIVE       = "active"
    EXPERIMENTAL = "experimental"
    DEPRECATED   = "deprecated"


# ═══════════════════════════════════════════════════════════════════════════════
# FactorMeta dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FactorMeta:
    """
    单个因子的完整元数据描述（v4.0 Phase E1 扩展版）。

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
    skip_neutralize_cols : tuple[str, ...]
        中性化回归时跳过的解释变量列名。

    Phase E1 新增字段
    -----------------
    inputs : tuple[str, ...]
        因子计算所依赖的原始数据列名（如 ("收盘价", "复权因子")）。
        用于文档生成和 audit() 缺失报告。
        默认空元组 = 未声明。
    output_semantic : str
        输出值语义（如 "higher=stronger_momentum", "higher=cheaper_valuation"）。
        默认 "" = 未声明。
    forward_safe : bool | None
        是否已验证无前瞻偏差：
          True  = 已验证无前瞻
          False = 有已知前瞻偏差（需说明）
          None  = 未审核
        默认 None = 未审核。
    version : str
        因子实现版本（如 "2.9.1"）。默认 "" = 未声明。
    tags : tuple[str, ...]
        自由标签（如 ("hfq", "log_return", "reversal_adjusted")）。
        默认空元组。
    status : FactorStatus
        因子生命周期状态（active / experimental / deprecated）。
        默认 FactorStatus.ACTIVE。

    Notes
    -----
    frozen=True：构造后字段不可修改，防止运行时意外改写元数据。
    """
    # ── 原有必选字段 ─────────────────────────────────────────────────────────
    name                 : str
    fn                   : Callable[[pd.DataFrame], pd.Series]
    display_name         : str
    category             : FactorCategory

    # ── 原有可选字段（有默认值）──────────────────────────────────────────────
    direction            : int                  = +1
    warmup_days          : int                  = 252
    description          : str                  = ""
    neutral_by_default   : bool                 = True
    skip_neutralize_cols : tuple                = ()

    # ── Phase E1 新增字段（均有默认值，向后兼容）────────────────────────────
    inputs               : tuple                = ()
    output_semantic      : str                  = ""
    forward_safe         : Optional[bool]       = None    # None = 未审核
    version              : str                  = ""
    tags                 : tuple                = ()
    status               : FactorStatus         = FactorStatus.ACTIVE

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
        # 验证 status 类型
        if not isinstance(self.status, FactorStatus):
            raise TypeError(
                f"FactorMeta.status 必须为 FactorStatus 枚举，收到: {self.status!r}"
            )

    # ── 便捷属性 ────────────────────────────────────────────────────────────

    @property
    def is_long_short(self) -> bool:
        """direction == +1 时因子值越大 → 预期收益越高（做多）。"""
        return self.direction == +1

    @property
    def group(self) -> str:
        """返回分类名称字符串（与 category.value 相同），方便分组聚合。"""
        return self.category.value

    @property
    def is_active(self) -> bool:
        """是否处于 ACTIVE 状态。"""
        return self.status == FactorStatus.ACTIVE

    @property
    def missing_e1_fields(self) -> list[str]:
        """
        返回尚未填写的 Phase E1 元数据字段名列表。

        用于 registry.audit() 生成缺失报告。
        空列表 = E1 元数据完整。
        """
        missing = []
        if not self.inputs:
            missing.append("inputs")
        if not self.output_semantic:
            missing.append("output_semantic")
        if self.forward_safe is None:
            missing.append("forward_safe")
        if not self.version:
            missing.append("version")
        return missing

    def __repr__(self) -> str:
        return (
            f"FactorMeta(name={self.name!r}, category={self.category.value!r}, "
            f"direction={self.direction:+d}, warmup={self.warmup_days}d, "
            f"neutral={self.neutral_by_default}, status={self.status.value!r})"
        )
