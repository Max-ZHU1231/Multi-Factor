"""
factor_framework.core.panel
============================
TimestampedPanel —— 带语义标注的面板数据结构。

设计原则（v3.0 规范）
--------------------
- 继承 pd.DataFrame（不是组合），保证 .loc/.iloc/.index/.columns 等调用零修改。
- 通过 pandas 的 _metadata 机制，将语义字段附加到实例，在 slice/copy 等操作后保留。
- 提供 shift_to_t1()、align_with()、trim_warmup()、assert_valid() 四个核心方法。
- 两个专用异常类：TimingAlignmentError、SemanticCompatibilityError。

语义字段（_metadata 注册）
--------------------------
semantic         : str  —— "factor_observation" | "forward_return" | "price"
is_t1_shifted    : bool —— 是否已做 T+1 行位移（factor_observation 侧）
forward_days     : int | None —— 仅对 forward_return 有效，表示预测期
price_basis      : str | None —— "hfq" | "qfq" | "raw" | None
factor_name      : str | None —— 因子名称，用于调试输出
warmup_trimmed   : bool —— 是否已截断 warm-up 期

合法 align_with() 组合（见 _ALIGN_ALLOWED）
-------------------------------------------
factor_observation  × forward_return   ✅（IC计算、分层回测）
forward_return      × factor_observation ✅（同上，顺序互换）
factor_observation  × factor_observation ✅（因子合成、相关性分析）
price               × price              ✅（价格比较）
price               × forward_return     ❌（禁止）
price               × factor_observation ❌（禁止）

额外约束：factor_observation × forward_return 对齐时，
          factor_observation.is_t1_shifted 必须为 True，否则抛出 TimingAlignmentError。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 专用异常
# ═══════════════════════════════════════════════════════════════════════════════

class TimingAlignmentError(ValueError):
    """
    时间对齐语义不兼容时抛出。

    典型场景：factor_observation 面板尚未做 T+1 滞后，就与 forward_return 对齐。
    解决方法：在 align_with() 之前先调用 factor_panel.shift_to_t1()。
    """


class SemanticCompatibilityError(TypeError):
    """
    两个面板的 semantic 组合不在合法对齐组合表中时抛出。

    典型场景：price 面板直接与 factor_observation 对齐（应先构建 ReturnPanel）。
    """


# ═══════════════════════════════════════════════════════════════════════════════
# 合法对齐组合表
# ═══════════════════════════════════════════════════════════════════════════════

# 格式：(left_semantic, right_semantic) → allowed: bool
_ALIGN_ALLOWED: dict[tuple[str, str], bool] = {
    ("factor_observation", "forward_return"):   True,
    ("forward_return",     "factor_observation"): True,
    ("factor_observation", "factor_observation"): True,
    ("price",              "price"):            True,
    ("price",              "forward_return"):   False,   # 禁止
    ("price",              "factor_observation"): False, # 禁止
    ("forward_return",     "forward_return"):   True,
    ("forward_return",     "price"):            False,
    ("factor_observation", "price"):            False,
}

# 需要检查 is_t1_shifted 的组合（左侧必须已 shift）
_REQUIRE_T1_SHIFTED: set[tuple[str, str]] = {
    ("factor_observation", "forward_return"),
    ("forward_return",     "factor_observation"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TimestampedPanel
# ═══════════════════════════════════════════════════════════════════════════════

class TimestampedPanel(pd.DataFrame):
    """
    带语义标注的面板数据结构，继承 pd.DataFrame。

    Parameters
    ----------
    data         : 与 pd.DataFrame 相同的初始化数据
    semantic     : "factor_observation" | "forward_return" | "price"
    is_t1_shifted: 是否已做 T+1 行位移（默认 False）
    forward_days : 预测期（天），仅对 forward_return 有意义
    price_basis  : 价格复权方式，"hfq" / "qfq" / "raw" / None
    factor_name  : 因子名称（用于调试）
    warmup_trimmed: 是否已截断 warm-up 期（默认 False）
    **kwargs     : 透传给 pd.DataFrame.__init__

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_framework.core.panel import TimestampedPanel
    >>> df = pd.DataFrame({"A": [1.0, 2.0]}, index=["20200131", "20200229"])
    >>> tp = TimestampedPanel(df, semantic="factor_observation")
    >>> tp.semantic
    'factor_observation'
    >>> tp_t1 = tp.shift_to_t1()
    >>> tp_t1.is_t1_shifted
    True
    """

    # pandas _metadata 机制：列出需要在 copy/slice 后保留的自定义属性
    _metadata = [
        "semantic",
        "is_t1_shifted",
        "forward_days",
        "price_basis",
        "factor_name",
        "warmup_trimmed",
    ]

    def __init__(
        self,
        data=None,
        *args,
        semantic:       str = "factor_observation",
        is_t1_shifted:  bool = False,
        forward_days:   Optional[int] = None,
        price_basis:    Optional[str] = None,
        factor_name:    Optional[str] = None,
        warmup_trimmed: bool = False,
        **kwargs,
    ):
        super().__init__(data, *args, **kwargs)
        self.semantic       = semantic
        self.is_t1_shifted  = is_t1_shifted
        self.forward_days   = forward_days
        self.price_basis    = price_basis
        self.factor_name    = factor_name
        self.warmup_trimmed = warmup_trimmed

    @property
    def _constructor(self):
        """确保 pandas 内部操作（slice、copy 等）返回 TimestampedPanel 而非普通 DataFrame。"""
        def _c(*args, **kwargs):
            # 继承当前实例的元数据字段
            obj = TimestampedPanel(*args, **kwargs)
            for attr in self._metadata:
                object.__setattr__(obj, attr, getattr(self, attr, None))
            return obj
        return _c

    # ── 核心方法 ─────────────────────────────────────────────────────────────

    def shift_to_t1(self) -> "TimestampedPanel":
        """
        对当前面板做 T+1 行位移（pandas 行位移，等价于"下一个有数据的日期"）。

        语义
        ----
        - 因为 index 只包含有效交易日（停牌日不出现），pandas 的 shift(1) 在
          行位移语义上等价于"下一个交易日"，不需要引入 TradingCalendar 依赖。
        - 返回新的 TimestampedPanel，不修改原对象（immutable 语义）。
        - 若已做过 T+1，抛出 RuntimeError 防止重复调用。

        Returns
        -------
        TimestampedPanel，is_t1_shifted=True

        Raises
        ------
        RuntimeError : 若当前面板已做过 T+1（is_t1_shifted=True）
        """
        if self.is_t1_shifted:
            raise RuntimeError(
                f"[TimestampedPanel] 已做过 T+1 滞后（factor_name={self.factor_name!r}），"
                "请勿重复调用 shift_to_t1()。"
            )

        shifted = TimestampedPanel(
            super().shift(1),
            semantic       = self.semantic,
            is_t1_shifted  = True,
            forward_days   = self.forward_days,
            price_basis    = self.price_basis,
            factor_name    = self.factor_name,
            warmup_trimmed = self.warmup_trimmed,
        )
        return shifted

    def align_with(
        self,
        other: "TimestampedPanel",
    ) -> Tuple["TimestampedPanel", "TimestampedPanel"]:
        """
        验证两个面板的 semantic 组合是否合法，并返回 index 取交集后的对齐结果。

        合法性检查
        ----------
        1. (self.semantic, other.semantic) 必须在 _ALIGN_ALLOWED 中且为 True。
        2. 若组合为 factor_observation × forward_return（任意顺序），
           则 factor_observation 侧必须 is_t1_shifted=True。

        Parameters
        ----------
        other : 另一个 TimestampedPanel

        Returns
        -------
        (self_aligned, other_aligned) : index 取交集后的两个 TimestampedPanel

        Raises
        ------
        SemanticCompatibilityError : semantic 组合非法
        TimingAlignmentError       : factor_observation 未做 T+1 滞后
        """
        if not isinstance(other, TimestampedPanel):
            raise TypeError(
                f"align_with() 要求参数为 TimestampedPanel，"
                f"收到 {type(other).__name__}。"
            )

        combo = (self.semantic, other.semantic)
        allowed = _ALIGN_ALLOWED.get(combo)

        if allowed is None:
            # 未在表中定义的组合，默认禁止
            raise SemanticCompatibilityError(
                f"[TimestampedPanel] 未知语义组合：{combo}，"
                "请在 _ALIGN_ALLOWED 中明确声明。"
            )
        if not allowed:
            raise SemanticCompatibilityError(
                f"[TimestampedPanel] 不允许对齐语义组合 {combo}。\n"
                "  禁止原因：收益率面板必须由 ReturnPanel.build() 构建，"
                "不能让价格面板直接与因子/收益率面板对齐。"
            )

        # 额外约束：factor_observation × forward_return 时，factor 侧必须已 T+1
        if combo in _REQUIRE_T1_SHIFTED:
            # 确定哪一侧是 factor_observation
            factor_side = self if self.semantic == "factor_observation" else other
            if not factor_side.is_t1_shifted:
                raise TimingAlignmentError(
                    f"[TimestampedPanel] factor_observation 面板"
                    f"（factor_name={factor_side.factor_name!r}）"
                    f"尚未做 T+1 滞后，不能与 forward_return 对齐。\n"
                    "  修复方法：先调用 factor_panel.shift_to_t1()，"
                    "再执行 align_with()。"
                )

        # 取 index 交集对齐
        common_index = self.index.intersection(other.index)
        return self.loc[common_index], other.loc[common_index]

    def trim_warmup(self, n_days: int) -> "TimestampedPanel":
        """
        删除 index 中最早的 n_days 行，截断 warm-up 期。

        Parameters
        ----------
        n_days : 需要删除的最早行数

        Returns
        -------
        新的 TimestampedPanel，warmup_trimmed=True

        Raises
        ------
        ValueError : n_days >= len(self)（截断后面板为空）
        """
        if n_days <= 0:
            return self
        if n_days >= len(self):
            raise ValueError(
                f"[TimestampedPanel.trim_warmup] n_days={n_days} ≥ "
                f"面板行数={len(self)}，截断后面板为空。"
            )
        trimmed = TimestampedPanel(
            self.iloc[n_days:],
            semantic       = self.semantic,
            is_t1_shifted  = self.is_t1_shifted,
            forward_days   = self.forward_days,
            price_basis    = self.price_basis,
            factor_name    = self.factor_name,
            warmup_trimmed = True,
        )
        return trimmed

    def assert_valid(self) -> None:
        """
        对当前面板做完整性自检。

        检查项
        ------
        1. index 是否有序（严格递增）
        2. index 是否有重复
        3. 数值列是否全为 NaN

        Raises
        ------
        AssertionError : 任意检查项失败，错误信息包含诊断详情
        """
        # 1. index 有序（不要求类型，只要 monotonic_increasing）
        assert self.index.is_monotonic_increasing, (
            f"[TimestampedPanel.assert_valid] index 不是严格递增。\n"
            f"  前5行: {list(self.index[:5])}\n"
            f"  后5行: {list(self.index[-5:])}"
        )

        # 2. index 无重复
        dup = self.index[self.index.duplicated()]
        assert len(dup) == 0, (
            f"[TimestampedPanel.assert_valid] index 存在 {len(dup)} 个重复日期：{list(dup[:5])}"
        )

        # 3. 非全 NaN（至少有一个数值）
        assert self.notna().any().any(), (
            f"[TimestampedPanel.assert_valid] 面板全为 NaN "
            f"（semantic={self.semantic!r}, factor_name={self.factor_name!r}）"
        )

    # ── 便捷类方法 ───────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        semantic:       str = "factor_observation",
        is_t1_shifted:  bool = False,
        forward_days:   Optional[int] = None,
        price_basis:    Optional[str] = None,
        factor_name:    Optional[str] = None,
        warmup_trimmed: bool = False,
    ) -> "TimestampedPanel":
        """
        从普通 pd.DataFrame 构建 TimestampedPanel（零拷贝）。

        Parameters
        ----------
        df          : 原始面板 DataFrame
        semantic    : 语义标注
        其余参数    : 同 __init__

        Returns
        -------
        TimestampedPanel
        """
        obj = cls(
            df,
            semantic       = semantic,
            is_t1_shifted  = is_t1_shifted,
            forward_days   = forward_days,
            price_basis    = price_basis,
            factor_name    = factor_name,
            warmup_trimmed = warmup_trimmed,
        )
        return obj

    def __repr__(self) -> str:
        base = super().__repr__()
        meta = (
            f"  semantic={self.semantic!r}, "
            f"is_t1_shifted={self.is_t1_shifted}, "
            f"forward_days={self.forward_days}, "
            f"factor_name={self.factor_name!r}"
        )
        return f"TimestampedPanel(\n{meta}\n)\n{base}"
