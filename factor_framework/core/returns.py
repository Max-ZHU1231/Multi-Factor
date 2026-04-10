"""
factor_framework.core.returns
==============================
ReturnPanel —— 整个框架唯一被允许从价格面板构建收益率的模块。

设计原则（v3.0 规范 §3.2）
--------------------------
- 所有下游模块（ic.py、backtest.py）的收益率参数类型必须来自此模块。
- 使用 price.shift(-forward) / price - 1 而非 pct_change(forward).shift(-forward)，
  停牌 NaN 语义清晰，不会用前值填充。
- 收益率面板本身不做 T+1 滞后（T+1 在因子面板侧通过 shift_to_t1() 完成）。
- 入参的价格面板必须是后复权（price_basis="hfq"），否则抛出 ValueError。

与现有代码的关系
---------------
- FactorEngine.build_return_panel()：阶段二保留为兼容接口，内部调用 ReturnPanel.build()。
- 阶段三删除 build_return_panel()，直接使用 ReturnPanel.build()。
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_framework.core.panel import TimestampedPanel


class ReturnPanel:
    """
    收益率面板构建器（静态方法集合，无需实例化）。

    所有方法均返回 TimestampedPanel（semantic="forward_return"）。
    """

    @staticmethod
    def build(
        hfq_price_panel: TimestampedPanel,
        forward_days:    int,
    ) -> TimestampedPanel:
        """
        从后复权价格面板构建单个预测期的收益率面板。

        公式：ret[t] = price[t + forward] / price[t] - 1
        实现：price_panel.shift(-forward) / price_panel - 1

        Parameters
        ----------
        hfq_price_panel : TimestampedPanel（price_basis="hfq"）
                          若 price_basis 不为 "hfq"，抛出 ValueError
        forward_days    : 预测期（天数）

        Returns
        -------
        TimestampedPanel(
            semantic      = "forward_return",
            forward_days  = forward_days,
            is_t1_shifted = False,
        )

        Notes
        -----
        - T+1 滞后在因子面板侧完成（factor_panel.shift_to_t1()），
          收益率面板本身不做 T+1 滞后。
        - 尾部 forward_days 行因未来价格不存在而全为 NaN，调用方负责截断。
        """
        if not isinstance(hfq_price_panel, TimestampedPanel):
            raise TypeError(
                "ReturnPanel.build() 要求 hfq_price_panel 为 TimestampedPanel，"
                f"收到 {type(hfq_price_panel).__name__}。"
            )
        if hfq_price_panel.price_basis not in ("hfq", None):
            # price_basis=None 表示未知，宽松接受；只明确拒绝非 hfq 的
            raise ValueError(
                f"ReturnPanel.build() 要求后复权价格（price_basis='hfq'），"
                f"收到 price_basis={hfq_price_panel.price_basis!r}。\n"
                "  建议：使用 DataStore.get_price_panel(basis='hfq') 获取后复权价格。"
            )

        price = hfq_price_panel.replace(0, np.nan)
        ret   = price.shift(-forward_days) / price - 1

        return TimestampedPanel.from_dataframe(
            ret,
            semantic      = "forward_return",
            forward_days  = forward_days,
            is_t1_shifted = False,
            price_basis   = hfq_price_panel.price_basis,
        )

    @staticmethod
    def build_multi_forward(
        hfq_price_panel: TimestampedPanel,
        forward_list:    List[int],
    ) -> Dict[int, TimestampedPanel]:
        """
        批量构建多个预测期的收益率面板（共享同一份价格面板）。

        用途：IC 衰减分析（ICAnalyzer.compute 接收此字典）。

        Parameters
        ----------
        hfq_price_panel : TimestampedPanel（price_basis="hfq"）
        forward_list    : 预测期列表（天数），如 [1, 5, 10, 21, 60]

        Returns
        -------
        Dict[int, TimestampedPanel]：key=forward_days，value=收益率面板
        """
        return {
            fwd: ReturnPanel.build(hfq_price_panel, fwd)
            for fwd in forward_list
        }

    @staticmethod
    def from_raw_dataframe(
        price_df:     pd.DataFrame,
        forward_days: int,
        price_basis:  Optional[str] = None,
    ) -> TimestampedPanel:
        """
        从普通 pd.DataFrame（价格面板）构建收益率面板（向后兼容入口）。

        与 build() 的区别：入参是普通 pd.DataFrame 而非 TimestampedPanel，
        适用于尚未迁移到 TimestampedPanel 的调用方。

        Parameters
        ----------
        price_df     : 普通 pd.DataFrame（日期 × 股票，收盘价）
        forward_days : 预测期（天数）
        price_basis  : "hfq" | "qfq" | "raw" | None

        Returns
        -------
        TimestampedPanel(semantic="forward_return")
        """
        if price_basis not in ("hfq", None):
            warnings.warn(
                f"ReturnPanel.from_raw_dataframe() 收到 price_basis={price_basis!r}，"
                "建议使用后复权价格（price_basis='hfq'）以确保收益率计算正确。",
                stacklevel=2,
            )

        price = price_df.replace(0, np.nan)
        ret   = price.shift(-forward_days) / price - 1

        return TimestampedPanel.from_dataframe(
            ret,
            semantic      = "forward_return",
            forward_days  = forward_days,
            is_t1_shifted = False,
            price_basis   = price_basis,
        )
