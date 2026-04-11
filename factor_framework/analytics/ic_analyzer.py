"""
factor_framework.factors.ic_analyzer
======================================
ICAnalyzer —— 结构化 IC 分析封装，统一 ic_series / ic_stats / ic_decay
三者的调用、结果存储与格式化输出。

设计原则（v3.0 规范 §4.2）
--------------------------
- ICAnalyzer 接收已对齐的因子面板和收益率面板（或多期 return_panels 字典）。
- 内部调用 ic_analysis 模块的函数，将结果收拢到一个对象中。
- 提供 .summary() -> dict、.decay_df -> pd.DataFrame 两个主要输出接口。
- 不执行截面预处理（那是 TransformPipeline 的职责）。
- 向下游（FactorReport / FactorPipeline）提供统一的 IC 结果结构。

使用方式
--------
    from factor_framework.factors.ic_analyzer import ICAnalyzer

    analyzer = ICAnalyzer(
        factor_panel  = factor_panel,    # 已经过 TransformPipeline 处理
        return_panel  = return_panel,    # forward=21 的主收益率面板
        return_panels = {1: rp1, 5: rp5, 21: rp21},  # IC 衰减用
        method        = "rank",
        periods_per_year = 12,           # 月频=12
    )
    analyzer.run()

    print(analyzer.summary())
    print(analyzer.decay_df)
    print(analyzer.ic_series.mean())
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_framework.analytics.ic_analysis import (
    compute_ic,
    ic_stats,
    ic_decay,
    ic_significance,
)


class ICAnalyzer:
    """
    结构化 IC 分析器。

    Parameters
    ----------
    factor_panel     : (日期 × 股票) 因子面板（已完成截面预处理）
    return_panel     : (日期 × 股票) 主收益率面板（用于逐期 IC 计算）
    return_panels    : {forward: pd.DataFrame} 多期收益率字典（用于 IC 衰减）
                       若为 None，仅跳过衰减分析
    method           : IC 计算方式：'rank'（默认）或 'normal'
    periods_per_year : 年化期数（月频=12，日频=252）
    min_stocks       : 有效截面最少股票数（不足则 IC 置 NaN）
    lags             : Newey-West 修正的滞后阶数（None = 自动取 T^0.25）
    """

    def __init__(
        self,
        factor_panel:      pd.DataFrame,
        return_panel:      pd.DataFrame,
        return_panels:     Optional[Dict[int, pd.DataFrame]] = None,
        method:            str = "rank",
        periods_per_year:  int = 12,
        min_stocks:        int = 5,
        lags:              Optional[int] = None,
    ) -> None:
        self.factor_panel     = factor_panel
        self.return_panel     = return_panel
        self.return_panels    = return_panels
        self.method           = method
        self.periods_per_year = periods_per_year
        self.min_stocks       = min_stocks
        self.lags             = lags

        # 结果存储（run() 后填充）
        self._ic_series:  Optional[pd.Series]     = None
        self._ic_stats:   Optional[Dict]           = None
        self._ic_nw:      Optional[Dict]           = None
        self._decay_df:   Optional[pd.DataFrame]   = None
        self._ran:        bool                     = False

    # ── 执行分析 ─────────────────────────────────────────────────────────────

    def run(self) -> "ICAnalyzer":
        """
        执行全部 IC 分析，填充内部结果字段。支持链式调用。

        Returns
        -------
        self（支持 ICAnalyzer(...).run().summary()）
        """
        # 1. 逐期 IC
        self._ic_series = compute_ic(
            self.factor_panel,
            self.return_panel,
            method=self.method,
            min_stocks=self.min_stocks,
        )

        # 2. IC 统计指标
        self._ic_stats = ic_stats(
            self._ic_series,
            annualize_periods=self.periods_per_year,
        )

        # 3. Newey-West 修正显著性
        n_lags = self.lags if self.lags is not None else max(
            1, int(len(self._ic_series.dropna()) ** 0.25)
        )
        self._ic_nw = ic_significance(self._ic_series, lags=n_lags)

        # 4. IC 衰减（若提供了多期收益率面板）
        if self.return_panels is not None:
            self._decay_df = ic_decay(
                self.factor_panel,
                return_panels=self.return_panels,
                method=self.method,
            )

        self._ran = True
        return self

    # ── 结果属性 ─────────────────────────────────────────────────────────────

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError(
                "[ICAnalyzer] 请先调用 .run() 再访问结果属性。"
            )

    @property
    def ic_series(self) -> pd.Series:
        """逐期 IC 时间序列。"""
        self._check_ran()
        return self._ic_series

    @property
    def ic_stats_dict(self) -> Dict:
        """
        IC 核心统计字典：
        mean_ic / std_ic / icir / win_rate / t_stat / p_value /
        total_periods / annualized_icir
        """
        self._check_ran()
        return self._ic_stats

    @property
    def ic_nw(self) -> Dict:
        """Newey-West 修正 t 检验结果：nw_t_stat / nw_p_value。"""
        self._check_ran()
        return self._ic_nw

    @property
    def decay_df(self) -> Optional[pd.DataFrame]:
        """
        IC 衰减分析结果 DataFrame（行 = forward 期，列含 Mean IC 等统计量）。
        若初始化时未传入 return_panels，则为 None。
        """
        self._check_ran()
        return self._decay_df

    # ── 汇总输出 ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """
        返回 IC 分析汇总字典（适合存入 FactorReport 或打印输出）。

        字段
        ----
        mean_ic, std_ic, icir, annualized_icir,
        win_rate, t_stat, p_value,
        nw_t_stat, nw_p_value,
        total_periods
        """
        self._check_ran()
        return {
            **self._ic_stats,
            "nw_t_stat":  self._ic_nw.get("nw_t_stat"),
            "nw_p_value": self._ic_nw.get("nw_p_value"),
        }

    def print_summary(self, factor_name: str = "") -> None:
        """打印格式化 IC 分析摘要。"""
        self._check_ran()
        s = self.summary()
        header = f"── IC 分析摘要 {'[' + factor_name + ']' if factor_name else ''} ──"
        print(header)
        print(f"  平均 IC        : {s.get('mean_ic', float('nan')):.4f}")
        print(f"  IC 标准差      : {s.get('std_ic', float('nan')):.4f}")
        print(f"  ICIR           : {s.get('icir', float('nan')):.4f}")
        print(f"  年化 ICIR      : {s.get('annualized_icir', float('nan')):.4f}")
        print(f"  胜率           : {s.get('win_rate', float('nan')):.2%}")
        print(f"  t 统计量       : {s.get('t_stat', float('nan')):.4f}")
        print(f"  Newey-West t   : {s.get('nw_t_stat', float('nan')):.4f}")
        print(f"  有效期数       : {s.get('total_periods', 0)}")

    def __repr__(self) -> str:
        status = "已运行" if self._ran else "未运行（调用 .run()）"
        return (
            f"ICAnalyzer(method={self.method!r}, "
            f"periods_per_year={self.periods_per_year}, "
            f"status={status})"
        )
