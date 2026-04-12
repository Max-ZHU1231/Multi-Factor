"""
factor_framework.factors.ic_analyzer  [v4.0 COMPATIBILITY SHIM]
================================================================
⚠️  ICAnalyzer 已迁移至 factor_framework.analytics（v4.0）。
    旧路径将在 v4.2 移除，请更新 import：

    旧：from factor_framework.factors.ic_analyzer import ICAnalyzer
    新：from factor_framework.analytics import ICAnalyzer
"""

from __future__ import annotations
import warnings as _warnings
_warnings.warn(
    "factor_framework.factors.ic_analyzer has moved to factor_framework.analytics. "
    "The legacy import path will be removed in v4.2.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_framework.ic_analysis import (
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
        header = f"── IC Analysis Summary {'[' + factor_name + ']' if factor_name else ''} ──"
        print(f"[INFO] {header}")
        print(f"[INFO] Mean IC        : {s.get('mean_ic', float('nan')):.4f}")
        print(f"[INFO] IC Std         : {s.get('std_ic', float('nan')):.4f}")
        print(f"[INFO] ICIR           : {s.get('icir', float('nan')):.4f}")
        print(f"[INFO] Annualized ICIR: {s.get('annualized_icir', float('nan')):.4f}")
        print(f"[INFO] Win Rate       : {s.get('win_rate', float('nan')):.2%}")
        print(f"[INFO] t-stat         : {s.get('t_stat', float('nan')):.4f}")
        print(f"[INFO] Newey-West t   : {s.get('nw_t_stat', float('nan')):.4f}")
        print(f"[INFO] Valid Periods  : {s.get('total_periods', 0)}")

    def __repr__(self) -> str:
        status = "ran" if self._ran else "not-ran (call .run())"
        return (
            f"ICAnalyzer(method={self.method!r}, "
            f"periods_per_year={self.periods_per_year}, "
            f"status={status})"
        )
