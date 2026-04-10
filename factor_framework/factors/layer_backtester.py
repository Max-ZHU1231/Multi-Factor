"""
factor_framework.factors.layer_backtester
===========================================
LayerBacktester —— 分层回测封装，统一 layer_backtest / long_short_stats /
turnover_analysis 三者的调用、结果存储与格式化输出。

设计原则（v3.0 规范 §4.3）
--------------------------
- LayerBacktester 接收已处理的因子面板 + 收益率面板。
- 内部调用 backtest 模块函数，将结果收拢到一个对象。
- 提供 .summary() -> dict、.layer_ret -> pd.DataFrame、
  .nav -> pd.DataFrame（净值曲线）三个主要输出接口。
- 不执行截面预处理（TransformPipeline 的职责）。

使用方式
--------
    from factor_framework.factors.layer_backtester import LayerBacktester

    bt = LayerBacktester(
        factor_panel     = factor_panel,    # 已标准化
        return_panel     = return_panel,    # forward=21
        n_groups         = 5,
        direction        = 1,
        periods_per_year = 12,
        rf               = 0.0,
        cost_per_side    = 0.002,
    )
    bt.run()

    print(bt.summary())
    print(bt.layer_ret.head())
    print(bt.nav.tail())
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from factor_framework.backtest import (
    layer_backtest,
    long_short_stats,
    turnover_analysis,
)


class LayerBacktester:
    """
    分层回测封装器。

    Parameters
    ----------
    factor_panel     : (日期 × 股票) 因子面板（已完成截面预处理）
    return_panel     : (日期 × 股票) 收益率面板
    n_groups         : 分层数（默认 5）
    direction        : 因子方向（+1 或 -1，默认 +1）
    periods_per_year : 年化期数（月频=12，日频=252）
    rf               : 无风险利率（年化，默认 0.0）
    cost_per_side    : 单边交易成本（默认 0.002）
    """

    def __init__(
        self,
        factor_panel:      pd.DataFrame,
        return_panel:      pd.DataFrame,
        n_groups:          int   = 5,
        direction:         int   = 1,
        periods_per_year:  int   = 12,
        rf:                float = 0.0,
        cost_per_side:     float = 0.002,
    ) -> None:
        self.factor_panel     = factor_panel
        self.return_panel     = return_panel
        self.n_groups         = n_groups
        self.direction        = direction
        self.periods_per_year = periods_per_year
        self.rf               = rf
        self.cost_per_side    = cost_per_side

        # 结果存储（run() 后填充）
        self._layer_ret:  Optional[pd.DataFrame] = None
        self._ls_stats:   Optional[Dict]          = None
        self._turnover:   Optional[Dict]           = None
        self._nav:        Optional[pd.DataFrame]  = None
        self._ran:        bool                    = False

    # ── 执行回测 ─────────────────────────────────────────────────────────────

    def run(self) -> "LayerBacktester":
        """
        执行分层回测、多空统计、换手率分析，填充内部结果字段。

        Returns
        -------
        self（支持链式调用 LayerBacktester(...).run().summary()）
        """
        # 1. 分层回测
        self._layer_ret = layer_backtest(
            self.factor_panel,
            self.return_panel,
            n_groups  = self.n_groups,
            direction = self.direction,
        )

        # 2. 多空统计
        self._ls_stats = long_short_stats(
            self._layer_ret,
            periods_per_year = self.periods_per_year,
            rf               = self.rf,
        )

        # 3. 换手率 / 交易成本
        self._turnover = turnover_analysis(
            self.factor_panel,
            n_groups      = self.n_groups,
            direction     = self.direction,
            cost_per_side = self.cost_per_side,
        )

        # 4. 各层净值曲线（(1+r).cumprod()）
        if self._layer_ret is not None and not self._layer_ret.empty:
            self._nav = (1 + self._layer_ret.fillna(0)).cumprod()

        self._ran = True
        return self

    # ── 结果属性 ─────────────────────────────────────────────────────────────

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError(
                "[LayerBacktester] 请先调用 .run() 再访问结果属性。"
            )

    @property
    def layer_ret(self) -> pd.DataFrame:
        """各层收益率序列（行=日期，列=第1~N层）。"""
        self._check_ran()
        return self._layer_ret

    @property
    def ls_stats(self) -> Dict:
        """
        多空组合统计：
        ls_annual_return / ls_sharpe / ls_max_drawdown /
        ls_calmar / ls_win_rate / monotone_score
        """
        self._check_ran()
        return self._ls_stats

    @property
    def turnover(self) -> Dict:
        """换手率与交易成本：avg_turnover / avg_cost。"""
        self._check_ran()
        return self._turnover

    @property
    def nav(self) -> Optional[pd.DataFrame]:
        """各层净值曲线（从 1.0 起始的 cumprod）。"""
        self._check_ran()
        return self._nav

    # ── 汇总输出 ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """
        返回分层回测汇总字典（适合存入 FactorReport 或打印）。

        字段
        ----
        ls_annual_return, ls_sharpe, ls_max_drawdown, ls_calmar,
        ls_win_rate, monotone_score, avg_turnover, avg_cost
        """
        self._check_ran()
        return {
            **self._ls_stats,
            "avg_turnover": self._turnover.get("avg_turnover"),
            "avg_cost":     self._turnover.get("avg_cost"),
        }

    def print_summary(self, factor_name: str = "") -> None:
        """打印格式化分层回测摘要。"""
        self._check_ran()
        s = self.summary()
        header = f"── 分层回测摘要 {'[' + factor_name + ']' if factor_name else ''} ──"
        print(header)
        print(f"  多空年化收益   : {s.get('ls_annual_return', float('nan')):.2%}")
        print(f"  多空夏普比率   : {s.get('ls_sharpe', float('nan')):.4f}")
        print(f"  最大回撤       : {s.get('ls_max_drawdown', float('nan')):.2%}")
        print(f"  Calmar 比率    : {s.get('ls_calmar', float('nan')):.4f}")
        print(f"  胜率           : {s.get('ls_win_rate', float('nan')):.2%}")
        print(f"  单调性得分     : {s.get('monotone_score', float('nan')):.4f}")
        print(f"  平均换手率     : {s.get('avg_turnover', float('nan')):.2%}")
        print(f"  平均交易成本   : {s.get('avg_cost', float('nan')):.4f}")

    def __repr__(self) -> str:
        status = "已运行" if self._ran else "未运行（调用 .run()）"
        return (
            f"LayerBacktester(n_groups={self.n_groups}, "
            f"direction={self.direction}, "
            f"periods_per_year={self.periods_per_year}, "
            f"status={status})"
        )
