"""
pipeline.py
===========
端到端因子流水线（Pipeline）。

完整流程
--------
1. 加载 Stocks/ 中的股票数据（data_cleaner.load_and_clean）
2. 注册内置或自定义因子（FactorEngine.register）
3. 构建因子面板 + 收益率面板
4. 横截面标准化（可选：rank / zscore）
5. 因子中性化（可选：market_cap + industry）
6. IC 分析（compute_ic + ic_stats + ic_decay）
7. 分层回测（layer_backtest + long_short_stats）
8. 换手率与交易成本分析
9. 生成汇总报告（输出 dict / CSV / 打印）

使用方式
--------
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline(
    stocks_dir   = 'Stocks/',
    stock_basic  = '股票列表-stock_basic.csv',
)
pipe.register_factor('momentum_12_1', lambda df: ...)   # 或加载内置因子
report = pipe.run(
    factor_name      = 'momentum_12_1',
    start            = '20150101',
    end              = '20261231',
    forward          = 21,
    n_groups         = 5,
    neutralize       = True,
    standardize      = 'rank',
)
report.print_summary()
report.save('output/momentum_12_1_report.csv')
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from factor_framework.factor_engine import FactorEngine, FactorFn
from factor_framework.ic_analysis   import compute_ic, ic_stats, ic_decay, ic_significance
from factor_framework.backtest      import layer_backtest, long_short_stats, turnover_analysis
from factor_framework.neutralize    import neutralize_regression, neutralize_industry_zscore
from factor_framework.operators     import cs_rank, cs_zscore, cs_winsorize


# ═══════════════════════════════════════════════════════════════════════════════
# 报告对象
# ═══════════════════════════════════════════════════════════════════════════════

class FactorReport:
    """封装单个因子的完整检验结果。"""

    def __init__(
        self,
        factor_name:  str,
        ic_series:    pd.Series,
        ic_stats:     Dict,
        ic_nw:        Dict,
        ic_decay_df:  pd.DataFrame,
        layer_ret:    pd.DataFrame,
        ls_stats:     Dict,
        turnover:     Dict,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
    ):
        self.factor_name  = factor_name
        self.ic_series    = ic_series
        self.ic_stats_    = ic_stats
        self.ic_nw        = ic_nw
        self.ic_decay_df  = ic_decay_df
        self.layer_ret    = layer_ret
        self.ls_stats     = ls_stats
        self.turnover     = turnover
        self.factor_panel = factor_panel
        self.return_panel = return_panel

    # ── 打印汇总 ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """终端打印因子评估报告。"""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  因子名称: {self.factor_name}")
        print(sep)

        s = self.ic_stats_
        print(f"\n【IC 分析】")
        print(f"  Mean IC      : {s.get('mean_ic', 'N/A'):.4f}  "
              f"（合格 |IC| > 0.02，优秀 > 0.05）")
        print(f"  Std  IC      : {s.get('std_ic', 'N/A'):.4f}")
        print(f"  ICIR         : {s.get('icir', 'N/A'):.4f}  "
              f"（合格 > 0.5，优秀 > 1.0）")
        print(f"  IC 胜率      : {s.get('win_rate', 'N/A'):.1%}  "
              f"（合格 > 55%）")
        print(f"  t 统计量     : {s.get('t_stat', 'N/A'):.4f}  "
              f"（|t| > 2 显著）")
        print(f"  Newey-West t : {self.ic_nw.get('nw_t_stat', 'N/A')}")
        print(f"  有效期数     : {s.get('total_periods', 'N/A')}")

        print(f"\n【IC 衰减（不同预测期 Mean IC）】")
        print(self.ic_decay_df[["mean_ic", "icir"]].to_string())

        ls = self.ls_stats
        print(f"\n【分层回测 · 多空组合】")
        print(f"  年化收益     : {ls.get('ls_annual_return', 'N/A'):.2%}  "
              f"（合格 > 10%）")
        print(f"  年化夏普     : {ls.get('ls_sharpe', 'N/A'):.4f}  "
              f"（合格 > 1.0）")
        print(f"  最大回撤     : {ls.get('ls_max_drawdown', 'N/A'):.2%}  "
              f"（合格 < 30%）")
        print(f"  Calmar 比率  : {ls.get('ls_calmar', 'N/A'):.4f}  "
              f"（合格 > 0.5）")
        print(f"  多空胜率     : {ls.get('ls_win_rate', 'N/A'):.1%}  "
              f"（合格 > 55%）")
        print(f"  单调性得分   : {ls.get('monotone_score', 'N/A'):.4f}  "
              f"（越接近 1 越好）")

        print(f"\n【各层年化收益】")
        ann = ls.get("layer_annual_return")
        if ann is not None:
            for k, v in ann.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    print(f"  {k}:    N/A")
                else:
                    bar = "█" * max(0, int(v * 100))
                    print(f"  {k}: {v:>8.2%}  {bar}")

        t = self.turnover
        print(f"\n【换手率与交易成本】")
        print(f"  平均单边换手率: {t.get('avg_turnover', 'N/A'):.2%}")
        print(f"  每期估算成本  : {t.get('avg_cost', 'N/A'):.4%}")

        print(f"\n{sep}\n")

    # ── 保存 ─────────────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path = "output") -> None:
        """将各结果保存为 CSV 到 output_dir 目录。"""
        out = Path(output_dir) / self.factor_name
        out.mkdir(parents=True, exist_ok=True)

        self.ic_series.to_csv(out / "ic_series.csv", header=True)
        self.ic_decay_df.to_csv(out / "ic_decay.csv")
        self.layer_ret.to_csv(out / "layer_returns.csv")
        self.ls_stats["nav"].to_csv(out / "nav.csv")
        self.factor_panel.to_csv(out / "factor_panel.csv")

        # 汇总指标
        summary = {**self.ic_stats_, **self.ic_nw,
                   "ls_annual_return": self.ls_stats.get("ls_annual_return"),
                   "ls_sharpe":        self.ls_stats.get("ls_sharpe"),
                   "ls_max_drawdown":  self.ls_stats.get("ls_max_drawdown"),
                   "ls_calmar":        self.ls_stats.get("ls_calmar"),
                   "ls_win_rate":      self.ls_stats.get("ls_win_rate"),
                   "monotone_score":   self.ls_stats.get("monotone_score"),
                   "avg_turnover":     self.turnover.get("avg_turnover"),
                   "avg_cost":         self.turnover.get("avg_cost"),
                   }
        pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
        print(f"✓ 报告已保存至 {out}/")

    # ── 属性便捷访问 ─────────────────────────────────────────────────────────

    @property
    def summary_dict(self) -> Dict:
        ls = self.ls_stats
        return {
            "factor":           self.factor_name,
            **self.ic_stats_,
            "nw_t_stat":        self.ic_nw.get("nw_t_stat"),
            "ls_annual_return": ls.get("ls_annual_return"),
            "ls_sharpe":        ls.get("ls_sharpe"),
            "ls_max_drawdown":  ls.get("ls_max_drawdown"),
            "ls_calmar":        ls.get("ls_calmar"),
            "ls_win_rate":      ls.get("ls_win_rate"),
            "monotone_score":   ls.get("monotone_score"),
            "avg_turnover":     self.turnover.get("avg_turnover"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 主类
# ═══════════════════════════════════════════════════════════════════════════════

class FactorPipeline:
    """
    端到端多因子研究流水线。

    Parameters
    ----------
    stocks_dir  : Stocks/ 目录路径
    stock_basic : 股票列表 CSV（含 ts_code, industry 列）
    min_rows    : 新股最少有效行数
    verbose     : 显示进度条
    """

    def __init__(
        self,
        stocks_dir:  str | Path = "Stocks/",
        stock_basic: str | Path = "股票列表-stock_basic.csv",
        min_rows:    int = 60,
        verbose:     bool = True,
    ):
        self.engine = FactorEngine(
            stocks_dir  = stocks_dir,
            stock_basic = stock_basic,
            min_rows    = min_rows,
            verbose     = verbose,
        )

    # ── 因子注册 ──────────────────────────────────────────────────────────────

    def register_factor(self, name: str, func: FactorFn) -> "FactorPipeline":
        """注册单个自定义因子。支持链式调用。"""
        self.engine.register(name, func)
        return self

    def register_builtins(self, names: Optional[List[str]] = None) -> "FactorPipeline":
        """
        注册内置因子库中的因子。

        Parameters
        ----------
        names : 指定名称列表（None = 全部注册）
        """
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        targets = names or list(BUILTIN_FACTORS.keys())
        for n in targets:
            if n in BUILTIN_FACTORS:
                self.engine.register(n, BUILTIN_FACTORS[n])
            else:
                warnings.warn(f"内置因子 '{n}' 不存在，已跳过。")
        return self

    # ── 核心运行 ──────────────────────────────────────────────────────────────

    def run(
        self,
        factor_name:     str,
        start:           Optional[str] = None,
        end:             Optional[str] = None,
        forward:         int = 21,
        n_groups:        int = 5,
        direction:       int = 1,
        standardize:     Optional[str] = "rank",   # 'rank', 'zscore', None
        neutralize:      bool = False,              # 市值+行业回归中性化
        winsorize:       bool = True,               # 截面 MAD Winsorize
        ic_method:       str = "rank",
        ic_forward_list: List[int] = (1, 5, 10, 21, 60),
        periods_per_year:int = 252,
        rf:              float = 0.0,
        cost_per_side:   float = 0.002,
        symbols:         Optional[List[str]] = None,
    ) -> FactorReport:
        """
        执行完整的因子检验流程。

        Parameters
        ----------
        factor_name      : 已注册的因子名称
        start / end      : 日期范围（YYYYMMDD）
        forward          : 预测期（天）
        n_groups         : 分层数
        direction        : 因子方向（+1 或 -1）
        standardize      : 横截面标准化方式（'rank','zscore',None）
        neutralize       : 是否做市值+行业中性化
        winsorize        : 是否先做截面 MAD Winsorize
        ic_method        : IC 计算方式（'rank' 或 'normal'）
        ic_forward_list  : IC 衰减分析的预测期列表
        periods_per_year : 年化期数
        rf               : 无风险利率（年化）
        cost_per_side    : 单边交易成本
        symbols          : 指定股票列表（None = 全部）

        Returns
        -------
        FactorReport
        """
        print(f"\n[1/6] 构建因子面板: {factor_name} ...")
        factor_panel = self.engine.build_panel(
            factor_name, start=start, end=end, symbols=symbols
        )
        if factor_panel.empty:
            raise ValueError(f"因子 '{factor_name}' 面板为空，请检查因子函数或数据。")

        print(f"      因子面板: {factor_panel.shape[0]} 个交易日 × {factor_panel.shape[1]} 只股票")

        print(f"\n[2/6] 构建收益率面板（forward={forward}天）...")
        return_panel = self.engine.build_return_panel(
            forward=forward, start=start, end=end, symbols=symbols
        )

        # ── 截面预处理 ──────────────────────────────────────────────────────
        print(f"\n[3/6] 截面预处理（winsorize={winsorize}, standardize={standardize}, neutralize={neutralize}）...")

        if winsorize:
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_winsorize)

        if neutralize and self.engine.industry_map is not None:
            mktcap_panel = self.engine.build_panel("__mktcap__", start=start, end=end, symbols=symbols) \
                if "__mktcap__" in self.engine.registered() else pd.DataFrame()

            if mktcap_panel.empty:
                # 临时注册市值因子
                self.engine.register("__mktcap__", lambda df: df["总市值（万元）"])
                mktcap_panel = self.engine.build_panel("__mktcap__", start=start, end=end, symbols=symbols)
                del self.engine._registry["__mktcap__"]

            factor_panel = neutralize_regression(
                factor_panel,
                mktcap_panel,
                industry_map = self.engine.industry_map,
            )
        elif neutralize:
            warnings.warn("neutralize=True 但 industry_map 为空，跳过中性化。")

        if standardize == "rank":
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_rank)
        elif standardize == "zscore":
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_zscore)

        # ── IC 分析 ──────────────────────────────────────────────────────────
        print(f"\n[4/6] IC 分析（method={ic_method}）...")
        ic_series = compute_ic(factor_panel, return_panel, method=ic_method)
        ic_s      = ic_stats(ic_series, annualize_periods=periods_per_year)
        ic_nw     = ic_significance(ic_series, lags=max(1, int(len(ic_series) ** 0.25)))

        # IC 衰减（需要价格面板）
        self.engine.register("__close__", lambda df: df["收盘价"])
        close_panel = self.engine.build_panel("__close__", start=start, end=end, symbols=symbols)
        del self.engine._registry["__close__"]
        ic_decay_df = ic_decay(factor_panel, close_panel, forward_periods=ic_forward_list, method=ic_method)

        # ── 分层回测 ──────────────────────────────────────────────────────────
        print(f"\n[5/6] 分层回测（n_groups={n_groups}）...")
        layer_ret = layer_backtest(
            factor_panel, return_panel,
            n_groups=n_groups, direction=direction
        )
        ls_stats_ = long_short_stats(layer_ret, periods_per_year=periods_per_year, rf=rf)

        # ── 换手率分析 ────────────────────────────────────────────────────────
        print(f"\n[6/6] 换手率分析...")
        turnover_ = turnover_analysis(
            factor_panel, n_groups=n_groups, direction=direction,
            cost_per_side=cost_per_side
        )

        report = FactorReport(
            factor_name  = factor_name,
            ic_series    = ic_series,
            ic_stats     = ic_s,
            ic_nw        = ic_nw,
            ic_decay_df  = ic_decay_df,
            layer_ret    = layer_ret,
            ls_stats     = ls_stats_,
            turnover     = turnover_,
            factor_panel = factor_panel,
            return_panel = return_panel,
        )

        print("\n✓ 流程完成。")
        return report

    # ── 批量多因子运行 ─────────────────────────────────────────────────────────

    def run_batch(
        self,
        factor_names: List[str],
        **kwargs,
    ) -> pd.DataFrame:
        """
        批量检验多个因子，返回汇总 DataFrame（每行一个因子）。
        kwargs 透传给 run()。
        """
        rows = []
        for name in factor_names:
            try:
                report = self.run(name, **kwargs)
                rows.append(report.summary_dict)
            except Exception as e:
                warnings.warn(f"因子 '{name}' 检验失败: {e}")
                rows.append({"factor": name})
        return pd.DataFrame(rows).set_index("factor")
