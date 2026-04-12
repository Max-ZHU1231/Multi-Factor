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
from factor_framework.optimizer     import equal_weight, icir_weight, print_weights
from factor_framework.engine.cache         import CacheLayer
from factor_framework.engine.panel_builder import PanelBuilder
from factor_framework.research_config      import ResearchConfig

# IC 衰减诊断（懒加载，避免循环依赖）
def _lazy_diagnostics():
    from factor_framework.analytics.ic_decay_diagnostics import (
        ICDecayDiagnostics, DiagnosticReport,
    )
    return ICDecayDiagnostics, DiagnosticReport


def _lazy_advanced_diagnostics():
    from factor_framework.analytics.advanced_diagnostics import (
        run_advanced_diagnostics,
        AdvancedDiagnosticsReport,
    )
    return run_advanced_diagnostics, AdvancedDiagnosticsReport


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_monthly(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    将日频因子面板和收益率面板重采样到月末频率。

    逻辑
    ----
    - 取每月最后一个有效交易日的因子值（月末截面）
    - 对应的收益率面板取同一行（已是月度远期收益，无需再次聚合）
    - 两个面板按公共月末日期对齐

    这解决了"日频滚动模拟月度换仓导致换手率虚高"的问题：
    实盘每月换仓一次，而日频回测相当于每天都在换仓。
    """
    # 将 index 转为 pandas DatetimeIndex 以便 resample
    def _to_datetime(idx):
        # index 可能是 str '20200131' 或已是 datetime
        if pd.api.types.is_datetime64_any_dtype(idx):
            return idx
        try:
            return pd.to_datetime(idx, format="%Y%m%d")
        except Exception:
            return pd.to_datetime(idx)

    f_dt = factor_panel.copy()
    f_dt.index = _to_datetime(factor_panel.index)
    r_dt = return_panel.copy()
    r_dt.index = _to_datetime(return_panel.index)

    # 取每月末最后一个有值的日期（resample('ME').last()）
    try:
        f_m = f_dt.resample("ME").last()
        r_m = r_dt.resample("ME").last()
    except Exception:
        # pandas < 2.2 使用 'M'
        f_m = f_dt.resample("M").last()
        r_m = r_dt.resample("M").last()

    # 对齐
    common_idx = f_m.index.intersection(r_m.index)
    f_m = f_m.loc[common_idx]
    r_m = r_m.loc[common_idx]

    # 将 index 转回原始字符串格式（与下游兼容）
    str_idx = f_m.index.strftime("%Y%m%d")
    f_m.index = str_idx
    r_m.index = str_idx

    # 去掉全 NaN 行
    valid = f_m.dropna(how="all").index.intersection(r_m.dropna(how="all").index)
    return f_m.loc[valid], r_m.loc[valid]


def _resample_panel_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    """Resample a single panel to month-end using last valid row."""
    if panel is None or panel.empty:
        return panel
    p = panel.copy()
    if not pd.api.types.is_datetime64_any_dtype(p.index):
        try:
            p.index = pd.to_datetime(p.index, format="%Y%m%d")
        except Exception:
            p.index = pd.to_datetime(p.index)
    try:
        p_m = p.resample("ME").last()
    except Exception:
        p_m = p.resample("M").last()
    p_m.index = p_m.index.strftime("%Y%m%d")
    return p_m.dropna(how="all")


def _save_diag_report(report: object, out_dir: Path) -> None:
    """
    将 DiagnosticReport 的所有证据保存为标准化 CSV + JSON。

    目录结构
    --------
    out_dir/
      diagnostic_overview.csv          — 6 模块状态汇总
      final_judgement.json             — 最终结论与风险等级
      module1_alignment_audit.csv
      module2_incremental_ic.csv       — incr_ic_full DataFrame
      module2_cumulative_ic.csv
      module3_neutralized_compare.csv
      module4_survivorship_audit.csv
      module5_factor_horizon_profile.csv
      module5_incr_ic_by_k.csv
      module6_split_period_ic.csv
      module6_winsor_sensitivity.csv
      module6_regime_ic.csv
    """
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    overview_rows = []
    for r in report.results:
        overview_rows.append({
            "module_id":   r.module_id,
            "module_name": r.module_name,
            "passed":      r.passed,
            "risk_level":  r.risk_level,
            "conclusion":  r.conclusion,
        })
    pd.DataFrame(overview_rows).to_csv(out_dir / "diagnostic_overview.csv", index=False)

    # ── final verdict JSON ───────────────────────────────────────────────────
    fails = [r for r in report.results if r.passed is False]
    high_risk = [r for r in report.results if r.risk_level == "HIGH"]
    if len(fails) == 0:
        verdict, risk = "Genuine medium-horizon efficacy (all modules passed)", "LOW"
    elif any(r.module_id in (1, 2) for r in fails):
        verdict, risk = "Implementation bias (time alignment / cumulative stats issue)", "HIGH"
    elif len(fails) >= 3:
        verdict, risk = "Multi-source bias", "HIGH"
    elif high_risk:
        verdict, risk = "Driven by structural exposure", "MEDIUM"
    else:
        verdict, risk = "Partially plausible (further analysis recommended)", "MEDIUM"

    final = {
        "factor_name":   report.factor_name,
        "verdict":       verdict,
        "risk_level":    risk,
        "pass_count":    sum(1 for r in report.results if r.passed is True),
        "total_modules": len(report.results),
        "failed_modules": [f"M{r.module_id}" for r in fails],
    }
    with open(out_dir / "final_judgement.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # ── 各模块证据文件 ────────────────────────────────────────────────────────
    _MODULE_FILE_MAP = {
        1: [("evidence", "module1_alignment_audit.csv")],
        2: [("cumul_ic",       "module2_cumulative_ic.csv"),
            ("incr_ic_full",   "module2_incremental_ic.csv")],
        3: [("evidence",       "module3_neutralized_compare.csv")],
        4: [("evidence",       "module4_survivorship_audit.csv")],
        5: [("autocorr_by_lag_month",    "module5_factor_horizon_profile.csv"),
            ("incr_ic_by_k",             "module5_incr_ic_by_k.csv")],
        6: [("split_period_ic",    "module6_split_period_ic.csv"),
            ("winsor_sensitivity", "module6_winsor_sensitivity.csv"),
            ("regime_ic",          "module6_regime_ic.csv")],
    }

    for r in report.results:
        mappings = _MODULE_FILE_MAP.get(r.module_id, [])
        for ev_key, filename in mappings:
            # evidence 可能是 dict 或直接 DataFrame
            if isinstance(r.evidence, dict):
                ev = r.evidence.get(ev_key)
            elif ev_key == "evidence":
                ev = r.evidence
            else:
                ev = None
            if ev is None:
                continue
            try:
                if isinstance(ev, pd.DataFrame):
                    ev.to_csv(out_dir / filename)
                elif isinstance(ev, pd.Series):
                    ev.to_frame().to_csv(out_dir / filename)
            except Exception as _exc:
                warnings.warn(f"[diagnostics] Failed to save {filename}: {_exc}")

    print(f"[OK] IC decay diagnostics report saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# 报告对象
# ═══════════════════════════════════════════════════════════════════════════════

class FactorReport:
    """封装单个因子的完整检验结果。"""

    def __init__(
        self,
        factor_name:       str,
        ic_series:         pd.Series,
        ic_stats:          Dict,
        ic_nw:             Dict,
        ic_decay_df:       pd.DataFrame,
        layer_ret:         pd.DataFrame,
        ls_stats:          Dict,
        turnover:          Dict,
        factor_panel:      pd.DataFrame,
        return_panel:      pd.DataFrame,
        composite_weights: Optional[Dict[str, float]] = None,
        price_panel:       Optional[pd.DataFrame] = None,
        mktcap_panel:      Optional[pd.DataFrame] = None,
        industry_map:      Optional[pd.Series] = None,
    ):
        self.factor_name       = factor_name
        self.ic_series         = ic_series
        self.ic_stats_         = ic_stats
        self.ic_nw             = ic_nw
        self.ic_decay_df       = ic_decay_df
        self.composite_weights = composite_weights  # 仅多因子合成时有值
        self.layer_ret    = layer_ret
        self.ls_stats     = ls_stats
        self.turnover     = turnover
        self.factor_panel = factor_panel
        self.return_panel = return_panel
        # 诊断所需的补充面板（由 run(run_ic_decay_diagnostics=True) 时传入）
        self.price_panel  = price_panel
        self.mktcap_panel = mktcap_panel
        self.industry_map = industry_map
        # 诊断报告（运行后缓存，避免重复计算）
        self._diag_report: Optional[object] = None  # DiagnosticReport | None
        self._advanced_report: Optional[object] = None  # AdvancedDiagnosticsReport | None

    # ── IC 衰减诊断 ───────────────────────────────────────────────────────────

    def run_ic_diagnostics(
        self,
        forward_list: List[int] = (1, 5, 10, 21, 60),
        run_modules:  Optional[List[int]] = None,
        verbose:      bool = True,
        save_dir:     Optional[str | Path] = None,
    ) -> object:  # DiagnosticReport（懒加载，避免循环导入）
        """
        对本次回测结果运行 IC 衰减异常诊断（6 模块）。

        Parameters
        ----------
        forward_list : 诊断用预测期列表（默认与 ic_forward_list 一致）
        run_modules  : 指定运行哪些模块（默认全部 1-6）
        verbose      : 是否打印完整报告
        save_dir     : 若指定，自动保存诊断结果到该目录

        Returns
        -------
        DiagnosticReport

        Notes
        -----
        需要 ``price_panel``（原始收盘价）可用。
        通过 ``pipe.run(..., run_ic_decay_diagnostics=True)`` 自动填充，
        或手动调用 ``report.price_panel = price_df`` 后再调用此方法。
        """
        if self.price_panel is None:
            raise ValueError(
                "run_ic_diagnostics() requires price_panel.\n"
                "Use pipe.run(..., run_ic_decay_diagnostics=True) or set "
                "report.price_panel = price_df before calling this method."
            )

        ICDecayDiagnostics, _ = _lazy_diagnostics()
        diag = ICDecayDiagnostics(
            factor_panel = self.factor_panel,
            price_panel  = self.price_panel,
            forward_list = list(forward_list),
            industry_map = self.industry_map,
            mktcap_panel = self.mktcap_panel,
            ic_method    = "rank",
            factor_name  = self.factor_name,
        )
        report = diag.run_all(
            run_modules = run_modules,
            verbose     = verbose,
        )
        self._diag_report = report

        if save_dir is not None:
            _save_diag_report(report, Path(save_dir))

        return report

    # ── 打印汇总 ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """终端打印因子评估报告。"""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  Factor Name: {self.factor_name}")
        print(sep)

        # ── 合成权重（多因子时显示）────────────────────────────────────────
        if self.composite_weights:
            print(f"\n[Composite Factor Weights]")
            for name, w in sorted(self.composite_weights.items(), key=lambda x: -x[1]):
                bar = "█" * max(0, int(w * 40))
                print(f"  {name:<28} {w:>6.2%}  {bar}")
            print(f"  {'Total':<28} {sum(self.composite_weights.values()):>6.2%}")

        s = self.ic_stats_
        print(f"\n[IC Analysis]")
        print(f"  Mean IC      : {s.get('mean_ic', 'N/A'):.4f}  "
              f"(pass |IC| > 0.02, strong > 0.05)")
        print(f"  Std  IC      : {s.get('std_ic', 'N/A'):.4f}")
        print(f"  ICIR         : {s.get('icir', 'N/A'):.4f}  "
              f"(pass > 0.5, strong > 1.0)")
        print(f"  IC Win Rate  : {s.get('win_rate', 'N/A'):.1%}  "
              f"(pass > 55%)")
        print(f"  t-stat       : {s.get('t_stat', 'N/A'):.4f}  "
              f"(|t| > 2 is significant)")
        print(f"  Newey-West t : {self.ic_nw.get('nw_t_stat', 'N/A')}")
        print(f"  Valid Periods: {s.get('total_periods', 'N/A')}")

        print(f"\n[IC Decay (Mean IC by horizon)]")
        print(self.ic_decay_df[["mean_ic", "icir"]].to_string())

        ls = self.ls_stats
        print(f"\n[Layer Backtest · Long-Short]")
        print(f"  Annual Return: {ls.get('ls_annual_return', 'N/A'):.2%}  "
              f"(pass > 10%)")
        print(f"  Annual Sharpe: {ls.get('ls_sharpe', 'N/A'):.4f}  "
              f"(pass > 1.0)")
        print(f"  Max Drawdown : {ls.get('ls_max_drawdown', 'N/A'):.2%}  "
              f"(pass < 30%)")
        print(f"  Calmar Ratio : {ls.get('ls_calmar', 'N/A'):.4f}  "
              f"(pass > 0.5)")
        print(f"  LS Win Rate  : {ls.get('ls_win_rate', 'N/A'):.1%}  "
              f"(pass > 55%)")
        print(f"  Monotone Score: {ls.get('monotone_score', 'N/A'):.4f}  "
              f"(closer to 1 is better)")

        print(f"\n[Annualized Return by Layer]")
        ann = ls.get("layer_annual_return")
        if ann is not None:
            for k, v in ann.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    print(f"  {k}:    N/A")
                else:
                    bar = "█" * max(0, int(v * 100))
                    print(f"  {k}: {v:>8.2%}  {bar}")

        t = self.turnover
        print(f"\n[Turnover and Trading Cost]")
        print(f"  Avg One-way Turnover: {t.get('avg_turnover', 'N/A'):.2%}")
        print(f"  Estimated Cost/Period: {t.get('avg_cost', 'N/A'):.4%}")

        print(f"\n{sep}\n")

    def run_advanced_diagnostics(
        self,
        save_dir: Optional[str | Path] = None,
        *,
        n_groups: int = 5,
        direction: int = 1,
        periods_per_year: int = 12,
        peer_factors: Optional[List[str]] = None,
        min_cs_nobs: int = 20,
        corr_method: str = "spearman",
        high_corr_threshold: float = 0.7,
        nw_lag_rule: str = "t_pow_0.25",
        enable_wide_output: bool = True,
    ) -> object:
        """Run advanced diagnostics and optionally save outputs."""
        runner, _ = _lazy_advanced_diagnostics()
        out_dir = Path(save_dir) if save_dir is not None else Path("output") / self.factor_name / "advanced_diagnostics"
        adv_report = runner(
            self,
            output_dir=out_dir,
            n_groups=n_groups,
            direction=direction,
            periods_per_year=periods_per_year,
            peer_factors=peer_factors,
            min_cs_nobs=min_cs_nobs,
            corr_method=corr_method,
            high_corr_threshold=high_corr_threshold,
            nw_lag_rule=nw_lag_rule,
            enable_wide_output=enable_wide_output,
        )
        self._advanced_report = adv_report
        return adv_report

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

        # 合成权重（多因子时保存）
        if self.composite_weights:
            pd.Series(self.composite_weights, name="weight").to_csv(
                out / "composite_weights.csv", header=True
            )

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
        print(f"[OK] Report saved to {out}/")

        # ── 若诊断报告已运行，自动保存到 ic_decay_diagnostics/ 子目录 ───────
        if self._diag_report is not None:
            _save_diag_report(self._diag_report, out / "ic_decay_diagnostics")
        if self._advanced_report is not None:
            self._advanced_report.output_dir = out / "advanced_diagnostics"
            self._advanced_report.save()

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
    stocks_dir   : Stocks/ 目录路径
    stock_basic  : 股票列表 CSV（含 ts_code, industry 列）
    min_rows     : 新股最少有效行数
    verbose      : 显示进度条
    cache_dir    : 磁盘缓存目录（默认 "cache/"，None = 不启用 L2 Parquet 缓存）
                   指定后，计算超过 5 秒的面板将自动缓存到磁盘，
                   第二次运行直接从 Parquet 读取，大幅缩短耗时。
    min_calc_secs: L2 写入阈值（秒）；仅计算超过此时间的面板才写入磁盘
    store        : DataStore 实例（可选）；传入后优先通过 DataStore 读取数据，
                   默认自动构造 CSVDataStore(stocks_dir)
    """

    def __init__(
        self,
        stocks_dir:    str | Path = "Stocks/",
        stock_basic:   str | Path = "股票列表-stock_basic.csv",
        min_rows:      int = 60,
        verbose:       bool = True,
        cache_dir:     Optional[str | Path] = "cache/",
        min_calc_secs: float = 5.0,
        store=None,   # Optional[DataStore] — 避免循环导入，运行时检查
    ):
        # ── DataStore（B2: 自动构造 CSVDataStore 或使用传入的 store）───────
        if store is None:
            try:
                from factor_framework.data.store import CSVDataStore
                store = CSVDataStore(stocks_dir=str(stocks_dir))
            except Exception:
                store = None   # 构造失败时降级为 None（向后兼容）

        # ── 缓存层（B4: 默认 "cache/"，可传 None 禁用）──────────────────────
        _cache: Optional[CacheLayer] = None
        if cache_dir is not None:
            _cache = CacheLayer(
                cache_dir     = str(cache_dir),
                stocks_dir    = str(stocks_dir),
                enabled_l2    = True,
                min_calc_secs = min_calc_secs,
            )

        # ── PanelBuilder（含 FactorEngine）────────────────────────────────
        self._builder = PanelBuilder(
            stocks_dir  = stocks_dir,
            stock_basic = stock_basic,
            cache       = _cache,
            min_rows    = min_rows,
            verbose     = verbose,
            store       = store,
        )

        # ── 向后兼容：self.engine 指向底层 FactorEngine ──────────────────
        # 所有已有代码（register/apply_cross_section/industry_map 等）
        # 均通过 self.engine.xxx 访问，保持零改动。
        self.engine = self._builder.engine

        # ── Phase D: 暴露 cache 引用 + 最后一次运行的 manifest ──────────
        self._cache = _cache
        self.last_manifest = None  # type: Optional[object]  # RunManifest | None

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
                warnings.warn(f"Built-in factor '{n}' not found; skipped.")
        return self

    # ── 核心运行 ──────────────────────────────────────────────────────────────

    def run(
        self,
        factor_name:      Optional[str] = None,
        start:            Optional[str] = None,
        end:              Optional[str] = None,
        forward:          int = 21,
        n_groups:         int = 5,
        direction:        int = 1,
        standardize:      Optional[str] = "rank",   # 'rank', 'zscore', None
        neutralize:       bool = False,              # 市值+行业回归中性化
        winsorize:        bool = True,               # 截面 MAD Winsorize
        ic_method:        str = "rank",
        ic_forward_list:  List[int] = (1, 5, 10, 21, 60),
        periods_per_year: int = 252,
        rf:               float = 0.0,
        cost_per_side:    float = 0.002,
        symbols:          Optional[List[str]] = None,
        resample_monthly: bool = True,               # 月度重采样（推荐）
        config:           Optional[ResearchConfig] = None,  # Phase C: 结构化配置
        run_ic_decay_diagnostics: bool = False,      # 是否在回测后运行 IC 衰减诊断
        run_advanced_diagnostics: bool = False,      # 是否运行高级诊断包
        advanced_peer_factors: Optional[List[str]] = None,
        advanced_min_cs_nobs: int = 20,
        advanced_corr_method: str = "spearman",
        advanced_high_corr_threshold: float = 0.7,
        advanced_nw_lag_rule: str = "t_pow_0.25",
        advanced_enable_wide_output: bool = True,
    ) -> FactorReport:
        """
        执行完整的因子检验流程。

        双入口兼容（Phase C）
        ---------------------
        * **新方式**：``run(config=ResearchConfig(...))`` — 推荐，便于哈希和序列化
        * **旧方式**：``run(factor_name=..., forward=..., ...)`` — 继续支持，
          内部自动转换为 ResearchConfig，行为与之前完全一致

        Parameters
        ----------
        config           : ResearchConfig 实例（提供时所有其他参数被忽略）
        factor_name      : 已注册的因子名称（config=None 时必须提供）
        start / end      : 日期范围（YYYYMMDD）
        forward          : 预测期（天）
        n_groups         : 分层数
        direction        : 因子方向（+1 或 -1）
        standardize      : 横截面标准化方式（'rank','zscore',None）
        neutralize       : 是否做市值+行业中性化
        winsorize        : 是否先做截面 MAD Winsorize
        ic_method        : IC 计算方式（'rank' 或 'normal'）
        ic_forward_list  : IC 衰减分析的预测期列表
        periods_per_year : 年化期数（月度重采样后应传 12）
        rf               : 无风险利率（年化）
        cost_per_side    : 单边交易成本
        symbols          : 指定股票列表（None = 全部）
        resample_monthly : True = 每月末重采样一次（避免日频滚动模拟月度换仓，
                           换手率虚高）；False = 保留日频（适合短期因子）

        Returns
        -------
        FactorReport
        """
        # ── Phase C: 统一转换为 ResearchConfig ──────────────────────────
        if config is None:
            if not factor_name:
                raise ValueError(
                    "run() requires factor_name or config=ResearchConfig(...)."
                )
            config = ResearchConfig.from_kwargs(
                factor_name      = factor_name,
                start            = start,
                end              = end,
                forward          = forward,
                n_groups         = n_groups,
                direction        = direction,
                standardize      = standardize,
                neutralize       = neutralize,
                winsorize        = winsorize,
                ic_method        = ic_method,
                ic_forward_list  = tuple(ic_forward_list),
                periods_per_year = periods_per_year,
                rf               = rf,
                cost_per_side    = cost_per_side,
                symbols          = symbols,
                resample_monthly = resample_monthly,
                advanced_peer_factors        = advanced_peer_factors,
                advanced_min_cs_nobs         = advanced_min_cs_nobs,
                advanced_corr_method         = advanced_corr_method,
                advanced_high_corr_threshold = advanced_high_corr_threshold,
                advanced_nw_lag_rule         = advanced_nw_lag_rule,
                advanced_enable_wide_output  = advanced_enable_wide_output,
            )
        # 从 config 提取所有运行参数（统一来源）
        factor_name      = config.factor_name
        start            = config.start
        end              = config.end
        forward          = config.forward
        n_groups         = config.n_groups
        direction        = config.direction
        standardize      = config.standardize
        neutralize       = config.neutralize
        winsorize        = config.winsorize
        ic_method        = config.ic_method
        ic_forward_list  = config.ic_forward_list
        periods_per_year = config.periods_per_year
        rf               = config.rf
        cost_per_side    = config.cost_per_side
        symbols          = config.symbols
        resample_monthly = config.resample_monthly
        advanced_peer_factors        = config.advanced_peer_factors
        advanced_min_cs_nobs         = config.advanced_min_cs_nobs
        advanced_corr_method         = config.advanced_corr_method
        advanced_high_corr_threshold = config.advanced_high_corr_threshold
        advanced_nw_lag_rule         = config.advanced_nw_lag_rule
        advanced_enable_wide_output  = config.advanced_enable_wide_output
        import time as _time
        _run_start = _time.perf_counter()
        if self._cache is not None:
            self._cache.reset_stats()

        print(f"\n[1/6] Building factor panel: {factor_name} ...")
        factor_panel = self._builder.build_panel(
            factor_name, start=start, end=end, symbols=symbols
        )
        if factor_panel.empty:
            raise ValueError(f"Factor panel '{factor_name}' is empty; check factor function or source data.")

        print(f"      Factor panel: {factor_panel.shape[0]} trading days x {factor_panel.shape[1]} symbols")

        print(f"\n[2/6] Building return panel (forward={forward}, T+1 already applied) ...")
        return_panel = self._builder.build_return_panel(
            forward=forward, start=start, end=end, symbols=symbols
        )

        # ── 截断尾部 NaN（forward+1 行因 shift(-forward).shift(1) 无效）────
        # build_return_panel 内置 T+1 shift，尾部共 forward+1 行为全 NaN
        valid_ret_idx = return_panel.dropna(how="all").index
        n_dropped = len(return_panel) - len(valid_ret_idx)
        if n_dropped > 0:
            warnings.warn(
                f"[tail-trim] return_panel tail has {n_dropped} all-NaN trading days "
                f"(including T+1 lag); aligned rows were trimmed from factor_panel."
            )
            return_panel = return_panel.loc[valid_ret_idx]
            factor_panel = factor_panel.loc[factor_panel.index.intersection(valid_ret_idx)]

        # ── 截面预处理（在日频上执行，避免月度重采样后频率不匹配）──────────
        # BUG 11 FIX: mktcap_panel 必须 reindex 到已截断的日频 factor_panel.index，
        # 否则月度重采样后行数不匹配。
        # BUG 10 NOTE: T+1 已内置于 build_return_panel（price.shift(-fwd)/price-1 再
        # .shift(1)），即 return_panel[t] = t+1 日起持有的远期收益。故 factor_panel[t]
        # （t 日收盘计算）与 return_panel[t] 对齐是正确的，无需额外移位。
        print(f"\n[3/6] Cross-sectional preprocessing (winsorize={winsorize}, standardize={standardize}, neutralize={neutralize}) ...")

        if winsorize:
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_winsorize)

        if neutralize and self.engine.industry_map is not None:
            # 临时注册市值因子
            self.engine.register("__mktcap__", lambda df: df["总市值（万元）"])
            mktcap_panel = self._builder.build_panel("__mktcap__", start=start, end=end, symbols=symbols)
            del self.engine._registry["__mktcap__"]
            # BUG 11 FIX: 对齐到已截断的日频 factor_panel.index
            mktcap_panel = mktcap_panel.reindex(factor_panel.index)
            factor_panel = neutralize_regression(
                factor_panel,
                mktcap_panel,
                industry_map = self.engine.industry_map,
            )
        elif neutralize:
            warnings.warn("neutralize=True but industry_map is empty; neutralization skipped.")

        if standardize == "rank":
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_rank)
        elif standardize == "zscore":
            factor_panel = self.engine.apply_cross_section(factor_panel, cs_zscore)

        # ── IC 衰减分析（在月度重采样前用日频面板执行）─────────────────────
        # BUG 12 FIX: ic_decay 必须在 resample_monthly 之前调用，否则月末索引与日频
        # 价格面板的 intersection 几乎为空。
        # BUG 9 FIX: 为每个 ic_forward_list 中的 forward 构建收益率面板，
        # 传入 ic_decay 的 return_panels 参数，消除双路径不一致问题。
        # 注意：此处在日频面板上构建多个 forward 的收益率，与主 IC 路径同源。
        print(f"\n[4/6] IC analysis (method={ic_method}) ...")
        ic_return_panels: Dict[int, pd.DataFrame] = {}
        for _fwd in ic_forward_list:
            _rp = self._builder.build_return_panel(
                forward=_fwd, start=start, end=end, symbols=symbols
            )
            # 截断尾部全 NaN 行（forward+1 行因 T+1 shift 无效）
            _valid_idx = _rp.dropna(how="all").index
            ic_return_panels[_fwd] = _rp.loc[_valid_idx]
        ic_decay_df = ic_decay(
            factor_panel,
            return_panels=ic_return_panels,
            method=ic_method,
        )

        # ── 诊断所需面板（在 resample 前取原始日频数据）────────────────────
        # run_ic_decay_diagnostics=True 时，将 price_panel / mktcap_panel 传给 report
        _diag_price_panel  = None
        _diag_mktcap_panel = None
        _diag_industry_map = None
        _diag_factor_panel_raw = None  # 诊断需要未经 resample 的因子面板
        if run_ic_decay_diagnostics:
            try:
                self.engine.register("__close_diag__", lambda df: df["收盘价"])
                _diag_price_panel = self._builder.build_panel(
                    "__close_diag__", start=start, end=end, symbols=symbols
                )
                del self.engine._registry["__close_diag__"]
            except Exception as _pe:
                warnings.warn(f"[diagnostics] Failed to fetch price_panel: {_pe}")
            try:
                self.engine.register("__mktcap_diag__", lambda df: df["总市值（万元）"])
                _diag_mktcap_panel = self._builder.build_panel(
                    "__mktcap_diag__", start=start, end=end, symbols=symbols
                )
                del self.engine._registry["__mktcap_diag__"]
            except Exception:
                pass
            _diag_industry_map = self.engine.industry_map
            # 保存 resample 前的日频因子面板（诊断模块需要原始日频）
            _diag_factor_panel_raw = factor_panel.copy()

        # ── 月度重采样（可选，在截面预处理和 IC 衰减之后）──────────────────
        # BUG 13/15 NOTE: 月度重采样后 factor_panel/return_panel 为月末截面，
        # layer_backtest 和 turnover_analysis 中每行对应一个月，不存在日频滚动重叠。
        if resample_monthly:
            factor_panel, return_panel = _resample_monthly(factor_panel, return_panel)
            print(f"      [monthly-resample] {factor_panel.shape[0]} month-end snapshots")

        # ── IC 分析（月度重采样后执行，与 return_panel 频率一致）────────────
        ic_series = compute_ic(factor_panel, return_panel, method=ic_method)
        ic_s      = ic_stats(ic_series, annualize_periods=periods_per_year)
        ic_nw     = ic_significance(ic_series, lags=max(1, int(len(ic_series) ** 0.25)))

        # ── 分层回测 ──────────────────────────────────────────────────────────
        # BUG 14 NOTE: periods_per_year 由调用方指定（月度重采样后应传 12），
        # _annual_return 使用 total^(periods_per_year/n)-1，月频输入时正确年化。
        print(f"\n[5/6] Layer backtest (n_groups={n_groups}) ...")
        layer_ret = layer_backtest(
            factor_panel, return_panel,
            n_groups=n_groups, direction=direction
        )
        ls_stats_ = long_short_stats(layer_ret, periods_per_year=periods_per_year, rf=rf)

        # ── 换手率分析 ────────────────────────────────────────────────────────
        print(f"\n[6/6] Turnover analysis ...")
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
            price_panel  = _diag_price_panel,
            mktcap_panel = _diag_mktcap_panel,
            industry_map = _diag_industry_map,
        )

        # ── IC 衰减诊断（可选）──────────────────────────────────────────────
        if run_ic_decay_diagnostics and _diag_price_panel is not None:
            print(f"\n[+] IC decay diagnostics (6 modules) ...")
            try:
                _factor_panel_monthly = report.factor_panel
                # 诊断模块使用日频原始因子面板（resample 前），不含 T+1
                # _diag_factor_panel_raw 已经过 winsorize/neutralize/standardize
                # 但未经 T+1 shift（诊断模块内部按需 shift）
                report.factor_panel = _diag_factor_panel_raw if _diag_factor_panel_raw is not None else factor_panel
                diag_report = report.run_ic_diagnostics(
                    forward_list = list(ic_forward_list),
                    verbose      = True,
                )
                report.factor_panel = _factor_panel_monthly
                # save() 时自动保存到 ic_decay_diagnostics/ 子目录
            except Exception as _de:
                warnings.warn(f"[diagnostics] IC decay diagnostics failed (non-fatal): {_de}")

        if run_advanced_diagnostics:
            print(f"\n[+] Advanced Diagnostics Pack ...")
            try:
                # ── Auto-build peer factor panels (user-provided panels keep priority) ──
                explicit_peers = getattr(report, "peer_factor_panels", None)
                explicit_peers = explicit_peers if isinstance(explicit_peers, dict) else {}
                auto_peer_names = list(
                    advanced_peer_factors
                    if advanced_peer_factors is not None
                    else ["value_pb", "momentum_12_1", "vol_20d", "turnover_rate", "size_log_mktcap"]
                )
                auto_peer_panels: Dict[str, pd.DataFrame] = {}
                unavailable_peers: List[str] = []

                # Ensure mktcap panel for advanced modules if absent.
                if report.mktcap_panel is None:
                    try:
                        self.engine.register("__mktcap_adv__", lambda df: df["总市值（万元）"])
                        _mc = self._builder.build_panel("__mktcap_adv__", start=start, end=end, symbols=symbols)
                        del self.engine._registry["__mktcap_adv__"]
                        _mc = _resample_panel_monthly(_mc) if resample_monthly else _mc
                        report.mktcap_panel = _mc.reindex(report.factor_panel.index)
                    except Exception as _mce:
                        unavailable_peers.append("mktcap_panel")

                # Value exposure for HML construction in alpha models.
                if getattr(report, "value_panel", None) is None:
                    try:
                        from factor_framework.factor_zoo import BUILTIN_FACTORS
                        _tmp_reg = False
                        if "value_pb" not in self.engine._registry and "value_pb" in BUILTIN_FACTORS:
                            self.engine.register("value_pb", BUILTIN_FACTORS["value_pb"])
                            _tmp_reg = True
                        _vp = self._builder.build_panel("value_pb", start=start, end=end, symbols=symbols)
                        if _tmp_reg:
                            del self.engine._registry["value_pb"]
                        _vp = _resample_panel_monthly(_vp) if resample_monthly else _vp
                        report.value_panel = _vp.reindex(report.factor_panel.index)
                    except Exception:
                        report.value_panel = None

                # Build built-in peer factor panels; never override explicit input.
                from factor_framework.factor_zoo import BUILTIN_FACTORS
                for _peer in auto_peer_names:
                    if _peer in explicit_peers:
                        continue
                    try:
                        _tmp_reg = False
                        if _peer not in self.engine._registry and _peer in BUILTIN_FACTORS:
                            self.engine.register(_peer, BUILTIN_FACTORS[_peer])
                            _tmp_reg = True
                        _pp = self._builder.build_panel(_peer, start=start, end=end, symbols=symbols)
                        if _tmp_reg:
                            del self.engine._registry[_peer]
                        if winsorize:
                            _pp = self.engine.apply_cross_section(_pp, cs_winsorize)
                        if standardize == "rank":
                            _pp = self.engine.apply_cross_section(_pp, cs_rank)
                        elif standardize == "zscore":
                            _pp = self.engine.apply_cross_section(_pp, cs_zscore)
                        if resample_monthly:
                            _pp = _resample_panel_monthly(_pp)
                        auto_peer_panels[_peer] = _pp.reindex(report.factor_panel.index)
                    except Exception:
                        unavailable_peers.append(_peer)

                merged_peer_panels = dict(auto_peer_panels)
                merged_peer_panels.update(explicit_peers)
                report.peer_factor_panels = merged_peer_panels
                report.unavailable_peer_factors = sorted(
                    set(list(getattr(report, "unavailable_peer_factors", []) or []) + unavailable_peers)
                )

                report.run_advanced_diagnostics(
                    n_groups=n_groups,
                    direction=direction,
                    periods_per_year=periods_per_year,
                    peer_factors=auto_peer_names,
                    min_cs_nobs=advanced_min_cs_nobs,
                    corr_method=advanced_corr_method,
                    high_corr_threshold=advanced_high_corr_threshold,
                    nw_lag_rule=advanced_nw_lag_rule,
                    enable_wide_output=advanced_enable_wide_output,
                )
            except Exception as _ade:
                warnings.warn(f"[advanced_diagnostics] Failed (non-fatal): {_ade}")

        # ── Phase D: 生成 RunManifest ─────────────────────────────────────
        try:
            from factor_framework.manifest import RunManifest
            _ci = self._cache.cache_info() if self._cache is not None else {}
            self.last_manifest = RunManifest.create(
                factors    = [factor_name],
                cfg        = config,           # Phase C: 传入标准化的 ResearchConfig
                cache_info = _ci,
                start_time = _run_start,
                failures   = [],
                stocks_dir = self._builder.stocks_dir,
                git_sha    = (_ci.get("git_sha") if _ci else None),
            )
        except Exception as _mex:
            warnings.warn(f"[manifest] Generation failed (non-fatal): {_mex}")

        print("\n[OK] Pipeline completed.")
        return report

    # ── 批量多因子运行（面板预构建版）──────────────────────────────────────────

    def run_batch_from_panels(
        self,
        factor_panels:    Dict[str, pd.DataFrame],
        return_panel:     pd.DataFrame,
        close_panel:      Optional[pd.DataFrame] = None,
        forward:          int = 21,
        n_groups:         int = 5,
        direction:        int = 1,
        standardize:      Optional[str] = "rank",
        neutralize:       bool = False,
        winsorize:        bool = True,
        ic_method:        str = "rank",
        ic_forward_list:  List[int] = (1, 5, 10, 21, 60),
        periods_per_year: int = 252,
        rf:               float = 0.0,
        cost_per_side:    float = 0.002,
        resample_monthly: bool = True,
        ic_return_panels: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> Dict[str, "FactorReport"]:
        """
        对已预构建的因子面板字典批量执行检验流程（无重复读盘）。

        与逐因子调用 run() 相比，此方法：
        - 收益率面板和价格面板只传入一次（调用方负责构建）
        - 免去每因子重建 ThreadPoolExecutor 的开销
        - 截面预处理在此统一做（每因子仍独立标准化）

        Parameters
        ----------
        factor_panels     : {factor_name: raw_factor_panel}（build_panel_batch 的输出）
        return_panel      : 已构建的主 forward 收益率面板（含 T+1 滞后）
        close_panel       : （已废弃，保留向后兼容，传入会被忽略）
                            BUG-9 修复后，IC 衰减使用 ic_return_panels，不再需要
        ic_return_panels  : {forward_days: 收益率面板}，用于 IC 衰减分析（BUG-9 修复）
                            若为 None，则在方法内部自动构建
        resample_monthly  : True = 月度重采样（推荐，与 forward=21 月度换仓语义一致）
        其余参数          : 同 run()

        Returns
        -------
        dict: {factor_name: FactorReport}
        """
        # 截断尾部 NaN（一次性对齐，所有因子共用；T+1 内置后尾部多 1 行 NaN）
        valid_ret_idx = return_panel.dropna(how="all").index
        n_dropped = len(return_panel) - len(valid_ret_idx)
        if n_dropped > 0:
            warnings.warn(
                f"[tail-trim] return_panel tail has {n_dropped} all-NaN trading days "
                f"(including T+1 lag); aligned rows were trimmed from all factor panels."
            )
            return_panel = return_panel.loc[valid_ret_idx]

        # ── BUG-9 修复：构建多 forward 收益率面板（所有因子共用，一次性构建）──
        # 若调用方未传入 ic_return_panels，则从 builder 自动构建（带缓存）
        if ic_return_panels is None:
            ic_return_panels = {}
            for _fwd in ic_forward_list:
                _rp = self._builder.build_return_panel(forward=_fwd)
                _valid_idx = _rp.dropna(how="all").index
                ic_return_panels[_fwd] = _rp.loc[_valid_idx]

        reports: Dict[str, FactorReport] = {}
        total = len(factor_panels)

        for i, (factor_name, raw_panel) in enumerate(factor_panels.items(), 1):
            print(f"\n[{i:02d}/{total}] Evaluating factor: {factor_name} ...")
            try:
                if raw_panel.empty:
                    raise ValueError("factor panel is empty")

                # 对齐尾部截断（用 loc+intersection 避免 reindex 引入幽灵 NaN 行）
                common_idx   = raw_panel.index.intersection(valid_ret_idx)
                factor_panel = raw_panel.loc[common_idx]
                ret_panel    = return_panel.loc[common_idx]

                # ── 截面预处理（在日频上执行，避免月度重采样后频率不匹配）──
                # BUG 11 FIX: mktcap/neutralize 必须在 resample 前，在已截断的日频上做
                if winsorize:
                    factor_panel = self.engine.apply_cross_section(factor_panel, cs_winsorize)

                if neutralize and self.engine.industry_map is not None:
                    self.engine.register("__mktcap__", lambda df: df["总市值（万元）"])
                    mktcap_panel = self._builder.build_panel("__mktcap__")
                    del self.engine._registry["__mktcap__"]
                    # BUG 11 FIX: reindex 到已截断的日频 factor_panel.index
                    mktcap_panel = mktcap_panel.reindex(factor_panel.index)
                    factor_panel = neutralize_regression(
                        factor_panel, mktcap_panel,
                        industry_map=self.engine.industry_map,
                    )
                elif neutralize:
                    warnings.warn("neutralize=True but industry_map is empty; neutralization skipped.")

                if standardize == "rank":
                    factor_panel = self.engine.apply_cross_section(factor_panel, cs_rank)
                elif standardize == "zscore":
                    factor_panel = self.engine.apply_cross_section(factor_panel, cs_zscore)

                # ── IC 衰减分析（在月度重采样前用日频面板执行）──────────────
                # BUG 12 FIX: ic_decay 必须在 resample_monthly 之前调用
                # BUG 9  FIX: 使用 ic_return_panels（与主 IC 同源），消除双路径
                ic_decay_df = ic_decay(
                    factor_panel,
                    return_panels=ic_return_panels,
                    method=ic_method,
                )

                # ── 月度重采样（在截面预处理和 IC 衰减之后）────────────────
                # BUG 13/15 NOTE: 月度重采样后每行对应一个月，无日频滚动重叠
                if resample_monthly:
                    factor_panel, ret_panel = _resample_monthly(factor_panel, ret_panel)

                # ── IC 分析（月度重采样后与 ret_panel 频率一致）─────────────
                ic_series   = compute_ic(factor_panel, ret_panel, method=ic_method)
                ic_s        = ic_stats(ic_series, annualize_periods=periods_per_year)
                ic_nw       = ic_significance(
                    ic_series, lags=max(1, int(len(ic_series) ** 0.25))
                )

                # ── 分层回测 ────────────────────────────────────────────────
                layer_ret = layer_backtest(
                    factor_panel, ret_panel,
                    n_groups=n_groups, direction=direction,
                )
                ls_stats_ = long_short_stats(
                    layer_ret, periods_per_year=periods_per_year, rf=rf,
                )

                # ── 换手率 ──────────────────────────────────────────────────
                turnover_ = turnover_analysis(
                    factor_panel, n_groups=n_groups, direction=direction,
                    cost_per_side=cost_per_side,
                )

                reports[factor_name] = FactorReport(
                    factor_name  = factor_name,
                    ic_series    = ic_series,
                    ic_stats     = ic_s,
                    ic_nw        = ic_nw,
                    ic_decay_df  = ic_decay_df,
                    layer_ret    = layer_ret,
                    ls_stats     = ls_stats_,
                    turnover     = turnover_,
                    factor_panel = factor_panel,
                    return_panel = ret_panel,
                )
            except Exception as exc:
                warnings.warn(f"Factor '{factor_name}' evaluation failed: {exc}")

        return reports

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
                warnings.warn(f"Factor '{name}' evaluation failed: {e}")
                rows.append({"factor": name})
        return pd.DataFrame(rows).set_index("factor")

    # ── 多因子合成 ────────────────────────────────────────────────────────────

    def run_composite(
        self,
        factor_names:      List[str],
        method:            str = "equal",           # 'equal' | 'icir'
        icir_window:       Optional[int] = 12,      # ICIR 滚动窗口（期数）
        composite_name:    str = "composite",       # 合成因子名称（用于报告）
        start:             Optional[str] = None,
        end:               Optional[str] = None,
        forward:           int = 21,
        n_groups:          int = 5,
        direction:         int = 1,
        standardize:       Optional[str] = "rank",
        neutralize:        bool = False,
        winsorize:         bool = True,
        ic_method:         str = "rank",
        ic_forward_list:   List[int] = (1, 5, 10, 21, 60),
        periods_per_year:  int = 252,
        rf:                float = 0.0,
        cost_per_side:     float = 0.002,
        symbols:           Optional[List[str]] = None,
    ) -> FactorReport:
        """
        多因子合成流程：先对各单因子分别构建面板并计算 IC，
        再按指定方法合成为组合信号，最后执行完整的回测检验。

        Parameters
        ----------
        factor_names   : 已注册的因子名称列表
        method         : 合成方法
                         'equal' — 等权组合（§2.4.1）
                         'icir'  — ICIR 加权（§2.4.2）
        icir_window    : ICIR 加权时的滚动窗口期数（None = 全样本）
        composite_name : 合成因子在报告中的显示名称
        其余参数       : 同 run() 方法

        Returns
        -------
        FactorReport  包含合成因子的完整检验结果，
                      report.composite_weights 存储各因子权重。
        """
        if not factor_names:
            raise ValueError("factor_names cannot be empty.")

        method = method.lower().strip()
        if method not in ("equal", "icir"):
            raise ValueError(f"Unsupported composition method '{method}'. Choose 'equal' or 'icir'.")

        # ── Step 1：逐因子构建面板 ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Multi-factor composition  [{composite_name}]  method={method}")
        print(f"{'='*60}")

        factor_panels:  Dict[str, pd.DataFrame] = {}
        ic_series_dict: Dict[str, pd.Series]    = {}

        for i, name in enumerate(factor_names, 1):
            print(f"\n[Factor {i}/{len(factor_names)}] Building panel: {name} ...")
            raw_panel = self._builder.build_panel(
                name, start=start, end=end, symbols=symbols
            )
            if raw_panel.empty:
                warnings.warn(f"Factor '{name}' panel is empty; skipped.")
                continue

            # 截面预处理（每个因子独立处理）
            panel = raw_panel.copy()
            if winsorize:
                panel = self.engine.apply_cross_section(panel, cs_winsorize)
            if neutralize and self.engine.industry_map is not None:
                self.engine.register("__mktcap__", lambda df: df["总市值（万元）"])
                mktcap_panel = self._builder.build_panel(
                    "__mktcap__", start=start, end=end, symbols=symbols
                )
                del self.engine._registry["__mktcap__"]
                panel = neutralize_regression(
                    panel, mktcap_panel,
                    industry_map=self.engine.industry_map,
                )
            if standardize == "rank":
                panel = self.engine.apply_cross_section(panel, cs_rank)
            elif standardize == "zscore":
                panel = self.engine.apply_cross_section(panel, cs_zscore)

            factor_panels[name] = panel

        if not factor_panels:
            raise ValueError("All factor panels are empty; cannot compose.")

        # ── Step 2：构建收益率面板（公共一份）──────────────────────────────
        print(f"\nBuilding return panel (forward={forward}) ...")
        return_panel = self._builder.build_return_panel(
            forward=forward, start=start, end=end, symbols=symbols
        )

        # ── 截断尾部 NaN（与 run() 保持一致）─────────────────────────────
        valid_ret_idx = return_panel.dropna(how="all").index
        n_dropped = len(return_panel) - len(valid_ret_idx)
        if n_dropped > 0:
            warnings.warn(
                f"[tail-trim] return_panel tail has {n_dropped} all-NaN trading days "
                f"due to forward={forward} shift; aligned rows were trimmed from all factor panels."
            )
            return_panel  = return_panel.loc[valid_ret_idx]
            factor_panels = {
                name: panel.reindex(valid_ret_idx)
                for name, panel in factor_panels.items()
            }

        # ── Step 3：逐因子计算 IC（ICIR 加权需要）──────────────────────────
        if method == "icir":
            print(f"\nComputing IC per factor (ICIR rolling window={icir_window}) ...")
            for name, panel in factor_panels.items():
                ic_series_dict[name] = compute_ic(panel, return_panel, method=ic_method)

        # ── Step 4：合成因子 ─────────────────────────────────────────────────
        print(f"\nComposing factor (method={method}) ...")
        if method == "equal":
            composite_panel, weights = equal_weight(factor_panels)
        else:  # icir
            composite_panel, weights = icir_weight(
                factor_panels,
                ic_series_dict,
                window=icir_window,
            )

        # 打印权重
        icir_vals = None
        if method == "icir" and ic_series_dict:
            icir_vals = {}
            for name, ic in ic_series_dict.items():
                ic_clean = ic.dropna()
                if icir_window is not None:
                    ic_clean = ic_clean.iloc[-icir_window:] if len(ic_clean) >= icir_window else ic_clean
                if len(ic_clean) >= 2:
                    mean_ic = float(ic_clean.mean())
                    std_ic  = float(ic_clean.std(ddof=1))
                    icir_vals[name] = mean_ic / std_ic if std_ic > 0 else 0.0
        print_weights(weights, method={"equal": "Equal Weight", "icir": "ICIR Weight"}[method], icir_dict=icir_vals)

        # ── Step 5：合成因子的 IC 分析 ────────────────────────────────────
        print(f"Computing composite factor IC ...")
        ic_series = compute_ic(composite_panel, return_panel, method=ic_method)
        ic_s      = ic_stats(ic_series, annualize_periods=periods_per_year)
        ic_nw     = ic_significance(ic_series, lags=max(1, int(len(ic_series) ** 0.25)))

        # IC 衰减（BUG-9 修复：使用同源多 forward 收益率面板，不从 close_panel 重算）
        ic_ret_panels_composite: Dict[int, pd.DataFrame] = {}
        for _fwd in ic_forward_list:
            _rp = self._builder.build_return_panel(
                forward=_fwd, start=start, end=end, symbols=symbols
            )
            _valid_idx = _rp.dropna(how="all").index
            ic_ret_panels_composite[_fwd] = _rp.loc[_valid_idx]
        ic_decay_df = ic_decay(
            composite_panel,
            return_panels=ic_ret_panels_composite,
            method=ic_method,
        )

        # ── Step 6：分层回测 ──────────────────────────────────────────────
        print(f"Layer backtest (n_groups={n_groups}) ...")
        layer_ret = layer_backtest(
            composite_panel, return_panel,
            n_groups=n_groups, direction=direction
        )
        ls_stats_ = long_short_stats(layer_ret, periods_per_year=periods_per_year, rf=rf)

        # ── Step 7：换手率 ────────────────────────────────────────────────
        print(f"Turnover analysis ...")
        turnover_ = turnover_analysis(
            composite_panel, n_groups=n_groups, direction=direction,
            cost_per_side=cost_per_side
        )

        report = FactorReport(
            factor_name        = composite_name,
            ic_series          = ic_series,
            ic_stats           = ic_s,
            ic_nw              = ic_nw,
            ic_decay_df        = ic_decay_df,
            layer_ret          = layer_ret,
            ls_stats           = ls_stats_,
            turnover           = turnover_,
            factor_panel       = composite_panel,
            return_panel       = return_panel,
            composite_weights  = weights,
        )

        print("\n[OK] Multi-factor composition completed.")
        return report
