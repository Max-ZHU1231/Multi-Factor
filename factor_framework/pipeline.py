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

    # ── 打印汇总 ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """终端打印因子评估报告。"""
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  因子名称: {self.factor_name}")
        print(sep)

        # ── 合成权重（多因子时显示）────────────────────────────────────────
        if self.composite_weights:
            print(f"\n【因子合成权重】")
            for name, w in sorted(self.composite_weights.items(), key=lambda x: -x[1]):
                bar = "█" * max(0, int(w * 40))
                print(f"  {name:<28} {w:>6.2%}  {bar}")
            print(f"  {'合计':<28} {sum(self.composite_weights.values()):>6.2%}")

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
                warnings.warn(f"内置因子 '{n}' 不存在，已跳过。")
        return self

    # ── 核心运行 ──────────────────────────────────────────────────────────────

    def run(
        self,
        factor_name:      str,
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
        import time as _time
        _run_start = _time.perf_counter()
        if self._cache is not None:
            self._cache.reset_stats()

        print(f"\n[1/6] 构建因子面板: {factor_name} ...")
        factor_panel = self._builder.build_panel(
            factor_name, start=start, end=end, symbols=symbols
        )
        if factor_panel.empty:
            raise ValueError(f"因子 '{factor_name}' 面板为空，请检查因子函数或数据。")

        print(f"      因子面板: {factor_panel.shape[0]} 个交易日 × {factor_panel.shape[1]} 只股票")

        print(f"\n[2/6] 构建收益率面板（forward={forward}天，T+1 已内置）...")
        return_panel = self._builder.build_return_panel(
            forward=forward, start=start, end=end, symbols=symbols
        )

        # ── 截断尾部 NaN（forward+1 行因 shift(-forward).shift(1) 无效）────
        # build_return_panel 内置 T+1 shift，尾部共 forward+1 行为全 NaN
        valid_ret_idx = return_panel.dropna(how="all").index
        n_dropped = len(return_panel) - len(valid_ret_idx)
        if n_dropped > 0:
            warnings.warn(
                f"[尾部截断] return_panel 末尾 {n_dropped} 个交易日（含 T+1 滞后）"
                f"全为 NaN，已同步截断 factor_panel 对应行。"
            )
            return_panel = return_panel.loc[valid_ret_idx]
            factor_panel = factor_panel.loc[factor_panel.index.intersection(valid_ret_idx)]

        # ── 截面预处理（在日频上执行，避免月度重采样后频率不匹配）──────────
        # BUG 11 FIX: mktcap_panel 必须 reindex 到已截断的日频 factor_panel.index，
        # 否则月度重采样后行数不匹配。
        # BUG 10 NOTE: T+1 已内置于 build_return_panel（price.shift(-fwd)/price-1 再
        # .shift(1)），即 return_panel[t] = t+1 日起持有的远期收益。故 factor_panel[t]
        # （t 日收盘计算）与 return_panel[t] 对齐是正确的，无需额外移位。
        print(f"\n[3/6] 截面预处理（winsorize={winsorize}, standardize={standardize}, neutralize={neutralize}）...")

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
            warnings.warn("neutralize=True 但 industry_map 为空，跳过中性化。")

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
        print(f"\n[4/6] IC 分析（method={ic_method}）...")
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

        # ── 月度重采样（可选，在截面预处理和 IC 衰减之后）──────────────────
        # BUG 13/15 NOTE: 月度重采样后 factor_panel/return_panel 为月末截面，
        # layer_backtest 和 turnover_analysis 中每行对应一个月，不存在日频滚动重叠。
        if resample_monthly:
            factor_panel, return_panel = _resample_monthly(factor_panel, return_panel)
            print(f"      [月度重采样] {factor_panel.shape[0]} 个月末截面")

        # ── IC 分析（月度重采样后执行，与 return_panel 频率一致）────────────
        ic_series = compute_ic(factor_panel, return_panel, method=ic_method)
        ic_s      = ic_stats(ic_series, annualize_periods=periods_per_year)
        ic_nw     = ic_significance(ic_series, lags=max(1, int(len(ic_series) ** 0.25)))

        # ── 分层回测 ──────────────────────────────────────────────────────────
        # BUG 14 NOTE: periods_per_year 由调用方指定（月度重采样后应传 12），
        # _annual_return 使用 total^(periods_per_year/n)-1，月频输入时正确年化。
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

        # ── Phase D: 生成 RunManifest ─────────────────────────────────────
        try:
            from factor_framework.manifest import RunManifest
            _ci = self._cache.cache_info() if self._cache is not None else {}
            self.last_manifest = RunManifest.create(
                factors    = [factor_name],
                cfg        = {},           # 无 cfg 时传空 dict
                cache_info = _ci,
                start_time = _run_start,
                failures   = [],
                stocks_dir = self._builder.stocks_dir,
                git_sha    = (_ci.get("git_sha") if _ci else None),
            )
        except Exception as _mex:
            warnings.warn(f"[manifest] 生成失败（非致命）: {_mex}")

        print("\n✓ 流程完成。")
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
                f"[尾部截断] return_panel 末尾 {n_dropped} 个交易日（含 T+1 滞后）"
                f"全为 NaN，已同步截断所有因子面板对应行。"
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
            print(f"\n[{i:02d}/{total}] 检验因子: {factor_name} ...")
            try:
                if raw_panel.empty:
                    raise ValueError("因子面板为空")

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
                    warnings.warn("neutralize=True 但 industry_map 为空，跳过中性化。")

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
                warnings.warn(f"因子 '{factor_name}' 检验失败: {exc}")

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
                warnings.warn(f"因子 '{name}' 检验失败: {e}")
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
            raise ValueError("factor_names 不能为空。")

        method = method.lower().strip()
        if method not in ("equal", "icir"):
            raise ValueError(f"不支持的合成方法 '{method}'，请选择 'equal' 或 'icir'。")

        # ── Step 1：逐因子构建面板 ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  多因子合成流程  [{composite_name}]  方法={method}")
        print(f"{'='*60}")

        factor_panels:  Dict[str, pd.DataFrame] = {}
        ic_series_dict: Dict[str, pd.Series]    = {}

        for i, name in enumerate(factor_names, 1):
            print(f"\n[因子 {i}/{len(factor_names)}] 构建面板: {name} ...")
            raw_panel = self._builder.build_panel(
                name, start=start, end=end, symbols=symbols
            )
            if raw_panel.empty:
                warnings.warn(f"因子 '{name}' 面板为空，已跳过。")
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
            raise ValueError("所有因子面板均为空，无法合成。")

        # ── Step 2：构建收益率面板（公共一份）──────────────────────────────
        print(f"\n构建收益率面板（forward={forward} 天）...")
        return_panel = self._builder.build_return_panel(
            forward=forward, start=start, end=end, symbols=symbols
        )

        # ── 截断尾部 NaN（与 run() 保持一致）─────────────────────────────
        valid_ret_idx = return_panel.dropna(how="all").index
        n_dropped = len(return_panel) - len(valid_ret_idx)
        if n_dropped > 0:
            warnings.warn(
                f"[尾部截断] return_panel 末尾 {n_dropped} 个交易日因 "
                f"forward={forward} 天 shift 导致收益率全为 NaN，"
                f"已同步截断所有因子面板的对应行。"
            )
            return_panel  = return_panel.loc[valid_ret_idx]
            factor_panels = {
                name: panel.reindex(valid_ret_idx)
                for name, panel in factor_panels.items()
            }

        # ── Step 3：逐因子计算 IC（ICIR 加权需要）──────────────────────────
        if method == "icir":
            print(f"\n计算各因子 IC（ICIR 滚动窗口={icir_window} 期）...")
            for name, panel in factor_panels.items():
                ic_series_dict[name] = compute_ic(panel, return_panel, method=ic_method)

        # ── Step 4：合成因子 ─────────────────────────────────────────────────
        print(f"\n合成因子（方法={method}）...")
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
        print_weights(weights, method={"equal": "等权", "icir": "ICIR加权"}[method], icir_dict=icir_vals)

        # ── Step 5：合成因子的 IC 分析 ────────────────────────────────────
        print(f"计算合成因子 IC ...")
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
        print(f"分层回测（n_groups={n_groups}）...")
        layer_ret = layer_backtest(
            composite_panel, return_panel,
            n_groups=n_groups, direction=direction
        )
        ls_stats_ = long_short_stats(layer_ret, periods_per_year=periods_per_year, rf=rf)

        # ── Step 7：换手率 ────────────────────────────────────────────────
        print(f"换手率分析 ...")
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

        print("\n✓ 多因子合成流程完成。")
        return report
