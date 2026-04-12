"""
universes/membership.py
========================
按日期查询当期有效成分的工具层（无副作用、纯查询）。

设计
----
UniverseMembership 读取 DynamicUniverseBuilder 生成的快照表，
对外暴露两种查询接口：

  get_symbols(date)      → 返回该日期生效的成分股列表
  get_schedule()         → 返回完整的（effective_date, symbols）映射

生效逻辑
--------
快照中的 effective_date 是成分 **开始生效** 的日期。
对于查询日 D，使用最近一期 effective_date <= D 的快照成分。

                effective_date_1    effective_date_2
                      │                   │
    ──────────────────│───────────────────│───────────
    D < eff_1  → 无成分（返回 None 或 []，视 strict 参数）
    eff_1 ≤ D < eff_2  → 使用 eff_1 的快照
    D ≥ eff_2  → 使用 eff_2 的快照

防未来函数保证
--------------
  - 快照数据仅含 decision_date 可得的信息
  - get_symbols(D) 只使用 effective_date <= D 的快照
  - 任何 effective_date > D 的快照数据均不可见

使用方式
--------
    from universes.membership import UniverseMembership

    mem = UniverseMembership.from_parquet("cache/universe/top500_total_mktcap_sa.parquet")

    # 查询某日成分
    syms = mem.get_symbols("20230615")
    # → ['000001.SZ', '000002.SZ', ..., ...]  (≤500 只)

    # 查询调仓时间表
    schedule = mem.get_schedule()
    # → dict[str, list[str]]  {effective_date: [symbol, ...]}

    # 批量查询（生成 date → symbols 映射，适合回测循环）
    mapping = mem.build_date_symbol_map(["20230101", "20230601", "20231231"])
"""

from __future__ import annotations

import bisect
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class UniverseMembership:
    """
    日期驱动的成分查询器。

    Parameters
    ----------
    snapshots      : DynamicUniverseBuilder.build() 返回的 DataFrame，
                     或通过 from_parquet() 加载
    strict         : 若 True，查询日早于最早 effective_date 时抛 ValueError；
                     若 False（默认），返回 None / 空列表
    """

    def __init__(
        self,
        snapshots: pd.DataFrame,
        strict:    bool = False,
    ) -> None:
        required = {"effective_date", "symbol"}
        missing = required - set(snapshots.columns)
        if missing:
            raise ValueError(
                f"[UniverseMembership] Snapshot is missing required columns: {missing}. "
                f"Current columns: {list(snapshots.columns)}"
            )

        self._strict = strict

        # 按 effective_date 分组，构建有序的 (dates, symbols) 结构
        grouped = (
            snapshots
            .sort_values(["effective_date", "rank"] if "rank" in snapshots.columns else ["effective_date", "symbol"])
            .groupby("effective_date")["symbol"]
            .apply(list)
        )

        # 有序 effective_date 列表及对应 symbols 列表
        self._eff_dates: List[str] = sorted(grouped.index.tolist())
        self._symbols_by_eff: Dict[str, List[str]] = dict(grouped)

        # 原始快照（用于导出 schedule / info）
        self._snapshots = snapshots.copy()

    # ── 工厂方法 ──────────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(cls, path: str | Path, strict: bool = False) -> "UniverseMembership":
        """从 Parquet 文件加载快照并构建 UniverseMembership。"""
        df = pd.read_parquet(Path(path))
        return cls(df, strict=strict)

    @classmethod
    def from_builder(
        cls,
        snapshots: pd.DataFrame,
        strict: bool = False,
    ) -> "UniverseMembership":
        """直接从 DynamicUniverseBuilder.build() 结果构建。"""
        return cls(snapshots, strict=strict)

    # ── 核心查询 ──────────────────────────────────────────────────────────────

    def get_symbols(self, date: str) -> Optional[List[str]]:
        """
        返回 date 当天生效的成分股列表（ts_code 格式）。

        防未来函数：仅使用 effective_date <= date 的快照。

        Parameters
        ----------
        date : 'YYYYMMDD' 查询日期

        Returns
        -------
        list[str] | None
            - list  : 当日有效成分（按原快照 rank 排序）
            - None  : date 早于最早 effective_date（strict=False 时）

        Raises
        ------
        ValueError : strict=True 且 date 早于最早 effective_date
        """
        if not self._eff_dates:
            return None

        # 找最近一期 effective_date <= date
        idx = bisect.bisect_right(self._eff_dates, date) - 1

        if idx < 0:
            # date 早于所有 effective_date
            if self._strict:
                raise ValueError(
                    f"[UniverseMembership] Query date {date!r} is earlier than the first snapshot "
                    f"{self._eff_dates[0]!r}; no membership is available. "
                    f"Use strict=False to return None instead."
                )
            return None

        eff_date = self._eff_dates[idx]
        return list(self._symbols_by_eff[eff_date])

    def is_member(self, symbol: str, date: str) -> bool:
        """
        判断 symbol 在 date 当天是否在股票池中。

        Parameters
        ----------
        symbol : ts_code，如 '000001.SZ'
        date   : 'YYYYMMDD'

        Returns
        -------
        bool
        """
        syms = self.get_symbols(date)
        if syms is None:
            return False
        return symbol in syms

    def get_effective_date_for(self, date: str) -> Optional[str]:
        """
        返回 date 当天使用的快照的 effective_date。

        主要用于调试 / 追溯。
        """
        if not self._eff_dates:
            return None
        idx = bisect.bisect_right(self._eff_dates, date) - 1
        if idx < 0:
            return None
        return self._eff_dates[idx]

    # ── 批量查询 ──────────────────────────────────────────────────────────────

    def build_date_symbol_map(
        self,
        dates: List[str],
    ) -> Dict[str, Optional[List[str]]]:
        """
        批量构建 date → symbols 映射。

        适合回测循环：预先一次性查询所有日期，避免逐日调用 get_symbols()。

        Parameters
        ----------
        dates : 有序日期列表（'YYYYMMDD'）

        Returns
        -------
        dict[str, list[str] | None]
        """
        return {d: self.get_symbols(d) for d in dates}

    def filter_panel(
        self,
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        按各行的日期过滤 panel（行 = 日期，列 = symbol），
        将当日不在股票池内的 symbol 列置为 NaN。

        Parameters
        ----------
        panel : 因子面板，index = 日期（'YYYYMMDD'），columns = ts_code

        Returns
        -------
        过滤后的 panel（index/columns 不变，池外 NaN）
        """
        result = panel.copy().astype(float)
        for date in panel.index:
            date_str = str(date)
            syms = self.get_symbols(date_str)
            if syms is None:
                # 最早期无成分：全部置 NaN
                result.loc[date] = float("nan")
            else:
                syms_set = set(syms)
                cols_to_nan = [c for c in panel.columns if c not in syms_set]
                result.loc[date, cols_to_nan] = float("nan")

        return result

    # ── 时间表 / 统计 ─────────────────────────────────────────────────────────

    def get_schedule(self) -> Dict[str, List[str]]:
        """
        返回 {effective_date: [symbol, ...]} 映射（完整调仓时间表）。
        """
        return {d: list(self._symbols_by_eff[d]) for d in self._eff_dates}

    def rebalance_dates(self) -> List[str]:
        """返回所有 effective_date（换仓日列表）。"""
        return list(self._eff_dates)

    def summary(self) -> pd.DataFrame:
        """
        返回各快照的摘要统计 DataFrame：
          effective_date, decision_date, n_symbols, avg_rank
        """
        rows = []
        for eff_date in self._eff_dates:
            subset = self._snapshots[self._snapshots["effective_date"] == eff_date]
            decision = subset["decision_date"].iloc[0] if "decision_date" in subset.columns else None
            rows.append({
                "effective_date": eff_date,
                "decision_date":  decision,
                "n_symbols":      len(self._symbols_by_eff[eff_date]),
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n_dates = len(self._eff_dates)
        if n_dates == 0:
            return "UniverseMembership(empty)"
        first = self._eff_dates[0]
        last  = self._eff_dates[-1]
        n_syms = len(self._symbols_by_eff.get(last, []))
        return (
            f"UniverseMembership("
            f"rebalance_dates={n_dates}, "
            f"range={first}~{last}, "
            f"latest_n_symbols={n_syms})"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  便利函数：从配置构建 UniverseMembership
# ═════════════════════════════════════════════════════════════════════════════

def build_membership_from_config(
    cfg,
    root:    Path = Path("."),
    verbose: bool = True,
) -> Optional["UniverseMembership"]:
    """
    根据 ResearchConfig（或 ConfigNamespace）中的 universe_* 字段，
    构建并返回 UniverseMembership。

    支持的 universe_mode：
      - "all" / None           → 返回 None（全部股票，无限制）
      - "static_file"          → 从 universe_file 加载静态 CSV（返回 None，由 loader 处理）
      - "topn_mktcap_dynamic"  → 构建动态 topN 快照

    Parameters
    ----------
    cfg     : ResearchConfig 或 ConfigNamespace（含 universe_* 字段）
    root    : 项目根目录
    verbose : 是否打印进度

    Returns
    -------
    UniverseMembership | None
    """
    mode = getattr(cfg, "universe_mode", "all") or "all"

    if mode in ("all", "static_file"):
        return None  # 由 UniverseLoader / symbols 参数处理

    if mode != "topn_mktcap_dynamic":
        warnings.warn(
            f"[build_membership_from_config] Unknown universe_mode={mode!r}; "
            f"falling back to 'all'.",
            UserWarning,
        )
        return None

    # ── topn_mktcap_dynamic ─────────────────────────────────────────────────
    from universes.builder import DynamicUniverseBuilder

    top_n         = getattr(cfg, "universe_top_n",              500)
    metric        = getattr(cfg, "universe_metric",             "total_mktcap")
    freq          = getattr(cfg, "universe_rebalance_freq",     "semiannual")
    months        = getattr(cfg, "universe_rebalance_months",   None)
    lag           = getattr(cfg, "universe_effective_lag_days", 1)
    stocks_dir    = root / getattr(getattr(cfg, "data", cfg), "stocks_dir", "stocks/")
    trade_cal     = root / "交易日历-trade_cal.csv"
    cache_dir     = root / getattr(cfg, "universe_cache_dir",  "cache/universe/")

    # 缓存路径
    builder = DynamicUniverseBuilder(
        stocks_dir         = stocks_dir,
        trade_cal          = trade_cal,
        top_n              = top_n,
        metric             = metric,
        rebalance_freq     = freq,
        rebalance_months   = months,
        effective_lag_days = lag,
    )
    cache_file = cache_dir / f"{builder.universe_id}_{builder.config_hash}.parquet"

    if cache_file.exists():
        if verbose:
            print(f"[UniverseMembership] Loading cached snapshot: {cache_file}")
        snapshots = DynamicUniverseBuilder.load(cache_file)
    else:
        start = getattr(cfg, "start", None) or "20100101"
        end   = getattr(cfg, "end",   None) or "20991231"
        snapshots = builder.build(start=start, end=end, verbose=verbose)
        builder.save(snapshots, cache_file)
        if verbose:
            print(f"[UniverseMembership] Snapshot saved: {cache_file}")

    return UniverseMembership.from_builder(snapshots)
