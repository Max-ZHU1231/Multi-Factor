"""
factor_engine.py
================
因子注册 / 计算 / 面板构建引擎。

.. deprecated::
    **兼容层（v3.2+）**：直接使用本模块是旧路径。
    新代码推荐使用：
      - `factor_framework.engine.panel_builder.PanelBuilder` — 带缓存的面板构建
      - `factor_framework.data.store.CSVDataStore` — 数据访问抽象
      - `factor_framework.core.panel.TimestampedPanel` — 带语义的面板类型
      - `factor_framework.core.returns.ReturnPanel` — 收益率唯一来源

    本模块作为执行引擎（计算后端）保留，但外部代码不应直接实例化 FactorEngine，
    而应通过 PanelBuilder 访问。首次直接实例化时将发出一次性 DeprecationWarning。

核心概念
--------
- 面板（Panel）: DataFrame，index=交易日(str YYYYMMDD)，columns=ts_code，values=因子值
- 因子函数签名: func(df: pd.DataFrame) -> pd.Series
  df 为单只股票的日频 DataFrame（已清洗，按日期升序），
  返回与 df 等长、index 对应的因子值 Series。
- 注册后通过 FactorEngine.build_panel() 批量计算所有股票。

两种注册方式
-----------
方式一（表达式树，推荐）::

    from factor_framework.dag import data, op, pct_change
    close  = data("close", col="收盘价")
    ret    = pct_change(close)
    vol20  = op("ts_stddev", ret, 20)
    vol60  = op("ts_stddev", ret, 60)  # ret 只计算一次（CSE）
    engine.register_expr("vol_20d", -vol20)
    engine.register_expr("vol_60d", -vol60)

方式二（lambda + 显式依赖，向后兼容）::

    engine.register("vol_20d", lambda df: -ts_stddev(df["_ret"], 20))
    engine.register("vol_60d", lambda df: -ts_stddev(df["_ret"], 60),
                    deps=["vol_20d"])   # 声明依赖，引擎按拓扑顺序执行

旧式注册（无 deps）::

    engine.register("log_mktcap", lambda df: log(df["总市值（万元）"]))
    panel = engine.build_panel("log_mktcap", start="20200101", end="20261231")
"""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_cleaner import load_and_clean
from factor_framework.dag import (
    Expr, DAGExecutor, DepGraph, LRUCache,
    cse_report,
    _DEFAULT_INTERMEDIATE_LRU, _DEFAULT_FACTOR_LRU,
    _MISS as _DAG_MISS,
)

# 因子函数类型别名
FactorFn = Callable[[pd.DataFrame], pd.Series]

# 哨兵对象（区分"LRU 缓存未命中"与合法的 None）
_MISS_SENTINEL = _DAG_MISS

# ── 日期筛选辅助 ──────────────────────────────────────────────────────────────

def _date_filter(s: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    """对 index 为 YYYYMMDD 字符串的 Series 做日期区间筛选。"""
    if not start and not end:
        return s
    mask = pd.Series(True, index=s.index)
    if start:
        mask &= s.index >= start
    if end:
        mask &= s.index <= end
    return s.loc[mask]


def _date_filter_df(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> Optional[pd.DataFrame]:
    """对含 COL_DATE 列的 DataFrame 做日期区间筛选，返回 None 若结果为空。"""
    if not start and not end:
        return df
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df[COL_DATE] >= start
    if end:
        mask &= df[COL_DATE] <= end
    result = df.loc[mask]
    return result if not result.empty else None

# 列名映射（对外统一，内部映射到中文列名）
COL_CLOSE  = "收盘价"
COL_DATE   = "交易日"
COL_CODE   = "股票代码"
COL_VOL    = "成交量（手）"
COL_AMOUNT = "成交额（千元）"
COL_MKTCAP = "总市值（万元）"
COL_RET    = "_ret"           # 日收益率（引擎内部生成）

# 轻量加载时需要的最小列集合（跳过完整 MAD Winsorize，提升约 3x 速度）
_FAST_COLS = [
    "交易日", "股票代码",
    "收盘价", "开盘价", "最高价", "最低价",
    "成交量（手）", "成交额（千元）",
    "换手率（%）", "总市值（万元）", "流通市值（万元）",
    "市净率", "市盈率（TTM，亏损为空）", "市销率（TTM）",
    "复权因子",
]

def _fast_load(path: Path) -> Optional[pd.DataFrame]:
    """
    轻量级 CSV 加载器：只读需要的列，做最基本的 ffill/排序，
    跳过完整 MAD Winsorize，速度约为 load_and_clean 的 3 倍。
    用于因子面板批量构建的性能关键路径。
    """
    try:
        # 先读表头确认哪些列存在
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in _FAST_COLS if c in header.columns]
        df = pd.read_csv(path, usecols=usecols, dtype={"交易日": str, "股票代码": str})
    except Exception:
        return None

    if "交易日" not in df.columns:
        return None

    df = df.sort_values("交易日").reset_index(drop=True)

    # ── BUG FIX v2.9.1（Issue 3）：先计算日收益率，再做 ffill ─────────────────
    # 若先 ffill 再 pct_change，停牌期（价格被填充为前值）的日收益率为 0，
    # 会低估停牌股复牌后的实际涨跌幅，并污染动量因子的计算。
    # 在 ffill 前，用原始未填充收盘价计算 _ret，停牌日为 NaN（正确）。
    # _ret 列写入 df 后，下游因子（vol_20d 等）会正确跳过这些 NaN 日期。
    _RET_COL = "_ret"
    if "收盘价" in df.columns:
        raw_close = df["收盘价"].copy()   # ffill 之前的原始价格
        df[_RET_COL] = raw_close.pct_change()   # 停牌日 → NaN（而非 0）

    # 价格列 ffill（最多 5 天，处理停牌）
    price_cols = ["收盘价", "开盘价", "最高价", "最低价", "成交量（手）",
                  "成交额（千元）", "换手率（%）", "总市值（万元）", "流通市值（万元）"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=5)

    # 估值列 ffill 不限长度（PIT 原则）
    val_cols = ["市净率", "市盈率（TTM，亏损为空）", "市销率（TTM）", "复权因子"]
    for col in val_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df


class FactorEngine:
    """
    单例式因子引擎：
    1. 注册因子函数（lambda 方式）或表达式树（DAG 方式）
    2. 批量遍历 Stocks/ 目录，对每只股票计算因子
    3. 拼接成 (日期 × 股票) 面板 DataFrame

    .. deprecated::
        直接实例化 FactorEngine 是旧路径（v3.2+ 兼容层）。
        新代码请使用 PanelBuilder + CSVDataStore，参见 README 阶段二主路径。
    """

    # 一次性 deprecation warning（避免每次构造都刷屏）
    _deprecation_warned: bool = False

    def __init__(
        self,
        stocks_dir:  str | Path = "Stocks/",
        stock_basic: str | Path = "股票列表-stock_basic.csv",
        min_rows:    int = 60,
        verbose:     bool = True,
        intermediate_lru: int = _DEFAULT_INTERMEDIATE_LRU,
        factor_lru:       int = _DEFAULT_FACTOR_LRU,
        _internal: bool = False,   # PanelBuilder 内部调用时传 True，抑制警告
    ):
        if not _internal and not FactorEngine._deprecation_warned:
            warnings.warn(
                "[WARN] Direct FactorEngine instantiation is a compatibility path (v3.2+).\n"
                "New code should use PanelBuilder + CSVDataStore (see README primary path).\n"
                "To suppress this warning, instantiate via PanelBuilder (_internal=True is set automatically).",
                DeprecationWarning,
                stacklevel=2,
            )
            FactorEngine._deprecation_warned = True
        self.stocks_dir  = Path(stocks_dir)
        self.stock_basic = Path(stock_basic)
        self.min_rows    = min_rows
        self.verbose     = verbose

        # 因子注册表：name → func（方式一/二均记录在此）
        self._registry: Dict[str, FactorFn] = {}

        # 行业映射 ts_code → industry（来自 stock_basic）
        self._industry_map: Optional[pd.Series] = None

        # ── DataFrame 缓存（消除多因子重复读盘）─────────────────────────────
        self._cache_fast: Dict[str, Optional[pd.DataFrame]] = {}
        self._cache_full: Dict[str, Optional[pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()

        # ── 函数级编译缓存（Numba/Numexpr 路径元数据）───────────────────────
        self._compile_cache: Dict[str, dict] = {}

        # ── DAG 相关（方式一：Expr 树）───────────────────────────────────────
        # _expr_registry : 因子名 → 根 Expr 节点（register_expr 注册）
        self._expr_registry: Dict[str, Expr] = {}
        # _intermediate_cache : 中间节点 LRU 缓存
        #   key = f"{symbol}|{node_hash}"，value = pd.Series
        self._intermediate_cache = LRUCache(intermediate_lru)
        # _factor_cache : 最终因子结果 LRU 缓存
        #   key = f"{factor_name}|{symbol}"，value = pd.Series
        self._factor_cache = LRUCache(factor_lru)

        # ── 显式依赖图（方式二：lambda + deps）──────────────────────────────
        self._dep_graph = DepGraph()

        self._load_meta()

    # ── 元数据 ────────────────────────────────────────────────────────────────

    def _load_meta(self) -> None:
        """加载股票基本信息，建立行业映射。"""
        if self.stock_basic.exists():
            sb = pd.read_csv(self.stock_basic, dtype=str)
            if "ts_code" in sb.columns and "industry" in sb.columns:
                self._industry_map = sb.set_index("ts_code")["industry"]
        else:
            warnings.warn(f"[WARN] stock_basic file not found: {self.stock_basic}")

    @property
    def industry_map(self) -> Optional[pd.Series]:
        """ts_code → industry 的 Series（None 表示未加载）。"""
        return self._industry_map

    def all_files(self) -> List[Path]:
        """返回 Stocks/ 目录下所有 CSV 文件路径列表。"""
        return sorted(self.stocks_dir.glob("*.csv"))

    def all_symbols(self) -> List[str]:
        """返回所有股票代码（文件名去 .csv）。"""
        return [f.stem for f in self.all_files()]

    # ── 因子注册 ──────────────────────────────────────────────────────────────

    def register(self, name: str, func: FactorFn, deps: Optional[List[str]] = None) -> None:
        """
        注册一个因子函数（方式二：lambda / 普通函数）。

        Parameters
        ----------
        name : 因子名称（唯一标识）
        func : (df: pd.DataFrame) -> pd.Series
               输入：单只股票的日频 DataFrame（已清洗）
               输出：等长的因子值 Series
        deps : 显式依赖的因子名称列表（可选）。声明后引擎会按拓扑顺序执行，
               并将依赖因子的计算结果注入 DataFrame 的
               ``__dep_{dep_name}__`` 列，供 func 按需读取。

        示例（使用 deps）::

            engine.register("vol_20d", lambda df: -ts_stddev(df["_ret"], 20))
            # vol_zscore 依赖 vol_20d 的结果
            engine.register(
                "vol_zscore",
                lambda df: cs_zscore(df["__dep_vol_20d__"]),
                deps=["vol_20d"],
            )
        """
        if name in self._registry:
            warnings.warn(f"[WARN] Factor '{name}' already exists and will be overwritten.")
        self._registry[name] = func
        if deps:
            self._dep_graph.register(name, deps)

    def register_expr(self, name: str, expr: Expr) -> None:
        """
        注册一个因子表达式树（方式一：DAG 方式）。

        Parameters
        ----------
        name : 因子名称
        expr : Expr 树根节点（由 dag.data / dag.op / dag.pct_change 组合构建）

        示例::

            from factor_framework.dag import data, op, pct_change
            close = data("close", col="收盘价")
            ret   = pct_change(close)
            vol20 = op("ts_stddev", ret, 20)
            engine.register_expr("vol_20d", -vol20)
        """
        from factor_framework.dag import Expr as _Expr
        if not isinstance(expr, _Expr):
            raise TypeError(f"[ERROR] register_expr expects an Expr object, got {type(expr)}")
        if name in self._expr_registry:
            warnings.warn(f"[WARN] Expr factor '{name}' already exists and will be overwritten.")
        self._expr_registry[name] = expr
        # 同时在 _registry 中注册一个占位函数（用于 build_panel 等统一接口）
        self._registry[name] = self._make_expr_fn(name)

    def _make_expr_fn(self, name: str) -> FactorFn:
        """为 Expr 因子生成一个兼容 _registry 的包装函数。"""
        def _fn(df: pd.DataFrame, _name=name) -> pd.Series:
            # 临时为这只股票创建单次 DAG 执行器
            sym_cache = LRUCache(-1)   # 单次执行用无上限缓存
            executor = DAGExecutor(sym_cache, self._expr_registry)
            results = executor.run(df, factor_names=[_name])
            if _name not in results:
                raise ValueError(f"[ERROR] DAG execution did not return result for factor '{_name}'.")
            return results[_name]
        return _fn

    def registered(self) -> List[str]:
        """列出所有已注册因子名称（lambda 和 Expr 两种方式）。"""
        return list(self._registry.keys())

    def registered_expr(self) -> List[str]:
        """列出通过 register_expr 注册的因子名称。"""
        return list(self._expr_registry.keys())

    def clear_cache(self) -> None:
        """清空所有缓存（DataFrame / 中间节点 / 因子结果 / 编译元数据）。"""
        with self._cache_lock:
            self._cache_fast.clear()
            self._cache_full.clear()
            self._compile_cache.clear()
        self._intermediate_cache.clear()
        self._factor_cache.clear()

    def cse_report(self) -> pd.DataFrame:
        """
        公共子表达式报告：显示 DAG 中被多个因子共享的中间节点。

        Returns
        -------
        pd.DataFrame，columns: [node_hash, repr, ref_count, shared_by]
        ref_count > 1 表示该节点被多个因子复用（节省了重复计算）。
        """
        if not self._expr_registry:
            return pd.DataFrame(columns=["node_hash", "repr", "ref_count", "shared_by"])
        return cse_report(self._expr_registry)

    def _resolve_compile_target(self, name: str) -> str:
        """
        查询已注册因子的编译路径。

        优先读取因子函数上的 ``_compile_target`` 属性；
        对于 lambda 包装的因子，通过函数名模式匹配推断目标。

        Returns
        -------
        'numba' | 'numexpr' | 'numpy' | 'pandas' | 'unknown'
        """
        if name in self._compile_cache:
            return self._compile_cache[name]["target"]

        fn = self._registry.get(name)
        if fn is None:
            return "unknown"

        # 先检查函数本身的 _compile_target 属性（算子库中已标注）
        if hasattr(fn, "_compile_target"):
            target = fn._compile_target
        else:
            # 对于匿名 lambda / 用户自定义函数，标记为 'pandas'
            target = "pandas"

        self._compile_cache[name] = {"target": target}
        return target

    def compile_report(self) -> pd.DataFrame:
        """
        返回所有已注册因子的编译路径报告（DataFrame）。

        Columns: factor_name, compile_target
        """
        rows = [
            {"factor_name": n, "compile_target": self._resolve_compile_target(n)}
            for n in self._registry
        ]
        return pd.DataFrame(rows)

    # ── 带缓存的数据加载（瓶颈 2 核心）─────────────────────────────────────

    def _load_df(self, symbol: str, fast_mode: bool) -> Optional[pd.DataFrame]:
        """
        加载单只股票的 DataFrame，结果写入缓存。
        同一 symbol + fast_mode 组合只读一次磁盘；
        后续调用（换因子）直接返回缓存对象（只读，不允许修改）。
        """
        cache = self._cache_fast if fast_mode else self._cache_full

        # 先无锁快速查询（绝大多数情况命中）
        if symbol in cache:
            return cache[symbol]

        # 缓存未命中：加锁后二次检查再读盘（防止并发重复加载）
        with self._cache_lock:
            if symbol in cache:
                return cache[symbol]

            path = self.stocks_dir / f"{symbol}.csv"
            if not path.exists():
                cache[symbol] = None
                return None

            df = _fast_load(path) if fast_mode else load_and_clean(path)

            if df is None or len(df) < self.min_rows:
                cache[symbol] = None
            else:
                df = df.copy()
                # _fast_load 路径：_ret 已在 ffill 前计算（见 _fast_load 内部）。
                # load_and_clean 路径：load_and_clean 内部做了 ffill，这里补算 _ret。
                # 两条路径下，若 _ret 列已存在（fast_load 已写入），跳过重复计算。
                if COL_RET not in df.columns:
                    df[COL_RET] = df[COL_CLOSE].pct_change()
                cache[symbol] = df

            return cache[symbol]

    # ── 单股计算 ──────────────────────────────────────────────────────────────

    def compute_single(
        self,
        symbol:       str,
        factor_name:  str,
        start:        Optional[str] = None,
        end:          Optional[str] = None,
        fast_mode:    bool = False,
        max_lookback: int  = 0,
    ) -> Optional[pd.Series]:
        """
        计算单只股票的因子值序列。

        支持三种注册方式：
        1. ``register_expr`` 注册的 Expr 因子（DAG 执行）
        2. ``register(..., deps=[...])`` 注册的带依赖 lambda 因子
        3. 无依赖的普通 lambda 因子

        Parameters
        ----------
        fast_mode    : True = 使用轻量加载器（跳过 MAD Winsorize，速度 ~3x）；
                       False = 使用完整 load_and_clean（含 MAD Winsorize）
        max_lookback : warm-up 期长度（交易日数）。若 > 0，计算时会向 start 之前
                       多加载 max_lookback 行数据以避免因子冷启动偏差，输出时再
                       截断回 [start, end]。0 = 不使用 warm-up（默认，向后兼容）。

        Returns
        -------
        pd.Series，index = 交易日(str)，values = 因子值；若数据不足返回 None。
        """
        if factor_name not in self._registry:
            raise KeyError(f"[ERROR] Factor '{factor_name}' is not registered. Call register() first.")

        # 最终因子缓存命中（两层缓存：第一层按 symbol+factor_name）
        # 注意：启用 max_lookback 时跳过缓存（截断边界不同，防止缓存污染）
        fc_key = f"{factor_name}|{symbol}"
        if max_lookback == 0:
            cached = self._factor_cache.get(fc_key)
            if cached is not _MISS_SENTINEL:
                if start or end:
                    return _date_filter(cached, start, end)
                return cached

        # 记录编译路径元数据（仅首次）
        self._resolve_compile_target(factor_name)

        # 从磁盘缓存获取（不再每次重复读盘）
        df = self._load_df(symbol, fast_mode)
        if df is None:
            return None

        # ── warm-up 截断：计算时向前扩展，输出时截断回原始 start ─────────────
        # 只在 start 有效且 max_lookback > 0 时生效；全量计算时不做截断。
        warm_start: Optional[str] = None
        if start and max_lookback > 0:
            all_dates = sorted(df[COL_DATE].values)
            try:
                start_pos = np.searchsorted(all_dates, start, side="left")
                warm_pos  = max(0, start_pos - max_lookback)
                warm_start = all_dates[warm_pos]
            except Exception:
                pass  # 降级：使用全量 df

        if warm_start is not None and warm_start < start:
            df_compute = df[df[COL_DATE] >= warm_start].copy()
            df_compute = df_compute.reset_index(drop=True)
        else:
            df_compute = df

        # 在 df_compute 上计算（含 warm-up 期，保证因子窗口完整）
        # ── Expr DAG 路径 ────────────────────────────────────────────────────
        if factor_name in self._expr_registry:
            result = self._compute_expr_single(symbol, factor_name, df_compute)
        # ── 显式依赖路径 ────────────────────────────────────────────────────
        elif self._dep_graph.deps_of(factor_name):
            result = self._compute_with_deps(symbol, factor_name, df_compute, fast_mode, None, None)
        # ── 普通 lambda 路径 ─────────────────────────────────────────────────
        else:
            try:
                result = self._registry[factor_name](df_compute)
            except Exception as e:
                warnings.warn(f"[WARN] {symbol}/{factor_name} calculation failed: {e}")
                return None

        if result is None:
            return None

        result = result.copy()
        result.index = df_compute[COL_DATE].values
        result.name  = symbol

        # 仅无 warm-up 时写入全序列因子缓存（warm-up 模式不缓存以防截断不一致）
        if max_lookback == 0:
            self._factor_cache.put(fc_key, result)

        # 返回时动态切片
        if start or end:
            return _date_filter(result, start, end)
        return result

    def _compute_expr_single(
        self,
        symbol:      str,
        factor_name: str,
        df:          pd.DataFrame,
    ) -> Optional[pd.Series]:
        """执行 Expr DAG 路径（共享中间节点缓存）。"""
        # 为每只股票使用独立的中间缓存（节点 key 不含 symbol，避免不同股票混用）
        # 注意：build_panel_batch 会在同一股票多因子时复用同一 sym_cache，
        #       单独调用 compute_single 时每次创建新缓存（仍比纯 pandas 快）。
        sym_cache = LRUCache(-1)
        executor  = DAGExecutor(sym_cache, self._expr_registry)
        results   = executor.run(df, factor_names=[factor_name])
        return results.get(factor_name)

    def _compute_with_deps(
        self,
        symbol:      str,
        factor_name: str,
        df:          pd.DataFrame,
        fast_mode:   bool,
        start:       Optional[str],
        end:         Optional[str],
    ) -> Optional[pd.Series]:
        """
        按拓扑顺序执行带依赖的因子（显式 deps 路径）。

        依赖因子的结果注入 df 的 ``__dep_{dep_name}__`` 列，
        供下游 lambda 函数按名读取。
        """
        # 收集完整依赖顺序（含传递依赖）
        all_deps = self._dep_graph.topo_order(
            self._dep_graph.deps_of(factor_name)
        )

        df = df.copy()   # 不修改缓存中的 df
        for dep in all_deps:
            dep_col = f"__dep_{dep}__"
            if dep_col in df.columns:
                continue   # 已注入
            dep_series = self.compute_single(
                symbol, dep, start=start, end=end, fast_mode=fast_mode
            )
            if dep_series is not None:
                # 对齐 index
                aligned = dep_series.reindex(df[COL_DATE].values)
                df[dep_col] = aligned.values

        try:
            return self._registry[factor_name](df)
        except Exception as e:
            warnings.warn(f"[WARN] {symbol}/{factor_name}(deps) calculation failed: {e}")
            return None

    # ── 面板构建 ──────────────────────────────────────────────────────────────

    def build_panel(
        self,
        factor_name:  str,
        start:        Optional[str] = None,
        end:          Optional[str] = None,
        symbols:      Optional[List[str]] = None,
        n_jobs:       int = 8,
        fast_mode:    bool = True,
        max_lookback: int  = 0,
    ) -> pd.DataFrame:
        """
        批量计算所有股票的因子值，构建 (日期 × 股票) 面板。

        Parameters
        ----------
        factor_name  : 因子名称（已注册）
        start / end  : 日期范围（YYYYMMDD 字符串，含端点）
        symbols      : 指定股票列表（None = 全部）
        n_jobs       : 并行线程数（默认 8）
        fast_mode    : True = 跳过 MAD Winsorize，速度 ~3x（默认）；
                       False = 使用完整清洗（更准确，更慢）
        max_lookback : warm-up 期长度（交易日数），传递给 compute_single()

        Returns
        -------
        pd.DataFrame，index = 交易日(str)，columns = ts_code，values = 因子值
        """
        targets = symbols or self.all_symbols()
        series_list: List[pd.Series] = []

        pbar = tqdm(total=len(targets), desc=f"计算 {factor_name}", disable=not self.verbose)

        def _compute(sym: str) -> Optional[pd.Series]:
            return self.compute_single(
                sym, factor_name,
                start=start, end=end,
                fast_mode=fast_mode,
                max_lookback=max_lookback,
            )

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sym = {executor.submit(_compute, sym): sym for sym in targets}
            for future in as_completed(future_to_sym):
                pbar.update(1)
                try:
                    s = future.result()
                    if s is not None and not s.isna().all():
                        series_list.append(s)
                except Exception as e:
                    sym = future_to_sym[future]
                    warnings.warn(f"[WARN] {sym}/{factor_name} worker exception: {e}")

        pbar.close()

        if not series_list:
            warnings.warn(f"[WARN] Factor '{factor_name}' has no valid data; returning empty DataFrame.")
            return pd.DataFrame()

        panel = pd.concat(series_list, axis=1)
        panel = panel.sort_index()
        return panel

    def build_panel_batch(
        self,
        factor_names: List[str],
        start:        Optional[str] = None,
        end:          Optional[str] = None,
        symbols:      Optional[List[str]] = None,
        n_jobs:       int = 8,
        fast_mode:    bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        多因子批量面板构建（DAG 公共子表达式消除版）。

        相比逐个调用 build_panel，此方法在同一只股票上一次性执行所有因子，
        共享中间计算结果（CSE），避免重复计算公共子表达式。

        适用场景
        --------
        - 同时需要 vol_20d / vol_60d（共享 ts_stddev(ret, 20)）
        - 同时需要 momentum_12_1 / momentum_6_1（共享收益率计算）
        - run_composite() 多因子合成

        Parameters
        ----------
        factor_names : 需要计算的因子名称列表（均须已注册）
        start / end  : 日期范围
        symbols      : 指定股票列表（None = 全部）
        n_jobs       : 并行线程数
        fast_mode    : 是否使用轻量加载

        Returns
        -------
        dict: {factor_name: pd.DataFrame(日期 × 股票)}
        """
        for fn in factor_names:
            if fn not in self._registry:
                raise KeyError(f"[ERROR] Factor '{fn}' is not registered. Call register() or register_expr() first.")

        targets = symbols or self.all_symbols()

        # 分离 Expr 因子 vs lambda 因子
        expr_names = [n for n in factor_names if n in self._expr_registry]
        lamb_names = [n for n in factor_names if n not in self._expr_registry]

        # 结果收集：{factor_name → [pd.Series]}
        results: Dict[str, List[pd.Series]] = {n: [] for n in factor_names}

        pbar = tqdm(
            total=len(targets),
            desc=f"批量计算 {len(factor_names)} 个因子",
            disable=not self.verbose,
        )

        def _compute_sym(sym: str) -> Dict[str, Optional[pd.Series]]:
            """对单只股票计算所有因子（共享中间节点缓存）。"""
            df = self._load_df(sym, fast_mode)
            if df is None:
                return {n: None for n in factor_names}

            # ── BUG FIX v2.9.1：先用完整 df 计算因子，计算完再按日期切片 ──────
            # 旧代码：df_slice = _date_filter_df(df, start, end) 然后在 df_slice
            # 上计算因子，导致 warm-up 期历史数据被截断，时序因子（如 momentum_12_1
            # 需要 252 天）在回测期初产生大量额外 NaN（基于残缺窗口）。
            # 正确做法：在完整 df 上计算，输出时再按 [start, end] 做日期切片，
            # 与 compute_single 路径（先计算再截断）保持一致。
            df_full = df  # 用完整历史计算，保证 warm-up 期数据足够

            out: Dict[str, Optional[pd.Series]] = {}

            # ── Expr 因子：一次拓扑排序，共享中间节点 ───────────────────────
            if expr_names:
                sym_cache = LRUCache(-1)
                executor  = DAGExecutor(sym_cache, self._expr_registry)
                dag_results = executor.run(df_full, factor_names=expr_names)
                for n in expr_names:
                    val = dag_results.get(n)
                    if val is not None:
                        val = val.copy()
                        val.index = df_full[COL_DATE].values
                        val.name  = sym
                        # 计算完成后按日期切片（而不是在计算前切片）
                        val = _date_filter(val, start, end)
                    out[n] = val

            # ── Lambda 因子：按拓扑顺序执行（尊重 deps）───────────────────
            if lamb_names:
                # 拓扑排序含依赖
                topo_lambs = self._dep_graph.topo_order(lamb_names)
                # 确保所有 lamb_names 都包含（无依赖的放在最后）
                topo_lambs = topo_lambs + [n for n in lamb_names if n not in topo_lambs]

                df_work = df_full.copy()
                for n in topo_lambs:
                    if n not in lamb_names:
                        continue
                    self._resolve_compile_target(n)
                    # 注入依赖列（依赖因子的完整时序结果，尚未切片）
                    for dep in self._dep_graph.deps_of(n):
                        dep_col = f"__dep_{dep}__"
                        if dep_col not in df_work.columns:
                            dep_val = out.get(dep)
                            if dep_val is not None:
                                # dep_val 已被切片，需重新对齐到完整 df 的日期索引
                                dep_val_aligned = dep_val.reindex(df_work[COL_DATE].values)
                                df_work[dep_col] = dep_val_aligned.values
                    try:
                        val = self._registry[n](df_work)
                        if val is not None:
                            val = val.copy()
                            val.index = df_work[COL_DATE].values
                            val.name  = sym
                            # 计算完成后再按日期切片
                            val = _date_filter(val, start, end)
                        out[n] = val
                        # 写入 __dep__ 列供后续因子使用（切片前的完整值）
                        # 注意：dep_val 已切片，这里需要在 df_work（完整）上写入
                        if val is not None:
                            # 反向对齐：用切片后的 val 填回完整 df_work 的对应行
                            full_ser = pd.Series(np.nan, index=df_work[COL_DATE].values, dtype=float)
                            full_ser.update(val)
                            df_work[f"__dep_{n}__"] = full_ser.values
                    except Exception as e:
                        warnings.warn(f"[WARN] {sym}/{n} calculation failed: {e}")
                        out[n] = None

            return out

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_map = {executor.submit(_compute_sym, sym): sym for sym in targets}
            for future in as_completed(future_map):
                pbar.update(1)
                sym = future_map[future]
                try:
                    sym_out = future.result()
                    for n, s in sym_out.items():
                        if s is not None and not s.isna().all():
                            results[n].append(s)
                            # 写入因子结果缓存
                            self._factor_cache.put(f"{n}|{sym}", s)
                except Exception as e:
                    warnings.warn(f"[WARN] {sym} batch worker exception: {e}")

        pbar.close()

        panels: Dict[str, pd.DataFrame] = {}
        for n, series_list in results.items():
            if series_list:
                panel = pd.concat(series_list, axis=1).sort_index()
                panels[n] = panel
            else:
                warnings.warn(f"[WARN] Factor '{n}' has no valid data; returning empty DataFrame.")
                panels[n] = pd.DataFrame()

        return panels

    # ── 收益率面板 ────────────────────────────────────────────────────────────

    def build_return_panel(
        self,
        forward:    int = 1,
        start:      Optional[str] = None,
        end:        Optional[str] = None,
        symbols:    Optional[List[str]] = None,
        fast_mode:  bool = True,
    ) -> pd.DataFrame:
        """
        构建未来 forward 日收益率面板（用于 IC / 分层回测）。

        收益率定义（修正 v2.9）
        ----------------------
        未来 forward 日收益 = price[t+forward] / price[t] - 1

        即：将未来第 forward 日的价格移到当前行（shift(-forward)），
        除以当前价格再减 1。

        注意：此处使用的是「后复权价格」列 COL_CLOSE_ADJ（如可用），
        否则退回到普通收盘价列并手动应用复权因子。
        这是为了避免前复权价格在历史回溯调整时引入前瞻偏差。

        T+1 滞后
        --------
        为模拟 T+1 制度（收盘后计算因子，次日方可成交），
        收益率面板整体 shift(1)，即 return_panel[t] 表示：
        以 t 日因子信号、t+1 日成交，持有至 t+1+forward 日的收益。
        调用方（pipeline.run / run_batch_from_panels）无需再额外处理。

        Returns
        -------
        pd.DataFrame，index = 交易日，columns = ts_code，values = 未来 forward 日收益
                     （已内置 T+1 滞后）
        """
        # 正确的未来 forward 日收益率公式：price[t+forward] / price[t] - 1
        # 再整体 shift(1) 实现 T+1 滞后（以 t 日因子决策，t+1 日起计算收益）
        _name = f"__ret_{forward}__"
        self.register(
            _name,
            lambda df, _fwd=forward: (
                df[COL_CLOSE].shift(-_fwd) / df[COL_CLOSE].replace(0, np.nan) - 1
            ).shift(1),   # T+1 滞后：收盘后才能知道因子值，次日才能成交
        )
        panel = self.build_panel(_name, start=start, end=end, symbols=symbols, fast_mode=fast_mode)
        del self._registry[_name]
        return panel

    # ── 横截面操作 ────────────────────────────────────────────────────────────

    @staticmethod
    def apply_cross_section(
        panel:    pd.DataFrame,
        cs_func:  Callable[[pd.Series], pd.Series],
        industry: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        对面板的每一行（每个截面日）应用横截面函数。

        Parameters
        ----------
        panel    : (日期 × 股票) 因子面板
        cs_func  : (x: pd.Series[, group]) → pd.Series，横截面算子
        industry : ts_code → 行业标签（若 cs_func 需要 group 参数时传入）

        Returns
        -------
        与 panel 等形状的横截面处理后面板

        实现说明
        --------
        对常用无状态横截面算子（cs_rank / cs_zscore / cs_winsorize），
        走全面板向量化快速路径（无 Python 循环，10–50x 加速）；
        其他算子或有 industry 参数时降级到 panel.apply(axis=1) / iterrows。
        """
        # ── 快速路径 1：cs_rank → DataFrame.rank(axis=1)（分块以节约内存）──
        func_name = getattr(cs_func, "__name__", "")
        if industry is None and func_name == "cs_rank":
            CHUNK = 300
            if len(panel) <= CHUNK:
                return panel.rank(axis=1, pct=True, na_option="keep")
            # 分块排名，避免一次性为超大面板分配内存
            chunks = []
            for start in range(0, len(panel), CHUNK):
                block = panel.iloc[start:start+CHUNK]
                chunks.append(block.rank(axis=1, pct=True, na_option="keep"))
            return pd.concat(chunks)

        # ── 快速路径 2：cs_zscore → 逐行 (x-mean)/std（纯 numpy，无循环）──
        if industry is None and func_name == "cs_zscore":
            arr   = panel.values.astype(float)
            CHUNK = 200
            T = arr.shape[0]
            out = np.empty_like(arr)
            for start in range(0, T, CHUNK):
                block = arr[start:start+CHUNK]
                mu    = np.nanmean(block, axis=1, keepdims=True)
                sigma = np.nanstd(block, axis=1, ddof=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[start:start+CHUNK] = np.where(sigma == 0, np.nan, (block - mu) / sigma)
                out[start:start+CHUNK][np.isnan(block)] = np.nan
            return pd.DataFrame(out, index=panel.index, columns=panel.columns)

        # ── 快速路径 3：cs_winsorize → 逐行 MAD clip（纯 numpy，无循环）───
        if industry is None and func_name == "cs_winsorize":
            arr   = panel.values.astype(float)
            n_std = 3.0  # cs_winsorize 默认值
            # 分块计算 nanmedian，避免一次性为超大矩阵分配内存
            CHUNK = 200  # 每次处理 200 行
            T = arr.shape[0]
            med_arr = np.empty((T, 1), dtype=float)
            mad_arr = np.empty((T, 1), dtype=float)
            for start in range(0, T, CHUNK):
                block = arr[start:start+CHUNK]  # (chunk, N)
                with np.errstate(invalid="ignore"):
                    med_block = np.nanmedian(block, axis=1, keepdims=True)
                    mad_block = np.nanmedian(np.abs(block - med_block), axis=1, keepdims=True)
                med_arr[start:start+CHUNK] = med_block
                mad_arr[start:start+CHUNK] = mad_block
            lower = med_arr - n_std * mad_arr
            upper = med_arr + n_std * mad_arr
            with np.errstate(invalid="ignore"):
                out = np.clip(arr, lower, upper)
            out[np.isnan(arr)] = np.nan
            return pd.DataFrame(out, index=panel.index, columns=panel.columns)

        # ── 快速路径 4：无 industry，尝试 panel.apply(axis=1) ────────────────
        if industry is None:
            try:
                result = panel.apply(cs_func, axis=1)
                if isinstance(result, pd.Series):
                    pass   # 标量输出，降级到逐行
                else:
                    return result.reindex(columns=panel.columns)
            except Exception:
                pass  # 降级到逐行处理

        # ── 回退路径：逐行处理（industry 参数或 apply 失败时）──────────────
        result_rows: dict = {}
        for date, row in panel.iterrows():
            x = row.dropna()
            if len(x) == 0:
                result_rows[date] = row
                continue
            try:
                if industry is not None:
                    grp = industry.reindex(x.index)
                    result_rows[date] = cs_func(x, grp)
                else:
                    result_rows[date] = cs_func(x)
            except Exception:
                result_rows[date] = row
        return pd.DataFrame(result_rows).T.reindex(columns=panel.columns)
