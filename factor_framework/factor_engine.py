"""
factor_engine.py
================
因子注册 / 计算 / 面板构建引擎。

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
    """

    def __init__(
        self,
        stocks_dir:  str | Path = "Stocks/",
        stock_basic: str | Path = "股票列表-stock_basic.csv",
        min_rows:    int = 60,
        verbose:     bool = True,
        intermediate_lru: int = _DEFAULT_INTERMEDIATE_LRU,
        factor_lru:       int = _DEFAULT_FACTOR_LRU,
    ):
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
            warnings.warn(f"stock_basic 文件不存在: {self.stock_basic}")

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
            warnings.warn(f"因子 '{name}' 已存在，将被覆盖。")
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
            raise TypeError(f"register_expr 需要 Expr 对象，收到 {type(expr)}")
        if name in self._expr_registry:
            warnings.warn(f"Expr 因子 '{name}' 已存在，将被覆盖。")
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
                raise ValueError(f"DAG 执行后未找到因子 '{_name}' 的结果。")
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
                # 内部生成日收益率列，一次即可
                df = df.copy()
                df[COL_RET] = df[COL_CLOSE].pct_change()
                cache[symbol] = df

            return cache[symbol]

    # ── 单股计算 ──────────────────────────────────────────────────────────────

    def compute_single(
        self,
        symbol:      str,
        factor_name: str,
        start:       Optional[str] = None,
        end:         Optional[str] = None,
        fast_mode:   bool = False,
    ) -> Optional[pd.Series]:
        """
        计算单只股票的因子值序列。

        支持三种注册方式：
        1. ``register_expr`` 注册的 Expr 因子（DAG 执行）
        2. ``register(..., deps=[...])`` 注册的带依赖 lambda 因子
        3. 无依赖的普通 lambda 因子

        Parameters
        ----------
        fast_mode : True = 使用轻量加载器（跳过 MAD Winsorize，速度 ~3x）；
                    False = 使用完整 load_and_clean（含 MAD Winsorize）

        Returns
        -------
        pd.Series，index = 交易日(str)，values = 因子值；若数据不足返回 None。
        """
        if factor_name not in self._registry:
            raise KeyError(f"因子 '{factor_name}' 未注册，请先调用 register()。")

        # 最终因子缓存命中（两层缓存：第一层按 symbol+factor_name）
        fc_key = f"{factor_name}|{symbol}"
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

        # 在完整 df 上计算（保证缓存的是全序列，date filter 在输出时动态切片）
        # ── Expr DAG 路径 ────────────────────────────────────────────────────
        if factor_name in self._expr_registry:
            result = self._compute_expr_single(symbol, factor_name, df)
        # ── 显式依赖路径 ────────────────────────────────────────────────────
        elif self._dep_graph.deps_of(factor_name):
            result = self._compute_with_deps(symbol, factor_name, df, fast_mode, None, None)
        # ── 普通 lambda 路径 ─────────────────────────────────────────────────
        else:
            try:
                result = self._registry[factor_name](df)
            except Exception as e:
                warnings.warn(f"{symbol}/{factor_name} 计算失败: {e}")
                return None

        if result is None:
            return None

        result = result.copy()
        result.index = df[COL_DATE].values
        result.name  = symbol

        # 写入最终因子缓存（全序列，不含日期筛选）
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
            warnings.warn(f"{symbol}/{factor_name}(deps) 计算失败: {e}")
            return None

    # ── 面板构建 ──────────────────────────────────────────────────────────────

    def build_panel(
        self,
        factor_name: str,
        start:       Optional[str] = None,
        end:         Optional[str] = None,
        symbols:     Optional[List[str]] = None,
        n_jobs:      int = 8,
        fast_mode:   bool = True,
    ) -> pd.DataFrame:
        """
        批量计算所有股票的因子值，构建 (日期 × 股票) 面板。

        Parameters
        ----------
        factor_name : 因子名称（已注册）
        start / end : 日期范围（YYYYMMDD 字符串，含端点）
        symbols     : 指定股票列表（None = 全部）
        n_jobs      : 并行线程数（默认 8）
        fast_mode   : True = 跳过 MAD Winsorize，速度 ~3x（默认）；
                      False = 使用完整清洗（更准确，更慢）

        Returns
        -------
        pd.DataFrame，index = 交易日(str)，columns = ts_code，values = 因子值
        """
        targets = symbols or self.all_symbols()
        series_list: List[pd.Series] = []

        pbar = tqdm(total=len(targets), desc=f"计算 {factor_name}", disable=not self.verbose)

        def _compute(sym: str) -> Optional[pd.Series]:
            return self.compute_single(sym, factor_name, start=start, end=end, fast_mode=fast_mode)

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
                    warnings.warn(f"{sym}/{factor_name} 线程异常: {e}")

        pbar.close()

        if not series_list:
            warnings.warn(f"因子 '{factor_name}' 无有效数据，返回空 DataFrame。")
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
                raise KeyError(f"因子 '{fn}' 未注册，请先调用 register() 或 register_expr()。")

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

            df_slice = _date_filter_df(df, start, end)
            if df_slice is None:
                return {n: None for n in factor_names}

            out: Dict[str, Optional[pd.Series]] = {}

            # ── Expr 因子：一次拓扑排序，共享中间节点 ───────────────────────
            if expr_names:
                sym_cache = LRUCache(-1)
                executor  = DAGExecutor(sym_cache, self._expr_registry)
                dag_results = executor.run(df_slice, factor_names=expr_names)
                for n in expr_names:
                    val = dag_results.get(n)
                    if val is not None:
                        val = val.copy()
                        val.index = df_slice[COL_DATE].values
                        val.name  = sym
                    out[n] = val

            # ── Lambda 因子：按拓扑顺序执行（尊重 deps）───────────────────
            if lamb_names:
                # 拓扑排序含依赖
                topo_lambs = self._dep_graph.topo_order(lamb_names)
                # 确保所有 lamb_names 都包含（无依赖的放在最后）
                topo_lambs = topo_lambs + [n for n in lamb_names if n not in topo_lambs]

                df_work = df_slice.copy()
                for n in topo_lambs:
                    if n not in lamb_names:
                        continue
                    self._resolve_compile_target(n)
                    # 注入依赖列
                    for dep in self._dep_graph.deps_of(n):
                        dep_col = f"__dep_{dep}__"
                        if dep_col not in df_work.columns:
                            dep_val = out.get(dep)
                            if dep_val is not None:
                                df_work[dep_col] = dep_val.reindex(
                                    df_work[COL_DATE].values
                                ).values
                    try:
                        val = self._registry[n](df_work)
                        if val is not None:
                            val = val.copy()
                            val.index = df_work[COL_DATE].values
                            val.name  = sym
                        out[n] = val
                        # 写入 __dep__ 列供后续因子使用
                        if val is not None:
                            df_work[f"__dep_{n}__"] = val.values
                    except Exception as e:
                        warnings.warn(f"{sym}/{n} 计算失败: {e}")
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
                    warnings.warn(f"{sym} 批量计算线程异常: {e}")

        pbar.close()

        panels: Dict[str, pd.DataFrame] = {}
        for n, series_list in results.items():
            if series_list:
                panel = pd.concat(series_list, axis=1).sort_index()
                panels[n] = panel
            else:
                warnings.warn(f"因子 '{n}' 无有效数据，返回空 DataFrame。")
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

        Returns
        -------
        pd.DataFrame，index = 交易日，columns = ts_code，values = 未来 forward 日收益
        """
        # 临时注册收益率因子
        _name = f"__ret_{forward}__"
        self.register(_name, lambda df: df[COL_CLOSE].pct_change(forward).shift(-forward))
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
        """
        result_rows = {}
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
