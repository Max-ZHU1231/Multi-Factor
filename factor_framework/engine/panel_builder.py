"""
factor_framework.engine.panel_builder
=======================================
PanelBuilder —— 将"因子计算调度"与"缓存策略"整合的面板构建器。

设计原则（v3.0 规范 §3.4）
--------------------------
PanelBuilder 是 FactorEngine 和 CacheLayer 之间的协调层：

    PanelBuilder.build_panel(factor_name, ...)
        1. 生成缓存键（CacheLayer.make_key）
        2. 查询缓存（CacheLayer.get_panel） → 命中：直接返回
        3. 未命中：委托 FactorEngine.build_panel 执行计算（计时）
        4. 写入缓存（CacheLayer.put_panel，含计算耗时）
        5. 返回面板

PanelBuilder 同时承担 ReturnPanel 构建：

    PanelBuilder.build_return_panel(forward, ...)
        - 首先尝试从缓存读取（以 "__ret_{forward}__" 为 factor_name）
        - 未命中时委托 FactorEngine.build_return_panel 计算
        - 写入缓存

与现有 pipeline.py 的关系
-------------------------
- 阶段二目标：FactorPipeline 持有 PanelBuilder（而非直接持有 FactorEngine）
- PanelBuilder 对外暴露与 FactorEngine 完全相同的 build_panel / build_return_panel
  签名，做到 pipeline.py 零改动即可切换
- FactorEngine 实例由 PanelBuilder 内部管理，外部可通过 .engine 属性访问

使用方式
--------
    from factor_framework.engine.panel_builder import PanelBuilder
    from factor_framework.engine.cache import CacheLayer

    cache   = CacheLayer(cache_dir="cache/", stocks_dir="Stocks/")
    builder = PanelBuilder(
        stocks_dir  = "Stocks/",
        stock_basic = "股票列表-stock_basic.csv",
        cache       = cache,
    )
    panel = builder.build_panel("momentum_12_1", start="20200101", end="20251231")

不使用缓存（向后兼容 FactorEngine 用法）
----------------------------------------
    builder = PanelBuilder(stocks_dir="Stocks/", stock_basic="...", cache=None)
    panel = builder.build_panel("momentum_12_1")   # 与直接调用 engine 一致
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from factor_framework.engine.cache import CacheLayer


class PanelBuilder:
    """
    带缓存的因子面板构建器。

    Parameters
    ----------
    stocks_dir  : 股票 CSV 数据目录
    stock_basic : 股票基本信息 CSV（用于行业映射）
    cache       : CacheLayer 实例；None = 不使用缓存（兼容模式）
    min_rows    : 有效数据的最少行数
    verbose     : 是否打印进度条
    n_jobs      : 并行线程数
    store       : DataStore 实例（可选）；传入后优先通过 DataStore 读取数据
    """

    def __init__(
        self,
        stocks_dir:  str | Path = "Stocks/",
        stock_basic: str | Path = "股票列表-stock_basic.csv",
        cache:       Optional[CacheLayer] = None,
        min_rows:    int = 60,
        verbose:     bool = True,
        n_jobs:      int = 8,
        store=None,  # Optional[DataStore] — 避免循环导入，运行时检查类型
    ) -> None:
        self.stocks_dir  = Path(stocks_dir)
        self.stock_basic = Path(stock_basic)
        self.cache       = cache
        self.min_rows    = min_rows
        self.verbose     = verbose
        self.n_jobs      = n_jobs
        self.store       = store   # DataStore 实例（可选）

        # 延迟初始化 FactorEngine（避免在 import 时触发重量级初始化）
        self._engine: Optional[object] = None

    # ── FactorEngine 延迟初始化 ──────────────────────────────────────────────

    @property
    def engine(self):
        """返回底层 FactorEngine 实例（首次访问时初始化）。"""
        if self._engine is None:
            from factor_framework.factor_engine import FactorEngine
            self._engine = FactorEngine(
                stocks_dir  = self.stocks_dir,
                stock_basic = self.stock_basic,
                min_rows    = self.min_rows,
                verbose     = self.verbose,
                _internal   = True,   # 通过 PanelBuilder 调用，抑制 deprecation 警告
            )
        return self._engine

    # ── 注册代理（透传给 FactorEngine）──────────────────────────────────────

    def register(self, name: str, func, deps=None) -> "PanelBuilder":
        """注册 lambda/函数因子，透传给底层 FactorEngine。支持链式调用。"""
        self.engine.register(name, func, deps=deps)
        return self

    def register_expr(self, name: str, expr) -> "PanelBuilder":
        """注册 DAG Expr 因子，透传给底层 FactorEngine。支持链式调用。"""
        self.engine.register_expr(name, expr)
        return self

    def register_builtins(self, names=None) -> "PanelBuilder":
        """注册内置因子，透传逻辑与 FactorPipeline.register_builtins 一致。"""
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        targets = names or list(BUILTIN_FACTORS.keys())
        for n in targets:
            if n in BUILTIN_FACTORS:
                self.engine.register(n, BUILTIN_FACTORS[n])
            else:
                warnings.warn(f"内置因子 '{n}' 不存在，已跳过。")
        return self

    @property
    def industry_map(self):
        """ts_code → industry 的 Series（透传自 FactorEngine）。"""
        return self.engine.industry_map

    def all_symbols(self) -> List[str]:
        """返回所有股票代码（透传自 FactorEngine）。"""
        return self.engine.all_symbols()

    def apply_cross_section(self, panel, cs_func, industry=None):
        """横截面函数应用（透传自 FactorEngine）。"""
        return self.engine.apply_cross_section(panel, cs_func, industry=industry)

    # ── 缓存键生成 ───────────────────────────────────────────────────────────

    def _make_cache_key(
        self,
        factor_name: str,
        start:       Optional[str],
        end:         Optional[str],
        symbols:     Optional[List[str]],
    ) -> tuple:
        """
        生成 (v2_key, v1_legacy_key) 元组。

        v2 键包含 transform_config_hash / semantic_contract_version / git_sha，
        v1 键保留向后兼容（旧 Parquet 文件读取）。
        """
        _start   = start   or "ALL"
        _end     = end     or "ALL"
        _symbols = symbols or self.all_symbols()

        # v1 旧键（兼容现有 Parquet 缓存）
        v1_key = CacheLayer.make_key(factor_name, _start, _end, _symbols)

        # v2 新键（含语义版本等额外维度）
        if hasattr(self.cache, "make_key_v2"):
            v2_key = self.cache.make_key_v2(factor_name, _start, _end, _symbols)
        else:
            v2_key = v1_key   # 降级兼容

        return v2_key, v1_key

    # ── 核心接口：build_panel ────────────────────────────────────────────────

    def build_panel(
        self,
        factor_name:  str,
        start:        Optional[str] = None,
        end:          Optional[str] = None,
        symbols:      Optional[List[str]] = None,
        n_jobs:       Optional[int] = None,
        fast_mode:    bool = True,
        max_lookback: int = 0,
    ) -> pd.DataFrame:
        """
        构建因子面板，自动使用缓存加速（若 cache != None）。

        与 FactorEngine.build_panel() 签名完全兼容。

        流程
        ----
        1. 生成缓存键
        2. CacheLayer.get_panel() → 命中：直接返回（L1 或 L2）
        3. 未命中：FactorEngine.build_panel()（计时）
        4. CacheLayer.put_panel()（超过 min_calc_secs 才写 L2）
        5. 返回

        Parameters
        ----------
        factor_name  : 已注册的因子名称
        start / end  : 日期范围（YYYYMMDD）
        symbols      : 股票列表（None = 全部）
        n_jobs       : 并行数（None = 使用实例默认值）
        fast_mode    : 是否轻量加载
        max_lookback : warm-up 期行数

        Returns
        -------
        pd.DataFrame，index=交易日，columns=ts_code，values=因子值
        """
        _n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        # ── 缓存路径 ──────────────────────────────────────────────────────
        if self.cache is not None:
            key, legacy_key = self._make_cache_key(factor_name, start, end, symbols)
            panel = self.cache.get_panel(factor_name, key, legacy_key=legacy_key)
            if panel is not None:
                return panel

        # ── 计算路径 ──────────────────────────────────────────────────────
        t0    = time.perf_counter()
        panel = self.engine.build_panel(
            factor_name,
            start=start,
            end=end,
            symbols=symbols,
            n_jobs=_n_jobs,
            fast_mode=fast_mode,
            max_lookback=max_lookback,
        )
        elapsed = time.perf_counter() - t0

        # ── 写入缓存（使用 v2 新键）──────────────────────────────────────
        if self.cache is not None and not panel.empty:
            self.cache.put_panel(factor_name, key, panel, calc_secs=elapsed)

        return panel

    # ── 核心接口：build_return_panel ─────────────────────────────────────────

    def build_return_panel(
        self,
        forward:   int = 1,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        symbols:   Optional[List[str]] = None,
        fast_mode: bool = True,
    ) -> pd.DataFrame:
        """
        构建未来 forward 日收益率面板，自动使用缓存加速。

        与 FactorEngine.build_return_panel() 签名完全兼容。

        缓存键 factor_name 使用 "__ret_{forward}__" 以避免与业务因子冲突。

        Parameters
        ----------
        forward   : 预测期（天数）
        start/end : 日期范围
        symbols   : 股票列表
        fast_mode : 是否轻量加载

        Returns
        -------
        pd.DataFrame（含内置 T+1 滞后）
        """
        cache_name = f"__ret_{forward}__"

        # ── 缓存路径 ──────────────────────────────────────────────────────
        if self.cache is not None:
            key, legacy_key = self._make_cache_key(cache_name, start, end, symbols)
            panel = self.cache.get_panel(cache_name, key, legacy_key=legacy_key)
            if panel is not None:
                return panel

        # ── 计算路径 ──────────────────────────────────────────────────────
        t0    = time.perf_counter()
        panel = self.engine.build_return_panel(
            forward=forward, start=start, end=end,
            symbols=symbols, fast_mode=fast_mode,
        )
        elapsed = time.perf_counter() - t0

        # ── 写入缓存（v2 新键）───────────────────────────────────────────
        if self.cache is not None and not panel.empty:
            self.cache.put_panel(cache_name, key, panel, calc_secs=elapsed)

        return panel

    # ── 批量面板构建（透传 + 缓存）────────────────────────────────────────────

    def build_panel_batch(
        self,
        factor_names: List[str],
        start:        Optional[str] = None,
        end:          Optional[str] = None,
        symbols:      Optional[List[str]] = None,
        n_jobs:       Optional[int] = None,
        fast_mode:    bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量构建多个因子面板，逐一尝试缓存命中。

        对每个因子名独立做缓存查询/写入。
        若全部因子缓存命中，无需执行任何计算（第二次调用极快）。
        若任意因子未命中，委托 FactorEngine.build_panel_batch() 批量计算（共享 CSE）。

        Parameters
        ----------
        factor_names : 因子名称列表
        start/end    : 日期范围
        symbols      : 股票列表
        n_jobs       : 并行数
        fast_mode    : 是否轻量加载

        Returns
        -------
        Dict[str, pd.DataFrame]，{factor_name: panel}
        """
        _n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        result: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        # 第一步：缓存查询
        for fn in factor_names:
            if self.cache is not None:
                key, legacy_key = self._make_cache_key(fn, start, end, symbols)
                panel = self.cache.get_panel(fn, key, legacy_key=legacy_key)
                if panel is not None:
                    result[fn] = panel
                    continue
            missing.append(fn)

        if not missing:
            return result

        # 第二步：批量计算未命中的因子
        t0 = time.perf_counter()
        batch = self.engine.build_panel_batch(
            missing,
            start=start, end=end, symbols=symbols,
            n_jobs=_n_jobs, fast_mode=fast_mode,
        )
        elapsed = time.perf_counter() - t0
        # 平均耗时（每个因子视为计算时间的均等份额）
        per_factor_secs = elapsed / max(len(missing), 1)

        # 第三步：写入缓存（v2 新键）
        for fn, panel in batch.items():
            result[fn] = panel
            if self.cache is not None and not panel.empty:
                key, _ = self._make_cache_key(fn, start, end, symbols)
                self.cache.put_panel(fn, key, panel, calc_secs=per_factor_secs)

        return result
