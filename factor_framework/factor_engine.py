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

使用示例
--------
from factor_framework.factor_engine import FactorEngine
from factor_framework.operators import ts_mean, log

engine = FactorEngine(stocks_dir='Stocks/', stock_basic='股票列表-stock_basic.csv')
engine.register('log_mktcap', lambda df: log(df['总市值（万元）']))
engine.register('mom_20', lambda df: df['收盘价'].pct_change(20))
panel = engine.build_panel('log_mktcap', start='20200101', end='20261231')
"""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_cleaner import load_and_clean

# 因子函数类型别名
FactorFn = Callable[[pd.DataFrame], pd.Series]

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
    1. 注册因子函数
    2. 批量遍历 Stocks/ 目录，对每只股票计算因子
    3. 拼接成 (日期 × 股票) 面板 DataFrame
    """

    def __init__(
        self,
        stocks_dir:  str | Path = "Stocks/",
        stock_basic: str | Path = "股票列表-stock_basic.csv",
        min_rows:    int = 60,
        verbose:     bool = True,
    ):
        self.stocks_dir  = Path(stocks_dir)
        self.stock_basic = Path(stock_basic)
        self.min_rows    = min_rows
        self.verbose     = verbose

        # 因子注册表：name → func
        self._registry: Dict[str, FactorFn] = {}

        # 行业映射 ts_code → industry（来自 stock_basic）
        self._industry_map: Optional[pd.Series] = None

        # ── DataFrame 缓存（瓶颈 2：消除多因子重复读盘）────────────────────
        # 键：symbol（文件 stem），值：已清洗的 DataFrame 或 None（文件不存在/新股）
        # fast_mode=True / False 分别用独立缓存，避免混用
        self._cache_fast: Dict[str, Optional[pd.DataFrame]] = {}
        self._cache_full: Dict[str, Optional[pd.DataFrame]] = {}
        # 缓存锁（ThreadPoolExecutor 下保证写安全）
        self._cache_lock = threading.Lock()

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

    def register(self, name: str, func: FactorFn) -> None:
        """
        注册一个因子函数。

        Parameters
        ----------
        name : 因子名称（唯一标识）
        func : (df: pd.DataFrame) -> pd.Series
               输入：单只股票的日频 DataFrame（已清洗）
               输出：等长的因子值 Series（index = 整数，与 df 一致）
        """
        if name in self._registry:
            warnings.warn(f"因子 '{name}' 已存在，将被覆盖。")
        self._registry[name] = func

    def registered(self) -> List[str]:
        """列出所有已注册因子名称。"""
        return list(self._registry.keys())

    def clear_cache(self) -> None:
        """手动清空 DataFrame 缓存（释放内存）。"""
        with self._cache_lock:
            self._cache_fast.clear()
            self._cache_full.clear()

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

        # 从缓存获取（不再每次重复读盘）
        df = self._load_df(symbol, fast_mode)
        if df is None:
            return None

        # 日期筛选（在缓存副本上切片，不修改缓存）
        if start or end:
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df[COL_DATE] >= start
            if end:
                mask &= df[COL_DATE] <= end
            df = df.loc[mask]

        if df.empty:
            return None

        try:
            factor_vals = self._registry[factor_name](df)
        except Exception as e:
            warnings.warn(f"{symbol}/{factor_name} 计算失败: {e}")
            return None

        factor_vals = factor_vals.copy()
        factor_vals.index = df[COL_DATE].values
        factor_vals.name  = symbol
        return factor_vals

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
