"""
factor_framework.data.store
============================
DataStore 抽象层 —— 将"数据从哪里来"与"如何计算因子"解耦。

设计原则（v3.0 规范 §3.3）
--------------------------
- DataStore 是抽象基类，定义三个核心接口：
    get_price_panel()   : 返回后复权价格面板（TimestampedPanel, price_basis='hfq'）
    get_raw_df()        : 返回单只股票的清洗后 DataFrame（供因子计算用）
    list_symbols()      : 返回可用股票代码列表
- CSVDataStore 是面向当前 Stocks/ 目录结构的具体实现，
  内部复用 factor_engine._fast_load / data_cleaner.load_and_clean。
- 下游（PanelBuilder、ReturnPanel）均接收 DataStore，不直接碰文件路径，
  便于未来切换到数据库 / 云存储等数据源。

当前工作目录数据格式
-------------------
- Stocks/ 目录：每只股票一个 CSV，文件名 = ts_code（如 000001_SZ.csv）
- 必含列：交易日（str YYYYMMDD）、收盘价、复权因子
- CSVDataStore 优先使用 复权因子 × 原始收盘价 重建后复权序列；
  若复权因子列不存在，直接使用收盘价列（近似后复权）。

使用方式
--------
    store = CSVDataStore(stocks_dir="Stocks/")
    syms  = store.list_symbols()
    price_panel = store.get_price_panel(
        symbols=syms[:10], start="20200101", end="20251231"
    )
    raw_df = store.get_raw_df("000001_SZ")
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_framework.core.panel import TimestampedPanel


# ═══════════════════════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore(ABC):
    """
    数据存储抽象基类。

    所有数据源实现必须继承此类，实现以下三个抽象方法。
    """

    @abstractmethod
    def list_symbols(self) -> List[str]:
        """
        返回此数据源中所有可用的股票代码列表（ts_code 格式）。

        Returns
        -------
        List[str]，如 ['000001_SZ', '000002_SZ', ...]
        """

    @abstractmethod
    def get_raw_df(
        self,
        symbol:    str,
        fast_mode: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        返回单只股票的清洗后 DataFrame（供因子函数使用）。

        Parameters
        ----------
        symbol    : 股票代码（ts_code 格式）
        fast_mode : True = 轻量加载（跳过完整 MAD Winsorize，约 3x 提速）

        Returns
        -------
        pd.DataFrame 或 None（该股票数据不可用时）

        DataFrame 列要求（最小集合）
        ----------------------------
        交易日（str YYYYMMDD）、收盘价、复权因子（可选）
        """

    @abstractmethod
    def get_price_panel(
        self,
        symbols:   Optional[List[str]] = None,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
    ) -> TimestampedPanel:
        """
        构建后复权价格面板（TimestampedPanel, semantic='price', price_basis='hfq'）。

        Parameters
        ----------
        symbols : 股票代码列表（None = 全部）
        start   : 开始日期（YYYYMMDD，None = 全部）
        end     : 结束日期（YYYYMMDD，None = 全部）

        Returns
        -------
        TimestampedPanel(
            semantic    = 'price',
            price_basis = 'hfq',
            index       = 交易日（str YYYYMMDD），
            columns     = ts_code，
            values      = 后复权收盘价，
        )
        """


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 实现
# ═══════════════════════════════════════════════════════════════════════════════

class CSVDataStore(DataStore):
    """
    基于 Stocks/ 目录 CSV 文件的数据存储实现。

    Parameters
    ----------
    stocks_dir : CSV 数据目录（默认 "Stocks/"）
    fast_mode  : 默认是否使用轻量加载（可被 get_raw_df 的同名参数覆盖）

    目录结构要求
    -----------
    stocks_dir/
        000001_SZ.csv
        000002_SZ.csv
        ...
    每个 CSV 文件的文件名（去 .csv）即为 ts_code。
    """

    # 轻量加载需要的最小列集合（与 factor_engine._FAST_COLS 保持一致）
    _FAST_COLS = [
        "交易日", "股票代码",
        "收盘价", "开盘价", "最高价", "最低价",
        "成交量（手）", "成交额（千元）",
        "换手率（%）", "总市值（万元）", "流通市值（万元）",
        "市净率", "市盈率（TTM，亏损为空）", "市销率（TTM）",
        "复权因子",
    ]

    def __init__(
        self,
        stocks_dir: str | Path = "Stocks/",
        fast_mode:  bool = True,
    ) -> None:
        self.stocks_dir = Path(stocks_dir)
        self.fast_mode  = fast_mode

        # 符号列表延迟加载
        self._symbols: Optional[List[str]] = None

    # ── 符号列表 ──────────────────────────────────────────────────────────────

    def list_symbols(self) -> List[str]:
        """返回 stocks_dir 下所有 CSV 文件的文件名（去 .csv）列表。"""
        if self._symbols is None:
            self._symbols = sorted(
                f.stem for f in self.stocks_dir.glob("*.csv")
            )
        return list(self._symbols)

    # ── 单只股票原始 DataFrame ───────────────────────────────────────────────

    def get_raw_df(
        self,
        symbol:    str,
        fast_mode: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        """
        加载并返回单只股票的清洗后 DataFrame。

        Parameters
        ----------
        symbol    : 股票代码（ts_code，文件名去 .csv）
        fast_mode : True = 轻量加载；False = 完整 load_and_clean；
                    None = 使用实例默认值

        Returns
        -------
        pd.DataFrame（已排序）或 None（文件不存在 / 加载失败）
        """
        use_fast = fast_mode if fast_mode is not None else self.fast_mode
        csv_path = self.stocks_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return None

        if use_fast:
            return self._fast_load(csv_path)
        else:
            return self._full_load(csv_path)

    def _fast_load(self, path: Path) -> Optional[pd.DataFrame]:
        """
        轻量级加载：只读 _FAST_COLS，跳过完整 MAD Winsorize。
        与 factor_engine._fast_load 逻辑一致，在 DataStore 层统一管理。
        """
        try:
            header  = pd.read_csv(path, nrows=0)
            usecols = [c for c in self._FAST_COLS if c in header.columns]
            df = pd.read_csv(
                path,
                usecols=usecols,
                dtype={"交易日": str, "股票代码": str},
            )
        except Exception:
            return None

        if "交易日" not in df.columns:
            return None

        df = df.sort_values("交易日").reset_index(drop=True)

        # 先用原始未填充收盘价计算日收益率，停牌日为 NaN（正确语义）
        if "收盘价" in df.columns:
            raw_close = df["收盘价"].copy()
            df["_ret"] = raw_close.pct_change()

        # 价格列 ffill（最多 5 天，处理停牌）
        price_cols = [
            "收盘价", "开盘价", "最高价", "最低价",
            "成交量（手）", "成交额（千元）",
            "换手率（%）", "总市值（万元）", "流通市值（万元）",
        ]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=5)

        # 估值列 ffill 不限长度
        val_cols = ["市净率", "市盈率（TTM，亏损为空）", "市销率（TTM）", "复权因子"]
        for col in val_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        return df

    def _full_load(self, path: Path) -> Optional[pd.DataFrame]:
        """
        完整加载：调用 data_cleaner.load_and_clean（含 MAD Winsorize）。
        """
        try:
            from data_cleaner import load_and_clean
            df = load_and_clean(str(path))
            if df is None or df.empty:
                return None
            return df.sort_values("交易日").reset_index(drop=True)
        except Exception as exc:
            warnings.warn(
                f"[CSVDataStore] 完整加载失败 ({path.name}): {exc}，"
                "尝试降级到轻量加载。",
                stacklevel=2,
            )
            return self._fast_load(path)

    # ── 价格面板 ──────────────────────────────────────────────────────────────

    def get_price_panel(
        self,
        symbols:   Optional[List[str]] = None,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
    ) -> TimestampedPanel:
        """
        构建后复权价格面板。

        后复权价格计算：
        - 若数据含 '复权因子' 列：hfq_close = 收盘价 × 复权因子
        - 否则：hfq_close = 收盘价（近似，发出 warning）

        Returns
        -------
        TimestampedPanel(semantic='price', price_basis='hfq')

        index = YYYYMMDD 字符串，columns = ts_code，values = 后复权收盘价
        """
        targets = symbols or self.list_symbols()
        series_list: Dict[str, pd.Series] = {}

        for sym in targets:
            df = self.get_raw_df(sym, fast_mode=True)
            if df is None or "交易日" not in df.columns or "收盘价" not in df.columns:
                continue

            # 日期筛选
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df["交易日"] >= start
            if end:
                mask &= df["交易日"] <= end
            df_slice = df.loc[mask]
            if df_slice.empty:
                continue

            # 后复权价格
            close = df_slice["收盘价"].copy()
            if "复权因子" in df_slice.columns:
                adj_factor = df_slice["复权因子"].fillna(1.0)
                hfq_close  = close * adj_factor
            else:
                warnings.warn(
                    f"[CSVDataStore] {sym}: 无复权因子列，使用原始收盘价（近似后复权）。",
                    stacklevel=2,
                )
                hfq_close = close

            s = pd.Series(hfq_close.values, index=df_slice["交易日"].values, name=sym)
            series_list[sym] = s

        if not series_list:
            # 返回空 TimestampedPanel
            empty_df = pd.DataFrame()
            return TimestampedPanel.from_dataframe(
                empty_df, semantic="price", price_basis="hfq"
            )

        raw_df = pd.DataFrame(series_list)
        raw_df.index.name = "交易日"
        raw_df = raw_df.sort_index()

        return TimestampedPanel.from_dataframe(
            raw_df, semantic="price", price_basis="hfq"
        )
