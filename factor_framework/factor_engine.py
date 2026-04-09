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

import warnings
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

    # ── 单股计算 ──────────────────────────────────────────────────────────────

    def compute_single(
        self,
        symbol:      str,
        factor_name: str,
        start:       Optional[str] = None,
        end:         Optional[str] = None,
    ) -> Optional[pd.Series]:
        """
        计算单只股票的因子值序列。

        Returns
        -------
        pd.Series，index = 交易日(str)，values = 因子值；若数据不足返回 None。
        """
        if factor_name not in self._registry:
            raise KeyError(f"因子 '{factor_name}' 未注册，请先调用 register()。")

        path = self.stocks_dir / f"{symbol}.csv"
        if not path.exists():
            return None

        df = load_and_clean(path)
        if df is None or len(df) < self.min_rows:
            return None

        # 内部生成日收益率列（供动量类因子使用）
        df[COL_RET] = df[COL_CLOSE].pct_change()

        # 日期筛选
        if start:
            df = df[df[COL_DATE] >= start]
        if end:
            df = df[df[COL_DATE] <= end]
        if df.empty:
            return None

        try:
            factor_vals = self._registry[factor_name](df)
        except Exception as e:
            warnings.warn(f"{symbol}/{factor_name} 计算失败: {e}")
            return None

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
        n_jobs:      int = 1,
    ) -> pd.DataFrame:
        """
        批量计算所有股票的因子值，构建 (日期 × 股票) 面板。

        Parameters
        ----------
        factor_name : 因子名称（已注册）
        start / end : 日期范围（YYYYMMDD 字符串，含端点）
        symbols     : 指定股票列表（None = 全部）
        n_jobs      : 暂支持 1（单进程），保留扩展接口

        Returns
        -------
        pd.DataFrame，index = 交易日(str)，columns = ts_code，values = 因子值
        """
        targets = symbols or self.all_symbols()
        series_list = []

        iterable = tqdm(targets, desc=f"计算 {factor_name}", disable=not self.verbose)
        for sym in iterable:
            s = self.compute_single(sym, factor_name, start=start, end=end)
            if s is not None and not s.isna().all():
                series_list.append(s)

        if not series_list:
            warnings.warn(f"因子 '{factor_name}' 无有效数据，返回空 DataFrame。")
            return pd.DataFrame()

        panel = pd.concat(series_list, axis=1)
        panel = panel.sort_index()
        return panel

    # ── 收益率面板 ────────────────────────────────────────────────────────────

    def build_return_panel(
        self,
        forward:  int = 1,
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        symbols:  Optional[List[str]] = None,
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
        panel = self.build_panel(_name, start=start, end=end, symbols=symbols)
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
