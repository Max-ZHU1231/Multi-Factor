"""
universes/loader.py
===================
股票池（Universe）加载器。

支持的股票池
-----------
  all         全部股票（数据目录中的所有 CSV，默认行为）
  hs300       沪深300成分股（universes/hs300.csv）
  <文件路径>  用户自定义 CSV 文件，必须含 ``code`` 列

文件格式
--------
CSV，至少含以下列：
  code     : 股票代码（6 位数字，不含交易所后缀）
  name     : 股票名称（可选）
  exchange : 交易所/板块（可选）
  weight   : 权重 %（可选）

ts_code 映射规则
----------------
代码 XXXXXX + 交易所判断：
  6xxxxx → XXXXXX.SH
  0xxxxx / 3xxxxx / 002xxx / 001xxx / 003xxx → XXXXXX.SZ
  688xxx / 689xxx → XXXXXX.SH（科创板）
  8xxxxx / 9xxxxx / 43xxxx → XXXXXX.BJ（北交所）

用法
----
from universes.loader import UniverseLoader

# 加载沪深300
syms = UniverseLoader.load("hs300")
# → ['000001.SZ', '000002.SZ', ..., '688981.SH']

# 加载全部（与 None 等价）
syms = UniverseLoader.load("all")   # → None（下游按全部处理）

# 用户自定义文件
syms = UniverseLoader.load("my_pool.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

# 内置股票池目录
_UNIVERSE_DIR = Path(__file__).parent

# 内置股票池别名 → 文件名
_BUILTIN_ALIAS: dict[str, str] = {
    "hs300":  "hs300.csv",
    "csi300": "hs300.csv",   # 英文别名
    "沪深300": "hs300.csv",  # 中文别名
}


def _code_to_ts_code(code: str) -> str:
    """
    将 6 位股票代码转换为 ts_code（带交易所后缀）。

    规则：
      6xxxxx          → XXXXXX.SH （上交所主板 + 科创板）
      688xxx / 689xxx → XXXXXX.SH （科创板，已含于上条）
      000xxx / 001xxx / 002xxx / 003xxx / 300xxx → XXXXXX.SZ （深交所）
      8xxxxx / 9xxxxx / 43xxxx     → XXXXXX.BJ （北交所）
    """
    code = code.strip().zfill(6)
    if code.startswith("6"):
        return f"{code}.SH"
    elif code.startswith(("0", "1", "2", "3")):
        return f"{code}.SZ"
    elif code.startswith(("8", "9", "43")):
        return f"{code}.BJ"
    else:
        # 默认深交所
        return f"{code}.SZ"


class UniverseLoader:
    """股票池加载工具类（全静态方法）。"""

    @staticmethod
    def load(
        universe: Optional[str],
        stocks_dir: Optional[Path] = None,
    ) -> Optional[List[str]]:
        """
        加载股票池，返回 ts_code 列表（如 '000001.SZ'）。

        Parameters
        ----------
        universe   : 股票池名称 / 文件路径 / None
                     - None 或 "all"  → 返回 None（下游使用全部股票）
                     - "hs300"        → 内置沪深300
                     - 文件路径       → 自定义 CSV
        stocks_dir : 数据目录（用于过滤不存在的股票，可选）

        Returns
        -------
        list[str] | None
            - None   表示不限制（全部股票）
            - list   已过滤并排序的 ts_code 列表
        """
        if universe is None or universe.strip().lower() == "all":
            return None  # 不限制，下游使用全部股票

        # 解析文件路径
        csv_path = UniverseLoader._resolve_path(universe)
        if csv_path is None:
            raise FileNotFoundError(
                f"[UniverseLoader] 找不到股票池文件: {universe!r}\n"
                f"  内置别名: {list(_BUILTIN_ALIAS)}\n"
                f"  也可传入 CSV 文件路径（含 'code' 列）"
            )

        # 读取 CSV
        df = pd.read_csv(csv_path, dtype=str)
        if "code" not in df.columns:
            raise ValueError(
                f"[UniverseLoader] 文件 {csv_path} 缺少 'code' 列。"
                f"  现有列: {list(df.columns)}"
            )

        # 转换为 ts_code
        ts_codes = [_code_to_ts_code(c) for c in df["code"].dropna()]
        ts_codes = sorted(set(ts_codes))  # 去重 + 排序

        # 可选：过滤掉数据目录中不存在的股票
        if stocks_dir is not None:
            stocks_dir = Path(stocks_dir)
            available = {f.stem for f in stocks_dir.glob("*.csv")}
            before = len(ts_codes)
            ts_codes = [c for c in ts_codes if c in available]
            after = len(ts_codes)
            if after < before:
                import warnings
                warnings.warn(
                    f"[UniverseLoader] 股票池 {universe!r}: "
                    f"{before - after} 只股票在数据目录中无对应文件，已跳过。"
                    f"  有效股票数: {after}",
                    stacklevel=2,
                )

        return ts_codes

    @staticmethod
    def list_builtins() -> dict[str, Path]:
        """返回所有内置股票池及其文件路径。"""
        result = {}
        for alias, fname in _BUILTIN_ALIAS.items():
            p = _UNIVERSE_DIR / fname
            if p.exists():
                result[alias] = p
        return result

    @staticmethod
    def info(universe: str, stocks_dir: Optional[Path] = None) -> dict:
        """
        返回股票池信息字典，不实际过滤（调试用）。

        Returns
        -------
        dict with keys: name, path, total, available (if stocks_dir given)
        """
        csv_path = UniverseLoader._resolve_path(universe)
        if csv_path is None:
            return {"name": universe, "error": "文件不存在"}

        df = pd.read_csv(csv_path, dtype=str)
        ts_codes = [_code_to_ts_code(c) for c in df["code"].dropna()]
        result: dict = {
            "name":    universe,
            "path":    str(csv_path),
            "total":   len(ts_codes),
            "columns": list(df.columns),
        }
        if stocks_dir is not None:
            available = {f.stem for f in Path(stocks_dir).glob("*.csv")}
            result["available"] = sum(1 for c in ts_codes if c in available)
            result["missing"]   = result["total"] - result["available"]
        return result

    @staticmethod
    def _resolve_path(universe: str) -> Optional[Path]:
        """将别名或路径解析为实际 CSV 文件 Path，找不到返回 None。"""
        # 内置别名
        alias_key = universe.strip().lower()
        if alias_key in _BUILTIN_ALIAS:
            p = _UNIVERSE_DIR / _BUILTIN_ALIAS[alias_key]
            return p if p.exists() else None

        # 直接文件路径
        p = Path(universe)
        if p.exists():
            return p

        # 相对于 universes/ 目录
        p2 = _UNIVERSE_DIR / universe
        if p2.exists():
            return p2

        return None
