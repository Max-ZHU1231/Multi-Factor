"""
factor_framework.engine.cache
==============================
CacheLayer —— 两级缓存架构（L1 内存 LRU + L2 Parquet 磁盘）。

设计（v3.0 规范 §3.5）
----------------------
L1：内存 LRU 缓存
    - 沿用现有 LRUCache 接口（factor_engine.py 中已有实现）
    - key = "{factor_name}|{symbol}"，value = pd.Series
    - 容量通过 max_size 参数控制

L2：磁盘 Parquet 缓存（新增）
    - 缓存目标：计算时间超过阈值的因子面板（DataFrame，日期 × 股票）
    - 存储路径：{cache_dir}/{factor_name}/{key_hash}.parquet
    - 失效机制：比较 Parquet mtime 与数据源 CSV 目录下最新文件 mtime
    - 失效粒度：因子级（一个面板整体缓存，不做股票级拆分）
    - 缓存键：因子名 + start + end + sorted(symbols) 列表的 MD5 哈希

协作协议（与 PanelBuilder）
--------------------------
PanelBuilder.build_panel() 调用顺序：
    1. CacheLayer.get_panel(factor_name, cache_key) → 命中则直接返回
    2. （未命中）执行计算
    3. CacheLayer.put_panel(factor_name, cache_key, panel) → 写入磁盘

CacheLayer 对 PanelBuilder 是透明的加速层，不改变 PanelBuilder 的逻辑。

使用方式
--------
    cache = CacheLayer(cache_dir="cache/", stocks_dir="Stocks/")

    key = cache.make_key("momentum_12_1", "20200101", "20251231", symbols)
    panel = cache.get_panel("momentum_12_1", key)
    if panel is None:
        panel = compute(...)              # 实际计算
        cache.put_panel("momentum_12_1", key, panel)
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class CacheLayer:
    """
    两级缓存层：L1 内存 LRU + L2 Parquet 磁盘。

    Parameters
    ----------
    cache_dir      : Parquet 文件根目录（默认 "cache/"）
    stocks_dir     : 股票 CSV 数据目录（用于 mtime 失效检测）
    enabled_l2     : 是否启用 L2 磁盘缓存（默认 True）
    min_calc_secs  : 计算时间超过此阈值才写入 L2（秒，默认 5.0）
    """

    def __init__(
        self,
        cache_dir:     str = "cache/",
        stocks_dir:    str = "Stocks/",
        enabled_l2:    bool = True,
        min_calc_secs: float = 5.0,
    ) -> None:
        self.cache_dir     = Path(cache_dir)
        self.stocks_dir    = Path(stocks_dir)
        self.enabled_l2    = enabled_l2
        self.min_calc_secs = min_calc_secs

        # L1 内存缓存：{key: pd.DataFrame}（面板级）
        self._l1: Dict[str, pd.DataFrame] = {}

        # 数据源最新 mtime（延迟加载，首次查询时计算）
        self._source_mtime: Optional[float] = None

    # ── 缓存键 ────────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(
        factor_name: str,
        start:       str,
        end:         str,
        symbols:     List[str],
    ) -> str:
        """
        生成磁盘缓存键（MD5 哈希，包含所有决定输出的参数）。

        key 组成：因子名 + start + end + sorted(symbols) 列表。
        任何参数变化均产生不同的 key，不会读到错误的缓存。

        Parameters
        ----------
        factor_name : 因子名称
        start       : 开始日期（YYYYMMDD）
        end         : 结束日期（YYYYMMDD）
        symbols     : 股票列表

        Returns
        -------
        16位 MD5 哈希字符串
        """
        raw = f"{factor_name}|{start}|{end}|" + ",".join(sorted(symbols))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    # ── L2 路径 ───────────────────────────────────────────────────────────────

    def _parquet_path(self, factor_name: str, key: str) -> Path:
        """返回因子对应的 Parquet 缓存文件路径。"""
        factor_dir = self.cache_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        return factor_dir / f"{key}.parquet"

    # ── 数据源 mtime ─────────────────────────────────────────────────────────

    def _get_source_mtime(self) -> float:
        """
        返回 stocks_dir 中所有 CSV 文件的最新修改时间（mtime）。

        延迟加载：首次调用时扫描，后续复用（进程级缓存）。
        若需强制刷新，调用 invalidate_source_mtime()。
        """
        if self._source_mtime is not None:
            return self._source_mtime

        stocks_path = self.stocks_dir
        if not stocks_path.exists():
            # 数据目录不存在，返回 0（任何 Parquet 都视为比数据源新）
            self._source_mtime = 0.0
            return 0.0

        mtimes = []
        for f in stocks_path.rglob("*.csv"):
            try:
                mtimes.append(f.stat().st_mtime)
            except OSError:
                pass

        self._source_mtime = max(mtimes) if mtimes else 0.0
        return self._source_mtime

    def invalidate_source_mtime(self) -> None:
        """强制重新扫描数据源 mtime（当数据更新后调用）。"""
        self._source_mtime = None

    # ── L2 有效性检查 ─────────────────────────────────────────────────────────

    def _is_l2_valid(self, parquet_path: Path) -> bool:
        """
        判断 Parquet 缓存文件是否有效。

        有效条件：
        1. 文件存在
        2. Parquet 文件的 mtime > 数据源最新 CSV 的 mtime
        """
        if not parquet_path.exists():
            return False
        try:
            parquet_mtime = parquet_path.stat().st_mtime
        except OSError:
            return False
        source_mtime = self._get_source_mtime()
        return parquet_mtime > source_mtime

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def get_panel(
        self,
        factor_name: str,
        key:         str,
    ) -> Optional[pd.DataFrame]:
        """
        从缓存中读取因子面板。

        查询顺序：L1 内存 → L2 磁盘 Parquet。
        若 L2 命中，自动提升到 L1。

        Parameters
        ----------
        factor_name : 因子名称
        key         : make_key() 生成的缓存键

        Returns
        -------
        pd.DataFrame（命中）或 None（未命中 / 缓存失效）
        """
        l1_key = f"{factor_name}|{key}"

        # L1 命中
        if l1_key in self._l1:
            return self._l1[l1_key]

        # L2 命中
        if self.enabled_l2:
            parquet_path = self._parquet_path(factor_name, key)
            if self._is_l2_valid(parquet_path):
                try:
                    panel = pd.read_parquet(parquet_path)
                    self._l1[l1_key] = panel  # 提升到 L1
                    return panel
                except Exception as e:
                    warnings.warn(
                        f"[CacheLayer] 读取 Parquet 失败 ({parquet_path}): {e}，"
                        "已忽略缓存，将重新计算。",
                        stacklevel=2,
                    )

        return None

    def put_panel(
        self,
        factor_name: str,
        key:         str,
        panel:       pd.DataFrame,
        calc_secs:   float = 0.0,
    ) -> None:
        """
        将因子面板写入缓存。

        Parameters
        ----------
        factor_name : 因子名称
        key         : make_key() 生成的缓存键
        panel       : 因子面板（日期 × 股票）
        calc_secs   : 实际计算耗时（秒）；仅超过 min_calc_secs 的才写入 L2

        Notes
        -----
        L1 无论 calc_secs 如何都会写入（面板已在内存中，无开销）。
        L2 仅在 calc_secs >= min_calc_secs 且 enabled_l2=True 时写入。
        """
        l1_key = f"{factor_name}|{key}"
        self._l1[l1_key] = panel

        if self.enabled_l2 and calc_secs >= self.min_calc_secs:
            parquet_path = self._parquet_path(factor_name, key)
            try:
                panel.to_parquet(parquet_path)
            except Exception as e:
                warnings.warn(
                    f"[CacheLayer] 写入 Parquet 失败 ({parquet_path}): {e}，"
                    "此次计算结果仅保存在 L1 内存缓存中。",
                    stacklevel=2,
                )

    def clear_l1(self) -> None:
        """清空 L1 内存缓存（释放内存）。"""
        self._l1.clear()

    def clear_l2(self, factor_name: Optional[str] = None) -> int:
        """
        清空 L2 磁盘缓存。

        Parameters
        ----------
        factor_name : 若指定，只清除该因子的缓存；否则清除全部

        Returns
        -------
        删除的文件数量
        """
        count = 0
        if factor_name is not None:
            target_dir = self.cache_dir / factor_name
        else:
            target_dir = self.cache_dir

        if not target_dir.exists():
            return 0

        for f in target_dir.rglob("*.parquet"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        return count

    def cache_info(self) -> Dict[str, object]:
        """
        返回缓存状态摘要（用于调试/监控）。

        Returns
        -------
        dict:
            l1_entries    : L1 内存中的面板数量
            l2_files      : L2 磁盘中的 Parquet 文件数量
            l2_total_mb   : L2 磁盘总大小（MB）
            source_mtime  : 数据源最新 mtime（延迟加载，首次调用时触发）
        """
        l2_files = list(self.cache_dir.rglob("*.parquet")) if self.cache_dir.exists() else []
        total_bytes = sum(f.stat().st_size for f in l2_files if f.exists())
        return {
            "l1_entries":   len(self._l1),
            "l2_files":     len(l2_files),
            "l2_total_mb":  round(total_bytes / 1024 / 1024, 2),
            "source_mtime": self._source_mtime,
            "enabled_l2":   self.enabled_l2,
        }
