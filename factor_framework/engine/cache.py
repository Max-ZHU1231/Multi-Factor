"""
factor_framework.engine.cache
==============================
CacheLayer —— 两级缓存架构（L1 内存 LRU + L2 Parquet 磁盘）。

设计（v3.0 规范 §3.5 + v4.0 Phase D 升级）
-----------------------------------------
L1：内存 LRU 缓存
    - key = "{factor_name}|{symbol}"，value = pd.Series

L2：磁盘 Parquet 缓存
    - 存储路径：{cache_dir}/{factor_name}/{key_hash}.parquet
    - 失效机制：Parquet mtime > 数据源最新 CSV mtime
    - **v4.0 升级**：缓存键扩展为 CacheKeyV2，加入
        transform_config_hash、semantic_contract_version、git_sha
      同时保留对旧键格式的向后兼容读取（一个过渡版本）

缓存键版本策略（Phase D D1）
----------------------------
新键（v2）：对 factor_name + start + end + sorted(symbols)
           + transform_config_hash + semantic_contract_version + git_sha
           取 MD5，前 16 位为磁盘文件名

旧键（v1）：仅对 factor_name + start + end + sorted(symbols) 取 MD5

查询顺序：v2 新键 → v1 旧键（backward-compat）→ 重新计算
命中来源由 get_panel() 返回值的第二元素携带：
    "new_key_hit" | "legacy_key_hit" | "recompute"
    (为保持与现有 PanelBuilder 的零改动兼容，hit-source 写入
     CacheLayer.last_hit_source 属性，get_panel() 仍返回 DataFrame|None)

协作协议（与 PanelBuilder）
--------------------------
PanelBuilder.build_panel() 调用顺序：
    1. CacheLayer.get_panel(factor_name, new_key) → 命中则直接返回
    2. （未命中）执行计算
    3. CacheLayer.put_panel(factor_name, new_key, panel) → 写入磁盘

使用方式
--------
    cache = CacheLayer(cache_dir="cache/", stocks_dir="Stocks/")

    key = cache.make_key_v2("momentum_12_1", "20200101", "20251231",
                             symbols, transform_cfg_hash="abc", git_sha="def")
    panel = cache.get_panel("momentum_12_1", key)
    if panel is None:
        panel = compute(...)
        cache.put_panel("momentum_12_1", key, panel)
    print(cache.last_hit_source)  # "new_key_hit" | "legacy_key_hit" | "recompute"
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

# ── 版本常量 ─────────────────────────────────────────────────────────────────
#   每次改变因子变换逻辑的 **语义** 时，递增此版本号以使旧缓存自动失效
SEMANTIC_CONTRACT_VERSION: str = "4.0"


def _get_git_sha(repo_root: Optional[Path] = None, length: int = 8) -> str:
    """
    返回当前 git HEAD 的短 SHA（8 位）。

    若不在 git 仓库中（或 git 命令失败），返回 "unknown"。
    """
    import subprocess
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "--short", str(length), "HEAD"],
            capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class CacheLayer:
    """
    两级缓存层：L1 内存 LRU + L2 Parquet 磁盘。

    Parameters
    ----------
    cache_dir      : Parquet 文件根目录（默认 "cache/"）
    stocks_dir     : 股票 CSV 数据目录（用于 mtime 失效检测）
    enabled_l2     : 是否启用 L2 磁盘缓存（默认 True）
    min_calc_secs  : 计算时间超过此阈值才写入 L2（秒，默认 5.0）
    transform_config_hash : 变换配置的哈希（用于 v2 缓存键，默认 ""）
    git_sha        : 当前 git SHA（用于 v2 缓存键；None = 自动获取）
    repo_root      : git 仓库根目录（自动获取 git_sha 时使用）
    """

    def __init__(
        self,
        cache_dir:             str = "cache/",
        stocks_dir:            str = "Stocks/",
        enabled_l2:            bool = True,
        min_calc_secs:         float = 5.0,
        transform_config_hash: str = "",
        git_sha:               Optional[str] = None,
        repo_root:             Optional[Path] = None,
    ) -> None:
        self.cache_dir             = Path(cache_dir)
        self.stocks_dir            = Path(stocks_dir)
        self.enabled_l2            = enabled_l2
        self.min_calc_secs         = min_calc_secs
        self.transform_config_hash = transform_config_hash
        self.git_sha = (
            git_sha if git_sha is not None
            else _get_git_sha(repo_root)
        )

        # L1 内存缓存：{key: pd.DataFrame}（面板级）
        self._l1: Dict[str, pd.DataFrame] = {}

        # 数据源最新 mtime（延迟加载）
        self._source_mtime: Optional[float] = None

        # 最后一次 get_panel 的命中来源
        # "new_key_hit" | "legacy_key_hit" | "recompute"
        self.last_hit_source: str = "recompute"

        # 累计统计计数器
        self._stats: Dict[str, int] = {
            "new_key_hit":    0,
            "legacy_key_hit": 0,
            "recompute":      0,
        }

    # ── 缓存键（v1 — 旧格式，向后兼容）──────────────────────────────────────

    @staticmethod
    def make_key(
        factor_name: str,
        start:       str,
        end:         str,
        symbols:     List[str],
    ) -> str:
        """
        生成 **v1** 磁盘缓存键（MD5，仅含基础维度）。

        v1 键组成：factor_name + start + end + sorted(symbols)
        任何参数变化均产生不同 key。

        .. note::
            新代码应使用 :meth:`make_key_v2`，此方法保留以支持旧缓存读取。
        """
        raw = f"{factor_name}|{start}|{end}|" + ",".join(sorted(symbols))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def make_key_v2(
        self,
        factor_name:           str,
        start:                 str,
        end:                   str,
        symbols:               List[str],
        transform_config_hash: Optional[str] = None,
        git_sha:               Optional[str] = None,
    ) -> str:
        """
        生成 **v2** 磁盘缓存键（MD5，含语义版本 + 变换配置 + git SHA）。

        v2 键组成：
          factor_name + start + end + sorted(symbols)
          + transform_config_hash (默认使用实例属性)
          + SEMANTIC_CONTRACT_VERSION
          + git_sha (默认使用实例属性)

        任何影响因子值语义的变化（包括变换逻辑、合约版本、代码版本）
        均会产生不同 key，自动规避缓存污染。

        Parameters
        ----------
        factor_name           : 因子名称
        start / end           : 日期范围（YYYYMMDD）
        symbols               : 股票列表
        transform_config_hash : 变换配置哈希（None = 使用实例默认值）
        git_sha               : git 短 SHA（None = 使用实例默认值）

        Returns
        -------
        16 位 MD5 哈希字符串
        """
        _tcfg = transform_config_hash if transform_config_hash is not None \
                else self.transform_config_hash
        _sha  = git_sha if git_sha is not None else self.git_sha

        raw = (
            f"{factor_name}|{start}|{end}|"
            + ",".join(sorted(symbols))
            + f"|tcfg={_tcfg}"
            + f"|contract={SEMANTIC_CONTRACT_VERSION}"
            + f"|git={_sha}"
        )
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
        legacy_key:  Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        从缓存中读取因子面板。

        查询顺序（Phase D D1 升级版）：
          1. L1 内存（新键）
          2. L2 磁盘（新键，v2）          → hit_source = "new_key_hit"
          3. L2 磁盘（旧键 legacy_key，v1）→ hit_source = "legacy_key_hit"
          4. 返回 None                    → hit_source = "recompute"

        命中来源写入 ``self.last_hit_source`` 并累计到 ``self._stats``。

        Parameters
        ----------
        factor_name : 因子名称
        key         : make_key_v2() 生成的 v2 键
        legacy_key  : make_key() 生成的 v1 键（可选；为 None 时跳过 v1 查询）

        Returns
        -------
        pd.DataFrame（命中）或 None（未命中 / 缓存失效）
        """
        l1_key = f"{factor_name}|{key}"

        # ── 1. L1 命中（新键）──────────────────────────────────────────────
        if l1_key in self._l1:
            self.last_hit_source = "new_key_hit"
            self._stats["new_key_hit"] += 1
            return self._l1[l1_key]

        if self.enabled_l2:
            # ── 2. L2 命中（新键 v2）──────────────────────────────────────
            parquet_path = self._parquet_path(factor_name, key)
            if self._is_l2_valid(parquet_path):
                try:
                    panel = pd.read_parquet(parquet_path)
                    self._l1[l1_key] = panel
                    self.last_hit_source = "new_key_hit"
                    self._stats["new_key_hit"] += 1
                    return panel
                except Exception as e:
                    warnings.warn(
                        f"[CacheLayer] 读取 Parquet 失败 ({parquet_path}): {e}，"
                        "已忽略缓存，将尝试旧键或重新计算。",
                        stacklevel=2,
                    )

            # ── 3. L2 命中（旧键 v1，向后兼容）────────────────────────────
            if legacy_key is not None:
                legacy_path = self._parquet_path(factor_name, legacy_key)
                if self._is_l2_valid(legacy_path):
                    try:
                        panel = pd.read_parquet(legacy_path)
                        # 提升到 L1（使用新键索引，避免再次旧键查找）
                        self._l1[l1_key] = panel
                        self.last_hit_source = "legacy_key_hit"
                        self._stats["legacy_key_hit"] += 1
                        warnings.warn(
                            f"[CacheLayer] 因子 '{factor_name}' 命中旧版缓存键（v1）。"
                            f"建议在下次计算后删除旧缓存文件以使用 v2 键。",
                            stacklevel=2,
                        )
                        return panel
                    except Exception as e:
                        warnings.warn(
                            f"[CacheLayer] 读取旧键 Parquet 失败 ({legacy_path}): {e}，"
                            "将重新计算。",
                            stacklevel=2,
                        )

        self.last_hit_source = "recompute"
        self._stats["recompute"] += 1
        return None

    def put_panel(
        self,
        factor_name: str,
        key:         str,
        panel:       pd.DataFrame,
        calc_secs:   float = 0.0,
    ) -> None:
        """
        将因子面板写入缓存（L1 始终写入；L2 仅在满足阈值时写入）。

        Parameters
        ----------
        factor_name : 因子名称
        key         : make_key_v2() 生成的缓存键
        panel       : 因子面板（日期 × 股票）
        calc_secs   : 实际计算耗时（秒）；超过 min_calc_secs 才写 L2

        Notes
        -----
        L1 无论 calc_secs 如何都会写入。
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
        返回缓存状态摘要（用于调试/监控 / run_manifest）。

        Returns
        -------
        dict:
            l1_entries      : L1 内存中的面板数量
            l2_files        : L2 磁盘中的 Parquet 文件数量
            l2_total_mb     : L2 磁盘总大小（MB）
            source_mtime    : 数据源最新 mtime
            enabled_l2      : L2 是否启用
            stats           : 命中统计 {"new_key_hit", "legacy_key_hit", "recompute"}
            git_sha         : 当前 git SHA
            contract_version: SEMANTIC_CONTRACT_VERSION
        """
        l2_files = list(self.cache_dir.rglob("*.parquet")) if self.cache_dir.exists() else []
        total_bytes = sum(f.stat().st_size for f in l2_files if f.exists())
        return {
            "l1_entries":       len(self._l1),
            "l2_files":         len(l2_files),
            "l2_total_mb":      round(total_bytes / 1024 / 1024, 2),
            "source_mtime":     self._source_mtime,
            "enabled_l2":       self.enabled_l2,
            "stats":            dict(self._stats),
            "git_sha":          self.git_sha,
            "contract_version": SEMANTIC_CONTRACT_VERSION,
        }

    def reset_stats(self) -> None:
        """重置命中计数器（用于每次 pipeline.run() 开始时）。"""
        for k in self._stats:
            self._stats[k] = 0
