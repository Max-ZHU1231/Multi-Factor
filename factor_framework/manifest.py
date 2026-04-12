"""
factor_framework.manifest
==========================
RunManifest — 每次 pipeline.run() / mf batch 运行后自动生成的结构化记录。

D2 规范字段
-----------
  run_id               唯一运行 ID（uuid4 短形式，8 位）
  timestamp            ISO-8601 UTC 时间戳
  config_hash          有效配置的 SHA-256（前 16 位）
  git_sha              代码版本（git HEAD 短 SHA，8 位；不在 git 仓库则 "unknown"）
  data_snapshot_id     数据快照标识（数据目录路径 + 最新 CSV mtime 的 MD5，16 位）
  pipeline_version     "4.0"（硬编码，与 SEMANTIC_CONTRACT_VERSION 对齐）
  factors              本次运行的因子名称列表
  date_range           {"start": "YYYYMMDD", "end": "YYYYMMDD"}
  cache_stats          CacheLayer.cache_info() 输出（含命中统计）
  exit_status          "success" | "partial_failure" | "failure"
  failures             失败因子列表（exit_status != "success" 时非空）
  forward              预测期（天）
  n_groups             分层数
  run_duration_secs    运行耗时（秒，保留 2 位小数）

使用方式
--------
    from factor_framework.manifest import RunManifest

    mf = RunManifest.create(
        factors    = ["momentum_12_1", "value_pb"],
        cfg        = cfg_namespace,
        cache_info = cache.cache_info(),
        start_time = t0,
        failures   = [],
    )
    mf.save("artifacts/run_manifest.json")
    print(mf.to_dict())
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 与 cache.py 保持同源
PIPELINE_VERSION: str = "4.0"


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_short(obj: Any, length: int = 16) -> str:
    """
    对任意 JSON 可序列化对象取 SHA-256，返回前 length 个十六进制字符。

    若对象不可序列化，则对 repr(obj) 取哈希。
    """
    try:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        raw = repr(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _data_snapshot_id(stocks_dir: Path, length: int = 16) -> str:
    """
    计算数据快照标识：stocks_dir 路径 + 目录下最新 CSV 的 mtime → MD5。

    用于 manifest 中标记"使用了哪份数据"。
    若目录不存在，返回 "unknown"。
    """
    if not stocks_dir.exists():
        return "unknown"

    mtimes = []
    for f in stocks_dir.rglob("*.csv"):
        try:
            mtimes.append((str(f), f.stat().st_mtime))
        except OSError:
            pass

    if not mtimes:
        return "unknown"

    latest_mtime = max(m for _, m in mtimes)
    raw = f"{stocks_dir.resolve()}|{latest_mtime:.3f}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:length]


def _config_hash(cfg, length: int = 16) -> str:
    """
    计算配置对象的哈希（SHA-256 前 length 位）。

    优先级（从高到低）：
    1. ResearchConfig — 调用 to_stable_dict()（排序键、去瞬时字段，跨会话稳定）
    2. 有 .to_dict()  — 使用 .to_dict()（ConfigNamespace 等）
    3. dict / 其他   — 直接 dict() 或 repr()

    .. note::
        使用 ``to_stable_dict()`` 而非 ``to_dict()`` 是为了确保 symbols 等大列表
        不参与哈希（避免股票列表细微变化导致哈希漂移）。
    """
    try:
        # ResearchConfig: 使用稳定字典（排序+去瞬时字段）
        if hasattr(cfg, "to_stable_dict"):
            d = cfg.to_stable_dict()
        elif hasattr(cfg, "to_dict"):
            d = cfg.to_dict()
        else:
            d = dict(cfg)
    except Exception:
        d = repr(cfg)
    return _sha256_short(d, length)


# ═══════════════════════════════════════════════════════════════════════════════
#  RunManifest
# ═══════════════════════════════════════════════════════════════════════════════

class RunManifest:
    """
    单次 pipeline 运行的完整记录。

    通常通过 :meth:`create` 工厂方法构造，而不是直接 ``__init__``。
    """

    # 所有 D2 规范字段
    REQUIRED_FIELDS = frozenset({
        "run_id", "timestamp", "config_hash", "git_sha",
        "data_snapshot_id", "pipeline_version",
        "factors", "date_range", "cache_stats",
        "exit_status", "failures", "forward", "n_groups",
        "run_duration_secs",
    })

    def __init__(self, data: Dict[str, Any]) -> None:
        missing = self.REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"RunManifest missing required fields: {sorted(missing)}")
        self._data = data

    # ── 工厂方法 ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        factors:      List[str],
        cfg,                          # ConfigNamespace 或 dict
        cache_info:   Dict[str, Any],
        start_time:   float,          # time.perf_counter() 或 time.time()
        failures:     Optional[List[str]] = None,
        stocks_dir:   Optional[Path] = None,
        git_sha:      Optional[str] = None,
    ) -> "RunManifest":
        """
        从运行上下文构造 RunManifest。

        Parameters
        ----------
        factors    : 本次运行的因子列表
        cfg        : 有效配置（ConfigNamespace 或 dict）
        cache_info : CacheLayer.cache_info() 的输出
        start_time : 运行开始时的时间戳（time.perf_counter()）
        failures   : 失败因子列表（可选，默认 []）
        stocks_dir : 数据目录（用于 data_snapshot_id；None = 从 cfg 推断）
        git_sha    : 覆盖 git SHA（None = 从 cache_info 或 git 命令获取）
        """
        failures = failures or []

        # exit_status 推断
        if not failures:
            exit_status = "success"
        elif len(failures) < len(factors):
            exit_status = "partial_failure"
        else:
            exit_status = "failure"

        # 推断 stocks_dir
        if stocks_dir is None:
            try:
                _sd = cfg.data.stocks_dir if hasattr(cfg, "data") else cfg.get("data", {}).get("stocks_dir", "Stocks/")
                stocks_dir = Path(_sd)
            except Exception:
                stocks_dir = Path("Stocks/")

        # git_sha
        if git_sha is None:
            git_sha = cache_info.get("git_sha", "unknown")

        # 提取 start / end / forward / n_groups
        def _get(obj, *keys, default=None):
            for k in keys:
                try:
                    obj = getattr(obj, k)
                except Exception:
                    try:
                        obj = obj[k]
                    except Exception:
                        return default
            return obj

        start_date = _get(cfg, "backtest", "start", default="")
        end_date   = _get(cfg, "backtest", "end",   default="")
        forward    = _get(cfg, "backtest", "forward", default=0)
        n_groups   = _get(cfg, "backtest", "n_groups", default=5)

        data: Dict[str, Any] = {
            "run_id":           uuid.uuid4().hex[:8],
            "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
            "config_hash":      _config_hash(cfg),
            "git_sha":          str(git_sha),
            "data_snapshot_id": _data_snapshot_id(stocks_dir),
            "pipeline_version": PIPELINE_VERSION,
            "factors":          list(factors),
            "date_range":       {"start": str(start_date), "end": str(end_date)},
            "cache_stats":      cache_info,
            "exit_status":      exit_status,
            "failures":         list(failures),
            "forward":          int(forward) if forward else 0,
            "n_groups":         int(n_groups) if n_groups else 5,
            "run_duration_secs": round(time.perf_counter() - start_time, 2),
        }
        return cls(data)

    # ── 序列化 ────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """返回完整字段字典（浅拷贝）。"""
        return dict(self._data)

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self._data, indent=indent, ensure_ascii=False, default=str)

    def save(self, path: str | Path) -> Path:
        """
        将 manifest 写入 JSON 文件。

        若目录不存在，自动创建。返回实际写入的 Path。

        Parameters
        ----------
        path : 目标文件路径（如 "artifacts/run_manifest.json"）
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """
        从 JSON 文件加载 RunManifest。

        Parameters
        ----------
        path : manifest JSON 文件路径
        """
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data)

    # ── 快捷访问 ─────────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return (
            f"RunManifest(run_id={self._data['run_id']!r}, "
            f"factors={self._data['factors']}, "
            f"exit_status={self._data['exit_status']!r})"
        )

    @property
    def run_id(self) -> str:
        return self._data["run_id"]

    @property
    def exit_status(self) -> str:
        return self._data["exit_status"]

    @property
    def factors(self) -> List[str]:
        return list(self._data["factors"])

    @property
    def config_hash(self) -> str:
        return self._data["config_hash"]

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return dict(self._data["cache_stats"])

    def print_summary(self) -> None:
        """终端打印 manifest 摘要（用于 mf single/batch 运行结束时）。"""
        d = self._data
        cs = d.get("cache_stats", {})
        stats = cs.get("stats", {})
        print("\n" + "─" * 56)
        print(f"  run_id       : {d['run_id']}")
        print(f"  timestamp    : {d['timestamp']}")
        print(f"  git_sha      : {d['git_sha']}")
        print(f"  config_hash  : {d['config_hash']}")
        print(f"  snapshot_id  : {d['data_snapshot_id']}")
        print(f"  factors      : {', '.join(d['factors'])}")
        print(f"  date_range   : {d['date_range']['start']} ~ {d['date_range']['end']}")
        print(f"  exit_status  : {d['exit_status']}")
        if d["failures"]:
            print(f"  failed_factors: {', '.join(d['failures'])}")
        print(f"  duration     : {d['run_duration_secs']}s")
        if stats:
            print(f"  cache_hit(v2): {stats.get('new_key_hit', 0)}")
            print(f"  cache_hit(old): {stats.get('legacy_key_hit', 0)}")
            print(f"  recompute    : {stats.get('recompute', 0)}")
        print("─" * 56)
