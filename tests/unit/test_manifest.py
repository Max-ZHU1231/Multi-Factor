"""
tests/unit/test_manifest.py
============================
Phase D · D3 — 最小测试套件

覆盖范围
--------
1. manifest 字段完整性     — create() 产出包含所有 REQUIRED_FIELDS
2. config hash 稳定性      — 同一 cfg 两次 _config_hash() 结果相同
3. config 变更 → hash 变化  — 修改 forward 值后 config_hash 不同
4. 旧缓存向后兼容           — get_panel(v2_key, legacy_key=v1_key) 读取 v1 Parquet
                              后 last_hit_source == "legacy_key_hit"
5. v2 键优先               — v2 Parquet 存在时命中 v2，last_hit_source == "new_key_hit"
6. reset_stats             — 重置后所有计数器归零
7. manifest save/load 往返  — JSON 序列化后重新加载字段完全一致
8. exit_status 推断         — failures=[] → "success"；部分失败 → "partial_failure"；
                              全失败 → "failure"
"""
from __future__ import annotations

import json
import time
import types
from pathlib import Path

import pandas as pd
import pytest

# ── 被测模块 ──────────────────────────────────────────────────────────────────
from factor_framework.manifest import (
    RunManifest,
    _config_hash,
    _sha256_short,
    PIPELINE_VERSION,
)
from factor_framework.engine.cache import CacheLayer, SEMANTIC_CONTRACT_VERSION


# ═════════════════════════════════════════════════════════════════════════════
#  辅助工具
# ═════════════════════════════════════════════════════════════════════════════

def _make_cfg(forward: int = 5, n_groups: int = 5, start: str = "20220101", end: str = "20231231"):
    """构造轻量级配置命名空间，模拟 ConfigNamespace 结构。"""
    backtest = types.SimpleNamespace(
        start=start, end=end, forward=forward, n_groups=n_groups,
        periods_per_year=252, rf=0.0, cost_per_side=0.001, resample_monthly=False,
    )
    data = types.SimpleNamespace(stocks_dir="Stocks/")
    preprocess = types.SimpleNamespace(standardize=True, winsorize=True, neutralize=False)
    ic = types.SimpleNamespace(method="rank")
    output = types.SimpleNamespace(batch="output/batch/")
    cfg = types.SimpleNamespace(backtest=backtest, data=data,
                                 preprocess=preprocess, ic=ic, output=output)
    # 模拟 ConfigNamespace.to_dict()
    def to_dict(self=cfg):
        return {
            "backtest": {
                "start": cfg.backtest.start, "end": cfg.backtest.end,
                "forward": cfg.backtest.forward, "n_groups": cfg.backtest.n_groups,
                "periods_per_year": cfg.backtest.periods_per_year,
                "rf": cfg.backtest.rf,
            },
            "preprocess": {
                "standardize": cfg.preprocess.standardize,
                "winsorize": cfg.preprocess.winsorize,
                "neutralize": cfg.preprocess.neutralize,
            },
            "ic": {"method": cfg.ic.method},
        }
    cfg.to_dict = to_dict
    return cfg


def _make_cache(tmp_path: Path, git_sha: str = "testsha1") -> CacheLayer:
    """在临时目录中创建禁用 L2 自动写入门槛的 CacheLayer（min_calc_secs=0）。"""
    return CacheLayer(
        cache_dir=str(tmp_path / "cache"),
        stocks_dir=str(tmp_path / "stocks"),
        enabled_l2=True,
        min_calc_secs=0.0,
        transform_config_hash="testhash",
        git_sha=git_sha,
    )


def _dummy_panel() -> pd.DataFrame:
    """返回一个简单的 2×3 因子面板（日期 × 股票）。"""
    idx = pd.to_datetime(["2022-01-04", "2022-01-05"])
    cols = ["000001.SZ", "000002.SZ", "000004.SZ"]
    return pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=idx, columns=cols)


# ═════════════════════════════════════════════════════════════════════════════
#  Test 1 — manifest 字段完整性
# ═════════════════════════════════════════════════════════════════════════════

def test_manifest_has_all_required_fields(tmp_path):
    """RunManifest.create() 必须包含所有 REQUIRED_FIELDS。"""
    cfg = _make_cfg()
    cache = _make_cache(tmp_path)
    cache_info = cache.cache_info()
    t0 = time.perf_counter()

    mf = RunManifest.create(
        factors=["momentum_12_1", "value_pb"],
        cfg=cfg,
        cache_info=cache_info,
        start_time=t0,
        failures=[],
        stocks_dir=tmp_path / "stocks",
        git_sha="abc12345",
    )

    d = mf.to_dict()
    missing = RunManifest.REQUIRED_FIELDS - set(d.keys())
    assert not missing, f"manifest 缺少字段: {sorted(missing)}"


def test_manifest_field_types(tmp_path):
    """验证核心字段的类型。"""
    cfg = _make_cfg()
    t0 = time.perf_counter()
    mf = RunManifest.create(
        factors=["momentum"],
        cfg=cfg,
        cache_info={},
        start_time=t0,
    )
    d = mf.to_dict()

    assert isinstance(d["run_id"], str) and len(d["run_id"]) == 8
    assert isinstance(d["timestamp"], str) and "T" in d["timestamp"]
    assert isinstance(d["factors"], list)
    assert isinstance(d["date_range"], dict)
    assert "start" in d["date_range"] and "end" in d["date_range"]
    assert isinstance(d["run_duration_secs"], float)
    assert isinstance(d["cache_stats"], dict)
    assert d["pipeline_version"] == PIPELINE_VERSION


# ═════════════════════════════════════════════════════════════════════════════
#  Test 2 — config hash 稳定性
# ═════════════════════════════════════════════════════════════════════════════

def test_config_hash_stability():
    """同一配置对象多次调用 _config_hash() 应得到相同结果。"""
    cfg = _make_cfg(forward=5)
    h1 = _config_hash(cfg)
    h2 = _config_hash(cfg)
    assert h1 == h2, "同一 cfg 的 config_hash 应稳定（幂等性）"


def test_config_hash_deterministic_dict():
    """纯 dict 配置也应产生稳定 hash。"""
    d = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    assert _config_hash(d) == _config_hash(d)


# ═════════════════════════════════════════════════════════════════════════════
#  Test 3 — config 变更 → hash 变化
# ═════════════════════════════════════════════════════════════════════════════

def test_config_hash_changes_with_forward():
    """修改 forward 参数后，config_hash 必须不同。"""
    cfg_5  = _make_cfg(forward=5)
    cfg_10 = _make_cfg(forward=10)
    h5  = _config_hash(cfg_5)
    h10 = _config_hash(cfg_10)
    assert h5 != h10, "不同 forward 应产生不同 config_hash"


def test_config_hash_changes_with_n_groups():
    """修改 n_groups 参数后，config_hash 必须不同。"""
    h5  = _config_hash(_make_cfg(n_groups=5))
    h10 = _config_hash(_make_cfg(n_groups=10))
    assert h5 != h10, "不同 n_groups 应产生不同 config_hash"


# ═════════════════════════════════════════════════════════════════════════════
#  Test 4 — 旧缓存向后兼容（legacy_key_hit）
# ═════════════════════════════════════════════════════════════════════════════

def test_legacy_cache_compat(tmp_path):
    """
    将面板以 v1 键写入磁盘后，用 get_panel(v2_key, legacy_key=v1_key) 查询：
    - 应命中并返回数据
    - last_hit_source 应为 "legacy_key_hit"
    """
    cache = _make_cache(tmp_path)
    panel = _dummy_panel()
    symbols = ["000001.SZ", "000002.SZ", "000004.SZ"]

    v1_key = CacheLayer.make_key("momentum", "20220101", "20231231", symbols)
    v2_key = cache.make_key_v2("momentum", "20220101", "20231231", symbols)

    assert v1_key != v2_key, "v1 键和 v2 键应不同"

    # 以 v1 键写入 Parquet（模拟旧缓存）
    v1_path = cache._parquet_path("momentum", v1_key)
    panel.to_parquet(v1_path)
    # 让 Parquet mtime > source mtime（source 目录不存在，mtime=0）
    # 不需要额外操作，stocks_dir 不存在时 source_mtime=0

    result = cache.get_panel("momentum", v2_key, legacy_key=v1_key)

    assert result is not None, "应命中 v1 旧缓存"
    assert cache.last_hit_source == "legacy_key_hit"
    assert cache._stats["legacy_key_hit"] == 1
    assert cache._stats["new_key_hit"] == 0
    pd.testing.assert_frame_equal(result, panel)


def test_legacy_cache_miss_without_legacy_key(tmp_path):
    """不传 legacy_key 时，即使 v1 缓存存在，也应返回 None。"""
    cache = _make_cache(tmp_path)
    panel = _dummy_panel()
    symbols = ["000001.SZ", "000002.SZ"]

    v1_key = CacheLayer.make_key("momentum", "20220101", "20231231", symbols)
    v1_path = cache._parquet_path("momentum", v1_key)
    panel.to_parquet(v1_path)

    v2_key = cache.make_key_v2("momentum", "20220101", "20231231", symbols)
    result = cache.get_panel("momentum", v2_key)  # 不传 legacy_key

    assert result is None
    assert cache.last_hit_source == "recompute"


# ═════════════════════════════════════════════════════════════════════════════
#  Test 5 — v2 键优先（new_key_hit）
# ═════════════════════════════════════════════════════════════════════════════

def test_v2_key_hit_takes_priority(tmp_path):
    """
    v2 和 v1 缓存同时存在时，v2 应优先命中，last_hit_source == "new_key_hit"。
    """
    cache = _make_cache(tmp_path)
    panel_v1 = _dummy_panel()
    panel_v2 = _dummy_panel() * 2  # 内容不同，方便区分
    symbols = ["000001.SZ", "000002.SZ", "000004.SZ"]

    v1_key = CacheLayer.make_key("momentum", "20220101", "20231231", symbols)
    v2_key = cache.make_key_v2("momentum", "20220101", "20231231", symbols)

    # 写入 v1 Parquet
    v1_path = cache._parquet_path("momentum", v1_key)
    panel_v1.to_parquet(v1_path)

    # 写入 v2 Parquet
    v2_path = cache._parquet_path("momentum", v2_key)
    panel_v2.to_parquet(v2_path)

    result = cache.get_panel("momentum", v2_key, legacy_key=v1_key)

    assert result is not None
    assert cache.last_hit_source == "new_key_hit"
    assert cache._stats["new_key_hit"] == 1
    assert cache._stats["legacy_key_hit"] == 0
    pd.testing.assert_frame_equal(result, panel_v2)


def test_v2_l1_hit(tmp_path):
    """L1 内存中存在 v2 键时，应直接命中 L1（new_key_hit）。"""
    cache = _make_cache(tmp_path)
    panel = _dummy_panel()
    symbols = ["000001.SZ"]
    v2_key = cache.make_key_v2("mom", "20220101", "20231231", symbols)

    # 先 put_panel 写入 L1
    cache.put_panel("mom", v2_key, panel)
    # 清除 _stats 后重新查询（确认来自 L1）
    cache.reset_stats()

    result = cache.get_panel("mom", v2_key)

    assert result is not None
    assert cache.last_hit_source == "new_key_hit"
    assert cache._stats["new_key_hit"] == 1


# ═════════════════════════════════════════════════════════════════════════════
#  Test 6 — reset_stats
# ═════════════════════════════════════════════════════════════════════════════

def test_reset_stats(tmp_path):
    """reset_stats() 后所有计数器应归零。"""
    cache = _make_cache(tmp_path)
    panel = _dummy_panel()
    symbols = ["000001.SZ"]
    v2_key = cache.make_key_v2("mom", "20220101", "20231231", symbols)

    # 先产生一些统计
    cache.put_panel("mom", v2_key, panel)
    cache.get_panel("mom", v2_key)           # new_key_hit +1
    cache.get_panel("mom", "nonexistent")    # recompute +1
    assert cache._stats["new_key_hit"] >= 1

    cache.reset_stats()

    assert cache._stats["new_key_hit"] == 0
    assert cache._stats["legacy_key_hit"] == 0
    assert cache._stats["recompute"] == 0


def test_reset_stats_does_not_clear_l1(tmp_path):
    """reset_stats() 只重置统计计数器，不应影响 L1 缓存内容。"""
    cache = _make_cache(tmp_path)
    panel = _dummy_panel()
    symbols = ["000001.SZ"]
    v2_key = cache.make_key_v2("mom", "20220101", "20231231", symbols)

    cache.put_panel("mom", v2_key, panel)
    n_l1_before = len(cache._l1)

    cache.reset_stats()

    assert len(cache._l1) == n_l1_before, "reset_stats 不应清除 L1"


# ═════════════════════════════════════════════════════════════════════════════
#  Test 7 — manifest save / load 往返
# ═════════════════════════════════════════════════════════════════════════════

def test_manifest_save_load_roundtrip(tmp_path):
    """RunManifest 保存为 JSON 后再加载，所有字段应完全一致。"""
    cfg = _make_cfg()
    t0 = time.perf_counter()
    mf_orig = RunManifest.create(
        factors=["momentum_12_1"],
        cfg=cfg,
        cache_info={"stats": {"new_key_hit": 3, "legacy_key_hit": 1, "recompute": 2}},
        start_time=t0,
        failures=[],
        git_sha="deadbeef",
    )

    out_path = tmp_path / "run_manifest.json"
    mf_orig.save(out_path)
    assert out_path.exists()

    # 验证原始 JSON 可解析
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(raw.keys()) >= RunManifest.REQUIRED_FIELDS

    # 从磁盘重新加载
    mf_loaded = RunManifest.load(out_path)
    d_orig   = mf_orig.to_dict()
    d_loaded = mf_loaded.to_dict()

    assert d_orig == d_loaded, "save/load 往返后字段应完全一致"


def test_manifest_save_creates_parent_dirs(tmp_path):
    """save() 应自动创建不存在的父目录。"""
    cfg = _make_cfg()
    t0 = time.perf_counter()
    mf = RunManifest.create(factors=["f"], cfg=cfg, cache_info={}, start_time=t0)

    deep_path = tmp_path / "nested" / "deeply" / "run_manifest.json"
    mf.save(deep_path)
    assert deep_path.exists()


def test_manifest_load_invalid_raises():
    """从缺少必填字段的 JSON 加载 RunManifest 应抛出 ValueError。"""
    import tempfile, os
    bad_data = {"run_id": "abc", "timestamp": "2024-01-01T00:00:00+00:00"}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(bad_data, f)
        fname = f.name
    try:
        with pytest.raises(ValueError, match="缺少必填字段"):
            RunManifest.load(fname)
    finally:
        os.unlink(fname)


# ═════════════════════════════════════════════════════════════════════════════
#  Test 8 — exit_status 推断
# ═════════════════════════════════════════════════════════════════════════════

def _create_mf(factors, failures):
    cfg = _make_cfg()
    t0  = time.perf_counter()
    return RunManifest.create(
        factors=factors, cfg=cfg, cache_info={},
        start_time=t0, failures=failures,
    )


def test_exit_status_success():
    """无失败 → exit_status == 'success'。"""
    mf = _create_mf(["f1", "f2"], [])
    assert mf.exit_status == "success"
    assert mf["exit_status"] == "success"


def test_exit_status_partial_failure():
    """部分失败 → exit_status == 'partial_failure'。"""
    mf = _create_mf(["f1", "f2", "f3"], ["f2"])
    assert mf.exit_status == "partial_failure"


def test_exit_status_full_failure():
    """全部失败 → exit_status == 'failure'。"""
    mf = _create_mf(["f1", "f2"], ["f1", "f2"])
    assert mf.exit_status == "failure"


def test_exit_status_single_factor_success():
    """单因子无失败 → exit_status == 'success'。"""
    mf = _create_mf(["momentum"], [])
    assert mf.exit_status == "success"


def test_exit_status_single_factor_failure():
    """单因子失败 → exit_status == 'failure'（非 partial）。"""
    mf = _create_mf(["momentum"], ["momentum"])
    assert mf.exit_status == "failure"


# ═════════════════════════════════════════════════════════════════════════════
#  Test — cache_info() 返回格式
# ═════════════════════════════════════════════════════════════════════════════

def test_cache_info_contains_stats_and_version(tmp_path):
    """cache_info() 应包含 stats、git_sha、contract_version。"""
    cache = _make_cache(tmp_path, git_sha="abc12345")
    info = cache.cache_info()

    assert "stats" in info
    assert "git_sha" in info
    assert "contract_version" in info
    assert info["git_sha"] == "abc12345"
    assert info["contract_version"] == SEMANTIC_CONTRACT_VERSION
    assert set(info["stats"].keys()) == {"new_key_hit", "legacy_key_hit", "recompute"}


# ═════════════════════════════════════════════════════════════════════════════
#  Test — _sha256_short 辅助函数
# ═════════════════════════════════════════════════════════════════════════════

def test_sha256_short_length():
    """_sha256_short() 应返回指定长度的十六进制字符串。"""
    for length in [8, 16, 32]:
        result = _sha256_short({"key": "value"}, length=length)
        assert len(result) == length, f"期望长度 {length}，实际 {len(result)}"
        assert all(c in "0123456789abcdef" for c in result)


def test_sha256_short_deterministic():
    """相同输入应产生相同哈希。"""
    obj = {"a": [1, 2, 3], "b": {"nested": True}}
    assert _sha256_short(obj) == _sha256_short(obj)


def test_sha256_short_different_inputs():
    """不同输入应产生不同哈希（以高概率）。"""
    h1 = _sha256_short({"forward": 5})
    h2 = _sha256_short({"forward": 10})
    assert h1 != h2
