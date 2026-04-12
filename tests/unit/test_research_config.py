"""
tests/unit/test_research_config.py
====================================
Phase C — ResearchConfig 最小测试套件

覆盖范围
--------
1. 默认构造                — 所有必选/可选字段有正确默认值
2. from_kwargs 与直接构造结果一致（run() 旧入口兼容性）
3. from_dict 基本用途      — 从字典恢复正确字段
4. schema_version 缺失    — 自动注入 CURRENT_SCHEMA_VERSION + 触发 UserWarning
5. schema_version 过旧    — 触发 UserWarning，版本升级到当前
6. schema_version 过新    — 触发 UserWarning，降级到当前（向前兼容）
7. 非法 schema_version    — 抛出 ValueError
8. to_stable_dict 键排序  — 所有键按字母序排列
9. to_stable_dict 去瞬时  — symbols / ic_forward_list 不出现在 stable_dict
10. config_hash 稳定性    — 同一 ResearchConfig 两次哈希相同
11. forward 变化 → hash 变化
12. n_groups 变化 → hash 变化
13. symbols 不影响 hash   — 不同 symbols 产生相同 config_hash
14. kwargs 与 config 路由等价 — 同参数走两条路 manifest config_hash 一致
15. validate() 通过/失败   — 合法配置通过，非法 factor_name/forward/direction 抛出
16. upgrade_config no-op  — 有效 v1.0 dict 原样返回（字段不变）
"""
from __future__ import annotations

import warnings

import pytest

from factor_framework.research_config import (
    ResearchConfig,
    CURRENT_SCHEMA_VERSION,
    upgrade_config,
)
from factor_framework.manifest import _config_hash


# ═════════════════════════════════════════════════════════════════════════════
#  辅助工具
# ═════════════════════════════════════════════════════════════════════════════

def _rc(**overrides) -> ResearchConfig:
    """构造最小合法 ResearchConfig（factor_name 为必须）。"""
    defaults = dict(factor_name="momentum_12_1", start="20220101", end="20231231")
    defaults.update(overrides)
    return ResearchConfig(**defaults)


# ═════════════════════════════════════════════════════════════════════════════
#  1. 默认值
# ═════════════════════════════════════════════════════════════════════════════

def test_default_schema_version():
    rc = ResearchConfig(factor_name="f")
    assert rc.schema_version == CURRENT_SCHEMA_VERSION


def test_default_field_values():
    rc = ResearchConfig(factor_name="f")
    assert rc.forward == 21
    assert rc.n_groups == 5
    assert rc.direction == 1
    assert rc.standardize == "rank"
    assert rc.neutralize is False
    assert rc.winsorize is True
    assert rc.ic_method == "rank"
    assert rc.periods_per_year == 252
    assert rc.rf == 0.0
    assert rc.cost_per_side == 0.002
    assert rc.resample_monthly is True
    assert rc.symbols is None


# ═════════════════════════════════════════════════════════════════════════════
#  2. from_kwargs 与直接构造一致
# ═════════════════════════════════════════════════════════════════════════════

def test_from_kwargs_equals_direct():
    direct = ResearchConfig(
        factor_name="momentum_12_1",
        start="20220101", end="20231231",
        forward=10, n_groups=5,
    )
    via_kwargs = ResearchConfig.from_kwargs(
        factor_name="momentum_12_1",
        start="20220101", end="20231231",
        forward=10, n_groups=5,
    )
    assert direct.to_stable_dict() == via_kwargs.to_stable_dict()


def test_from_kwargs_ignores_unknown():
    """额外的 kwargs 应被静默忽略，不抛异常。"""
    rc = ResearchConfig.from_kwargs(
        factor_name="f", start=None, end=None,
        some_unknown_param="ignored",
    )
    assert rc.factor_name == "f"


# ═════════════════════════════════════════════════════════════════════════════
#  3. from_dict
# ═════════════════════════════════════════════════════════════════════════════

def test_from_dict_basic():
    d = {
        "factor_name": "value_pb",
        "start": "20200101",
        "end": "20231231",
        "forward": 5,
        "schema_version": CURRENT_SCHEMA_VERSION,
    }
    rc = ResearchConfig.from_dict(d)
    assert rc.factor_name == "value_pb"
    assert rc.forward == 5
    assert rc.start == "20200101"


def test_from_dict_ic_forward_list_becomes_tuple():
    """ic_forward_list 从 list 转为 tuple。"""
    d = {"factor_name": "f", "schema_version": CURRENT_SCHEMA_VERSION,
         "ic_forward_list": [1, 5, 21]}
    rc = ResearchConfig.from_dict(d)
    assert isinstance(rc.ic_forward_list, tuple)
    assert rc.ic_forward_list == (1, 5, 21)


def test_from_dict_ignores_unknown_fields():
    """from_dict 不应因未知字段抛出 TypeError。"""
    d = {
        "factor_name": "f",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "future_field_xyz": "ignored",
    }
    rc = ResearchConfig.from_dict(d)
    assert rc.factor_name == "f"


# ═════════════════════════════════════════════════════════════════════════════
#  4. schema_version 缺失 → 警告 + 自动注入
# ═════════════════════════════════════════════════════════════════════════════

def test_missing_schema_version_warns_and_injects():
    d = {"factor_name": "f", "forward": 21}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rc = ResearchConfig.from_dict(d)
    assert any("schema_version" in str(x.message) for x in w), \
        "缺少 schema_version 时应触发 UserWarning"
    assert rc.schema_version == CURRENT_SCHEMA_VERSION


# ═════════════════════════════════════════════════════════════════════════════
#  5. schema_version 过旧 → 警告 + 升级
# ═════════════════════════════════════════════════════════════════════════════

def test_old_schema_version_warns_and_upgrades():
    d = {"factor_name": "f", "schema_version": "0.1"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        upgraded = upgrade_config(d)
    assert any("below minimum supported version" in str(x.message) or
               "0.1" in str(x.message) for x in w), \
        "过旧版本应触发 UserWarning"
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION


# ═════════════════════════════════════════════════════════════════════════════
#  6. schema_version 过新 → 警告 + 降级到当前
# ═════════════════════════════════════════════════════════════════════════════

def test_future_schema_version_warns_and_downgrades():
    d = {"factor_name": "f", "schema_version": "99.0"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        upgraded = upgrade_config(d)
    assert any("newer than current code version" in str(x.message) or
               "99.0" in str(x.message) for x in w), \
        "未来版本应触发 UserWarning"
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION


# ═════════════════════════════════════════════════════════════════════════════
#  7. 非法 schema_version 格式 → ValueError
# ═════════════════════════════════════════════════════════════════════════════

def test_invalid_schema_version_format_raises():
    d = {"factor_name": "f", "schema_version": "not-a-version"}
    with pytest.raises(ValueError, match="Invalid schema_version format"):
        upgrade_config(d)


# ═════════════════════════════════════════════════════════════════════════════
#  8. to_stable_dict 键排序
# ═════════════════════════════════════════════════════════════════════════════

def test_stable_dict_keys_are_sorted():
    rc = _rc()
    d = rc.to_stable_dict()
    keys = list(d.keys())
    assert keys == sorted(keys), f"to_stable_dict() 的键应按字母序排列，实际: {keys}"


# ═════════════════════════════════════════════════════════════════════════════
#  9. to_stable_dict 去瞬时字段
# ═════════════════════════════════════════════════════════════════════════════

def test_stable_dict_excludes_transient_fields():
    rc = _rc(symbols=["000001.SZ", "000002.SZ"])
    d = rc.to_stable_dict()
    assert "symbols" not in d, "symbols 不应出现在 stable_dict（瞬时字段）"
    assert "ic_forward_list" not in d, "ic_forward_list 不应出现在 stable_dict（瞬时字段）"


def test_stable_dict_includes_schema_version():
    rc = _rc()
    d = rc.to_stable_dict()
    assert "schema_version" in d, "schema_version 应出现在 stable_dict（用于跨版本隔离）"


# ═════════════════════════════════════════════════════════════════════════════
#  10. config_hash 稳定性
# ═════════════════════════════════════════════════════════════════════════════

def test_config_hash_stability():
    rc = _rc(forward=21)
    h1 = _config_hash(rc)
    h2 = _config_hash(rc)
    assert h1 == h2, "同一 ResearchConfig 的 config_hash 应幂等"


def test_config_hash_uses_stable_dict(monkeypatch):
    """_config_hash(ResearchConfig) 应走 to_stable_dict() 分支。"""
    rc = _rc()
    called = []

    original = rc.to_stable_dict
    def patched():
        called.append(True)
        return original()
    monkeypatch.setattr(rc, "to_stable_dict", patched)

    _config_hash(rc)
    assert called, "_config_hash 应调用 to_stable_dict() 而非 to_dict()"


# ═════════════════════════════════════════════════════════════════════════════
#  11. forward 变化 → hash 变化
# ═════════════════════════════════════════════════════════════════════════════

def test_config_hash_changes_with_forward():
    h5  = _config_hash(_rc(forward=5))
    h10 = _config_hash(_rc(forward=10))
    assert h5 != h10, "不同 forward 应产生不同 config_hash"


# ═════════════════════════════════════════════════════════════════════════════
#  12. n_groups 变化 → hash 变化
# ═════════════════════════════════════════════════════════════════════════════

def test_config_hash_changes_with_n_groups():
    h5  = _config_hash(_rc(n_groups=5))
    h10 = _config_hash(_rc(n_groups=10))
    assert h5 != h10, "不同 n_groups 应产生不同 config_hash"


# ═════════════════════════════════════════════════════════════════════════════
#  13. symbols 不影响 config_hash
# ═════════════════════════════════════════════════════════════════════════════

def test_symbols_does_not_affect_config_hash():
    """symbols 是瞬时字段，不同 symbols 列表不应改变 config_hash。"""
    rc_all    = _rc(symbols=None)
    rc_subset = _rc(symbols=["000001.SZ", "000002.SZ"])
    assert _config_hash(rc_all) == _config_hash(rc_subset), \
        "symbols 不同时 config_hash 应相同（symbols 不参与哈希）"


# ═════════════════════════════════════════════════════════════════════════════
#  14. kwargs 路由与 config 路由 manifest hash 一致
# ═════════════════════════════════════════════════════════════════════════════

def test_kwargs_and_config_produce_same_hash():
    """
    from_kwargs（模拟旧 CLI 调用）与直接构造 ResearchConfig 的 config_hash 应一致。
    """
    params = dict(
        factor_name="momentum_12_1",
        start="20220101", end="20231231",
        forward=21, n_groups=5,
        standardize="rank", winsorize=True, neutralize=False,
        ic_method="rank", periods_per_year=252,
        rf=0.0, cost_per_side=0.002, resample_monthly=True,
    )
    rc_direct  = ResearchConfig(**params)
    rc_kwargs  = ResearchConfig.from_kwargs(**params)
    assert _config_hash(rc_direct) == _config_hash(rc_kwargs), \
        "两条路由产生的 config_hash 应完全一致"


# ═════════════════════════════════════════════════════════════════════════════
#  15. validate()
# ═════════════════════════════════════════════════════════════════════════════

def test_validate_passes_for_valid_config():
    """合法配置 validate() 应返回 self（支持链式调用）。"""
    rc = _rc()
    assert rc.validate() is rc


def test_validate_fails_empty_factor_name():
    rc = ResearchConfig(factor_name="")
    with pytest.raises(ValueError, match="factor_name"):
        rc.validate()


def test_validate_fails_forward_zero():
    rc = _rc(forward=0)
    with pytest.raises(ValueError, match="forward"):
        rc.validate()


def test_validate_fails_forward_negative():
    rc = _rc(forward=-1)
    with pytest.raises(ValueError, match="forward"):
        rc.validate()


def test_validate_fails_n_groups_one():
    rc = _rc(n_groups=1)
    with pytest.raises(ValueError, match="n_groups"):
        rc.validate()


def test_validate_fails_bad_direction():
    rc = _rc(direction=0)
    with pytest.raises(ValueError, match="direction"):
        rc.validate()


def test_validate_fails_bad_standardize():
    rc = _rc(standardize="invalid")
    with pytest.raises(ValueError, match="standardize"):
        rc.validate()


def test_validate_fails_bad_ic_method():
    rc = _rc(ic_method="bad")
    with pytest.raises(ValueError, match="ic_method"):
        rc.validate()


# ═════════════════════════════════════════════════════════════════════════════
#  16. upgrade_config no-op for valid v1.0
# ═════════════════════════════════════════════════════════════════════════════

def test_upgrade_config_noop_for_current_version():
    """有效的 v1.0 配置不应触发 warning，字段应原样保留。"""
    d = {
        "factor_name": "momentum",
        "forward": 21,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "custom_extra": "preserved",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = upgrade_config(d)
    assert not w, f"当前版本配置不应触发 warning，实际: {[str(x.message) for x in w]}"
    assert result["forward"] == 21
    assert result["custom_extra"] == "preserved"
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION


def test_upgrade_config_does_not_mutate_input():
    """upgrade_config 应返回深拷贝，不修改原始 dict。"""
    original = {"factor_name": "f", "schema_version": "0.1"}
    _ = upgrade_config(original)
    assert original["schema_version"] == "0.1", "原始 dict 不应被修改"


# ═════════════════════════════════════════════════════════════════════════════
#  repr / to_dict
# ═════════════════════════════════════════════════════════════════════════════

def test_repr_contains_factor_name():
    rc = _rc(factor_name="value_pb")
    assert "value_pb" in repr(rc)


def test_to_dict_contains_all_fields():
    rc = _rc()
    d = rc.to_dict()
    expected_keys = {
        "factor_name", "start", "end", "forward", "n_groups", "direction",
        "standardize", "neutralize", "winsorize", "ic_method",
        "periods_per_year", "rf", "cost_per_side", "resample_monthly",
        "schema_version", "symbols", "ic_forward_list",
    }
    missing = expected_keys - set(d.keys())
    assert not missing, f"to_dict() 缺少字段: {sorted(missing)}"


def test_to_dict_ic_forward_list_is_list():
    """to_dict() 中 ic_forward_list 应序列化为 list（JSON 友好）。"""
    rc = _rc()
    d = rc.to_dict()
    assert isinstance(d["ic_forward_list"], list)
