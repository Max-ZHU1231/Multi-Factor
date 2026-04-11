"""
tests/unit/test_phase_e1.py
============================
Phase E1 — FactorMeta 扩展 + FactorRegistry.audit() 最小测试套件

覆盖范围
--------
1.  FactorMeta E1 字段默认值         — 空元组/空字符串/None/ACTIVE
2.  FactorMeta 带 E1 字段构造        — 字段正确存储
3.  missing_e1_fields — 全缺失        — 返回 4 个字段名
4.  missing_e1_fields — 全填写        — 返回空列表
5.  missing_e1_fields — 部分缺失      — 只报缺失的字段
6.  FactorStatus 枚举值              — ACTIVE/EXPERIMENTAL/DEPRECATED
7.  FactorMeta status 默认为 ACTIVE
8.  FactorMeta status=DEPRECATED     — is_active == False
9.  FactorMeta status 非法类型       — 抛出 TypeError
10. registry.register() warning-only — 缺失 E1 字段仍成功注册，但触发 UserWarning
11. registry.register() 完整 E1      — 注册成功，无 E1 警告
12. audit() 返回 AuditReport          — 字段类型和计数正确
13. audit(print_report=False) 不打印
14. AuditReport.completeness_pct     — 完整因子占比计算正确
15. AuditReport.to_df()              — 返回 DataFrame，有 missing_fields 列
16. 28 个内置因子 E1 字段全部完整    — REGISTRY.audit() incomplete == 0
17. 28 个内置因子 status 均为 ACTIVE
18. 28 个内置因子 forward_safe 均为 True
19. 28 个内置因子 version 均为 "2.9.1"
20. summary_df() 含新 E1 列          — status/version/forward_safe/inputs 均在列中
21. list_active() 只返回 ACTIVE 因子
"""
from __future__ import annotations

import warnings

import pytest

from factor_framework.factors.meta import FactorCategory, FactorMeta, FactorStatus
from factor_framework.factors.registry import (
    REGISTRY,
    AuditReport,
    FactorRegistry,
    _E1_FIELDS,
)

# ── 导入内置因子（触发 REGISTRY 的完整填充）─────────────────────────────────
import factor_framework.factor_zoo  # noqa: F401 — side-effect: populates REGISTRY


# ═════════════════════════════════════════════════════════════════════════════
#  辅助工具
# ═════════════════════════════════════════════════════════════════════════════

def _minimal_meta(**overrides) -> FactorMeta:
    """构造最小合法 FactorMeta（原有必选字段）。"""
    defaults = dict(
        name="test_factor",
        fn=lambda df: df["收盘价"],
        display_name="测试因子",
        category=FactorCategory.CUSTOM,
    )
    defaults.update(overrides)
    return FactorMeta(**defaults)


def _full_e1_meta(**overrides) -> FactorMeta:
    """构造完整 E1 字段的 FactorMeta。"""
    defaults = dict(
        name="test_full",
        fn=lambda df: df["收盘价"],
        display_name="完整测试因子",
        category=FactorCategory.CUSTOM,
        inputs=("收盘价",),
        output_semantic="higher=better",
        forward_safe=True,
        version="1.0.0",
        tags=("test",),
        status=FactorStatus.ACTIVE,
    )
    defaults.update(overrides)
    return FactorMeta(**defaults)


def _fresh_registry() -> FactorRegistry:
    """返回空的独立 FactorRegistry（不影响全局 REGISTRY）。"""
    return FactorRegistry()


# ═════════════════════════════════════════════════════════════════════════════
#  1–2. FactorMeta E1 字段构造
# ═════════════════════════════════════════════════════════════════════════════

def test_e1_default_inputs():
    m = _minimal_meta()
    assert m.inputs == ()


def test_e1_default_output_semantic():
    m = _minimal_meta()
    assert m.output_semantic == ""


def test_e1_default_forward_safe_is_none():
    m = _minimal_meta()
    assert m.forward_safe is None, "未审核时 forward_safe 应为 None"


def test_e1_default_version_is_empty():
    m = _minimal_meta()
    assert m.version == ""


def test_e1_default_tags_empty():
    m = _minimal_meta()
    assert m.tags == ()


def test_e1_default_status_active():
    m = _minimal_meta()
    assert m.status == FactorStatus.ACTIVE


def test_e1_fields_stored_correctly():
    m = _full_e1_meta()
    assert m.inputs == ("收盘价",)
    assert m.output_semantic == "higher=better"
    assert m.forward_safe is True
    assert m.version == "1.0.0"
    assert m.tags == ("test",)
    assert m.status == FactorStatus.ACTIVE


# ═════════════════════════════════════════════════════════════════════════════
#  3–5. missing_e1_fields
# ═════════════════════════════════════════════════════════════════════════════

def test_missing_e1_all_fields():
    """全部 E1 字段为默认空值时，missing_e1_fields 应返回 4 个字段名。"""
    m = _minimal_meta()
    missing = m.missing_e1_fields
    assert set(missing) == {"inputs", "output_semantic", "forward_safe", "version"}, \
        f"全缺失时应返回全部 4 个 E1 字段，实际: {missing}"


def test_missing_e1_none_fields():
    """全部 E1 字段填写后，missing_e1_fields 应返回空列表。"""
    m = _full_e1_meta()
    assert m.missing_e1_fields == []


def test_missing_e1_partial_fields():
    """仅填写 inputs + version，仍缺少 output_semantic + forward_safe。"""
    m = _minimal_meta(inputs=("收盘价",), version="1.0")
    missing = m.missing_e1_fields
    assert set(missing) == {"output_semantic", "forward_safe"}


def test_missing_e1_forward_safe_false_counts_as_filled():
    """forward_safe=False（已知有偏差，显式标记）不应算作缺失。"""
    m = _minimal_meta(
        inputs=("收盘价",), output_semantic="higher=better",
        forward_safe=False, version="1.0",
    )
    assert m.missing_e1_fields == [], \
        "forward_safe=False 是显式声明，不应被报告为缺失"


# ═════════════════════════════════════════════════════════════════════════════
#  6–9. FactorStatus 枚举
# ═════════════════════════════════════════════════════════════════════════════

def test_factor_status_values():
    assert FactorStatus.ACTIVE.value       == "active"
    assert FactorStatus.EXPERIMENTAL.value == "experimental"
    assert FactorStatus.DEPRECATED.value   == "deprecated"


def test_is_active_true():
    assert _minimal_meta().is_active is True


def test_is_active_false_for_deprecated():
    m = _minimal_meta(status=FactorStatus.DEPRECATED)
    assert m.is_active is False


def test_is_active_false_for_experimental():
    m = _minimal_meta(status=FactorStatus.EXPERIMENTAL)
    assert m.is_active is False


def test_invalid_status_raises():
    with pytest.raises(TypeError, match="status"):
        FactorMeta(
            name="f", fn=lambda df: df["收盘价"],
            display_name="f", category=FactorCategory.CUSTOM,
            status="not_a_status",  # type: ignore
        )


# ═════════════════════════════════════════════════════════════════════════════
#  10–11. registry.register() warning 行为
# ═════════════════════════════════════════════════════════════════════════════

def test_register_missing_e1_warns_but_succeeds():
    """缺失 E1 字段时，register() 发出 UserWarning，但注册仍成功。"""
    reg = _fresh_registry()
    m = _minimal_meta(name="incomplete_factor")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.register(m)
    # 注册成功
    assert reg.get("incomplete_factor") is m, "注册应成功（warning-only）"
    # 触发 E1 警告
    e1_warnings = [x for x in w if "Phase E1" in str(x.message) or "缺少字段" in str(x.message)]
    assert e1_warnings, "缺失 E1 字段时应触发 UserWarning"


def test_register_complete_e1_no_warning():
    """E1 字段完整时，register() 不触发 E1 相关 UserWarning。"""
    reg = _fresh_registry()
    m = _full_e1_meta(name="complete_factor")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.register(m)
    e1_warnings = [x for x in w if "Phase E1" in str(x.message) or "缺少字段" in str(x.message)]
    assert not e1_warnings, f"完整 E1 不应触发警告，实际: {[str(x.message) for x in e1_warnings]}"


def test_register_overwrite_warns():
    """同名注册时应触发 overwrite 警告。"""
    reg = _fresh_registry()
    m1 = _full_e1_meta(name="dup")
    m2 = _full_e1_meta(name="dup")
    reg.register(m1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.register(m2)
    overwrite_warns = [x for x in w if "already exists" in str(x.message)]
    assert overwrite_warns, "重复注册应触发 overwrite UserWarning"


# ═════════════════════════════════════════════════════════════════════════════
#  12–15. AuditReport + audit()
# ═════════════════════════════════════════════════════════════════════════════

def test_audit_returns_audit_report():
    reg = _fresh_registry()
    reg.register(_minimal_meta(name="f1"))  # 不完整
    reg.register(_full_e1_meta(name="f2"))  # 完整
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = reg.audit(print_report=False)
    assert isinstance(report, AuditReport)


def test_audit_counts_correct():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_minimal_meta(name="f1"))   # 不完整
        reg.register(_minimal_meta(name="f2"))   # 不完整
        reg.register(_full_e1_meta(name="f3"))   # 完整
        report = reg.audit(print_report=False)
    assert report.total == 3
    assert report.complete == 1
    assert report.incomplete == 2


def test_audit_completeness_pct():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_full_e1_meta(name="f1"))
        reg.register(_full_e1_meta(name="f2"))
        report = reg.audit(print_report=False)
    assert report.completeness_pct == 100.0


def test_audit_completeness_pct_partial():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_full_e1_meta(name="f1"))   # 完整
        reg.register(_minimal_meta(name="f2"))   # 不完整
        report = reg.audit(print_report=False)
    assert report.completeness_pct == 50.0


def test_audit_empty_registry():
    """空注册表 audit() 不应抛异常，completeness=100%。"""
    reg = _fresh_registry()
    report = reg.audit(print_report=False)
    assert report.total == 0
    assert report.completeness_pct == 100.0


def test_audit_to_df_shape():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_minimal_meta(name="f1"))
        reg.register(_full_e1_meta(name="f2"))
    report = reg.audit(print_report=False)
    df = report.to_df()
    # 只有 f1 是不完整的
    assert len(df) == 1
    assert "missing_fields" in df.columns
    assert "n_missing" in df.columns


def test_audit_missing_by_field_counts():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_minimal_meta(name="f1"))
        reg.register(_minimal_meta(name="f2"))
    report = reg.audit(print_report=False)
    # 两个因子都缺全部 E1 字段
    for field_name in _E1_FIELDS:
        assert len(report.missing_by_field[field_name]) == 2, \
            f"字段 {field_name!r} 应有 2 个因子缺失"


# ═════════════════════════════════════════════════════════════════════════════
#  16–19. 28 个内置因子 E1 全覆盖验证
# ═════════════════════════════════════════════════════════════════════════════

def test_all_28_builtin_factors_e1_complete():
    """所有 28 个内置因子的 E1 元数据应全部完整（incomplete == 0）。"""
    report = REGISTRY.audit(print_report=False)
    if report.incomplete > 0:
        df = report.to_df()
        pytest.fail(
            f"以下因子 E1 元数据不完整（共 {report.incomplete} 个）:\n"
            f"{df.to_string()}"
        )


def test_all_builtin_factors_status_active():
    """所有内置因子 status 应为 ACTIVE。"""
    non_active = [
        m.name for m in REGISTRY.list_all()
        if m.status != FactorStatus.ACTIVE
    ]
    assert not non_active, f"以下因子 status 不是 ACTIVE: {non_active}"


def test_all_builtin_factors_forward_safe_true():
    """所有内置因子 forward_safe 应为 True（已验证无前瞻偏差）。"""
    not_safe = [
        m.name for m in REGISTRY.list_all()
        if m.forward_safe is not True
    ]
    assert not not_safe, f"以下因子 forward_safe 不是 True: {not_safe}"


def test_all_builtin_factors_version_2_9_1():
    """所有内置因子 version 应为 '2.9.1'。"""
    wrong_ver = [
        f"{m.name}={m.version!r}"
        for m in REGISTRY.list_all()
        if m.version != "2.9.1"
    ]
    assert not wrong_ver, f"以下因子 version 不是 '2.9.1': {wrong_ver}"


# ═════════════════════════════════════════════════════════════════════════════
#  20. summary_df() 含 E1 列
# ═════════════════════════════════════════════════════════════════════════════

def test_summary_df_contains_e1_columns():
    df = REGISTRY.summary_df()
    e1_cols = {"status", "version", "forward_safe", "inputs", "output_semantic",
               "tags", "missing_e1"}
    missing_cols = e1_cols - set(df.columns)
    assert not missing_cols, f"summary_df() 缺少 E1 列: {sorted(missing_cols)}"


def test_summary_df_no_missing_e1_for_builtins():
    """内置因子的 missing_e1 列应全为空字符串。"""
    df = REGISTRY.summary_df()
    non_empty = df[df["missing_e1"] != ""]
    assert non_empty.empty, \
        f"以下内置因子 missing_e1 不为空:\n{non_empty['missing_e1'].to_string()}"


# ═════════════════════════════════════════════════════════════════════════════
#  21. list_active()
# ═════════════════════════════════════════════════════════════════════════════

def test_list_active_excludes_deprecated():
    reg = _fresh_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.register(_full_e1_meta(name="active_f",     status=FactorStatus.ACTIVE))
        reg.register(_full_e1_meta(name="deprecated_f", status=FactorStatus.DEPRECATED))
        reg.register(_full_e1_meta(name="exp_f",        status=FactorStatus.EXPERIMENTAL))
    active = reg.list_active()
    names = [m.name for m in active]
    assert "active_f"     in names
    assert "deprecated_f" not in names
    assert "exp_f"        not in names


def test_list_active_all_builtins_included():
    """内置因子全 ACTIVE，list_active() 应与 list_all() 数量相同。"""
    assert len(REGISTRY.list_active()) == len(REGISTRY.list_all())


# ═════════════════════════════════════════════════════════════════════════════
#  repr + 冻结验证
# ═════════════════════════════════════════════════════════════════════════════

def test_repr_contains_status():
    m = _full_e1_meta()
    assert "active" in repr(m), "repr 应包含 status 字段"


def test_factormeta_is_frozen():
    """FactorMeta 是 frozen dataclass，不应允许字段修改。"""
    m = _minimal_meta()
    with pytest.raises((AttributeError, TypeError)):
        m.inputs = ("收盘价",)  # type: ignore  — frozen dataclass should raise
