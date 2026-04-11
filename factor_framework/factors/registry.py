"""
factor_framework.factors.registry
===================================
FactorRegistry -- unified factor registration centre (v4.0 Phase E1).
_CompatDict    -- zero-breakage compatibility shim for BUILTIN_FACTORS.

Dependency direction
--------------------
registry.py  <--  factor_zoo.py   (one-way; no circular import)

Phase E1 changes
----------------
* register() emits UserWarning when Phase E1 fields are missing
  (inputs / output_semantic / forward_safe / version).
  Registration still SUCCEEDS — warning-only, never fails.
* audit() — returns AuditReport with per-factor missing-field breakdown
  and prints a human-readable table to stdout.
* summary_df() — extended to include all E1 columns.

FactorRegistry API
------------------
register(meta)             -> None            (store / overwrite FactorMeta)
get(name)                  -> FactorMeta|None
get_fn(name)               -> Callable|None
list_by_category(category) -> List[FactorMeta] sorted by name
list_all()                 -> List[FactorMeta] sorted by name
to_compat_dict()           -> _CompatDict     (for BUILTIN_FACTORS)
summary_df()               -> pd.DataFrame    (metadata overview, incl. E1)
audit()                    -> AuditReport     (E1 completeness report)

Global singleton
----------------
REGISTRY : FactorRegistry  -- empty at module load; populated by factor_zoo.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from factor_framework.factors.meta import FactorCategory, FactorMeta, FactorStatus

# ─────────────────────────────────────────────────────────────────────────────
# Phase E1 audit fields (must match FactorMeta.missing_e1_fields logic)
# ─────────────────────────────────────────────────────────────────────────────
_E1_FIELDS = ("inputs", "output_semantic", "forward_safe", "version")


# ─────────────────────────────────────────────────────────────────────────────
# AuditReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """
    Phase E1 元数据完整性审计报告。

    Attributes
    ----------
    total         : 注册的因子总数
    complete      : E1 元数据已完整填写的因子数
    incomplete    : 有缺失字段的因子数
    missing_by_field : {字段名: [因子名列表]}
    rows          : 每因子缺失字段详情 [{name, missing}]
    """
    total:            int
    complete:         int
    incomplete:       int
    missing_by_field: Dict[str, List[str]]
    rows:             List[dict]

    @property
    def completeness_pct(self) -> float:
        """元数据完整度百分比（0–100）。"""
        return 100.0 * self.complete / self.total if self.total > 0 else 100.0

    def print_report(self) -> None:
        """终端打印可读的审计报告。"""
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  Phase E1 元数据审计报告")
        print(sep)
        print(f"  总因子数     : {self.total}")
        print(f"  完整         : {self.complete}  ({self.completeness_pct:.1f}%)")
        print(f"  有缺失字段   : {self.incomplete}")
        print(sep)

        if not self.rows:
            print("  ✓ 所有因子 E1 元数据均已完整填写。")
        else:
            # 按缺失字段数（多→少）排序
            sorted_rows = sorted(self.rows, key=lambda r: -len(r["missing"]))
            # 列宽
            name_w = max(len(r["name"]) for r in sorted_rows) + 2
            for r in sorted_rows:
                missing_str = ", ".join(r["missing"])
                print(f"  {'[缺失]':<8}  {r['name']:<{name_w}}  缺少: {missing_str}")

        print(sep)
        if self.missing_by_field:
            print("  按字段汇总缺失数:")
            for f in _E1_FIELDS:
                n = len(self.missing_by_field.get(f, []))
                bar = "█" * n
                print(f"    {f:<20} {n:>3}  {bar}")
        print(sep + "\n")

    def to_df(self) -> pd.DataFrame:
        """将缺失明细转换为 DataFrame（name × 缺失字段列表）。"""
        if not self.rows:
            return pd.DataFrame(columns=["name", "missing_fields", "n_missing"])
        rows = [
            {"name": r["name"],
             "missing_fields": ", ".join(r["missing"]),
             "n_missing": len(r["missing"])}
            for r in self.rows
        ]
        return pd.DataFrame(rows).set_index("name").sort_values("n_missing", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# _CompatDict
# ─────────────────────────────────────────────────────────────────────────────

class _CompatDict(dict):
    def __init__(self, registry, mapping):
        super().__init__(mapping)
        object.__setattr__(self, "_registry_ref", registry)

    @property
    def registry(self):
        return object.__getattribute__(self, "_registry_ref")

    def get_meta(self, name):
        return self.registry.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# FactorRegistry
# ─────────────────────────────────────────────────────────────────────────────

class FactorRegistry:
    """
    因子元数据注册中心（Phase E1 升级版）。

    注册时若 E1 字段缺失，发出 UserWarning（warning-only，不阻止注册）。
    使用 audit() 获取批量缺失报告，便于补齐 28 个内置因子的元数据。
    """

    def __init__(self):
        self._meta_table: Dict[str, FactorMeta] = {}

    def register(self, meta: FactorMeta) -> None:
        """
        注册因子元数据。

        Phase E1：若 E1 字段（inputs/output_semantic/forward_safe/version）
        有任意缺失，发出 UserWarning。**注册仍会成功**（warning-only）。
        """
        if not isinstance(meta, FactorMeta):
            raise TypeError(f"register() expects FactorMeta, got {type(meta).__name__!r}")
        if meta.name in self._meta_table:
            warnings.warn(
                f"[FactorRegistry] factor {meta.name!r} already exists, overwriting.",
                UserWarning, stacklevel=2,
            )
        self._meta_table[meta.name] = meta

        # ── Phase E1: warn on missing metadata (non-blocking) ────────────
        missing = meta.missing_e1_fields
        if missing:
            warnings.warn(
                f"[FactorRegistry] '{meta.name}' 注册成功，但 Phase E1 元数据"
                f" 缺少字段: {missing}。"
                f" 请补充以提高文档和审计覆盖率。",
                UserWarning,
                stacklevel=2,
            )

    def get(self, name: str) -> Optional[FactorMeta]:
        return self._meta_table.get(name)

    def get_fn(self, name: str) -> Optional[Callable]:
        meta = self._meta_table.get(name)
        return meta.fn if meta is not None else None

    def __contains__(self, name: str) -> bool:
        return name in self._meta_table

    def __len__(self) -> int:
        return len(self._meta_table)

    def __iter__(self):
        return iter(self._meta_table)

    def list_by_category(self, category: FactorCategory) -> List[FactorMeta]:
        return sorted(
            [m for m in self._meta_table.values() if m.category == category],
            key=lambda m: m.name,
        )

    def list_all(self) -> List[FactorMeta]:
        return sorted(self._meta_table.values(), key=lambda m: m.name)

    def list_active(self) -> List[FactorMeta]:
        """返回所有 status=ACTIVE 的因子（过滤掉 experimental/deprecated）。"""
        return [m for m in self.list_all() if m.status == FactorStatus.ACTIVE]

    def to_compat_dict(self) -> _CompatDict:
        mapping = {name: meta.fn for name, meta in self._meta_table.items()}
        return _CompatDict(self, mapping)

    # ── summary_df（含 E1 字段）────────────────────────────────────────────

    def summary_df(self) -> pd.DataFrame:
        """
        返回所有注册因子的元数据总览 DataFrame（含 Phase E1 字段）。

        列：name / display_name / category / direction / warmup_days /
            neutral_by_default / status / version / forward_safe /
            inputs / output_semantic / tags / missing_e1 / description
        """
        rows = []
        for meta in self.list_all():
            rows.append({
                "name":                meta.name,
                "display_name":        meta.display_name,
                "category":            meta.category.value,
                "direction":           meta.direction,
                "warmup_days":         meta.warmup_days,
                "neutral_by_default":  meta.neutral_by_default,
                "skip_neutralize_cols": ", ".join(meta.skip_neutralize_cols),
                # E1 新增
                "status":              meta.status.value,
                "version":             meta.version,
                "forward_safe":        meta.forward_safe,
                "inputs":              ", ".join(meta.inputs) if meta.inputs else "",
                "output_semantic":     meta.output_semantic,
                "tags":                ", ".join(meta.tags) if meta.tags else "",
                "missing_e1":         ", ".join(meta.missing_e1_fields),
                "description":         meta.description,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("name")
        return df

    # ── Phase E1: audit() ─────────────────────────────────────────────────

    def audit(self, print_report: bool = True) -> AuditReport:
        """
        检查所有已注册因子的 Phase E1 元数据完整性。

        对每个因子调用 meta.missing_e1_fields（检查 inputs/output_semantic/
        forward_safe/version），汇总缺失情况，生成 AuditReport。

        Parameters
        ----------
        print_report : bool
            True（默认）= 立即打印可读报告到 stdout；
            False       = 静默返回 AuditReport 对象。

        Returns
        -------
        AuditReport
            包含 total/complete/incomplete/missing_by_field/rows。

        示例
        ----
            from factor_framework.factors.registry import REGISTRY
            report = REGISTRY.audit()
            df = report.to_df()   # 转为 DataFrame 分析
        """
        missing_by_field: Dict[str, List[str]] = {f: [] for f in _E1_FIELDS}
        rows = []

        for meta in self.list_all():
            missing = meta.missing_e1_fields
            if missing:
                rows.append({"name": meta.name, "missing": missing})
                for f in missing:
                    if f in missing_by_field:
                        missing_by_field[f].append(meta.name)

        total      = len(self._meta_table)
        incomplete = len(rows)
        complete   = total - incomplete

        report = AuditReport(
            total            = total,
            complete         = complete,
            incomplete       = incomplete,
            missing_by_field = missing_by_field,
            rows             = rows,
        )
        if print_report:
            report.print_report()
        return report

    def __repr__(self) -> str:
        cats: Dict[str, int] = {}
        for m in self._meta_table.values():
            cats[m.category.value] = cats.get(m.category.value, 0) + 1
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
        return f"FactorRegistry(n={len(self)}, [{cat_str}])"


REGISTRY = FactorRegistry()
