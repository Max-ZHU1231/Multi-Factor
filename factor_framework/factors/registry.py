"""
factor_framework.factors.registry
===================================
FactorRegistry -- unified factor registration centre (v3.0 spec sec 2.2).
_CompatDict    -- zero-breakage compatibility shim for BUILTIN_FACTORS.

Dependency direction
--------------------
registry.py  <--  factor_zoo.py   (one-way; no circular import)

factor_zoo._register_builtins() imports this module directly after all its
own functions are defined, then calls REGISTRY.register(FactorMeta(...)) for
each of the 28 built-in factors.  This means REGISTRY is fully populated by
the time any user code imports factor_zoo.

FactorRegistry API
------------------
register(meta)             -> None            (store / overwrite FactorMeta)
get(name)                  -> FactorMeta|None
get_fn(name)               -> Callable|None
list_by_category(category) -> List[FactorMeta] sorted by name
list_all()                 -> List[FactorMeta] sorted by name
to_compat_dict()           -> _CompatDict     (for BUILTIN_FACTORS)
summary_df()               -> pd.DataFrame    (metadata overview)

Global singleton
----------------
REGISTRY : FactorRegistry  -- empty at module load; populated by factor_zoo.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterator, List, Optional

import pandas as pd

from factor_framework.factors.meta import FactorCategory, FactorMeta


class _CompatDict(dict):
    def __init__(self, registry, mapping):
        super().__init__(mapping)
        object.__setattr__(self, "_registry_ref", registry)

    @property
    def registry(self):
        return object.__getattribute__(self, "_registry_ref")

    def get_meta(self, name):
        return self.registry.get(name)


class FactorRegistry:
    def __init__(self):
        self._meta_table = {}

    def register(self, meta):
        if not isinstance(meta, FactorMeta):
            raise TypeError(f"register() expects FactorMeta, got {type(meta).__name__!r}")
        if meta.name in self._meta_table:
            warnings.warn(f"[FactorRegistry] factor {meta.name!r} already exists, overwriting.", UserWarning, stacklevel=2)
        self._meta_table[meta.name] = meta

    def get(self, name):
        return self._meta_table.get(name)

    def get_fn(self, name):
        meta = self._meta_table.get(name)
        return meta.fn if meta is not None else None

    def __contains__(self, name):
        return name in self._meta_table

    def __len__(self):
        return len(self._meta_table)

    def __iter__(self):
        return iter(self._meta_table)

    def list_by_category(self, category):
        return sorted([m for m in self._meta_table.values() if m.category == category], key=lambda m: m.name)

    def list_all(self):
        return sorted(self._meta_table.values(), key=lambda m: m.name)

    def to_compat_dict(self):
        mapping = {name: meta.fn for name, meta in self._meta_table.items()}
        return _CompatDict(self, mapping)

    def summary_df(self):
        rows = []
        for meta in self.list_all():
            rows.append({
                "name": meta.name,
                "display_name": meta.display_name,
                "category": meta.category.value,
                "direction": meta.direction,
                "warmup_days": meta.warmup_days,
                "neutral_by_default": meta.neutral_by_default,
                "skip_neutralize_cols": ", ".join(meta.skip_neutralize_cols),
                "description": meta.description,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("name")
        return df

    def __repr__(self):
        cats = {}
        for m in self._meta_table.values():
            cats[m.category.value] = cats.get(m.category.value, 0) + 1
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
        return f"FactorRegistry(n={len(self)}, [{cat_str}])"


REGISTRY = FactorRegistry()
