# Migration Guide: v3.x → v4.0

Estimated effort: **15–30 minutes** for a typical research script.

> **Compatibility window**: all v3 entry-points and import paths work in v4.0 and v4.1,
> emitting `DeprecationWarning`. They are **removed in v4.2**.

---

## Part 1 — Entry-point changes

### Old (v3) → still works in v4, but warns

```bash
python factor_analysis.py --factor momentum_12_1   # DeprecationWarning
python backtest_demo.py                             # DeprecationWarning
python scripts/run_analysis.py --factor momentum_12_1
python scripts/run_batch.py --config config/default.yaml
python scripts/run_factor_screening.py
```

### New (v4) — unified `mf` CLI

```bash
# one-time install
pip install -e .

# single factor: IC analysis + layered backtest
mf single --factor momentum_12_1

# multiple factors
mf single --factor momentum_12_1 value_pb size_log_mktcap

# full batch (all 28 registered factors)
mf batch

# targeted batch with date range
mf batch --factors momentum_12_1 value_pb --start 2021-01-01 --end 2024-12-31

# validation
mf validate --suite lookahead
mf validate --suite all

# cache management
mf cache info
mf cache gc --days 30
mf cache clear
```

Without install:
```bash
python -m factor_framework.cli single --factor momentum_12_1
```

**Exit codes:** `0` success · `1` runtime error · `2` argument error

---

## Part 2 — ResearchConfig (replacing scatter kwargs)

### Old (v3) — keyword arguments on `pipe.run()`

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline("stocks/stocks/")
pipe.register_builtins()
results = pipe.run(
    factor_name="momentum_12_1",
    start="2020-01-01",
    end="2024-12-31",
    forward_days=20,
    n_groups=5,
    cost_bps=10,
    resample_monthly=True,
)
```

### New (v4) — `ResearchConfig` object

```python
from factor_framework.pipeline import FactorPipeline
from factor_framework.research_config import ResearchConfig

rc = ResearchConfig(
    factor_name="momentum_12_1",
    start="2020-01-01",
    end="2024-12-31",
    forward_days=20,
    n_groups=5,
    cost_bps=10,
    resample_monthly=True,
)

pipe = FactorPipeline("stocks/stocks/")
pipe.register_builtins()
results = pipe.run(config=rc)

# config is the single source of truth
print(rc.config_hash())       # deterministic 8-char hex
print(rc.to_stable_dict())    # canonical serialisable form
```

### Load from YAML

```python
import yaml
from factor_framework.research_config import ResearchConfig

with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
rc = ResearchConfig.from_dict(cfg)      # auto-upgrades old schema versions
```

### v3 kwargs still work (no breakage)

`pipe.run(factor_name=..., start=..., ...)` continues to work in v4.0 — kwargs are
auto-converted to `ResearchConfig` internally. No `DeprecationWarning` is emitted for
this path; full removal is not planned before v4.2.

---

## Part 3 — Import path changes

### Moved modules

| v3 import | v4 canonical import | Shim until |
|-----------|--------------------|--------------------|
| `from factor_framework.ic_analysis import compute_ic` | `from factor_framework.analytics.ic_analysis import compute_ic` | v4.2 |
| `from factor_framework.ic_analysis import ic_decay` | `from factor_framework.analytics.ic_analysis import ic_decay` | v4.2 |
| `from factor_framework.neutralize import neutralize_regression` | `from factor_framework.transform.neutralize import neutralize_regression` | v4.2 |
| `from factor_framework.optimizer import equal_weight` | `from factor_framework.optimize.optimizer import equal_weight` | v4.2 |
| `from factor_framework.factors.transform import TransformPipeline` | `from factor_framework.transform import TransformPipeline` | v4.2 |
| `from factor_framework.factors.ic_analyzer import ICAnalyzer` | `from factor_framework.analytics.ic_analyzer import ICAnalyzer` | v4.2 |
| `from factor_framework.factors.layer_backtester import LayerBacktester` | `from factor_framework.analytics.layer_backtester import LayerBacktester` | v4.2 |

All old paths work in v4.0 / v4.1 but emit `DeprecationWarning` on import.

### Top-level convenience imports (unchanged)

```python
# factor_framework/__init__.py re-exports these — still valid
from factor_framework import FactorPipeline, FactorEngine
from factor_framework import compute_ic, ic_stats, ic_decay
from factor_framework import orthogonalize
```

---

## Part 4 — Manifest and cache key

### run_manifest.json (new in v4.0)

Every `pipe.run()` call populates `pipe.last_manifest`. `mf single` / `mf batch` write
`{output_dir}/run_manifest.json`:

```json
{
  "schema_version": "1.0",
  "run_id": "20260411_143022_a3f2",
  "timestamp": "2026-04-11T14:30:22",
  "config_hash": "3a9f2b1c",
  "git_sha": "3beb975",
  "data_snapshot_id": "stocks/stocks/::mtime=1744358422",
  "pipeline_version": "4.0.0rc1",
  "factors": ["momentum_12_1"],
  "date_range": ["2020-01-01", "2024-12-31"],
  "cache_stats": {"hits": 3, "misses": 1, "legacy_hits": 0},
  "exit_status": "success",
  "elapsed_s": 42.3,
  "environment": {"python": "3.13", "platform": "Windows"},
  "key_metrics": {"mean_ic": 0.042, "icir": 0.38, "ls_sharpe": 1.12}
}
```

```python
from factor_framework.manifest import RunManifest
m = RunManifest.load("artifacts/run_manifest.json")
m.print_summary()
```

### Cache key upgrade (v1 → v2)

v4.0 adds three dimensions to the cache key:

| Dimension | v3 (key v1) | v4 (key v2) |
|-----------|-------------|-------------|
| Factor ID + time range + universe hash | ✅ | ✅ |
| Transform config hash | ❌ | ✅ |
| Semantic contract version | ❌ | ✅ |
| Git SHA | ❌ | ✅ |

**No action needed** — v4.0 tries the v2 key first, then falls back to the v1 key. Your
existing cache files are still used. The manifest records which key was hit
(`new_key_hit` / `legacy_key_hit` / `recompute`).

To force a full rebuild with v2 keys:
```bash
mf cache clear && mf batch
```

---

## Part 5 — Factor metadata (E1 fields, optional in v4.0)

All 28 built-in factors now carry full metadata. Custom factors may omit these fields —
a `UserWarning` is emitted at registration but registration always succeeds.

```python
from factor_framework.factors.meta import FactorMeta, FactorStatus

meta = FactorMeta(
    name="my_factor",
    fn=my_fn,
    display_name="My Factor",
    category="custom",
    # E1 fields — optional in v4.0, recommended for audit coverage
    inputs=("close", "volume"),
    output_semantic="higher=better_signal",
    forward_safe=True,
    version="1.0.0",
    tags=("custom",),
    status=FactorStatus.ACTIVE,
)
```

Audit completeness across all registered factors:
```python
from factor_framework.factors.registry import REGISTRY
report = REGISTRY.audit()          # prints table; returns AuditReport
df     = report.to_df()            # DataFrame for further analysis
```

---

## Deprecation timeline

| Version | Policy |
|---------|--------|
| **v4.0 (current)** | All old paths work; emit `DeprecationWarning` |
| **v4.1** | E1 meta fields enforced (strict gate); old paths still work |
| **v4.2** | Old entry-points and import shims **removed** |

---

## Quick reference

```
v3                                    v4
──────────────────────────────────────────────────────────
python factor_analysis.py          →  mf single --factor NAME
python scripts/run_analysis.py     →  mf single --factor NAME
python scripts/run_batch.py        →  mf batch
from factor_framework.ic_analysis  →  from factor_framework.analytics.ic_analysis
from factor_framework.neutralize   →  from factor_framework.transform.neutralize
from factor_framework.optimizer    →  from factor_framework.optimize.optimizer
pipe.run(factor_name=..., ...)     →  pipe.run(config=ResearchConfig(...))
```
