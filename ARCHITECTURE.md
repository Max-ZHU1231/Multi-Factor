# Architecture — Multi-Factor Research Framework (v4.0)

> **v4.0 Migration Summary**: New canonical sub-packages (`analytics`, `transform`, `optimize`,
> `reporting`, `cli`) added inside `factor_framework/`. Old import paths are kept as
> compatibility shims (DeprecationWarning) and will be removed in v4.2.
> See `MIGRATION.md` for import update guide.

## Overview

```
User
 │
 ├─ mf <command>                ← v4.0 PRIMARY CLI  (pip install -e .)
 │   ├─ mf screen               single-factor IC / layer-backtest
 │   ├─ mf batch                full-batch factor validation
 │   ├─ mf validate             look-ahead / data-quality validation
 │   ├─ mf cache                cache management
 │   └─ mf report               artifact report generation
 │
 ├─ scripts/run_analysis.py     ← legacy CLI shim (kept through v4.1)
 ├─ scripts/run_batch.py        ← legacy CLI shim (kept through v4.1)
 │
 └─ config/default.yaml         ← single config source (YAML)
    config/loader.py            ← ConfigNamespace (dot-access)
```

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLI Layer  (factor_framework/cli/)                              [v4.0 NEW] │
│  cli/main.py  →  mf screen / batch / validate / cache / report             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  Config Layer  (config/)                                                    │
│  default.yaml → load_config() → ConfigNamespace                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│  Pipeline Layer  (factor_framework/pipeline.py)                             │
│  FactorPipeline                                                             │
│   ├── register_builtins()                                                   │
│   ├── run(factor_name, ...)  → FactorReport                                │
│   └── run_batch_from_panels(...) → {name: FactorReport}                    │
└──────┬───────────────────────────────────────────┬──────────────────────────┘
       │                                           │
┌──────▼──────────────┐                 ┌──────────▼───────────────────────┐
│  Data Layer         │                 │  Compute Layer                   │
│  data/store.py      │                 │  engine/panel_builder.py         │
│  CSVDataStore       │──── feeds ─────►│  PanelBuilder                    │
│   └── get_price_    │                 │   ├── build(name, start, end)    │
│       panel()       │                 │   └── engine: FactorEngine       │
│  → TimestampedPanel │                 │                                  │
│    (semantic=price) │                 │  engine/cache.py                 │
└─────────────────────┘                 │  CacheLayer  (L1 mem + L2 Parquet│
                                        └──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Analytics Layer  (factor_framework/analytics/)                  [v4.0 NEW] │
│  analytics/ic_analysis.py   — compute_ic, ic_stats, ic_decay, ...          │
│  analytics/ic_analyzer.py   — ICAnalyzer                                   │
│  analytics/layer_backtester.py — LayerBacktester                           │
│                                                                             │
│  OLD SHIMS (DeprecationWarning, removed v4.2):                              │
│    factor_framework/ic_analysis.py       → analytics.ic_analysis           │
│    factor_framework/factors/ic_analyzer.py → analytics.ic_analyzer         │
│    factor_framework/factors/layer_backtester.py → analytics.layer_backtester│
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Transform Layer  (factor_framework/transform/)                  [v4.0 NEW] │
│  transform/transform.py  — TransformPipeline                               │
│  transform/neutralize.py — neutralize_regression, neutralize_industry_zscore│
│                                                                             │
│  OLD SHIMS (DeprecationWarning, removed v4.2):                              │
│    factor_framework/neutralize.py         → transform.neutralize            │
│    factor_framework/factors/transform.py  → transform.transform             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Optimize Layer  (factor_framework/optimize/)                    [v4.0 NEW] │
│  optimize/optimizer.py — equal_weight, icir_weight, print_weights           │
│                                                                             │
│  OLD SHIM (DeprecationWarning, removed v4.2):                               │
│    factor_framework/optimizer.py          → optimize.optimizer              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Reporting Layer  (factor_framework/reporting/)                  [v4.0 stub]│
│  Phase D implementation: RunManifest, HTML/PDF report generation            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Core Layer  (core/)                                                        │
│  core/panel.py  → TimestampedPanel   (semantic + align_with guard)         │
│  core/returns.py → ReturnPanel        (forward return construction)         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Factor Library  (factors/)                                                 │
│  factors/momentum.py    — momentum_12_1, momentum_6_1, ...                 │
│  factors/volatility.py  — vol_20d, vol_60d, vol_skew, ...                  │
│  factors/value.py       — value_pb, value_pe_ttm, size_log_mktcap          │
│  factors/volume.py      — amihud_illiquidity, rsi_14, macd_signal          │
│  factors/registry.py    — REGISTRY (global catalogue)                      │
│  factors/meta.py        — FactorMeta + FactorCategory                      │
│  factors/transform.py   — [SHIM] → transform.TransformPipeline             │
│  factors/ic_analyzer.py — [SHIM] → analytics.ICAnalyzer                   │
│  factors/layer_backtester.py — [SHIM] → analytics.LayerBacktester          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Operator / JIT Layer                                                       │
│  operators.py  — cs_rank, cs_zscore, cs_winsorize, ts_mean, ...            │
│  jit_ops.py    — Numba JIT / Numexpr / pandas three-tier fallback          │
│  dag.py        — expression tree, CSE, DAGExecutor                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Layout (v4.0)

```
Multi Factor/
│
├── config/                         # Config layer
│   ├── default.yaml                #   Single config source of truth
│   ├── loader.py                   #   load_config(), ConfigNamespace
│   └── __init__.py
│
├── scripts/                        # Legacy CLI shims (kept through v4.1)
│   ├── run_analysis.py             #   ⚠️ Use: mf screen
│   ├── run_batch.py                #   ⚠️ Use: mf batch
│   └── run_validation.py           #   ⚠️ Use: mf validate
│
├── factor_framework/               # Core library
│   │
│   ├── cli/                        # ← NEW (v4.0) Unified CLI
│   │   ├── main.py                 #   mf entry-point + sub-command stubs
│   │   └── __init__.py
│   │
│   ├── analytics/                  # ← NEW (v4.0) Analytics sub-package
│   │   ├── ic_analysis.py          #   compute_ic, ic_stats, ic_decay, ...
│   │   ├── ic_analyzer.py          #   ICAnalyzer
│   │   ├── layer_backtester.py     #   LayerBacktester
│   │   └── __init__.py
│   │
│   ├── transform/                  # ← NEW (v4.0) Transform sub-package
│   │   ├── transform.py            #   TransformPipeline
│   │   ├── neutralize.py           #   neutralize_regression, industry_zscore
│   │   └── __init__.py
│   │
│   ├── optimize/                   # ← NEW (v4.0) Optimize sub-package
│   │   ├── optimizer.py            #   equal_weight, icir_weight, print_weights
│   │   └── __init__.py
│   │
│   ├── reporting/                  # ← NEW (v4.0) Reporting sub-package (stub)
│   │   └── __init__.py             #   Phase D: RunManifest, HTML/PDF reports
│   │
│   ├── pipeline.py                 #   FactorPipeline (top-level orchestrator)
│   ├── backtest.py                 #   layer_backtest, long_short_stats (in-place v4.0)
│   │
│   ├── core/                       # Semantic type system
│   │   ├── panel.py                #   TimestampedPanel + alignment guards
│   │   └── returns.py              #   ReturnPanel
│   │
│   ├── engine/                     # Compute + cache
│   │   ├── panel_builder.py        #   PanelBuilder (parallel + cache)
│   │   └── cache.py                #   CacheLayer (L1 memory + L2 Parquet)
│   │
│   ├── data/                       # Data access
│   │   └── store.py                #   DataStore ABC + CSVDataStore
│   │
│   ├── factors/                    # Factor sub-package
│   │   ├── meta.py                 #   FactorMeta + FactorCategory
│   │   ├── registry.py             #   REGISTRY global catalogue
│   │   ├── momentum.py             #   momentum_* factor functions
│   │   ├── volatility.py           #   vol_* factor functions
│   │   ├── value.py                #   value_*, size_* factor functions
│   │   ├── volume.py               #   amihud_*, rsi_*, macd_* factor functions
│   │   ├── transform.py            #   [SHIM v4.0] → transform.TransformPipeline
│   │   ├── ic_analyzer.py          #   [SHIM v4.0] → analytics.ICAnalyzer
│   │   └── layer_backtester.py     #   [SHIM v4.0] → analytics.LayerBacktester
│   │
│   ├── ic_analysis.py              #   [SHIM v4.0] → analytics.ic_analysis
│   ├── neutralize.py               #   [SHIM v4.0] → transform.neutralize
│   ├── optimizer.py                #   [SHIM v4.0] → optimize.optimizer
│   │
│   ├── operators.py                #   cs_rank, ts_mean, ts_stddev, ...
│   ├── jit_ops.py                  #   Three-tier JIT acceleration
│   ├── dag.py                      #   Expression tree + CSE + DAGExecutor
│   │
│   ├── factor_engine.py            #   ⚠️ DEPRECATED — use PanelBuilder
│   └── factor_zoo.py               #   ⚠️ DEPRECATED — use factors/{cat}.py
│
├── tests/                          # ← v4.0 Tiered test structure
│   ├── unit/                       #   Fast, isolated function-level tests
│   │   ├── test_new_import_paths.py #  New + legacy path import verification
│   │   ├── test_data_cleaner.py
│   │   └── test_data_quality.py
│   ├── integration/                #   Cross-module flow tests (may load data)
│   │   └── test_lookahead_bias.py
│   ├── regression/                 #   IC/Sharpe snapshot baselines
│   │   └── test_factor_baselines.py #  provisional v1: 10 factors
│   ├── slow/                       #   Large-sample tests (run on demand: -m slow)
│   ├── test_factor_framework.py    #   550 unit + integration tests
│   └── conftest.py
│
├── validation/                     # Path-consistency / bias validation (legacy)
│   └── test_lookahead_bias.py
│
├── artifacts/                      # Output artifacts (gitignored)
│
├── conftest.py                     # Root conftest (sys.path setup)
├── pyproject.toml                  # PEP 517 build + pytest config (v4.0.0)
├── MIGRATION.md                    # Migration guide
└── ARCHITECTURE.md                 # This file

Raw data:
└── stocks/stocks/                  # Raw OHLCV CSVs (one file per stock)
```

---

## Import Path Migration Guide (v4.0)

| Old path (shim — removed v4.2) | New canonical path |
|--------------------------------|-------------------|
| `from factor_framework.ic_analysis import compute_ic` | `from factor_framework.analytics import compute_ic` |
| `from factor_framework.ic_analysis import ic_decay` | `from factor_framework.analytics import ic_decay` |
| `from factor_framework.neutralize import neutralize_regression` | `from factor_framework.transform import neutralize_regression` |
| `from factor_framework.optimizer import equal_weight` | `from factor_framework.optimize import equal_weight` |
| `from factor_framework.factors.transform import TransformPipeline` | `from factor_framework.transform import TransformPipeline` |
| `from factor_framework.factors.ic_analyzer import ICAnalyzer` | `from factor_framework.analytics import ICAnalyzer` |
| `from factor_framework.factors.layer_backtester import LayerBacktester` | `from factor_framework.analytics import LayerBacktester` |

---

## Data-Flow for a Single Factor Run

```
mf screen --factor momentum_12_1
    │
    │  1. load_config()          reads config/default.yaml (+ CLI overrides)
    │
    │  2. FactorPipeline(...)    constructs with CSVDataStore + PanelBuilder
    │
    │  3. pipe.register_builtins([factor_name])
    │         └─► REGISTRY.register(FactorMeta(...))
    │
    │  4. pipe.run(factor_name, forward=21, ...)
    │         │
    │         ├─ PanelBuilder.build(factor_name, start, end)
    │         │       ├─ CacheLayer.get(key)  ← L2 Parquet hit?
    │         │       ├─ FactorEngine.build_panel(...)   ← compute
    │         │       │       └─ DAGExecutor(expr_tree)
    │         │       │             └─ CSVDataStore.load_one(symbol)
    │         │       │                   └─ load_and_clean(csv_path)
    │         │       └─ CacheLayer.put(key, panel)  → .parquet
    │         │
    │         ├─ ReturnPanel.build(price_ts, forward_days=21)
    │         │
    │         ├─ factor_ts.shift_to_t1()          ← T+1 alignment
    │         │
    │         ├─ analytics.compute_ic(factor_t1, ret_ts)
    │         ├─ analytics.ic_decay(factor_t1, return_panels={...})
    │         ├─ analytics.LayerBacktester(factor_t1, ret_ts, n_groups=5)
    │         └─ transform.neutralize_regression(factor_t1, mktcap, industry)
    │
    │  5. report.print_summary()
    │     report.save(output_dir)
    ▼
  artifacts/  (or output/)
```

---

## Semantic Type System (v3.2+)

Every `TimestampedPanel` carries a `semantic` tag that analysis functions
validate at call time:

| semantic | Created by | Valid pairing |
|----------|-----------|---------------|
| `"price"` | `CSVDataStore.get_price_panel()` | input to `ReturnPanel.build()` only |
| `"factor_observation"` | `PanelBuilder.build()` | must call `.shift_to_t1()` before analysis |
| `"forward_return"` | `ReturnPanel.build()` | pairs with `factor_observation` (T+1-shifted) |

Violations raise `TimingAlignmentError` or `SemanticCompatibilityError` immediately.

---

## Cache Architecture

```
CacheLayer
 ├── L1: in-memory dict  { key → DataFrame }   (per-process lifetime)
 └── L2: Parquet files   cache/<factor>/<md5>.parquet
           key = MD5( factor_name + start + end + sorted_symbols )
```

Policy:
- Write to L2 only if panel computation took ≥ `min_calc_secs` (default 5 s)
- Reads check L1 first, then L2, then compute
- `CacheLayer.clear_l2()` deletes all Parquet files
- `cache_dir=None` disables L2; L1 always active

---

## Testing Layout (v4.0)

| Suite | Location | Scope | Run |
|-------|----------|-------|-----|
| Unit | `tests/unit/` | Fast, isolated function-level tests | `pytest tests/unit/` |
| Integration | `tests/integration/` | Cross-module flow, may load data | `pytest tests/integration/` |
| Regression | `tests/regression/` | IC/Sharpe snapshot baselines | `pytest tests/regression/` |
| Slow | `tests/slow/` | Large-sample, on demand | `pytest -m slow` |
| Framework | `tests/test_factor_framework.py` | 550 unit + integration tests | `pytest tests/` |

Run all fast tests:
```bash
pytest tests/unit/ tests/integration/ -q
```

Run regression baselines:
```bash
pytest tests/regression/ -v
```

---

## Release Milestones

| Version | Milestone |
|---------|-----------|
| v3.0 | `factor_framework/` package, FactorEngine, IC analysis, layer backtest |
| v3.1 | DataStore, PanelBuilder, CacheLayer |
| v3.2 | Factor sub-package, TransformPipeline, ICAnalyzer, LayerBacktester |
| v3.3 | **config/**, **scripts/** main path, tests/ governance, deprecation markers |
| **v4.0** | **analytics/, transform/, optimize/, cli/ sub-packages; tiered tests; `mf` CLI** |
| v4.1 | CLI Phase B (full sub-command impl.), Manifest + cache-key upgrade (Phase D) |
| v4.2 | Remove all v3.x shims (`ic_analysis.py`, `neutralize.py`, `optimizer.py`, etc.) |


## Overview

```
User
 │
 ├─ scripts/run_analysis.py     ← single-factor CLI entry point (v3.3+)
 ├─ scripts/run_batch.py        ← multi-factor batch CLI (v3.3+)
 │        │
 │        ▼
 │   config/default.yaml        ← single config source (YAML)
 │   config/loader.py           ← ConfigNamespace (dot-access)
 │
 └─ [legacy] factor_analysis.py  ← DEPRECATED (v3.3+)
    [legacy] backtest_demo.py    ← DEPRECATED (v3.3+)
```

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Entry Layer  (scripts/)                                                    │
│  run_analysis.py  ──────────────────────┐                                  │
│  run_batch.py ──────────────────────────┤                                  │
│                                         ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Config Layer  (config/)                                             │  │
│  │  default.yaml → load_config() → ConfigNamespace                     │  │
│  └──────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                           │
│  ┌──────────────────────────────▼───────────────────────────────────────┐  │
│  │  Pipeline Layer  (factor_framework/pipeline.py)                      │  │
│  │  FactorPipeline                                                      │  │
│  │   ├── register_builtins()                                            │  │
│  │   ├── run(factor_name, ...)  → FactorReport                         │  │
│  │   └── run_batch_from_panels(...) → {name: FactorReport}             │  │
│  └──────┬───────────────────────────────────────────┬───────────────────┘  │
│         │                                           │                       │
│  ┌──────▼──────────────┐                 ┌──────────▼───────────────────┐  │
│  │  Data Layer         │                 │  Compute Layer               │  │
│  │  data/store.py      │                 │  engine/panel_builder.py     │  │
│  │  CSVDataStore       │──── feeds ─────►│  PanelBuilder                │  │
│  │   └── get_price_    │                 │   ├── build(name, start, end)│  │
│  │       panel()       │                 │   └── engine: FactorEngine   │  │
│  │  → TimestampedPanel │                 │                              │  │
│  │    (semantic=price) │                 │  engine/cache.py             │  │
│  └─────────────────────┘                 │  CacheLayer                  │  │
│                                          │   ├── L1: memory dict        │  │
│                                          │   └── L2: Parquet disk       │  │
│                                          └──────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Core Layer  (core/)                                                 │  │
│  │  core/panel.py  → TimestampedPanel   (semantic + align_with guard)  │  │
│  │  core/returns.py → ReturnPanel        (forward return construction)  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Analysis Layer                                                      │  │
│  │  ic_analysis.py  → compute_ic(), ic_stats(), ic_decay()             │  │
│  │  backtest.py     → layer_backtest(), long_short_stats()             │  │
│  │  neutralize.py   → neutralize_regression()                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Factor Library  (factors/)                                          │  │
│  │  factors/momentum.py    — momentum_12_1, momentum_6_1, ...          │  │
│  │  factors/volatility.py  — vol_20d, vol_60d, vol_skew, ...           │  │
│  │  factors/value.py       — value_pb, value_pe_ttm, size_log_mktcap   │  │
│  │  factors/volume.py      — amihud_illiquidity, rsi_14, macd_signal   │  │
│  │  factors/registry.py    — REGISTRY (global catalogue)               │  │
│  │  factors/meta.py        — FactorMeta + FactorCategory               │  │
│  │  factors/transform.py   — TransformPipeline                         │  │
│  │  factors/ic_analyzer.py — ICAnalyzer                                │  │
│  │  factors/layer_backtester.py — LayerBacktester                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Operator / JIT Layer                                                │  │
│  │  operators.py  — cs_rank, cs_zscore, cs_winsorize, ts_mean, ...     │  │
│  │  jit_ops.py    — Numba JIT / Numexpr / pandas three-tier fallback   │  │
│  │  dag.py        — expression tree, CSE, DAGExecutor                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Compatibility Layer  [DEPRECATED — will be removed in v4.0]        │  │
│  │  factor_engine.py    ← use PanelBuilder instead                     │  │
│  │  factor_zoo.py       ← use factors/{category}.py instead            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Layout (v3.3)

```
Multi Factor/
│
├── config/                         # ← NEW (v3.3) Config layer
│   ├── default.yaml                #   Single config source of truth
│   ├── loader.py                   #   load_config(), ConfigNamespace
│   └── __init__.py
│
├── scripts/                        # ← Entry points (v3.3 main path)
│   ├── run_analysis.py             #   Single-factor CLI
│   ├── run_batch.py                #   Multi-factor batch CLI
│   └── run_validation.py           #   13-point DoD validation
│
├── factor_framework/               # Core library
│   ├── pipeline.py                 #   FactorPipeline (top-level orchestrator)
│   │
│   ├── core/                       # ← NEW (v3.2) Semantic type system
│   │   ├── panel.py                #   TimestampedPanel + alignment guards
│   │   └── returns.py              #   ReturnPanel
│   │
│   ├── engine/                     # ← NEW (v3.1) Compute + cache
│   │   ├── panel_builder.py        #   PanelBuilder (parallel + cache)
│   │   └── cache.py                #   CacheLayer (L1 memory + L2 Parquet)
│   │
│   ├── data/                       # ← NEW (v3.1) Data access
│   │   └── store.py                #   DataStore ABC + CSVDataStore
│   │
│   ├── factors/                    # ← NEW (v3.2) Factor sub-package
│   │   ├── meta.py                 #   FactorMeta + FactorCategory
│   │   ├── registry.py             #   REGISTRY global catalogue
│   │   ├── momentum.py             #   momentum_* factor functions
│   │   ├── volatility.py           #   vol_* factor functions
│   │   ├── value.py                #   value_*, size_* factor functions
│   │   ├── volume.py               #   amihud_*, rsi_*, macd_* factor functions
│   │   ├── transform.py            #   TransformPipeline
│   │   ├── ic_analyzer.py          #   ICAnalyzer
│   │   └── layer_backtester.py     #   LayerBacktester
│   │
│   ├── ic_analysis.py              #   compute_ic, ic_stats, ic_decay
│   ├── backtest.py                 #   layer_backtest, long_short_stats
│   ├── neutralize.py               #   neutralize_regression
│   ├── operators.py                #   cs_rank, ts_mean, ts_stddev, ...
│   ├── jit_ops.py                  #   Three-tier JIT acceleration
│   ├── dag.py                      #   Expression tree + CSE + DAGExecutor
│   ├── optimizer.py                #   equal_weight, icir_weight
│   │
│   ├── factor_engine.py            #   ⚠️ DEPRECATED — use PanelBuilder
│   └── factor_zoo.py               #   ⚠️ DEPRECATED — use factors/{cat}.py
│
├── tests/                          # ← NEW (v3.3) Consolidated test dir
│   ├── test_factor_framework.py    #   550 unit tests
│   ├── test_data_cleaner.py        #   data cleaner tests (+ real-data)
│   ├── test_data_quality.py        #   data quality checks (+ real-data)
│   ├── conftest.py
│   └── __init__.py
│
├── validation/                     # Path-consistency / bias validation
│   └── test_lookahead_bias.py      #   69 look-ahead bias tests
│
├── artifacts/                      # ← NEW (v3.3) Output artifacts (gitignored)
│
├── conftest.py                     # Root conftest (sys.path setup)
├── pyproject.toml                  # PEP 517 build + pytest config
├── MIGRATION.md                    # ← NEW (v3.3) Migration guide
├── ARCHITECTURE.md                 # ← NEW (v3.3) This file
│
├── factor_analysis.py              # ⚠️ DEPRECATED entry point
├── backtest_demo.py                # ⚠️ DEPRECATED demo script
│
└── stocks/stocks/                  # Raw OHLCV CSVs (one file per stock)
```

---

## Data-Flow for a Single Factor Run

```
scripts/run_analysis.py
    │
    │  1. load_config()          reads config/default.yaml (+ CLI overrides)
    │
    │  2. FactorPipeline(...)    constructs with CSVDataStore + PanelBuilder
    │
    │  3. pipe.register_builtins([factor_name])
    │         └─► REGISTRY.register(FactorMeta(...))
    │
    │  4. pipe.run(factor_name, forward=21, ...)
    │         │
    │         ├─ PanelBuilder.build(factor_name, start, end)
    │         │       ├─ CacheLayer.get(key)  ← L2 Parquet hit?
    │         │       ├─ FactorEngine.build_panel(...)   ← compute
    │         │       │       └─ DAGExecutor(expr_tree)
    │         │       │             └─ CSVDataStore.load_one(symbol)
    │         │       │                   └─ load_and_clean(csv_path)
    │         │       └─ CacheLayer.put(key, panel)  → .parquet
    │         │
    │         ├─ ReturnPanel.build(price_ts, forward_days=21)
    │         │
    │         ├─ factor_ts.shift_to_t1()          ← T+1 alignment
    │         │
    │         ├─ compute_ic(factor_t1, ret_ts)    ← alignment guard
    │         ├─ ic_decay(factor_t1, return_panels={...})
    │         ├─ layer_backtest(factor_t1, ret_ts, n_groups=5)
    │         └─ neutralize_regression(factor_t1, mktcap, industry)
    │
    │  5. report.print_summary()
    │     report.save(output_dir)
    ▼
  artifacts/  (or output/)
```

---

## Semantic Type System (v3.2+)

Every `TimestampedPanel` carries a `semantic` tag that the analysis functions
validate at call time:

| semantic | Created by | Valid pairing |
|----------|-----------|---------------|
| `"price"` | `CSVDataStore.get_price_panel()` | input to `ReturnPanel.build()` only |
| `"factor_observation"` | `PanelBuilder.build()` | must call `.shift_to_t1()` before analysis |
| `"forward_return"` | `ReturnPanel.build()` | pairs with `factor_observation` (T+1-shifted) |

Violations raise `TimingAlignmentError` or `SemanticCompatibilityError` immediately.

---

## Cache Architecture

```
CacheLayer
 ├── L1: in-memory dict  { key → DataFrame }   (per-process lifetime)
 └── L2: Parquet files   cache/<factor>/<md5>.parquet
           key = MD5( factor_name + start + end + sorted_symbols )
```

Policy:
- Write to L2 only if panel computation took ≥ `min_calc_secs` (default 5 s)
- Reads check L1 first, then L2, then compute
- `CacheLayer.clear_l2()` deletes all Parquet files
- `cache_dir=None` disables L2; L1 always active

---

## Testing Layout

| Suite | Location | Count | Scope |
|-------|----------|-------|-------|
| Unit + integration | `tests/test_factor_framework.py` | 550 | All `factor_framework` abstractions |
| Data cleaner | `tests/test_data_cleaner.py` | ~240 | `load_and_clean`, real-data sample |
| Data quality | `tests/test_data_quality.py` | ~240 | Tick-level quality checks |
| Look-ahead bias | `validation/test_lookahead_bias.py` | 69 | Path consistency, T+1 convention |

Run all:
```bash
python -m pytest               # uses testpaths from pyproject.toml
```

---

## Release Milestones

| Version | Milestone |
|---------|-----------|
| v3.0 | `factor_framework/` package, FactorEngine, IC analysis, layer backtest |
| v3.1 | DataStore, PanelBuilder, CacheLayer |
| v3.2 | Factor sub-package, TransformPipeline, ICAnalyzer, LayerBacktester |
| v3.3 | **config/**, **scripts/** main path, tests/ governance, deprecation markers |
| v4.0 | Remove `factor_analysis.py`, `backtest_demo.py`, `factor_zoo.py`, `price_panel=` arg |
