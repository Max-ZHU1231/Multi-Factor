# Architecture — Multi-Factor Research Framework (v3.3)

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
