# Multi Factor — A-share Multi-Factor Research Framework

> **v3.3 · Phase 3 complete** · Python 3.10+ · 1027 tests passing

A production-grade quantitative research framework for Chinese A-share markets.
Build factor panels, run IC analysis, perform layer backtests, and assemble
multi-factor composites — all in a clean, extensible Python API.

---

## Quick Start (v3.3+)

```bash
# 1. Clone & install (editable)
git clone <repo-url>
cd "Multi Factor"
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# 2. Run a single factor analysis (new main path)
python scripts/run_analysis.py

# 3. Batch-test all 28 built-in factors
python scripts/run_batch.py

# 4. Preview the resolved config (no analysis run)
python scripts/run_analysis.py --show-config

# 5. Override parameters on the fly
python scripts/run_analysis.py --start 20180101 --end 20231231 --forward 21 --n-groups 5
```

> ⚠️ `python factor_analysis.py` and `python backtest_demo.py` still work but
> emit a `DeprecationWarning`. They will be removed in v4.0. See [MIGRATION.md](MIGRATION.md).

---

## Configuration (v3.3+)

All parameters live in `config/default.yaml`. There is no need to edit any
Python script to change analysis parameters.

```yaml
# config/default.yaml (excerpt)
data:
  stocks_dir: "stocks/stocks/"
  stock_basic: "股票列表-stock_basic.csv"

backtest:
  start: "20200101"
  end:   "20251231"
  forward: 21          # forward-return horizon (trading days)
  n_groups: 5
  direction: 1
  periods_per_year: 12
  rf: 0.02
  cost_per_side: 0.002
  resample_monthly: true

cache:
  cache_dir: "cache/"
  min_calc_secs: 5.0

output:
  base_dir: "artifacts/"
```

### Override priority

```
CLI flags  >  --config my_override.yaml  >  config/default.yaml
```

```bash
# Use a custom config file
python scripts/run_analysis.py --config research/exp1.yaml

# Override single values without a file
python scripts/run_analysis.py --forward 5 --no-cache
```

### Programmatic access

```python
from config import load_config

cfg = load_config()
print(cfg.backtest.forward)           # 21
print(cfg['backtest.n_groups'])       # 5  (dotted key also works)

# Merge a custom YAML + CLI-style overrides
cfg = load_config("my.yaml", overrides={"backtest.forward": 10})
```

---

## Project Layout

```
Multi Factor/
├── config/                     # ← NEW v3.3 — Config layer
│   ├── default.yaml            #   Single source of truth for all parameters
│   ├── loader.py               #   load_config(), ConfigNamespace (dot-access)
│   └── __init__.py
├── scripts/                    # ← v3.3 main path (use these, not root scripts)
│   ├── run_analysis.py         #   Single-factor CLI entry point
│   ├── run_batch.py            #   Multi-factor batch CLI
│   └── run_validation.py       #   13-point DoD validation
├── factor_framework/           # Core library
│   ├── core/                   #   TimestampedPanel, ReturnPanel
│   ├── engine/                 #   PanelBuilder, CacheLayer
│   ├── data/                   #   DataStore, CSVDataStore
│   ├── factors/                #   28 built-in factors by category + registry
│   ├── pipeline.py             #   FactorPipeline (end-to-end orchestrator)
│   ├── ic_analysis.py          #   compute_ic, ic_stats, ic_decay
│   ├── backtest.py             #   layer_backtest, long_short_stats
│   ├── neutralize.py           #   neutralize_regression
│   ├── operators.py            #   cs_rank, cs_zscore, ts_mean, ts_stddev, ...
│   ├── jit_ops.py              #   Three-tier JIT acceleration
│   ├── dag.py                  #   Expression tree + CSE + DAGExecutor
│   ├── optimizer.py            #   equal_weight, icir_weight
│   ├── factor_engine.py        #   ⚠️ DEPRECATED — use PanelBuilder
│   └── factor_zoo.py           #   ⚠️ DEPRECATED — use factors/{cat}.py
├── tests/                      # ← NEW v3.3 — Consolidated test directory
│   ├── test_factor_framework.py
│   ├── test_data_cleaner.py
│   ├── test_data_quality.py
│   └── conftest.py
├── validation/
│   └── test_lookahead_bias.py  # 69 look-ahead / path-consistency tests
├── artifacts/                  # Output artifacts (gitignored)
├── MIGRATION.md                # ← NEW v3.3 — Migration guide
├── ARCHITECTURE.md             # ← NEW v3.3 — System architecture
├── factor_analysis.py          # ⚠️ DEPRECATED entry point
├── backtest_demo.py            # ⚠️ DEPRECATED demo script
├── pyproject.toml              # Build config + pytest settings
└── stocks/stocks/              # Raw OHLCV CSVs (one file per stock)
```

---

## Built-in Factors (28)

### Momentum / Reversal (6)

| Name | Description |
|------|-------------|
| `momentum_12_1` | 12-month return skipping last month |
| `momentum_6_1` | 6-month return skipping last month |
| `momentum_1m` | 1-month return |
| `momentum_52w_high` | Price / 52-week high ratio |
| `reversal_1w` | 1-week return reversal |
| `reversal_1m` | 1-month return reversal |

### Volatility (4)

| Name | Description |
|------|-------------|
| `vol_20d` | 20-day rolling return std |
| `vol_60d` | 60-day rolling return std |
| `vol_skew` | 60-day return skewness |
| `downside_vol` | 60-day downside deviation |

### Value / Size (5)

| Name | Description |
|------|-------------|
| `value_pb` | Price-to-book reciprocal (B/P) |
| `value_pe_ttm` | Earnings-to-price (E/P TTM) |
| `value_ps_ttm` | Sales-to-price (S/P TTM) |
| `size_log_mktcap` | Log total market cap |
| `size_log_free_cap` | Log free-float market cap |

### Volume / Liquidity / Technical (13)

| Name | Description |
|------|-------------|
| `amihud_illiquidity` | Amihud (2002) illiquidity ratio |
| `turnover_rate` | 20-day average turnover rate |
| `vol_price_corr` | Volume-price correlation (20d) |
| `vwap_deviation` | Close / VWAP deviation |
| `price_strength` | Close relative to 20-day high-low range |
| `bid_ask_spread_proxy` | (High - Low) / Close proxy spread |
| `zero_return_ratio` | Proportion of zero-return days (60d) |
| `pastor_stambaugh` | Pastor-Stambaugh liquidity factor |
| `order_imbalance` | Volume-weighted order imbalance |
| `rsi_14` | 14-day RSI |
| `macd_signal` | MACD signal line |
| `bb_position` | Position within Bollinger Bands |
| `volume_trend` | 5-day vs 20-day volume ratio |

---

## API Reference

### TransformPipeline

```python
from factor_framework.factors.transform import TransformPipeline

tp = (TransformPipeline()
      .winsorize(n_std=3.0)
      .neutralize(mktcap_panel, industry_map)
      .standardize("rank"))          # or "zscore"

clean = tp.transform(raw_panel)
print(tp.step_names)  # ['winsorize', 'neutralize', 'standardize']
print(len(tp))        # 3

# Custom step
tp.register_step("log", lambda p: np.log1p(p.clip(lower=0)))
```

### ICAnalyzer

```python
from factor_framework.factors.ic_analyzer import ICAnalyzer

az = ICAnalyzer(
    factor_panel,
    return_panel,
    return_panels={1: rp1, 5: rp5, 21: rp21},
    method="rank",
    periods_per_year=12,
).run()

az.print_summary("momentum_12_1")
az.ic_stats_dict   # mean_ic, std_ic, icir, win_rate, t_stat, ...
az.ic_nw           # nw_t_stat, nw_p_value
az.decay_df        # IC decay DataFrame
```

### LayerBacktester

```python
from factor_framework.factors.layer_backtester import LayerBacktester

bt = LayerBacktester(
    factor_panel, return_panel,
    n_groups=5, direction=1,
    periods_per_year=12, cost_per_side=0.002,
).run()

bt.print_summary("momentum_12_1")
bt.ls_stats    # ls_annual_return, ls_sharpe, ls_max_drawdown, ...
bt.turnover    # avg_turnover, avg_cost
bt.nav         # cumulative NAV series
```

### FactorPipeline

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline(
    stocks_dir  = "stocks/stocks/",
    stock_basic = "股票列表-stock_basic.csv",
    cache_dir   = "cache/",
)
pipe.register_builtins()

report = pipe.run(
    factor_name="momentum_12_1",
    start="20180101", end="20231231",
    forward=21, n_groups=5,
    neutralize=True, standardize="rank",
    resample_monthly=True, periods_per_year=12,
)
report.print_summary()
report.save("output/")

# Multi-factor composite
report = pipe.run_composite(
    ["momentum_12_1", "value_pb", "vol_20d"],
    method="icir", composite_name="composite_v1",
)
```

### FactorRegistry

```python
from factor_framework.factors.registry import REGISTRY
from factor_framework.factors.meta import FactorMeta, FactorCategory

REGISTRY.register(FactorMeta(
    name="my_factor",
    category=FactorCategory.MOMENTUM,
    fn=lambda df: df["close"].pct_change(21),
    doc="21-day price change",
))

meta = REGISTRY["my_factor"]
moms = REGISTRY.by_category(FactorCategory.MOMENTUM)
```

---

## Three-Tier JIT Acceleration

| Tier | Backend | Operators | Speedup |
|------|---------|-----------|---------|
| 1 | Numba JIT | `ts_mean`, `ts_sum`, `ts_rank` | 5-10x vs pandas |
| 2 | Numexpr | `log`, `sqrt`, `power` | 2-4x vs NumPy |
| 3 | pandas/NumPy | all others | baseline |

Graceful degradation when Numba/Numexpr are unavailable.

```python
from factor_framework.jit_ops import warmup
warmup()   # call at startup to pre-compile JIT kernels
```

---

## Testing

```bash
# Full test suite (uses testpaths from pyproject.toml)
python -m pytest

# Specific suite
python -m pytest tests/test_factor_framework.py -q

# Category filter
python -m pytest -k "TransformPipeline or ICAnalyzer or LayerBacktester" -v

# Look-ahead bias only
python -m pytest validation/test_lookahead_bias.py -v
```

| Suite | Location | Tests |
|-------|----------|-------|
| Framework unit + integration | `tests/test_factor_framework.py` | ~550 |
| Data cleaner | `tests/test_data_cleaner.py` | ~243 |
| Data quality | `tests/test_data_quality.py` | ~243 |
| Look-ahead bias | `validation/test_lookahead_bias.py` | 69 |
| **Total** | | **~1027** |

---

## Phase 2 Architecture — Main Path

> v3.3 wires all Phase 2 abstractions together end-to-end.

```
FactorPipeline
│
├── DataStore (CSVDataStore)      ← data access layer
│     └── get_price_panel()       → TimestampedPanel(semantic='price')
│
├── PanelBuilder                  ← computation + cache coordinator
│     ├── store: DataStore
│     ├── cache: CacheLayer (L1 memory + L2 Parquet)
│     └── engine: FactorEngine (_internal=True, no deprecation warn)
│
├── compute_ic(factor, return)    ← B1 guard: align_with() if TimestampedPanel
│     └── TimingAlignmentError / SemanticCompatibilityError on bad semantics
│
├── layer_backtest(factor, return) ← B1 guard: same as above
│
└── ic_decay(return_panels=...)   ← B3: main path (no DeprecationWarning)
      └── ic_decay(price_panel=...) raises DeprecationWarning (legacy)
```

**Recommended usage (v3.3+):**

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline(
    stocks_dir  = "stocks/stocks/",
    stock_basic = "股票列表-stock_basic.csv",
    # cache_dir defaults to "cache/" — L2 Parquet cache auto-enabled
    # store defaults to CSVDataStore(stocks_dir) — auto-constructed
)
pipe.register_builtins(["momentum_12_1"])
report = pipe.run("momentum_12_1", forward=21)
report.print_summary()
```

---

## Caching

### Default behavior (v3.3+)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cache_dir` | `"cache/"` | L2 Parquet cache **enabled by default** |
| `min_calc_secs` | `5.0` | Only cache panels that took > 5 s to compute |

The cache is stored under `cache/<factor_name>/<key>.parquet`. Keys are MD5 hashes of `(factor_name, start, end, sorted_symbols)`, so any change in inputs produces a new key.

### Disabling the cache

```python
pipe = FactorPipeline(..., cache_dir=None)   # L2 disabled; L1 still active
```

### Clearing the cache

```python
from factor_framework.engine.cache import CacheLayer
cache = CacheLayer(cache_dir="cache/", stocks_dir="stocks/stocks/")
cache.clear_l2()          # delete all Parquet files
cache.clear_l1()          # release memory
print(cache.cache_info()) # {'l1_entries': 0, ...}
```

---

## Time-Semantic Conventions

All `TimestampedPanel` instances carry a `semantic` attribute that encodes the
meaning of the data and enforces alignment rules at runtime:

| semantic | Description | Key constraints |
|----------|-------------|-----------------|
| `"price"` | Raw/adjusted closing prices | Cannot align with factor or return panels |
| `"factor_observation"` | Factor values observed at market close on date t | Must call `.shift_to_t1()` before aligning with `forward_return` |
| `"forward_return"` | Forward return starting at t (or t+1 if T+1-shifted) | Must originate from `ReturnPanel.build()` |

### T+1 shift convention

`factor_observation` panels must be shifted one day forward before pairing with
`forward_return` panels, to prevent look-ahead bias:

```python
from factor_framework.core.panel import TimestampedPanel
from factor_framework.core.returns import ReturnPanel
from factor_framework.ic_analysis import compute_ic

factor_ts = TimestampedPanel.from_dataframe(factor_df, semantic="factor_observation", factor_name="mom")
factor_t1 = factor_ts.shift_to_t1()        # shift factor by 1 day (use t's factor to predict t+1 return)

ret_ts = ReturnPanel.build(price_ts, forward_days=21)   # forward_return panel (no T+1 here)

ic = compute_ic(factor_t1, ret_ts)         # align_with() validates semantics automatically
```

If you pass an un-shifted `factor_observation` against a `forward_return`, `compute_ic` and
`layer_backtest` will raise `TimingAlignmentError` immediately.

### DoD validation

```bash
python scripts/run_validation.py   # 13/13 checks, exits 0 on pass
```

---

## Release History

### v3.3 — Phase 3: Reorganisation (current)
- **Config layer**: `config/default.yaml` + `config/loader.py` — single source of truth for all parameters
- **Main-path convergence**: `scripts/run_analysis.py` and `scripts/run_batch.py` are real CLI entry points consuming `config/default.yaml`
- **Directory governance**: `tests/` directory consolidated; `pyproject.toml` testpaths updated; `artifacts/` output dir
- **Deprecation markers**: `factor_analysis.py`, `backtest_demo.py`, `factor_engine.py`, `factor_zoo.py` all emit `DeprecationWarning` at runtime
- **Docs**: `MIGRATION.md`, `ARCHITECTURE.md`, README updated
- **+408 new tests → 1027 total**

### v3.3 — Phase 2 DoD
- **B1** `compute_ic` / `layer_backtest` wire `align_with()` semantic guard for `TimestampedPanel`
- **B2** `PanelBuilder(store=...)` accepts `DataStore`; `FactorPipeline` auto-constructs `CSVDataStore`
- **B3** `ic_decay(price_panel=...)` legacy path emits `DeprecationWarning`
- **B4** `cache_dir` default changed from `None` to `"cache/"` — L2 cache enabled out-of-the-box
- **C1** `FactorEngine` direct instantiation emits one-time `DeprecationWarning`; `PanelBuilder` passes `_internal=True` to suppress it
- `scripts/run_validation.py` — 13-point DoD validation script
- **+31 new tests → 619 total**

### v3.2 — Phase 3
- Factor sub-package with category files (`momentum`, `volatility`, `value`, `volume`)
- `TransformPipeline` — composable cross-section transform pipeline
- `ICAnalyzer` — structured IC analysis with Newey-West t-stats and IC decay
- `LayerBacktester` — layer backtest wrapper with long-short stats
- `FactorRegistry` + `FactorMeta` + `FactorCategory`
- `scripts/`, `analysis/`, `validation/` directory structure
- `pyproject.toml` with PEP 517 build config
- **+34 new tests → 588 total**

### v3.1 — Phase 2
- `DataStore`: unified data access layer
- `PanelBuilder`: parallel panel construction
- `CacheLayer`: L1 memory + L2 Parquet two-tier cache
- `FactorPipeline.cache_dir` parameter
- 554 tests

### v3.0 — Phase 1
- `factor_framework/` package extracted from monolithic script
- `FactorEngine`, IC analysis, layer backtest, neutralization, optimizer
- Three-tier JIT acceleration

---

## License

MIT

---

## Further Reading

- **[MIGRATION.md](MIGRATION.md)** — Step-by-step guide for migrating from v3.x to v3.3+: entry-point changes, API mapping table, common errors
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Full system architecture diagram, layer descriptions, data-flow walkthrough, cache design
