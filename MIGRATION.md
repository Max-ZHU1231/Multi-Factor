# Migration Guide — v3.x → v3.3+

This document maps **old interfaces** to their **v3.3 replacements** and lists
common pitfalls encountered during migration.

---

## 1. Entry-Point Migration

### Running a single-factor analysis

| Old (< v3.3) | New (v3.3+) |
|---|---|
| `python factor_analysis.py` | `python scripts/run_analysis.py` |
| `python backtest_demo.py` | `python scripts/run_batch.py` |
| Hard-coded `CFG = dict(...)` inside scripts | `config/default.yaml` + `--config my.yaml` |

**Before:**
```bash
python factor_analysis.py
```

**After:**
```bash
# Use defaults from config/default.yaml
python scripts/run_analysis.py

# Override individual parameters on the command line
python scripts/run_analysis.py --start 20180101 --end 20231231 --forward 21 --n-groups 5

# Use a custom YAML config
python scripts/run_analysis.py --config my_config.yaml

# Preview resolved config (no analysis run)
python scripts/run_analysis.py --show-config
```

### Running a batch of factors

| Old | New |
|---|---|
| `python backtest_demo.py` | `python scripts/run_batch.py` |
| `python scripts/run_batch.py --cache-dir cache/` | `python scripts/run_batch.py` (cache_dir now in `config/default.yaml`) |

---

## 2. Data-Access Migration

### Loading stock data

| Old | New |
|---|---|
| `from data_cleaner import load_and_clean` (called directly) | `from factor_framework.data.store import CSVDataStore` |
| `load_and_clean(path)` → raw DataFrame | `store.get_price_panel(start, end)` → `TimestampedPanel` |

**Before:**
```python
from data_cleaner import load_and_clean
df = load_and_clean("stocks/stocks/000001_SZ.csv")
```

**After:**
```python
from factor_framework.data.store import CSVDataStore
store = CSVDataStore(stocks_dir="stocks/stocks/", stock_basic="股票列表-stock_basic.csv")
price_panel = store.get_price_panel(start="20200101", end="20251231")
# Returns TimestampedPanel(semantic="price")
```

---

## 3. Factor Computation Migration

### Using FactorEngine directly

| Old | New |
|---|---|
| `from factor_framework.factor_engine import FactorEngine` | `from factor_framework.engine.panel_builder import PanelBuilder` |
| `engine = FactorEngine(stocks_dir=...)` | `builder = PanelBuilder(store=store, cache_dir="cache/")` |
| `engine.build_panel(factor_name, start, end)` | `builder.build(factor_name, start, end)` |

**Before:**
```python
from factor_framework.factor_engine import FactorEngine

engine = FactorEngine(stocks_dir="stocks/stocks/")
engine.register("vol_20d", lambda df: df["收盘价"].pct_change().rolling(20).std())
panel = engine.build_panel("vol_20d", start="20200101", end="20251231")
```

**After:**
```python
from factor_framework.data.store import CSVDataStore
from factor_framework.engine.panel_builder import PanelBuilder
from factor_framework.factors.registry import REGISTRY

store   = CSVDataStore(stocks_dir="stocks/stocks/", stock_basic="股票列表-stock_basic.csv")
builder = PanelBuilder(store=store, cache_dir="cache/")

panel = builder.build("vol_20d", start="20200101", end="20251231")
# Returns TimestampedPanel(semantic="factor_observation")
```

### Using the full pipeline (recommended)

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline(
    stocks_dir  = "stocks/stocks/",
    stock_basic = "股票列表-stock_basic.csv",
)
pipe.register_builtins(["momentum_12_1", "vol_20d", "value_pb"])
report = pipe.run("momentum_12_1", forward=21)
report.print_summary()
```

---

## 4. Factor-Definition Migration

### Importing built-in factor functions

| Old | New |
|---|---|
| `from factor_framework.factor_zoo import momentum_12_1` | `from factor_framework.factors.momentum import momentum_12_1` |
| `from factor_framework.factor_zoo import vol_20d` | `from factor_framework.factors.volatility import vol_20d` |
| `from factor_framework.factor_zoo import value_pb` | `from factor_framework.factors.value import value_pb` |
| `from factor_framework.factor_zoo import amihud_illiquidity` | `from factor_framework.factors.volume import amihud_illiquidity` |

The old `factor_zoo` module still works as a compatibility shim but will be
removed in **v4.0**.

---

## 5. IC Analysis Migration

| Old | New |
|---|---|
| `ic_decay(price_panel=close_panel, ...)` | `ic_decay(return_panels={1: rp1, 5: rp5, 21: rp21})` |
| `compute_ic(factor_df, return_df)` (raw DataFrames) | `compute_ic(factor_ts.shift_to_t1(), return_ts)` |

The old `price_panel=` argument to `ic_decay()` emits a `DeprecationWarning`
in v3.3 and will be removed in v4.0.

**Before:**
```python
from factor_framework.ic_analysis import compute_ic, ic_decay

ic = compute_ic(factor_panel, return_panel)
decay = ic_decay(factor_panel, close_panel, forward_list=[1, 5, 21])
```

**After:**
```python
from factor_framework.ic_analysis import compute_ic, ic_decay
from factor_framework.core.panel import TimestampedPanel
from factor_framework.core.returns import ReturnPanel

factor_ts = TimestampedPanel.from_dataframe(factor_df, semantic="factor_observation", factor_name="mom")
factor_t1 = factor_ts.shift_to_t1()     # T+1 shift — mandatory

price_ts  = TimestampedPanel.from_dataframe(price_df, semantic="price")
ret_ts    = ReturnPanel.build(price_ts, forward_days=21)

ic = compute_ic(factor_t1, ret_ts)

# IC decay with pre-built return panels (no price_panel= arg)
return_panels = {
    1:  ReturnPanel.build(price_ts, forward_days=1),
    5:  ReturnPanel.build(price_ts, forward_days=5),
    21: ReturnPanel.build(price_ts, forward_days=21),
}
decay = ic_decay(factor_t1, return_panels=return_panels)
```

---

## 6. Config Migration

### Centralising hard-coded parameters

v3.3 introduces `config/default.yaml` as the single source of truth.
All `CFG = dict(...)` blocks in old scripts should be replaced.

**Before (in any script):**
```python
CFG = dict(
    stocks_dir       = "Stocks/",
    start            = "20200101",
    end              = "20251231",
    forward          = 21,
    n_groups         = 5,
    ...
)
```

**After:**
```python
from config import load_config

cfg = load_config()                         # reads config/default.yaml
# or:
cfg = load_config("my_override.yaml",       # merge a custom YAML on top
                  overrides={"backtest.forward": 10})

stocks_dir = cfg.data.stocks_dir
forward    = cfg.backtest.forward
```

CLI equivalent (no Python change needed):
```bash
python scripts/run_analysis.py --forward 10 --start 20180101
```

### Three-level override priority

```
CLI flags  >  user_config YAML  >  config/default.yaml
```

---

## 7. Common Migration Errors

### Error: `TimingAlignmentError`

**Symptom:**
```
factor_framework.core.panel.TimingAlignmentError:
  factor_observation panel must be shifted to T+1 before aligning with forward_return
```

**Cause:** You passed a raw `factor_observation` panel directly to `compute_ic`
or `layer_backtest` without calling `.shift_to_t1()`.

**Fix:**
```python
factor_t1 = factor_ts.shift_to_t1()   # ← add this
ic = compute_ic(factor_t1, ret_ts)
```

### Error: `SemanticCompatibilityError`

**Symptom:**
```
factor_framework.core.panel.SemanticCompatibilityError:
  Cannot align panels with semantics 'price' and 'forward_return'
```

**Cause:** You tried to align a price panel with a return panel.
`compute_ic` and `layer_backtest` accept only `factor_observation`/`forward_return` pairs.

**Fix:** Build a `ReturnPanel` from your price panel first:
```python
ret_ts = ReturnPanel.build(price_ts, forward_days=21)
```

### Error: `DeprecationWarning: ic_decay(price_panel=...)`

**Cause:** You're using the legacy `price_panel=` signature.

**Fix:** Pre-build return panels for each horizon (see §5 above).

### Error: `DeprecationWarning` from `FactorEngine` direct construction

**Cause:** `FactorEngine(stocks_dir=...)` emits a one-time deprecation warning.

**Fix:** Use `PanelBuilder` instead, or suppress with:
```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    engine = FactorEngine(stocks_dir=..., _internal=True)
```

---

## 8. Versioned Retirement Plan

| Version | Action |
|---------|--------|
| **v3.3** (current) | `factor_analysis.py`, `backtest_demo.py`, `factor_engine.py`, `factor_zoo.py` emit `DeprecationWarning`; full backward compatibility maintained |
| **v3.x** | Compatibility layer preserved; new features added to `scripts/` + `factor_framework/factors/` only |
| **v4.0** (planned) | Remove `factor_analysis.py`, `backtest_demo.py` entry points; remove `price_panel=` argument from `ic_decay()`; remove `factor_zoo.py` shim |

---

## 9. Quick Reference

```
Old                             →  New
──────────────────────────────────────────────────────────────────────────
python factor_analysis.py       →  python scripts/run_analysis.py
python backtest_demo.py         →  python scripts/run_batch.py
factor_engine.FactorEngine      →  engine.panel_builder.PanelBuilder
factor_zoo.*                    →  factors.{momentum,volatility,value,volume}.*
ic_decay(price_panel=p)         →  ic_decay(return_panels={1:rp1, 21:rp21})
compute_ic(factor_df, ret_df)   →  compute_ic(factor_ts.shift_to_t1(), ret_ts)
CFG = dict(start=..., end=...)  →  config/default.yaml + load_config()
```
