content = """\
# Multi Factor — A-share Multi-Factor Research Framework

> **v4.0.0rc1** · Python 3.10+ · 239 tests passing · Phase A/B/C/D/E1/E2 complete

A production-grade quantitative research framework for Chinese A-share markets.
Build factor panels, run IC analysis, perform layer backtests, and assemble
multi-factor composites — all through a single unified CLI or clean Python API.

---

## Quick Start

```bash
# 1. Clone & install (editable)
git clone <repo-url>
cd "Multi Factor"
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[dev]"

# 2. Single factor: IC analysis + layered backtest
mf single --factor momentum_12_1

# 3. Multiple factors
mf single --factor momentum_12_1 value_pb size_log_mktcap

# 4. Full batch (all 28 built-in factors)
mf batch

# 5. Validation suites
mf validate --suite lookahead
```

Also: `python -m factor_framework.cli single --factor momentum_12_1`

> ⚠️  Deprecated entry-points (`factor_analysis.py`, `scripts/run_*.py`) still work
> but emit `DeprecationWarning` — **removed in v4.2**.
> See [MIGRATION_v3_to_v4.md](MIGRATION_v3_to_v4.md).

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `mf single --factor NAME [NAME ...]` | IC + layered backtest for one or more factors |
| `mf batch [--factors NAME ...]` | Run all 28 factors or a named subset |
| `mf validate --suite {lookahead\\|quality\\|all}` | Run validation suites |
| `mf cache {info\\|gc\\|clear}` | Cache management |

Exit codes: **0** success · **1** runtime error · **2** argument error

---

## Programmatic API

### ResearchConfig (v4.0)

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
)
pipe = FactorPipeline("stocks/stocks/")
pipe.register_builtins()
results = pipe.run(config=rc)
print(rc.config_hash())   # deterministic 8-char hex
```

### FactorRegistry audit (E1)

```python
from factor_framework.factors.registry import REGISTRY
report = REGISTRY.audit()
print(f"Complete: {report.completeness_pct:.0f}%")  # 100% for all 28 built-ins
df = report.to_df()
```

### RunManifest

```python
from factor_framework.manifest import RunManifest
m = RunManifest.load("artifacts/run_manifest.json")
m.print_summary()   # run_id, config_hash, git_sha, cache_stats, key_metrics
```

---

## Built-in Factors (28)

| Category | Factors |
|----------|---------|
| Momentum / Reversal | momentum_12_1, momentum_6_1, momentum_1m, momentum_52w_high, reversal_1w, reversal_1m |
| Volatility | vol_20d, vol_60d, vol_skew, downside_vol |
| Value / Size | value_pb, value_pe_ttm, value_ps_ttm, size_log_mktcap, size_log_free_cap |
| Liquidity | amihud_illiquidity, turnover_rate, vol_price_corr, vwap_deviation, bid_ask_spread_proxy, zero_return_ratio, pastor_stambaugh, order_imbalance |
| Technical | price_strength, rsi_14, macd_signal, bb_position, volume_trend |

All 28 carry full E1 metadata: `inputs`, `output_semantic`, `forward_safe=True`, `version="2.9.1"`.

---

## Project Layout

```
Multi Factor/
├── factor_framework/
│   ├── analytics/          ← canonical: IC analysis, ICAnalyzer, LayerBacktester
│   ├── transform/          ← canonical: TransformPipeline, neutralize
│   ├── optimize/           ← canonical: equal_weight, icir_weight
│   ├── cli/                ← unified mf CLI
│   ├── factors/            ← FactorMeta, FactorRegistry, 28 built-ins
│   ├── pipeline.py         ← FactorPipeline
│   ├── manifest.py         ← RunManifest
│   ├── research_config.py  ← ResearchConfig
│   ├── ic_analysis.py      ← ⚠️ DEPRECATED shim (→ analytics.ic_analysis)
│   ├── neutralize.py       ← ⚠️ DEPRECATED shim (→ transform.neutralize)
│   └── optimizer.py        ← ⚠️ DEPRECATED shim (→ optimize.optimizer)
├── tests/unit/             ← 239 fast unit tests
├── validation/             ← look-ahead bias tests
├── CHANGELOG.md
├── MIGRATION_v3_to_v4.md
├── deprecations.yaml       ← E2 canonical deprecation registry
└── pyproject.toml
```

---

## Testing

```bash
pytest tests/unit/           # 239 tests, ~3 s
pytest tests/                # full suite
pytest tests/ -m "not slow"  # skip large-sample tests
pytest validation/ -v        # look-ahead bias
```

| Suite | Count |
|-------|-------|
| test_cli.py | 29 |
| test_manifest.py | 24 |
| test_research_config.py | 33 |
| test_phase_e1.py | 36 |
| test_deprecations.py | 92 |
| other | 25 |
| **Total** | **239** |

---

## Deprecation Policy

| Version | Action |
|---------|--------|
| **v4.0 (current)** | Old paths work; emit `DeprecationWarning` |
| **v4.1** | E1 meta fields enforced (strict gate) |
| **v4.2** | Old entry-points and import shims **removed** |

Full list: [deprecations.yaml](deprecations.yaml) · Migration: [MIGRATION_v3_to_v4.md](MIGRATION_v3_to_v4.md)

---

## Release History

See [CHANGELOG.md](CHANGELOG.md) for full per-phase details.

| Version | Highlights |
|---------|-----------|
| **v4.0.0rc1** | Unified `mf` CLI, ResearchConfig, RunManifest, cache key v2, FactorMeta E1 fields, registry.audit(), deprecation governance |
| v3.3.0 | Config layer, scripts/ entry points, 1027 tests |
| v3.2.0 | Factor sub-package, TransformPipeline, ICAnalyzer, LayerBacktester |

---

## License

MIT
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)
print("OK")
