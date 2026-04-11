# Changelog

All notable changes to **Multi Factor** are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · Versioning: [SemVer](https://semver.org/).

---

## [4.0.0rc1] — 2026-04-11

### Summary
v4.0 is a full architectural upgrade. Core capabilities (factor compute, IC analysis,
layered backtest, composite optimisation, semantic look-ahead guard, cache acceleration)
are **preserved and backwards-compatible**. New capabilities are additive. All v3 entry-points
continue to work in v4.0 / v4.1, emitting `DeprecationWarning`. Planned removal: **v4.2**.

---

### Phase A — Directory convergence (commit `1462441`)
- New canonical sub-packages: `analytics/`, `transform/`, `optimize/`, `reporting/` (stub), `cli/`.
- Compatibility shims at all old paths emit `DeprecationWarning` (removal planned v4.2).
- Test directory restructured: `tests/unit/`, `tests/integration/`, `tests/regression/`, `tests/slow/`.
- `tests/regression/baseline_v1_provisional.json` — IC/Sharpe provisional baseline (27 factors, pending data-snapshot lock).
- `ARCHITECTURE.md` updated to v4.0 layer diagram.

### Phase B — Unified CLI (commit `7c5ef45`)
- **`mf` CLI** (`factor_framework/cli/main.py`): subcommands `single`, `batch`, `validate`, `cache`, `report` (stub).
- `pip install -e .` registers the `mf` console script; `python -m factor_framework.cli` also works.
- Exit-code contract: `0` success · `1` runtime error · `2` argument error.
- `docs/cli-contract.md` — full CLI contract documentation.
- 29 CLI acceptance tests (`tests/unit/test_cli.py`).

### Phase D — Manifest + cache key v2 (commit `9118cad`)
- **`factor_framework/manifest.py`** — `RunManifest` dataclass (14 fields: `run_id`, `timestamp`,
  `config_hash`, `git_sha`, `data_snapshot_id`, `pipeline_version`, `factors`, `date_range`,
  `cache_stats`, `exit_status`, `elapsed_s`, `environment`, `key_metrics`, `schema_version`).
- `pipe.run()` populates `pipe.last_manifest`; `mf single` / `mf batch` write `run_manifest.json`.
- **`CacheKeyV2`**: adds `transform_config_hash`, `semantic_contract_version`, `git_sha` dimensions.
- **Legacy fallback**: `get_panel()` tries v2 key first, falls back to v1 — existing caches still usable.
- `CacheLayer.stats()` / `reset_stats()` — cache hit-rate observability.
- 24 manifest + cache tests (`tests/unit/test_manifest.py`).

### Phase C — ResearchConfig (commit `7b3d042`)
- **`factor_framework/research_config.py`** — `ResearchConfig` dataclass (14 fields, `schema_version="1.0"`).
- `ResearchConfig.from_kwargs()` — convert legacy scatter-kwargs to config object.
- `ResearchConfig.from_dict()` + `upgrade_config()` — version-aware deserialise with automatic migration.
- `ResearchConfig.to_stable_dict()` — deterministic hash base (sorted keys, ephemeral fields excluded).
- `pipe.run(config=rc)` — single source of truth for all pipeline parameters; old kwargs auto-convert.
- `manifest._config_hash()` uses `to_stable_dict()` — symbols list no longer pollutes config hash.
- 33 ResearchConfig tests (`tests/unit/test_research_config.py`).

### Phase E1 — FactorMeta extension (commit `3beb975`)
- **`FactorMeta`** extended with 6 new optional fields: `inputs`, `output_semantic`, `forward_safe`,
  `version`, `tags`, `status` (all with safe defaults; fully backward-compatible).
- **`FactorStatus`** enum: `ACTIVE` · `EXPERIMENTAL` · `DEPRECATED`.
- `FactorMeta.missing_e1_fields` — lists unfilled E1 fields; `is_active` property.
- **`FactorRegistry.audit()`** → `AuditReport`: per-factor completeness, `completeness_pct`,
  `print_report()`, `to_df()`.
- `FactorRegistry.register()` emits `UserWarning` on missing E1 fields (warning-only; never blocks).
- `FactorRegistry.list_active()` — filters to `status == ACTIVE`.
- **All 28 built-in factors** fully populated: `inputs`, `output_semantic`, `forward_safe=True`,
  `version="2.9.1"`, `tags`, `status=ACTIVE`.
- 36 E1 tests (`tests/unit/test_phase_e1.py`).

### Phase E2 — Deprecation governance (commit TBD → this rc1 commit)
- **`deprecations.yaml`** — canonical registry of all deprecated interfaces (11 entries):
  `id`, `kind`, `target`, `replacement`, `warn_since`, `remove_in`, `migration_note`, `warn_template`.
- All 6 module shims already emit `DeprecationWarning` with `v4.2` removal notice; text confirmed
  consistent with `deprecations.yaml`.
- `tests/unit/test_deprecations.py` — validates YAML structure, version format, no duplicates,
  and that each shim emits `DeprecationWarning` mentioning the new path and `v4.2`.

---

### Fixed (across all phases)
- `ic_decay()` cache-key collision: different `fwd` values with equal `len(valid_ret_rows)` shared
  the wrong `common_stocks` set → `KeyError`. Fixed by keying on `fwd` directly.
- `apply_cross_section` chunked processing: `np.nanmedian` on large panels exhausted memory on
  Windows; replaced with 300-row chunk loop using `np.partition`-based MAD.
- `ic_decay` rank-IC: factor panel now ranked once per `fwd` (not per `(fwd, method)`), halving peak memory.

### Security / Reproducibility
- `ResearchConfig.validate()` rejects invalid `forward_days`, `n_groups`, negative costs.
- `CacheKeyV2` includes `git_sha` — prevents silent cache poisoning across code versions.
- `run_manifest.json` records `data_snapshot_id` (path + mtime digest) — implicit data version made explicit.

---

### Deprecated (all removed in v4.2)
| Interface | Replacement | Deprecated since |
|-----------|-------------|-----------------|
| `factor_analysis.py` (root script) | `mf single --factor NAME` | v3.3 |
| `backtest_demo.py` (root script) | `mf single --factor NAME` | v3.3 |
| `scripts/run_analysis.py` | `mf single` | v4.0 |
| `scripts/run_batch.py` | `mf batch` | v4.0 |
| `scripts/run_factor_screening.py` | `mf batch` | v4.0 |
| `factor_framework.ic_analysis` | `factor_framework.analytics.ic_analysis` | v4.0 |
| `factor_framework.neutralize` | `factor_framework.transform.neutralize` | v4.0 |
| `factor_framework.optimizer` | `factor_framework.optimize.optimizer` | v4.0 |
| `factor_framework.factors.transform` | `factor_framework.transform` | v4.0 |
| `factor_framework.factors.ic_analyzer` | `factor_framework.analytics.ic_analyzer` | v4.0 |
| `factor_framework.factors.layer_backtester` | `factor_framework.analytics.layer_backtester` | v4.0 |

---

## [3.3.0] — 2026-03-xx

### Added
- `scripts/run_analysis.py` — CLI entry point consuming `config/default.yaml`.
- `config/` layer: `default.yaml` + `loader.py` (`ConfigNamespace`, 3-level priority merge).
- `tests/` directory: all test files migrated from root; `conftest.py` at root and `tests/`.
- `MIGRATION.md`, `ARCHITECTURE.md` (v3.3).
- `DeprecationWarning` on `factor_analysis.py` and `backtest_demo.py`.

### Changed
- `pyproject.toml`: `3.2.x` → `3.3.0`; `testpaths` updated.

---

## [3.2.0] — 2026-02-xx

### Added
- `factor_framework/factors/` sub-package: `FactorMeta`, `FactorRegistry`, `TransformPipeline`,
  `ICAnalyzer`, `LayerBacktester`.
- `factor_framework/core/panel.py`: `TimestampedPanel` semantic contract guard.
- `CacheLayer` L1/L2 design.

---

## [3.1.0] — earlier

Initial public factor library with 28 built-in factors, IC analysis,
layered backtest, and composite optimisation.
