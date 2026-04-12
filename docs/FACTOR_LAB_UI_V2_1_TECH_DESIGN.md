# Factor Lab UI v2.1 — Technical Design

## 0. Document Version and Goal
### 0.1 Version Info (v2.1)
- Version: `2.1`
- Status: Draft for implementation
- Scope: UI shell and service layer on top of existing `factor_framework`

### 0.2 Why Reorganized
- Prioritize reuse of existing pipeline, diagnostics, CLI, and config chain.
- Minimize invasive changes and avoid duplicate computation logic.

### 0.3 Goals and Non-Goals
- **Goals:** readable UI, async execution, contract-first rendering, reproducible runs.
- **Non-goals:** rewriting backtest core, changing existing output paths, breaking old CLI behavior.

### 0.4 Glossary
- `FactorReport`: single-factor result container from pipeline run.
- `AdvancedDiagnostics`: advanced statistical diagnostics package.
- `PIT`: point-in-time membership/data policy.
- `OOS`: out-of-sample evaluation windows.

## 1. Current Baseline (As-Is)
### 1.1 Existing Capabilities
- Pipeline, advanced diagnostics, CLI commands, and config override chain are available.

### 1.2 Stable Output Contracts
- Base outputs under `output/<factor>/`.
- Advanced outputs under `output/<factor>/advanced_diagnostics/`.

### 1.3 Known Gaps
- PIT components for `ZZ500/ZZ1000` need full data source coverage.
- Some advanced modules need paper-grade edge-case hardening.

### 1.4 Constraints
- Minimal intrusion, backward compatible, reversible by config flags.

## 2. Target Architecture (To-Be)
### 2.1 Architecture
- UI shell + backend service + reused `factor_framework` execution engine.

### 2.2 Boundaries
- UI: rendering and user interaction.
- Backend service: task orchestration, IO normalization, progress and status.
- Framework: compute-only, contract outputs, deterministic behavior.

### 2.3 Data Flow
- Browse -> run/inspect -> diagnose -> optional registry/admission.

### 2.4 Principles
- Read-mostly path first.
- Contract-first integration.
- Graceful degradation on missing data.

## 3. Reuse Mapping (Core)
### 3.1 UI-to-Capability Mapping
- Dashboard -> aggregated summaries.
- Factor detail -> `summary.csv`, `ic_series.csv`, `nav.csv`, advanced outputs.
- Relationship page -> `factor_corr_matrix.csv` (+ optional wide matrix).

### 3.2 CLI/Config-to-UI Mapping
- UI controls map to `ResearchConfig` and `advanced.*` fields.
- Precedence: `UI runtime override > user cfg > default.yaml`.

### 3.3 Advanced Output-to-Widget Mapping
- Status cards: `advanced_summary.json`, `timing.csv`.
- Statistical tables: `fm_summary.csv`, `alpha_models.csv`, `orthogonal_ic.csv`.
- Correlation views: `factor_corr_matrix.csv`, `factor_corr_matrix_wide.csv`.

### 3.4 No-New-Compute Priority
- Reuse pipeline outputs directly before adding any new engine logic.

## 4. Output Contract and Data Dictionary
### 4.1 Base Contract
- `summary.csv`, `ic_series.csv`, `ic_decay.csv`, `layer_returns.csv`, `nav.csv`.

### 4.2 Advanced Contract
- 16+ files in `advanced_diagnostics/` with stable names.

### 4.3 Field Semantics
- `is_placeholder`: schema exists but module not fully implemented.
- `status`: readable execution/data sufficiency state (`ok`, `insufficient_windows`, etc).

### 4.4 Compatibility Strategy
- Keep old fields; append new fields only.

### 4.5 Empty/Error Contract
- Explicit values such as `missing_membership` and `insufficient_windows`.

## 5. Phased Plan (Reordered)
### 5.1 Phase A (Week 1): Read-only Dashboard MVP
- Three-source factor library merge.
- Factor report page (including advanced diagnostics display).
- Relationship page powered by existing correlation outputs.
- Chart downsampling and WebGL rendering path.

### 5.2 Phase B (Week 2): Async Tasks and Progress
- `BacktestService` with singleton executor.
- Task state machine and `tasks/<task_id>/` contract.
- Cache-hit strategy and force-rerun control.
- Progress callback first; log heartbeat fallback.

### 5.3 Phase C (Week 3): Admission and Version Loop
- AdmissionGate.
- VersionManager.
- `factor_zoo_generated.py` sync automation.
- Log viewer.

### 5.4 Phase D (Parallel Data Engineering, 1-2 weeks)
- PIT data for `HS300/ZZ500/ZZ1000`.
- Paper-grade `size_bucket_report`.
- Boundary hardening for rolling OOS, regime, and parameter sensitivity.

## 6. Task System Design
### 6.1 TaskStatus State Machine
- `queued -> running -> succeeded|failed|cancelled|timeout`.

### 6.2 TaskMeta
- `task_id`, `created_at`, `status`, `config_snapshot`, `progress`, `result_path`, `error`.

### 6.3 Task Directory Contract
- `progress.json`, `result.json`, `error.json`, optional logs.

### 6.4 Concurrency and Budget
- Default `max_workers=2`, bounded memory and CPU policy.

### 6.5 Cancel/Timeout/Retry
- Idempotent cancel request, hard timeout guard, retry with capped attempts.

## 7. Config System Design
### 7.1 `config/default.yaml`
- `advanced.*` defaults remain centralized.

### 7.2 `ResearchConfig`
- Schema-validated fields and stable hash semantics.

### 7.3 Override Priority
- `CLI/UI > cfg file > default`.

### 7.4 UI Pass-through Strategy
- UI writes explicit overrides only; do not mutate project defaults.

### 7.5 Reproducibility
- Store run config snapshot per task/run artifact.

## 8. Security and Isolation
### 8.1 Default Security
- DSL-first; advanced Python mode disabled by default.

### 8.2 AST Guardrails
- Block imports/ops outside allow-list.

### 8.3 RestrictedPython
- Recommended for production if user script execution is enabled.

### 8.4 Risk and Audit
- Action-level audit logs for task lifecycle and config changes.

## 9. Charts and Performance
### 9.1 Downsampling
- Trigger by point threshold; preserve extrema and trend.

### 9.2 WebGL
- Use for dense line/scatter rendering.

### 9.3 Large Tables
- Pagination and lazy loading.

### 9.4 Metrics
- Track first paint, interaction latency, and throughput.

## 10. Data Assets and PIT Governance
### 10.1 Universe Standard
- Required columns: `trade_date`, `index_code`, `ts_code`.

### 10.2 Data Source Onboarding
- Source metadata, update cadence, and retention policy for HS300/ZZ500/ZZ1000.

### 10.3 PIT Validation
- No future membership leakage (`effective_date <= query_date` rule).

### 10.4 Coverage Monitoring
- Track mean/min coverage and missing periods by index bucket.

## 11. Testing and Acceptance
### 11.1 Unit Matrix
- Service logic, parser/config, and output contract checks.

### 11.2 Integration
- `CLI -> pipeline -> output -> UI` end-to-end.

### 11.3 Regression
- Old parameters and old outputs stay compatible.

### 11.4 UAT Checklist
- Phase-wise acceptance criteria and release gates.

## 12. Release and Rollback
### 12.1 Release Strategy
- Feature flags and gradual rollout.

### 12.2 Versioning
- Semantic tags and change log.

### 12.3 Rollback
- Config-level rollback and feature-disable path.

### 12.4 Runbook
- Common failures, diagnostics, and operator actions.

## 13. Risk Register (Prioritized)
### 13.1 High
- Data definition drift, task contention, sandbox/security bypass.

### 13.2 Medium
- Performance under heavy UI usage and user misconfiguration.

### 13.3 Low
- UI polish and non-critical usability issues.

### 13.4 Mitigation Matrix
- Each risk has owner, trigger, and mitigation action.

## 14. Appendix
### 14.1 Final File Layout
- To be finalized after Phase B task runtime contract lands.

### 14.2 API/Service Draft
- Service endpoints and payload schemas.

### 14.3 Example Configs
- Personal mode and team mode presets.

### 14.4 Example Outputs
- Include `advanced_summary.json` and `timing.csv` snapshots.

