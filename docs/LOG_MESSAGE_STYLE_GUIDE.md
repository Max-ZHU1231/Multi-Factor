# Log Message Style Guide

## Scope
- Applies to runtime-visible messages only:
  - `print(...)`
  - `warnings.warn(...)`
  - raised exception messages (`raise ...("...")`)
  - CLI `help` / `epilog` text

## Prefix Convention
- Use one of:
  - `[INFO]` for normal progress/state messages
  - `[WARN]` for recoverable issues or degraded paths
  - `[ERROR]` for failures that require attention

## Tone and Tense
- Keep messages short, direct, and actionable.
- Prefer present tense.
- Avoid ambiguous phrasing and mixed-language output.

## Error Message Structure
- Include **reason** + **suggested action** whenever feasible.
- Recommended pattern:
  - `"[ERROR] <what failed>: <reason>. <suggested action>."`

## Compatibility Rules
- Do not change public API/output field names solely for localization.
- Do not rename variables/files for message-style cleanup.
- For output contracts, append new fields rather than breaking old ones.

## Examples
- Good:
  - `[INFO] Building return panel (forward=21)...`
  - `[WARN] Universe file missing; fallback to full universe.`
  - `[ERROR] Factor 'x' is not registered. Call register() first.`
- Avoid:
  - mixed Chinese/English in one run path
  - long narrative logs without actionable guidance
  - error text without root cause or next step

