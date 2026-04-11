#!/usr/bin/env python
"""
scripts/run_batch.py  [已弃用 — 将在 v4.2 移除]
=================================================
⚠️  此脚本已弃用，请改用:
    mf batch [参数]

参数映射
--------
  --start    →  mf batch --start
  --end      →  mf batch --end
  --forward  →  mf batch --forward
  --n-groups →  mf batch --n-groups
  --config   →  mf batch --config
  --output   →  mf batch --output
  --no-cache →  mf batch --no-cache

此脚本现在仅作为向后兼容的 shim，内部转发给 ``mf batch``。
业务逻辑已完全迁移至 factor_framework/cli/main.py。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import warnings

# ── 弃用警告 ─────────────────────────────────────────────────────────────────
warnings.warn(
    "\n⚠️  scripts/run_batch.py 已弃用，将在 v4.2 移除。\n"
    "   请改用: mf batch [参数]\n"
    "   完整文档: docs/cli-contract.md\n",
    DeprecationWarning,
    stacklevel=1,
)
print(
    "\n⚠️  scripts/run_batch.py 已弃用，将在 v4.2 移除。\n"
    "   请改用: mf batch [参数]\n",
    file=sys.stderr,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "[已弃用] scripts/run_batch.py — 转发至 mf batch。\n"
            "请直接使用: mf batch [参数]"
        )
    )
    parser.add_argument("--config",      default=None)
    parser.add_argument("--start",       default=None)
    parser.add_argument("--end",         default=None)
    parser.add_argument("--forward",     type=int, default=None)
    parser.add_argument("--n-groups",    type=int, default=None, dest="n_groups")
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--output",      default=None)
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()

    # Build forwarded argv for  mf batch
    fwd = [sys.executable, "-m", "factor_framework.cli", "batch"]

    if args.config:      fwd += ["--config",   args.config]
    if args.start:       fwd += ["--start",    args.start]
    if args.end:         fwd += ["--end",      args.end]
    if args.forward:     fwd += ["--forward",  str(args.forward)]
    if args.n_groups:    fwd += ["--n-groups", str(args.n_groups)]
    if args.output:      fwd += ["--output",   args.output]
    if args.no_cache:    fwd += ["--no-cache"]
    if args.show_config: fwd += ["--show-config"]

    result = subprocess.run(fwd, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

