#!/usr/bin/env python
"""
scripts/run_analysis.py  [已弃用 — 将在 v4.2 移除]
=====================================================
⚠️  此脚本已弃用，请改用:
    mf single --factor <name> [参数]

参数映射
--------
  --start    →  mf single --start
  --end      →  mf single --end
  --forward  →  mf single --forward
  --n-groups →  mf single --n-groups
  --config   →  mf single --config
  --output   →  mf single --output
  --no-cache →  mf single --no-cache

此脚本现在仅作为向后兼容的 shim，内部转发给 ``mf single``。
业务逻辑已完全迁移至 factor_framework/cli/main.py。
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

# ── 弃用警告 ─────────────────────────────────────────────────────────────────
warnings.warn(
    "\n⚠️  scripts/run_analysis.py 已弃用，将在 v4.2 移除。\n"
    "   请改用: mf single --factor <name> [参数]\n"
    "   完整文档: docs/cli-contract.md\n",
    DeprecationWarning,
    stacklevel=1,
)
print(
    "\n⚠️  scripts/run_analysis.py 已弃用，将在 v4.2 移除。\n"
    "   请改用: mf single --factor <name> [参数]\n",
    file=sys.stderr,
)

import argparse

ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "[已弃用] scripts/run_analysis.py — 转发至 mf single。\n"
            "请直接使用: mf single --factor <name> [参数]"
        )
    )
    p.add_argument("--config",      default=None)
    p.add_argument("--start",       default=None)
    p.add_argument("--end",         default=None)
    p.add_argument("--forward",     type=int, default=None)
    p.add_argument("--n-groups",    type=int, default=None, dest="n_groups")
    p.add_argument("--output",      default=None)
    p.add_argument("--no-cache",    action="store_true")
    p.add_argument("--show-config", action="store_true")
    # Legacy positional: ignored silently
    p.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Build forwarded argv for  mf single
    fwd = [sys.executable, "-m", "factor_framework.cli", "single"]

    # run_analysis had no --factor concept (ran all); forward to batch instead
    # if no factor info is available, warn and redirect to mf batch
    print(
        "  ℹ️  run_analysis.py 没有 --factor 参数。\n"
        "     将转发至 mf batch （与原来行为等价）。",
        file=sys.stderr,
    )
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


warnings.filterwarnings("ignore")

# ── 项目根目录 ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config, print_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多因子全量分析（新主路径，消费 config/default.yaml）"
    )
 p.add_argument("--config", default=None, help=" YAML ")
 p.add_argument("--start", default=None, help=" YYYYMMDD")
 p.add_argument("--end", default=None, help=" YYYYMMDD")
 p.add_argument("--forward", type=int, default=None, help="（）")
 p.add_argument("--n-groups", type=int, default=None, dest="n_groups", help="")
 p.add_argument("--output", default=None, help="")
 p.add_argument("--no-cache", action="store_true", help=" L2 ")
 p.add_argument("--show-config", action="store_true", help="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── 1. 构建 CLI overrides ────────────────────────────────────────────────
    overrides: dict = {}
    if args.start:    overrides["backtest.start"]   = args.start
    if args.end:      overrides["backtest.end"]     = args.end
    if args.forward:  overrides["backtest.forward"] = args.forward
    if args.n_groups: overrides["backtest.n_groups"] = args.n_groups
    if args.output:   overrides["output.factor_analysis"] = args.output
    if args.no_cache: overrides["cache.cache_dir"]  = None

    # ── 2. 加载配置（default → user → CLI）─────────────────────────────────
    cfg = load_config(user_config=args.config, overrides=overrides)

    if args.show_config:
        print("\n【】")
        print_config(cfg)
        return

    # ── 3. 参数回显 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(" Multi-Factor (v3.3 )")
    print("=" * 64)
    print(f" : {cfg.data.stocks_dir}")
    print(f" : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f" : {cfg.backtest.forward} ")
    print(f" : {cfg.backtest.n_groups}")
    print(f" : {cfg.cache.cache_dir}")
    print(f" : {cfg.output.factor_analysis}")
    print("=" * 64 + "\n")

    # ── 4. 将配置转为 factor_analysis.py 兼容的 CFG dict ───────────────────
    from factor_analysis import main as _legacy_main, CFG as _legacy_cfg

    # 用配置文件值覆盖 legacy CFG（仅覆盖 default.yaml 中有对应的字段）
    _legacy_cfg.update({
        "stocks_dir":      cfg.data.stocks_dir,
        "stock_basic":     cfg.data.stock_basic,
        "start":           cfg.backtest.start,
        "end":             cfg.backtest.end,
        "forward":         cfg.backtest.forward,
        "n_groups":        cfg.backtest.n_groups,
        "ic_method":       cfg.ic.method,
        "standardize":     cfg.preprocess.standardize,
        "winsorize":       cfg.preprocess.winsorize,
        "neutralize":      cfg.preprocess.neutralize,
        "periods_per_year": cfg.backtest.periods_per_year,
        "rf":              cfg.backtest.rf,
        "cost_per_side":   cfg.backtest.cost_per_side,
        "output_dir":      Path(cfg.output.factor_analysis),
        "n_jobs":          cfg.parallel.n_jobs,
    })

    # ── 5. 执行主分析逻辑 ────────────────────────────────────────────────────
    _legacy_main()


if __name__ == "__main__":
    main()

