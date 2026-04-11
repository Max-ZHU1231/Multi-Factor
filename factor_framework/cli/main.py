"""
factor_framework.cli.main
==========================
Primary CLI entry-point for the Multi-Factor framework (v4.0).

Registered as ``mf`` via pyproject.toml [project.scripts].
Install with:  pip install -e .

Sub-commands
------------
  mf screen   — run single-factor IC / layer-backtest screening
  mf batch    — full-batch factor validation (all registered factors)
  mf validate — look-ahead / data-quality validation suite
  mf cache    — cache management (info / clear / gc)
  mf report   — report generation from saved artifacts

Phase A stub
------------
Full implementation deferred to Phase B (CLI convergence).
Running any sub-command currently prints a helpful "not yet implemented"
message and exits with code 0 so that CI pipelines can detect the entry-point.
"""
from __future__ import annotations

import argparse
import sys


# ── sub-command stubs ────────────────────────────────────────────────────────

def _cmd_screen(args: argparse.Namespace) -> int:
    print("[mf screen] Phase B — not yet implemented.")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    print("[mf batch] Phase B — not yet implemented.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    print("[mf validate] Phase B — not yet implemented.")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    print("[mf cache] Phase B — not yet implemented.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    print("[mf report] Phase B — not yet implemented.")
    return 0


# ── parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mf",
        description="Multi-Factor Research Framework — v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mf screen --factor momentum_12_1\n"
            "  mf batch  --start 20200101 --end 20251231\n"
            "  mf validate --suite lookahead\n"
            "  mf cache  info\n"
            "  mf report --artifact artifacts/batch_results/\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="multi-factor 4.0.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # screen
    p_screen = sub.add_parser("screen", help="Single-factor IC / layer-backtest screening")
    p_screen.add_argument("--factor", nargs="+", help="Factor name(s) to screen")
    p_screen.add_argument("--start", default="20200101", help="Start date YYYYMMDD")
    p_screen.add_argument("--end",   default="20251231", help="End date YYYYMMDD")
    p_screen.set_defaults(func=_cmd_screen)

    # batch
    p_batch = sub.add_parser("batch", help="Full-batch factor validation")
    p_batch.add_argument("--start", default="20200101")
    p_batch.add_argument("--end",   default="20251231")
    p_batch.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    p_batch.set_defaults(func=_cmd_batch)

    # validate
    p_val = sub.add_parser("validate", help="Look-ahead / data-quality validation suite")
    p_val.add_argument("--suite", choices=["lookahead", "quality", "all"], default="all")
    p_val.set_defaults(func=_cmd_validate)

    # cache
    p_cache = sub.add_parser("cache", help="Cache management")
    p_cache.add_argument("action", choices=["info", "clear", "gc"], nargs="?", default="info")
    p_cache.set_defaults(func=_cmd_cache)

    # report
    p_rep = sub.add_parser("report", help="Report generation from saved artifacts")
    p_rep.add_argument("--artifact", help="Path to artifacts directory")
    p_rep.add_argument("--format", choices=["html", "pdf", "md"], default="html")
    p_rep.set_defaults(func=_cmd_report)

    return parser


# ── entry-point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
