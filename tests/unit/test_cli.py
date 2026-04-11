"""
tests/unit/test_cli.py
======================
Minimal acceptance tests for the ``mf`` CLI (Phase B).

These tests exercise the *parser layer only* — no data I/O, no factor
computation.  They run fast (~0.1 s) and are suitable for CI smoke checks.

Tested invariants
-----------------
1.  ``mf --help`` exits 0 and lists all required sub-commands.
2.  ``mf --version`` prints "multi-factor 4.0.0" and exits 0.
3.  ``mf single --help`` exits 0 and documents ``--factor``.
4.  ``mf batch --help``  exits 0 and documents ``--factors``.
5.  ``mf validate --help`` exits 0 and documents ``--suite``.
6.  ``mf cache --help``  exits 0 and documents ``info``.
7.  ``mf single`` (no ``--factor``) → exit-code 2.
8.  ``mf`` (no sub-command) → exit-code 0 (prints help).
9.  ``build_parser()`` builds a parser with all six sub-commands.
10. ``_cmd_report`` and ``_cmd_composite`` stubs return exit-code 0.
"""
from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ── import the module under test ─────────────────────────────────────────────
from factor_framework.cli.main import (
    _cmd_cache,
    _cmd_composite,
    _cmd_report,
    build_parser,
    main,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run(argv: list[str]) -> tuple[int, str, str]:
    """
    Invoke ``main(argv)`` in-process.
    Returns (exit_code, stdout_text, stderr_text).
    """
    stdout, stderr = StringIO(), StringIO()
    rc = 0
    with (
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
    ):
        try:
            main(argv)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 0
    return rc, stdout.getvalue(), stderr.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — parser / help
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelp:
    """Help output and version checks."""

    def test_top_level_help_exit_0(self):
        rc, out, _ = _run(["--help"])
        assert rc == 0

    def test_top_level_help_lists_single(self):
        rc, out, _ = _run(["--help"])
        assert "single" in out

    def test_top_level_help_lists_batch(self):
        rc, out, _ = _run(["--help"])
        assert "batch" in out

    def test_top_level_help_lists_validate(self):
        rc, out, _ = _run(["--help"])
        assert "validate" in out

    def test_top_level_help_lists_cache(self):
        rc, out, _ = _run(["--help"])
        assert "cache" in out

    def test_version_exit_0(self):
        rc, out, _ = _run(["--version"])
        assert rc == 0

    def test_version_string(self):
        rc, out, _ = _run(["--version"])
        assert "4.0.0" in out

    def test_single_help_exit_0(self):
        rc, out, _ = _run(["single", "--help"])
        assert rc == 0

    def test_single_help_mentions_factor(self):
        rc, out, _ = _run(["single", "--help"])
        assert "--factor" in out

    def test_batch_help_exit_0(self):
        rc, out, _ = _run(["batch", "--help"])
        assert rc == 0

    def test_batch_help_mentions_factors(self):
        rc, out, _ = _run(["batch", "--help"])
        assert "--factors" in out

    def test_validate_help_exit_0(self):
        rc, out, _ = _run(["validate", "--help"])
        assert rc == 0

    def test_validate_help_mentions_suite(self):
        rc, out, _ = _run(["validate", "--help"])
        assert "--suite" in out

    def test_cache_help_exit_0(self):
        rc, out, _ = _run(["cache", "--help"])
        assert rc == 0

    def test_cache_help_mentions_info(self):
        rc, out, _ = _run(["cache", "--help"])
        assert "info" in out

    def test_report_help_exit_0(self):
        rc, out, _ = _run(["report", "--help"])
        assert rc == 0

    def test_composite_help_exit_0(self):
        rc, out, _ = _run(["composite", "--help"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — exit codes for no-op invocations
# ═══════════════════════════════════════════════════════════════════════════════

class TestExitCodes:
    """Correct exit codes without running any computation."""

    def test_no_subcommand_exits_0(self):
        rc, _, _ = _run([])
        assert rc == 0

    def test_single_no_factor_exits_2(self):
        """``mf single`` without --factor must exit 2 (argument error)."""
        rc, _, _ = _run(["single"])
        assert rc == 2

    def test_report_stub_exits_0(self):
        args = SimpleNamespace()
        assert _cmd_report(args) == 0

    def test_composite_stub_exits_0(self):
        args = SimpleNamespace()
        assert _cmd_composite(args) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — build_parser structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildParser:
    """Verify the parser graph has the expected sub-commands."""

    EXPECTED_COMMANDS = {"single", "batch", "validate", "cache", "report", "composite"}

    def test_parser_has_all_subcommands(self):
        parser = build_parser()
        # Access registered sub-commands via the _subparsers action
        subparsers_action = next(
            a for a in parser._actions
            if hasattr(a, "_name_parser_map")
        )
        registered = set(subparsers_action._name_parser_map.keys())
        assert self.EXPECTED_COMMANDS.issubset(registered), (
            f"Missing sub-commands: {self.EXPECTED_COMMANDS - registered}"
        )

    def test_single_parser_has_factor_arg(self):
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions
            if hasattr(a, "_name_parser_map")
        )
        single_p = subparsers_action._name_parser_map["single"]
        arg_names = {a.dest for a in single_p._actions}
        assert "factor" in arg_names

    def test_batch_parser_has_parallel_arg(self):
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions
            if hasattr(a, "_name_parser_map")
        )
        batch_p = subparsers_action._name_parser_map["batch"]
        arg_names = {a.dest for a in batch_p._actions}
        assert "parallel" in arg_names


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests — cache command (no filesystem side-effects)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheCommand:
    """Cache sub-command: non-existent dir returns 0."""

    def test_cache_info_nonexistent_dir_exits_0(self, tmp_path):
        nonexistent = tmp_path / "no_such_cache"
        args = SimpleNamespace(action="info", dir=str(nonexistent), factor=None)
        rc = _cmd_cache(args)
        assert rc == 0

    def test_cache_info_empty_dir_exits_0(self, tmp_path):
        (tmp_path / "cache").mkdir()
        args = SimpleNamespace(action="info", dir=str(tmp_path / "cache"), factor=None)
        rc = _cmd_cache(args)
        assert rc == 0

    def test_cache_clear_empty_dir_exits_0(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        args = SimpleNamespace(action="clear", dir=str(cache_dir), factor=None, days=30)
        rc = _cmd_cache(args)
        assert rc == 0

    def test_cache_gc_empty_dir_exits_0(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        args = SimpleNamespace(action="gc", dir=str(cache_dir), factor=None, days=30)
        rc = _cmd_cache(args)
        assert rc == 0

    def test_cache_unknown_action_exits_2(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        args = SimpleNamespace(action="purge", dir=str(cache_dir), factor=None, days=30)
        rc = _cmd_cache(args)
        assert rc == 2
