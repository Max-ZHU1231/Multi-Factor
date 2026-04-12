"""
factor_framework.cli.main
==========================
Primary CLI entry-point for the Multi-Factor framework (v4.0).

Installed as ``mf`` via pyproject.toml [project.scripts].
Also reachable via:  python -m factor_framework.cli <command>

Sub-commands
------------
  mf single    - single-factor IC + layer-backtest screening
  mf batch     - full-batch factor validation (all registered factors)
  mf validate  - look-ahead / data-quality validation suite
  mf cache     - cache management (info / clear / gc)
  mf report    - report generation (Phase D stub)
  mf composite - multi-factor combination (v4.1 stub)

Exit codes
----------
  0  - success
  1  - runtime failure (data error, computation exception)
  2  - argument error (argparse handles automatically)

See docs/cli-contract.md for the full CLI contract.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress noisy deprecation warnings from legacy internals
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── project root (needed when invoked via python -m or scripts/) ─────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _err(msg: str) -> None:
    """Print error to stderr with [ERROR] prefix."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def _warn_deprecated(script: str, replacement: str) -> None:
    print(
        f"\n⚠️   {script} is deprecated and will be removed in v4.2.\n"
        f"    Please use: {replacement}\n",
        file=sys.stderr,
    )


def _load_cfg(args: argparse.Namespace):
    """Load config with CLI overrides applied."""
    from config.loader import load_config

    overrides: dict = {}
    if getattr(args, "start",    None): overrides["backtest.start"]    = args.start
    if getattr(args, "end",      None): overrides["backtest.end"]      = args.end
    if getattr(args, "forward",  None): overrides["backtest.forward"]  = args.forward
    if getattr(args, "n_groups", None): overrides["backtest.n_groups"] = args.n_groups
    if getattr(args, "no_cache", False): overrides["cache.cache_dir"]  = None
    # ── universe overrides ────────────────────────────────────────────────────
    if getattr(args, "universe",       None): overrides["universe.name"]              = args.universe
    if getattr(args, "universe_mode",  None): overrides["universe.mode"]              = args.universe_mode
    if getattr(args, "universe_top_n", None): overrides["universe.top_n"]             = args.universe_top_n
    if getattr(args, "universe_metric",None): overrides["universe.metric"]            = args.universe_metric
    if getattr(args, "universe_rebalance_freq", None):
        overrides["universe.rebalance_freq"] = args.universe_rebalance_freq
    if getattr(args, "universe_effective_lag_days", None) is not None:
        overrides["universe.effective_lag_days"] = args.universe_effective_lag_days
    force_exit = getattr(args, "force_exit_on_universe_drop", None)
    if force_exit is not None:
        overrides["universe.force_exit_on_drop"] = force_exit
    # ─────────────────────────────────────────────────────────────────────────
    # ── advanced diagnostics overrides ───────────────────────────────────────
    if getattr(args, "advanced_peer_factors", None):
        overrides["advanced.peer_factors"] = args.advanced_peer_factors
    if getattr(args, "advanced_min_cs_nobs", None) is not None:
        overrides["advanced.min_cs_nobs"] = args.advanced_min_cs_nobs
    if getattr(args, "advanced_corr_method", None):
        overrides["advanced.corr_method"] = args.advanced_corr_method
    if getattr(args, "advanced_high_corr_threshold", None) is not None:
        overrides["advanced.high_corr_threshold"] = args.advanced_high_corr_threshold
    if getattr(args, "advanced_nw_lag_rule", None):
        overrides["advanced.nw_lag_rule"] = args.advanced_nw_lag_rule
    if getattr(args, "advanced_enable_wide_output", None) is not None:
        overrides["advanced.enable_wide_output"] = args.advanced_enable_wide_output
    # ─────────────────────────────────────────────────────────────────────────
    if getattr(args, "output",   None):
        cmd = getattr(args, "command", "single")
        key = "output.batch" if cmd == "batch" else "output.factor_analysis"
        overrides[key] = args.output

    return load_config(
        user_config=getattr(args, "config", None),
        overrides=overrides,
    )


def _make_pipe(cfg, root: Path = _ROOT):
    """Construct a FactorPipeline from a loaded config."""
    from factor_framework.pipeline import FactorPipeline
    return FactorPipeline(
        stocks_dir  = root / cfg.data.stocks_dir,
        stock_basic = root / cfg.data.stock_basic,
        verbose     = cfg.parallel.verbose,
        cache_dir   = cfg.cache.cache_dir,
    )


def _print_header(title: str, cfg) -> None:
    uni = getattr(cfg, "universe", None)
    uni_mode = getattr(uni, "mode", "all") or "all"
    uni_name = getattr(uni, "name", None)
    if uni_mode == "topn_mktcap_dynamic":
        top_n  = getattr(uni, "top_n",  500)
        metric = getattr(uni, "metric", "total_mktcap")
        freq   = getattr(uni, "rebalance_freq", "semiannual")
        uni_display = f"topn_dynamic (top={top_n}, metric={metric}, freq={freq})"
    elif uni_name:
        uni_display = uni_name
    else:
        uni_display = uni_mode
    print("\n" + "=" * 64)
    print(f"[INFO] {title}")
    print("=" * 64)
    print(f"[INFO] Data Dir     : {cfg.data.stocks_dir}")
    print(f"[INFO] Universe     : {uni_display}")
    print(f"[INFO] Date Range   : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f"[INFO] Forward Days : {cfg.backtest.forward}")
    print(f"[INFO] N Groups     : {cfg.backtest.n_groups}")
    print(f"[INFO] Cache Dir    : {cfg.cache.cache_dir}")
    print("=" * 64 + "\n")


def _resolve_universe(cfg, root: Path = _ROOT):
    """
    从配置中解析股票池，返回 ts_code 列表（静态模式）、
    UniverseMembership 对象（动态模式）或 None（全部股票）。

    返回值类型：
      None                  → 全部股票
      list[str]             → 静态股票池（static_file / hs300 等）
      UniverseMembership    → 动态股票池（topn_mktcap_dynamic）
    """
    uni = getattr(cfg, "universe", None)
    mode = getattr(uni, "mode", "all") or "all"
    name = getattr(uni, "name", None)

    try:
        if mode == "topn_mktcap_dynamic":
            from universes.membership import build_membership_from_config
            return build_membership_from_config(cfg, root=root, verbose=True)

        # static_file 或 all（name 可能指向 hs300 等）
        from universes.loader import UniverseLoader
        # name 优先于 mode
        universe_key = name if name else (None if mode == "all" else mode)
        stocks_dir = root / cfg.data.stocks_dir
        return UniverseLoader.load(universe_key, stocks_dir=stocks_dir)

    except Exception as exc:
        import warnings as _w
        _w.warn(f"[WARN] [universe] Failed to load universe; fallback to full universe: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  mf single
# ═══════════════════════════════════════════════════════════════════════════════

def _save_manifest(pipe, cfg, factors, failures, start_time: float,
                   out_dir: Path) -> None:
    """
    Generate and save run_manifest.json to *out_dir*.
    Non-fatal: any exception is swallowed with a warning.
    """
    try:
        from factor_framework.manifest import RunManifest
        cache_info = pipe._cache.cache_info() if pipe._cache is not None else {}
        mf = RunManifest.create(
            factors    = list(factors),
            cfg        = cfg,
            cache_info = cache_info,
            start_time = start_time,
            failures   = list(failures),
            stocks_dir = pipe._builder.stocks_dir,
        )
        out_path = out_dir / "run_manifest.json"
        mf.save(out_path)
        mf.print_summary()
        print(f"[INFO] manifest -> {out_path}")
    except Exception as exc:
        import warnings as _w
        _w.warn(f"[WARN] [manifest] Save failed (non-fatal): {exc}")

def _cmd_single(args: argparse.Namespace) -> int:
    """Single-factor IC + layer-backtest screening."""
    if not args.factor:
        _err("--factor is required. Example: mf single --factor momentum_12_1")
        return 2

    try:
        cfg = _load_cfg(args)
    except Exception as exc:
        _err(f"Failed to load config: {exc}")
        return 1

    if getattr(args, "show_config", False):
        from config.loader import print_config
        print_config(cfg)
        return 0

    if not getattr(args, "quiet", False):
        _print_header("Multi-Factor Single-Factor Analysis (mf single)", cfg)

    import time as _time
    _t0 = _time.perf_counter()

    try:
        pipe = _make_pipe(cfg)
        pipe.register_builtins(args.factor)

        universe = _resolve_universe(cfg)
        # Unwrap dynamic membership to a flat symbol list for the panel builder.
        # The membership object is also kept for potential panel filtering later.
        from universes.membership import UniverseMembership as _UMem
        if isinstance(universe, _UMem):
            membership = universe
            # Collect all symbols that ever appeared in the universe
            symbols = sorted({s for syms in membership.get_schedule().values() for s in syms})
            uni_display = f"{len(symbols)} symbols (dynamic universe)"
        elif universe is not None:
            membership = None
            symbols = universe
            uni_display = f"{len(symbols)} symbols"
        else:
            membership = None
            symbols = None
            uni_display = "all symbols"
        if not getattr(args, "quiet", False):
            print(f"[INFO] Universe     : {uni_display}")

        out_base = Path(
            getattr(args, "output", None) or cfg.output.factor_analysis
        )

        results = []
        failures = []
        for factor_name in args.factor:
            factor_out = out_base / factor_name
            factor_out.mkdir(parents=True, exist_ok=True)
            try:
                if not getattr(args, "quiet", False):
                    print(f"\n{'─'*56}\n[INFO] Factor: {factor_name}\n{'─'*56}")
                from factor_framework.research_config import ResearchConfig
                rc = ResearchConfig.from_kwargs(
                    factor_name      = factor_name,
                    start            = cfg.backtest.start,
                    end              = cfg.backtest.end,
                    forward          = cfg.backtest.forward,
                    n_groups         = cfg.backtest.n_groups,
                    standardize      = cfg.preprocess.standardize,
                    winsorize        = cfg.preprocess.winsorize,
                    neutralize       = cfg.preprocess.neutralize,
                    ic_method        = cfg.ic.method,
                    periods_per_year = cfg.backtest.periods_per_year,
                    rf               = cfg.backtest.rf,
                    cost_per_side    = cfg.backtest.cost_per_side,
                    resample_monthly = cfg.backtest.resample_monthly,
                    symbols          = symbols,
                    advanced_peer_factors        = getattr(cfg.advanced, "peer_factors", None),
                    advanced_min_cs_nobs         = getattr(cfg.advanced, "min_cs_nobs", 20),
                    advanced_corr_method         = getattr(cfg.advanced, "corr_method", "spearman"),
                    advanced_high_corr_threshold = getattr(cfg.advanced, "high_corr_threshold", 0.7),
                    advanced_nw_lag_rule         = getattr(cfg.advanced, "nw_lag_rule", "t_pow_0.25"),
                    advanced_enable_wide_output  = getattr(cfg.advanced, "enable_wide_output", True),
                )
                _run_diag = getattr(args, "ic_decay_diagnostics", False)
                _run_adv = getattr(args, "advanced_diagnostics", False)
                report = pipe.run(
                    config=rc,
                    run_ic_decay_diagnostics=_run_diag,
                    run_advanced_diagnostics=_run_adv,
                )
                report.print_summary()
                # ── IC 衰减诊断（--ic-decay-diagnostics 标志）──────────────
                if _run_diag and report._diag_report is None:
                    # pipeline 内部诊断未能运行（price_panel 获取失败），在 CLI 层补充尝试
                    diag_dir = factor_out / "ic_decay_diagnostics"
                    try:
                        diag_report = report.run_ic_diagnostics(
                            forward_list = list(rc.ic_forward_list),
                            verbose      = True,
                            save_dir     = diag_dir,
                        )
                    except Exception as _de:
                        _err(f"IC decay diagnostics failed (non-fatal): {_de}")
                report.save(factor_out)
                results.append(factor_name)
            except Exception as exc:
                _err(f"Factor {factor_name!r} failed: {exc}")
                failures.append(factor_name)

        # ── Phase D: 写入 run_manifest.json ──────────────────────────────
        _save_manifest(
            pipe=pipe, cfg=cfg, factors=args.factor,
            failures=failures, start_time=_t0,
            out_dir=out_base,
        )

        if failures and not results:
            return 1
        return 0

    except Exception as exc:
        _err(f"Runtime error: {exc}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
#  mf batch
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_batch(args: argparse.Namespace) -> int:
    """Full-batch factor validation."""
    try:
        cfg = _load_cfg(args)
    except Exception as exc:
        _err(f"Failed to load config: {exc}")
        return 1

    if getattr(args, "show_config", False):
        from config.loader import print_config
        print_config(cfg)
        return 0

    if not getattr(args, "quiet", False):
        _print_header("Multi-Factor Batch Validation (mf batch)", cfg)

    import time as _time
    _t0 = _time.perf_counter()

    try:
        import pandas as pd
        from factor_framework.factor_zoo import BUILTIN_FACTORS

        pipe = _make_pipe(cfg)
        pipe.register_builtins()

        universe = _resolve_universe(cfg)
        from universes.membership import UniverseMembership as _UMem
        if isinstance(universe, _UMem):
            membership = universe
            symbols = sorted({s for syms in membership.get_schedule().values() for s in syms})
            uni_display = f"{len(symbols)} symbols (dynamic universe)"
        elif universe is not None:
            membership = None
            symbols = universe
            uni_display = f"{len(symbols)} symbols"
        else:
            membership = None
            symbols = None
            uni_display = "all symbols"
        if not getattr(args, "quiet", False):
            print(f"[INFO] Universe     : {uni_display}")

        factor_list = getattr(args, "factors", None) or list(BUILTIN_FACTORS.keys())
        out_dir = Path(getattr(args, "output", None) or cfg.output.batch)
        out_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        failures = []
        for name in factor_list:
            if not getattr(args, "quiet", False):
                print(f"\n{'='*60}\n[INFO] Factor: {name}\n{'='*60}")
            try:
                from factor_framework.research_config import ResearchConfig
                rc = ResearchConfig.from_kwargs(
                    factor_name      = name,
                    start            = cfg.backtest.start,
                    end              = cfg.backtest.end,
                    forward          = cfg.backtest.forward,
                    n_groups         = cfg.backtest.n_groups,
                    standardize      = cfg.preprocess.standardize,
                    winsorize        = cfg.preprocess.winsorize,
                    neutralize       = cfg.preprocess.neutralize,
                    ic_method        = cfg.ic.method,
                    periods_per_year = cfg.backtest.periods_per_year,
                    rf               = cfg.backtest.rf,
                    cost_per_side    = cfg.backtest.cost_per_side,
                    resample_monthly = cfg.backtest.resample_monthly,
                    symbols          = symbols,
                    advanced_peer_factors        = getattr(cfg.advanced, "peer_factors", None),
                    advanced_min_cs_nobs         = getattr(cfg.advanced, "min_cs_nobs", 20),
                    advanced_corr_method         = getattr(cfg.advanced, "corr_method", "spearman"),
                    advanced_high_corr_threshold = getattr(cfg.advanced, "high_corr_threshold", 0.7),
                    advanced_nw_lag_rule         = getattr(cfg.advanced, "nw_lag_rule", "t_pow_0.25"),
                    advanced_enable_wide_output  = getattr(cfg.advanced, "enable_wide_output", True),
                )
                report = pipe.run(config=rc)
                if getattr(args, "advanced_diagnostics", False):
                    report.run_advanced_diagnostics(
                        save_dir=out_dir / name / "advanced_diagnostics",
                        n_groups=cfg.backtest.n_groups,
                        direction=1,
                        periods_per_year=cfg.backtest.periods_per_year,
                        peer_factors=getattr(cfg.advanced, "peer_factors", None),
                        min_cs_nobs=getattr(cfg.advanced, "min_cs_nobs", 20),
                        corr_method=getattr(cfg.advanced, "corr_method", "spearman"),
                        high_corr_threshold=getattr(cfg.advanced, "high_corr_threshold", 0.7),
                        nw_lag_rule=getattr(cfg.advanced, "nw_lag_rule", "t_pow_0.25"),
                        enable_wide_output=getattr(cfg.advanced, "enable_wide_output", True),
                    )
                s = report.summary_dict
                s["factor"] = name
                s["error"] = ""
                summaries.append(s)
            except Exception as exc:
                _err(f"[skip] {name}: {exc}")
                summaries.append({"factor": name, "error": str(exc)})
                failures.append(name)

        if summaries:
            df = pd.DataFrame(summaries).set_index("factor")
            csv_path = out_dir / "factor_screening_summary.csv"
            df.to_csv(csv_path)
            print(f"\n[INFO] IC summary saved: {csv_path}")

        n_ok = len(summaries) - len(failures)
        print(f"\n[INFO] Done: {n_ok}/{len(summaries)} factors succeeded, {len(failures)} skipped.")

        # ── Phase D: 写入 run_manifest.json ──────────────────────────────
        _save_manifest(
            pipe=pipe, cfg=cfg, factors=factor_list,
            failures=failures, start_time=_t0,
            out_dir=out_dir,
        )

        return 1 if (n_ok == 0 and len(summaries) > 0) else 0

    except Exception as exc:
        _err(f"Runtime error: {exc}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
#  mf validate
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_validate(args: argparse.Namespace) -> int:
    """Run look-ahead / data-quality validation suite via pytest."""
    suite = getattr(args, "suite", "all")
    verbose_flag = ["-v"] if getattr(args, "verbose", False) else ["-q"]

    suite_map = {
        "lookahead": ["tests/integration/test_lookahead_bias.py"],
        "quality":   ["tests/unit/test_data_quality.py",
                      "tests/unit/test_data_cleaner.py"],
        "all":       ["tests/integration/test_lookahead_bias.py",
                      "tests/unit/test_data_quality.py",
                      "tests/unit/test_data_cleaner.py",
                      "tests/unit/test_new_import_paths.py"],
    }
    test_paths = suite_map.get(suite, suite_map["all"])

    cmd = [sys.executable, "-m", "pytest", "--tb=short"] + verbose_flag + test_paths
    print(f"[INFO] [mf validate] Running suite: {suite}")
    print(f"[INFO] Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_ROOT))
    return 0 if result.returncode == 0 else 1


# ═══════════════════════════════════════════════════════════════════════════════
#  mf cache
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_cache(args: argparse.Namespace) -> int:
    """Cache management: info / clear / gc."""
    action = getattr(args, "action", "info") or "info"
    cache_dir = Path(getattr(args, "dir", None) or _ROOT / "cache")
    factor_filter = getattr(args, "factor", None)

    if not cache_dir.exists():
        print(f"[WARN] Cache directory does not exist: {cache_dir}")
        return 0

    if action == "info":
        entries = list(cache_dir.glob("**/*.parquet"))
        if factor_filter:
            entries = [e for e in entries if factor_filter in str(e)]
        total_mb = sum(e.stat().st_size for e in entries) / 1024 / 1024
        print(f"[INFO] Cache directory : {cache_dir}")
        print(f"[INFO] Factor subdirs  : {len(list(cache_dir.iterdir()))}")
        print(f"[INFO] Parquet files   : {len(entries)}")
        print(f"[INFO] Total size      : {total_mb:.1f} MB")
        return 0

    if action == "clear":
        import shutil
        if factor_filter:
            targets = [d for d in cache_dir.iterdir()
                       if d.is_dir() and factor_filter in d.name]
        else:
            targets = [d for d in cache_dir.iterdir() if d.is_dir()]
        for t in targets:
            shutil.rmtree(t)
            print(f"[INFO] Deleted: {t.name}")
        print(f"[INFO] Cleared {len(targets)} cache directories.")
        return 0

    if action == "gc":
        import time
        max_age_days = getattr(args, "days", 30)
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for f in cache_dir.glob("**/*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        print(f"[INFO] GC complete: removed {removed} parquet files older than {max_age_days} days.")
        return 0

    _err(f"Unknown cache action: {action!r}")
    return 2


# ═══════════════════════════════════════════════════════════════════════════════
#  mf report  (Phase D stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_report(args: argparse.Namespace) -> int:
    print("[INFO] [mf report] Phase D - report generation will be implemented in v4.1.")
    print("[INFO] For now, inspect CSV files under artifacts/.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  mf composite  (v4.1 stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_composite(args: argparse.Namespace) -> int:
    print("[INFO] [mf composite] v4.1 - multi-factor composition will be implemented in v4.1.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser
# ═══════════════════════════════════════════════════════════════════════════════

def _add_common_backtest_args(p: argparse.ArgumentParser) -> None:
    """Shared backtest parameters."""
    p.add_argument("--start",    default=None, metavar="YYYYMMDD",
                   help="Backtest start date (default: 20200101)")
    p.add_argument("--end",      default=None, metavar="YYYYMMDD",
                   help="Backtest end date (default: 20251231)")
    p.add_argument("--forward",  type=int, default=None, metavar="N",
                   help="Forward horizon in trading days (default: 21)")
    p.add_argument("--n-groups", type=int, default=None, metavar="N",
                   dest="n_groups", help="Number of quantile groups (default: 5)")
    # ── 股票池参数 ────────────────────────────────────────────────────────────
    p.add_argument("--universe", default=None, metavar="NAME",
                   help="Static universe alias or CSV path (hs300 / my_pool.csv)")
    p.add_argument("--universe-mode", default=None,
                   dest="universe_mode",
                   choices=["all", "static_file", "topn_mktcap_dynamic"],
                   help="Universe mode (default: all)")
    p.add_argument("--universe-top-n", type=int, default=None,
                   dest="universe_top_n", metavar="N",
                   help="Top-N size for dynamic universe (default: 500; dynamic mode only)")
    p.add_argument("--universe-metric", default=None,
                   dest="universe_metric",
                   choices=["total_mktcap", "free_float_mktcap"],
                   help="Market-cap metric (default: total_mktcap)")
    p.add_argument("--universe-rebalance-freq", default=None,
                   dest="universe_rebalance_freq",
                   choices=["annual", "semiannual", "quarterly"],
                   help="Dynamic universe rebalance frequency (default: semiannual)")
    p.add_argument("--universe-effective-lag-days", type=int, default=None,
                   dest="universe_effective_lag_days", metavar="N",
                   help="Lag (trading days) from decision date to effective date (default: 1)")
    p.add_argument("--force-exit-on-universe-drop",
                   dest="force_exit_on_universe_drop",
                   type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=None, metavar="BOOL",
                   help="Force exit holdings dropped from universe (true/false, default: true)")
    # ─────────────────────────────────────────────────────────────────────────
    p.add_argument("--config",   default=None, metavar="PATH",
                   help="User YAML config path (overlaid on default.yaml)")
    p.add_argument("--show-config", action="store_true",
                   help="Print effective config and exit (code 0)")


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    """Shared IO parameters."""
    p.add_argument("--output",   default=None, metavar="DIR",
                   help="Override output directory")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable L2 parquet cache")
    p.add_argument("--quiet",    action="store_true",
                   help="Print summary only (quiet mode)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mf",
        description="Multi-Factor Research Framework - v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick Start\n"
            "--------\n"
            "  mf single --factor momentum_12_1\n"
            "  mf single --factor vwap_deviation --start 20210101\n"
            "  mf batch  --output artifacts/batch_results\n"
            "  mf batch  --universe hs300\n"
            "  mf batch  --universe-mode topn_mktcap_dynamic --universe-top-n 300\n"
            "  mf validate --suite lookahead\n"
            "  mf cache  info\n"
            "  mf cache  gc --days 14\n"
            "\n"
            "Universe\n"
            "------\n"
            "  --universe hs300                  static: CSI300 constituents (~300)\n"
            "  --universe my_pool.csv            static: custom CSV (requires code column)\n"
            "  --universe-mode topn_mktcap_dynamic  dynamic: top-N by market cap (default 500)\n"
            "  --universe-top-n 300              dynamic universe size\n"
            "  --universe-metric free_float_mktcap  rank by free-float market cap\n"
            "  (unspecified = full universe)\n"
            "\n"
            "Full contract: docs/cli-contract.md\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="multi-factor 4.0.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── single ────────────────────────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="Single-factor IC + layer backtest",
        description=(
            "Single-factor IC + layer backtest.\n"
            "Exit codes: 0=success  1=runtime error  2=argument error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n----\n"
            "  mf single --factor momentum_12_1\n"
            "  mf single --factor vwap_deviation price_strength --start 20210101\n"
            "  mf single --factor value_pb --forward 10 --no-cache\n"
            "  mf single --factor value_pb --universe hs300\n"
            "  mf single --factor value_pb --ic-decay-diagnostics\n"
        ),
    )
    p_single.add_argument(
        "--factor", nargs="+", metavar="NAME",
        help="Factor name(s), space-separated (required)"
    )
    p_single.add_argument(
        "--ic-decay-diagnostics", action="store_true",
        dest="ic_decay_diagnostics",
        help="Run IC-decay diagnostics (6 modules) after backtest; save to output/<factor>/ic_decay_diagnostics/",
    )
    p_single.add_argument(
        "--advanced-diagnostics", action="store_true",
        dest="advanced_diagnostics",
        help="Run Advanced Diagnostics Pack after backtest; save to output/<factor>/advanced_diagnostics/",
    )
    p_single.add_argument(
        "--advanced-peer-factors", nargs="+", metavar="NAME",
        dest="advanced_peer_factors",
        help="Peer factors for advanced diagnostics (user-provided has priority).",
    )
    p_single.add_argument(
        "--advanced-min-cs-nobs", type=int, default=None, metavar="N",
        dest="advanced_min_cs_nobs",
        help="Minimum cross-sectional sample size for correlation matrix (default: 20).",
    )
    p_single.add_argument(
        "--advanced-corr-method", choices=["spearman", "pearson"], default=None,
        dest="advanced_corr_method",
        help="Correlation method for correlation matrix (default: spearman).",
    )
    p_single.add_argument(
        "--advanced-high-corr-threshold", type=float, default=None, metavar="X",
        dest="advanced_high_corr_threshold",
        help="High-correlation threshold (default: 0.7).",
    )
    p_single.add_argument(
        "--advanced-nw-lag-rule", default=None, metavar="RULE",
        dest="advanced_nw_lag_rule",
        help="Newey-West lag rule: t_pow_0.25 or fixed_N (e.g., fixed_4).",
    )
    p_single.add_argument(
        "--advanced-enable-wide-output",
        dest="advanced_enable_wide_output",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=None, metavar="BOOL",
        help="Write factor_corr_matrix_wide.csv (true/false, default: true).",
    )
    _add_common_backtest_args(p_single)
    _add_common_io_args(p_single)
    p_single.set_defaults(func=_cmd_single)

    # ── batch ─────────────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Batch factor validation",
        description=(
            "Run IC + layer backtest for all registered factors and export summary CSV.\n"
            "Exit codes: 0=success(partial skips allowed)  1=all failed  2=argument error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n----\n"
            "  mf batch\n"
            "  mf batch --factors momentum_12_1 value_pb\n"
            "  mf batch --universe hs300\n"
            "  mf batch --universe hs300 --start 20210101 --parallel 4\n"
            "  mf batch --universe my_pool.csv\n"
        ),
    )
    p_batch.add_argument(
        "--factors", nargs="+", metavar="NAME",
        help="Subset of factors to run (default: all registered factors)"
    )
    p_batch.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Parallel worker count (-1 = CPU cores, default: 1)"
    )
    p_batch.add_argument(
        "--advanced-diagnostics", action="store_true",
        dest="advanced_diagnostics",
        help="Run advanced diagnostics for each factor (slower).",
    )
    p_batch.add_argument(
        "--advanced-peer-factors", nargs="+", metavar="NAME",
        dest="advanced_peer_factors",
        help="Peer factors for advanced diagnostics (user-provided has priority).",
    )
    p_batch.add_argument(
        "--advanced-min-cs-nobs", type=int, default=None, metavar="N",
        dest="advanced_min_cs_nobs",
        help="Minimum cross-sectional sample size for correlation matrix (default: 20).",
    )
    _add_common_backtest_args(p_batch)
    _add_common_io_args(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="look-ahead / data-quality validation suite",
        description="Run validation suites via pytest. Exit codes: 0=all pass  1=failure",
    )
    p_val.add_argument(
        "--suite",
        choices=["lookahead", "quality", "all"],
        default="all",
        help="Which validation suite to run (default: all)",
    )
    p_val.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show verbose assertion output"
    )
    p_val.set_defaults(func=_cmd_validate)

    # ── cache ─────────────────────────────────────────────────────────────────
    p_cache = sub.add_parser(
        "cache",
        help="Cache management (info / clear / gc)",
        description="Manage L2 parquet disk cache. Exit codes: 0=success  1=failure",
    )
    p_cache.add_argument(
        "action",
        choices=["info", "clear", "gc"],
        nargs="?",
        default="info",
        help="Action type (default: info)",
    )
    p_cache.add_argument(
        "--factor", default=None, metavar="NAME",
        help="Restrict to specific factor (for clear)"
    )
    p_cache.add_argument(
        "--dir", default=None, metavar="PATH",
        help="Cache directory (default: cache/)"
    )
    p_cache.add_argument(
        "--days", type=int, default=30, metavar="N",
        help="GC: remove entries not accessed for N days (default: 30)"
    )
    p_cache.set_defaults(func=_cmd_cache)

    # ── report  (Phase D stub) ────────────────────────────────────────────────
    p_rep = sub.add_parser(
        "report",
        help="Report generation (Phase D - planned for v4.1)",
    )
    p_rep.add_argument("--artifact", default=None, metavar="PATH",
                       help="Artifacts directory path")
    p_rep.add_argument("--format", choices=["html", "pdf", "md"], default="html",
                       help="Output format (default: html)")
    p_rep.set_defaults(func=_cmd_report)

    # ── composite  (v4.1 stub) ────────────────────────────────────────────────
    p_comp = sub.add_parser(
        "composite",
        help="Multi-factor composition (v4.1 - not implemented yet)",
    )
    p_comp.add_argument("--factors", nargs="+", metavar="NAME",
                        help="Factor list for composition")
    p_comp.add_argument("--method",
                        choices=["equal", "icir", "pca"], default="icir",
                        help="Composition method (default: icir)")
    p_comp.set_defaults(func=_cmd_composite)

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    """Main entry-point registered as `mf` console_script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
