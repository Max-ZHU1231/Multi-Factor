"""
factor_framework.cli.main
==========================
Primary CLI entry-point for the Multi-Factor framework (v4.0).

Installed as ``mf`` via pyproject.toml [project.scripts].
Also reachable via:  python -m factor_framework.cli <command>

Sub-commands
------------
  mf single    — single-factor IC + layer-backtest screening
  mf batch     — full-batch factor validation (all registered factors)
  mf validate  — look-ahead / data-quality validation suite
  mf cache     — cache management (info / clear / gc)
  mf report    — report generation (Phase D stub)
  mf composite — multi-factor combination (v4.1 stub)

Exit codes
----------
  0  — success
  1  — runtime failure (data error, computation exception)
  2  — argument error (argparse handles automatically)

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
        f"\n⚠️   {script} 已弃用，将在 v4.2 移除。\n"
        f"    请改用: {replacement}\n",
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
    print(f"  {title}")
    print("=" * 64)
    print(f"  数据目录  : {cfg.data.stocks_dir}")
    print(f"  股票池    : {uni_display}")
    print(f"  时间范围  : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f"  预测期    : {cfg.backtest.forward} 天")
    print(f"  分层数    : {cfg.backtest.n_groups}")
    print(f"  缓存目录  : {cfg.cache.cache_dir}")
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
        _w.warn(f"[universe] 股票池加载失败，将使用全部股票: {exc}")
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
        print(f"  manifest → {out_path}")
    except Exception as exc:
        import warnings as _w
        _w.warn(f"[manifest] 保存失败（非致命）: {exc}")

def _cmd_single(args: argparse.Namespace) -> int:
    """Single-factor IC + layer-backtest screening."""
    if not args.factor:
        _err("--factor 是必填参数。示例: mf single --factor momentum_12_1")
        return 2

    try:
        cfg = _load_cfg(args)
    except Exception as exc:
        _err(f"配置加载失败: {exc}")
        return 1

    if getattr(args, "show_config", False):
        from config.loader import print_config
        print_config(cfg)
        return 0

    if not getattr(args, "quiet", False):
        _print_header("Multi-Factor 单因子分析  (mf single)", cfg)

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
            uni_display = f"{len(symbols)} 只（动态池）"
        elif universe is not None:
            membership = None
            symbols = universe
            uni_display = f"{len(symbols)} 只"
        else:
            membership = None
            symbols = None
            uni_display = "全部"
        if not getattr(args, "quiet", False):
            print(f"  股票池    : {uni_display}")

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
                    print(f"\n{'─'*56}\n  因子: {factor_name}\n{'─'*56}")
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
                )
                _run_diag = getattr(args, "ic_decay_diagnostics", False)
                report = pipe.run(
                    config=rc,
                    run_ic_decay_diagnostics=_run_diag,
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
                        _err(f"IC 衰减诊断失败（非致命）: {_de}")
                report.save(factor_out)
                results.append(factor_name)
            except Exception as exc:
                _err(f"因子 {factor_name!r} 运行失败: {exc}")
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
        _err(f"运行异常: {exc}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
#  mf batch
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_batch(args: argparse.Namespace) -> int:
    """Full-batch factor validation."""
    try:
        cfg = _load_cfg(args)
    except Exception as exc:
        _err(f"配置加载失败: {exc}")
        return 1

    if getattr(args, "show_config", False):
        from config.loader import print_config
        print_config(cfg)
        return 0

    if not getattr(args, "quiet", False):
        _print_header("Multi-Factor 批量因子检验  (mf batch)", cfg)

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
            uni_display = f"{len(symbols)} 只（动态池）"
        elif universe is not None:
            membership = None
            symbols = universe
            uni_display = f"{len(symbols)} 只"
        else:
            membership = None
            symbols = None
            uni_display = "全部"
        if not getattr(args, "quiet", False):
            print(f"  股票池    : {uni_display}")

        factor_list = getattr(args, "factors", None) or list(BUILTIN_FACTORS.keys())
        out_dir = Path(getattr(args, "output", None) or cfg.output.batch)
        out_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        failures = []
        for name in factor_list:
            if not getattr(args, "quiet", False):
                print(f"\n{'='*60}\n因子: {name}\n{'='*60}")
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
                )
                report = pipe.run(config=rc)
                s = report.summary_dict
                s["factor"] = name
                s["error"] = ""
                summaries.append(s)
            except Exception as exc:
                _err(f"[跳过] {name}: {exc}")
                summaries.append({"factor": name, "error": str(exc)})
                failures.append(name)

        if summaries:
            df = pd.DataFrame(summaries).set_index("factor")
            csv_path = out_dir / "factor_screening_summary.csv"
            df.to_csv(csv_path)
            print(f"\n[OK] IC 汇总表已保存：{csv_path}")

        n_ok = len(summaries) - len(failures)
        print(f"\n完成：{n_ok}/{len(summaries)} 个因子成功，{len(failures)} 个跳过。")

        # ── Phase D: 写入 run_manifest.json ──────────────────────────────
        _save_manifest(
            pipe=pipe, cfg=cfg, factors=factor_list,
            failures=failures, start_time=_t0,
            out_dir=out_dir,
        )

        return 1 if (n_ok == 0 and len(summaries) > 0) else 0

    except Exception as exc:
        _err(f"运行异常: {exc}")
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
    print(f"[mf validate] 运行套件: {suite}")
    print(f"  命令: {' '.join(cmd)}\n")

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
        print(f"缓存目录不存在: {cache_dir}")
        return 0

    if action == "info":
        entries = list(cache_dir.glob("**/*.parquet"))
        if factor_filter:
            entries = [e for e in entries if factor_filter in str(e)]
        total_mb = sum(e.stat().st_size for e in entries) / 1024 / 1024
        print(f"缓存目录    : {cache_dir}")
        print(f"因子子目录  : {len(list(cache_dir.iterdir()))}")
        print(f"Parquet 文件: {len(entries)}")
        print(f"总大小      : {total_mb:.1f} MB")
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
            print(f"  已删除: {t.name}")
        print(f"[OK] 已清除 {len(targets)} 个缓存目录。")
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
        print(f"[OK] GC 完成：删除 {removed} 个超过 {max_age_days} 天的 Parquet 文件。")
        return 0

    _err(f"未知 cache action: {action!r}")
    return 2


# ═══════════════════════════════════════════════════════════════════════════════
#  mf report  (Phase D stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_report(args: argparse.Namespace) -> int:
    print("[mf report] Phase D — 报告生成功能将在 v4.1 实现。")
    print("  目前可直接查看 artifacts/ 目录中的 CSV 文件。")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  mf composite  (v4.1 stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_composite(args: argparse.Namespace) -> int:
    print("[mf composite] v4.1 — 多因子合成功能将在 v4.1 实现。")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser
# ═══════════════════════════════════════════════════════════════════════════════

def _add_common_backtest_args(p: argparse.ArgumentParser) -> None:
    """Shared backtest parameters."""
    p.add_argument("--start",    default=None, metavar="YYYYMMDD",
                   help="回测起始日期（默认: 20200101）")
    p.add_argument("--end",      default=None, metavar="YYYYMMDD",
                   help="回测截止日期（默认: 20251231）")
    p.add_argument("--forward",  type=int, default=None, metavar="N",
                   help="预测期（交易日，默认: 21）")
    p.add_argument("--n-groups", type=int, default=None, metavar="N",
                   dest="n_groups", help="分层数（默认: 5）")
    # ── 股票池参数 ────────────────────────────────────────────────────────────
    p.add_argument("--universe", default=None, metavar="NAME",
                   help="静态股票池别名或 CSV 路径（hs300 / my_pool.csv）")
    p.add_argument("--universe-mode", default=None,
                   dest="universe_mode",
                   choices=["all", "static_file", "topn_mktcap_dynamic"],
                   help="股票池模式（默认: all）")
    p.add_argument("--universe-top-n", type=int, default=None,
                   dest="universe_top_n", metavar="N",
                   help="动态池入选数量（默认: 500；仅 topn_mktcap_dynamic 生效）")
    p.add_argument("--universe-metric", default=None,
                   dest="universe_metric",
                   choices=["total_mktcap", "free_float_mktcap"],
                   help="市值指标（默认: total_mktcap）")
    p.add_argument("--universe-rebalance-freq", default=None,
                   dest="universe_rebalance_freq",
                   choices=["annual", "semiannual", "quarterly"],
                   help="动态池调仓频率（默认: semiannual）")
    p.add_argument("--universe-effective-lag-days", type=int, default=None,
                   dest="universe_effective_lag_days", metavar="N",
                   help="决策日到生效日的滞后交易日数（默认: 1）")
    p.add_argument("--force-exit-on-universe-drop",
                   dest="force_exit_on_universe_drop",
                   type=lambda x: x.lower() in ("1", "true", "yes"),
                   default=None, metavar="BOOL",
                   help="掉池持仓是否强制换仓（true/false，默认: true）")
    # ─────────────────────────────────────────────────────────────────────────
    p.add_argument("--config",   default=None, metavar="PATH",
                   help="用户 YAML 配置文件（叠加在 default.yaml 之上）")
    p.add_argument("--show-config", action="store_true",
                   help="打印有效配置后退出（退出码 0）")


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    """Shared IO parameters."""
    p.add_argument("--output",   default=None, metavar="DIR",
                   help="输出目录覆盖")
    p.add_argument("--no-cache", action="store_true",
                   help="禁用 L2 Parquet 缓存")
    p.add_argument("--quiet",    action="store_true",
                   help="只输出 summary，不打印进度")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mf",
        description="Multi-Factor Research Framework — v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "快速开始\n"
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
            "股票池\n"
            "------\n"
            "  --universe hs300                  静态：沪深300成分股（300只）\n"
            "  --universe my_pool.csv            静态：自定义CSV（含 code 列）\n"
            "  --universe-mode topn_mktcap_dynamic  动态：市值前N（默认500，防未来函数）\n"
            "  --universe-top-n 300              动态池入选数量\n"
            "  --universe-metric free_float_mktcap  使用流通市值排序\n"
            "  （不指定 = 全部股票，约5800只）\n"
            "\n"
            "完整契约文档: docs/cli-contract.md\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="multi-factor 4.0.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── single ────────────────────────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="单因子 IC + 分层回测筛选",
        description=(
            "单因子 IC + 分层回测筛选。\n"
            "退出码: 0=成功  1=运行失败  2=参数错误"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例\n----\n"
            "  mf single --factor momentum_12_1\n"
            "  mf single --factor vwap_deviation price_strength --start 20210101\n"
            "  mf single --factor value_pb --forward 10 --no-cache\n"
            "  mf single --factor value_pb --universe hs300\n"
            "  mf single --factor value_pb --ic-decay-diagnostics\n"
        ),
    )
    p_single.add_argument(
        "--factor", nargs="+", metavar="NAME",
        help="因子名称（可多个，用空格分隔）【必填】"
    )
    p_single.add_argument(
        "--ic-decay-diagnostics", action="store_true",
        dest="ic_decay_diagnostics",
        help="回测后自动运行 IC 衰减异常诊断（6 模块），结果保存至 output/<factor>/ic_decay_diagnostics/",
    )
    _add_common_backtest_args(p_single)
    _add_common_io_args(p_single)
    p_single.set_defaults(func=_cmd_single)

    # ── batch ─────────────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="全批量因子验证",
        description=(
            "对所有注册因子运行 IC + 分层回测，输出汇总 CSV。\n"
            "退出码: 0=成功(含部分跳过)  1=全部失败  2=参数错误"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例\n----\n"
            "  mf batch\n"
            "  mf batch --factors momentum_12_1 value_pb\n"
            "  mf batch --universe hs300\n"
            "  mf batch --universe hs300 --start 20210101 --parallel 4\n"
            "  mf batch --universe my_pool.csv\n"
        ),
    )
    p_batch.add_argument(
        "--factors", nargs="+", metavar="NAME",
        help="指定因子子集（默认：全部注册因子）"
    )
    p_batch.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="并行 worker 数（-1 = CPU 核数，默认 1）"
    )
    _add_common_backtest_args(p_batch)
    _add_common_io_args(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="look-ahead / 数据质量验证套件",
        description="通过 pytest 运行验证测试套件。退出码: 0=全部通过  1=有失败",
    )
    p_val.add_argument(
        "--suite",
        choices=["lookahead", "quality", "all"],
        default="all",
        help="运行哪个套件（默认: all）",
    )
    p_val.add_argument(
        "--verbose", "-v", action="store_true",
        help="输出详细断言信息"
    )
    p_val.set_defaults(func=_cmd_validate)

    # ── cache ─────────────────────────────────────────────────────────────────
    p_cache = sub.add_parser(
        "cache",
        help="缓存管理（info / clear / gc）",
        description="管理 L2 Parquet 磁盘缓存。退出码: 0=成功  1=失败",
    )
    p_cache.add_argument(
        "action",
        choices=["info", "clear", "gc"],
        nargs="?",
        default="info",
        help="操作类型（默认: info）",
    )
    p_cache.add_argument(
        "--factor", default=None, metavar="NAME",
        help="限定特定因子（用于 clear）"
    )
    p_cache.add_argument(
        "--dir", default=None, metavar="PATH",
        help="缓存目录（默认: cache/）"
    )
    p_cache.add_argument(
        "--days", type=int, default=30, metavar="N",
        help="GC：删除超过 N 天未访问的条目（默认: 30）"
    )
    p_cache.set_defaults(func=_cmd_cache)

    # ── report  (Phase D stub) ────────────────────────────────────────────────
    p_rep = sub.add_parser(
        "report",
        help="报告生成（Phase D — v4.1 实现）",
    )
    p_rep.add_argument("--artifact", default=None, metavar="PATH",
                       help="artifacts 目录路径")
    p_rep.add_argument("--format", choices=["html", "pdf", "md"], default="html",
                       help="输出格式（默认: html）")
    p_rep.set_defaults(func=_cmd_report)

    # ── composite  (v4.1 stub) ────────────────────────────────────────────────
    p_comp = sub.add_parser(
        "composite",
        help="多因子合成（v4.1 — 尚未实现）",
    )
    p_comp.add_argument("--factors", nargs="+", metavar="NAME",
                        help="参与合成的因子列表")
    p_comp.add_argument("--method",
                        choices=["equal", "icir", "pca"], default="icir",
                        help="合成方法（默认: icir）")
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
