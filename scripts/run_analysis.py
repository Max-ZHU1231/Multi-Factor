#!/usr/bin/env python
"""
scripts/run_analysis.py
=======================
全量因子分析入口脚本（v3.3 新主路径）。

用法
----
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"

# 使用 default.yaml 默认配置
.venv\\Scripts\\python.exe scripts/run_analysis.py

# 覆盖单个参数
.venv\\Scripts\\python.exe scripts/run_analysis.py --forward 10 --start 20210101

# 指定用户配置文件（叠加在 default.yaml 之上）
.venv\\Scripts\\python.exe scripts/run_analysis.py --config config/my_config.yaml

参数优先级
----------
CLI 参数 > --config 文件 > config/default.yaml

输出
----
output/factor_analysis/
    ic_summary.csv
    ic_series/<factor>.csv
    layer_stats/<factor>.csv
    plots/
        01_ic_timeseries.png
        02_cumulative_ic.png
        03_ic_summary_heatmap.png
        04_layer_nav/<factor>.png
        05_factor_corr_heatmap.png
        06_cluster_dendrogram.png
        07_cluster_heatmap.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

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
    p.add_argument("--config",   default=None,       help="用户 YAML 配置文件路径")
    p.add_argument("--start",    default=None,        help="开始日期 YYYYMMDD")
    p.add_argument("--end",      default=None,        help="结束日期 YYYYMMDD")
    p.add_argument("--forward",  type=int, default=None, help="预测期（交易日）")
    p.add_argument("--n-groups", type=int, default=None, dest="n_groups", help="分层数")
    p.add_argument("--output",   default=None,        help="输出目录覆盖")
    p.add_argument("--no-cache", action="store_true", help="禁用 L2 磁盘缓存")
    p.add_argument("--show-config", action="store_true", help="打印有效配置后退出")
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
        print("\n【有效配置】")
        print_config(cfg)
        return

    # ── 3. 参数回显 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Multi-Factor 全量因子分析  (v3.3 新主路径)")
    print("=" * 64)
    print(f"  数据目录  : {cfg.data.stocks_dir}")
    print(f"  时间范围  : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f"  预测期    : {cfg.backtest.forward} 天")
    print(f"  分层数    : {cfg.backtest.n_groups}")
    print(f"  缓存目录  : {cfg.cache.cache_dir}")
    print(f"  输出目录  : {cfg.output.factor_analysis}")
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

