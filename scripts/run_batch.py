#!/usr/bin/env python
"""
scripts/run_batch.py
====================
批量因子检验脚本：对所有内置因子运行 FactorPipeline，
输出 IC 汇总表和各因子分层回测统计。

用法
----
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"

# 使用 default.yaml 默认配置
.venv\\Scripts\\python.exe scripts/run_batch.py

# 覆盖参数
.venv\\Scripts\\python.exe scripts/run_batch.py --forward 10 --start 20210101

# 指定用户配置文件
.venv\\Scripts\\python.exe scripts/run_batch.py --config config/my_config.yaml

参数优先级: CLI > --config > config/default.yaml
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from config.loader import load_config, print_config
from factor_framework.pipeline   import FactorPipeline
from factor_framework.factor_zoo import BUILTIN_FACTORS


def main() -> None:
    parser = argparse.ArgumentParser(description="批量因子检验（新主路径，消费 config/default.yaml）")
    parser.add_argument("--config",   default=None, help="用户 YAML 配置文件路径")
    parser.add_argument("--start",    default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end",      default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--forward",  type=int, default=None, help="预测期（天）")
    parser.add_argument("--n-groups", type=int, default=None, dest="n_groups", help="分层数")
    parser.add_argument("--no-cache", action="store_true", help="禁用 L2 磁盘缓存")
    parser.add_argument("--output",   default=None, help="输出目录覆盖")
    parser.add_argument("--show-config", action="store_true", help="打印有效配置后退出")
    args = parser.parse_args()

    # ── 构建 CLI overrides ────────────────────────────────────────────────────
    overrides: dict = {}
    if args.start:    overrides["backtest.start"]   = args.start
    if args.end:      overrides["backtest.end"]     = args.end
    if args.forward:  overrides["backtest.forward"] = args.forward
    if args.n_groups: overrides["backtest.n_groups"] = args.n_groups
    if args.output:   overrides["output.batch"]     = args.output
    if args.no_cache: overrides["cache.cache_dir"]  = None

    # ── 加载配置 ─────────────────────────────────────────────────────────────
    cfg = load_config(user_config=args.config, overrides=overrides)

    if args.show_config:
        print("\n【有效配置】")
        print_config(cfg)
        return

    out_dir = Path(cfg.output.batch)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  Multi-Factor 批量因子检验  (v3.3 新主路径)")
    print("=" * 64)
    print(f"  数据目录  : {cfg.data.stocks_dir}")
    print(f"  时间范围  : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f"  预测期    : {cfg.backtest.forward} 天")
    print(f"  分层数    : {cfg.backtest.n_groups}")
    print(f"  缓存目录  : {cfg.cache.cache_dir}")
    print(f"  输出目录  : {out_dir}")
    print("=" * 64 + "\n")

    pipe = FactorPipeline(
        stocks_dir  = ROOT / cfg.data.stocks_dir,
        stock_basic = ROOT / cfg.data.stock_basic,
        verbose     = cfg.parallel.verbose,
        cache_dir   = cfg.cache.cache_dir,
    )
    pipe.register_builtins()

    summaries = []
    for name in list(BUILTIN_FACTORS.keys()):
        print(f"\n{'='*60}\n因子: {name}\n{'='*60}")
        try:
            report = pipe.run(
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
            )
            s = report.summary_dict
            s["factor"] = name
            summaries.append(s)
        except Exception as exc:
            print(f"  [跳过] {name}: {exc}")

    if summaries:
        df = pd.DataFrame(summaries).set_index("factor")
        csv_path = out_dir / "ic_summary.csv"
        df.to_csv(csv_path)
        print(f"\n✓ IC 汇总表已保存：{csv_path}")
        print(df[["mean_ic", "icir", "win_rate", "t_stat"]].to_string())


if __name__ == "__main__":
    main()
