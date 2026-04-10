#!/usr/bin/env python
"""
scripts/run_batch.py
====================
批量因子检验脚本：并行对所有内置因子运行 FactorPipeline，
输出 IC 汇总表和各因子分层回测统计。

用法
----
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"
.venv\\Scripts\\python.exe scripts/run_batch.py [--start YYYYMMDD] [--end YYYYMMDD]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from factor_framework.pipeline  import FactorPipeline
from factor_framework.factor_zoo import BUILTIN_FACTORS


def main() -> None:
    parser = argparse.ArgumentParser(description="批量因子检验")
    parser.add_argument("--start", default="20180101", help="开始日期（YYYYMMDD）")
    parser.add_argument("--end",   default="20261231", help="结束日期（YYYYMMDD）")
    parser.add_argument("--forward", type=int, default=21, help="预测期（天）")
    parser.add_argument(
        "--cache-dir", default=None,
        help="磁盘缓存目录（指定后第二次运行大幅加速，如 cache/）",
    )
    parser.add_argument("--output", default="output/batch_results", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = FactorPipeline(
        stocks_dir  = ROOT / "Stocks",
        stock_basic = ROOT / "股票列表-stock_basic.csv",
        verbose     = True,
        cache_dir   = args.cache_dir,
    )
    pipe.register_builtins()

    summaries = []
    for name in list(BUILTIN_FACTORS.keys()):
        print(f"\n{'='*60}\n因子: {name}\n{'='*60}")
        try:
            report = pipe.run(
                factor_name      = name,
                start            = args.start,
                end              = args.end,
                forward          = args.forward,
                resample_monthly = True,
                periods_per_year = 12,
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
