#!/usr/bin/env python
"""
scripts/run_factor_screening.py
================================
全量因子检验脚本：对因子库中所有 28 个内置因子运行完整的
IC 分析 + 分层回测，生成横向对比报告并保存到 artifacts/。

用法
----
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"

# 全量检验（使用 default.yaml 默认参数）
.venv\\Scripts\\python.exe scripts/run_factor_screening.py

# 覆盖时间范围
.venv\\Scripts\\python.exe scripts/run_factor_screening.py --start 20180101 --end 20251231

# 仅检验某一类因子
.venv\\Scripts\\python.exe scripts/run_factor_screening.py --category momentum

# 快速模式（减少股票数）
.venv\\Scripts\\python.exe scripts/run_factor_screening.py --max-stocks 500

参数优先级: CLI > --config > config/default.yaml
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 强制 stdout/stderr 使用 UTF-8（解决 Windows GBK 重定向问题）─────────────
# 用 reconfigure() (Python 3.7+) 或包装 buffer
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config.loader import load_config, print_config
from factor_framework.pipeline   import FactorPipeline
from factor_framework.factor_zoo import BUILTIN_FACTORS
from factor_framework.factors.registry import REGISTRY
from factor_framework.factors.meta import FactorCategory

# ── 因子分类中文标签 ──────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    FactorCategory.MOMENTUM:   "动量 / 反转",
    FactorCategory.REVERSAL:   "动量 / 反转",
    FactorCategory.VOLATILITY: "波动率",
    FactorCategory.VALUE:      "估值",
    FactorCategory.SIZE:       "规模",
    FactorCategory.VOLUME:     "量价",
    FactorCategory.LIQUIDITY:  "流动性",
    FactorCategory.TECHNICAL:  "技术分析",
}

# 展示列及中文别名
DISPLAY_COLS = [
    ("mean_ic",          "Mean IC"),
    ("std_ic",           "Std IC"),
    ("icir",             "ICIR"),
    ("win_rate",         "IC 胜率"),
    ("t_stat",           "t 统计量"),
    ("ls_annual_return", "多空年化收益"),
    ("ls_sharpe",        "多空夏普"),
    ("ls_max_drawdown",  "最大回撤"),
    ("monotone_score",   "单调性"),
    ("avg_turnover",     "平均换手率"),
]

GRADE_RULES = [
    # (mean_ic_min, icir_min, win_rate_min, monotone_min) → grade
    (0.04, 0.5, 0.55, 0.7, "★★★  优秀"),
    (0.02, 0.3, 0.52, 0.6, "★★   良好"),
    (0.01, 0.1, 0.50, 0.5, "★    合格"),
]


def _grade(row: pd.Series) -> str:
    mic  = abs(row.get("mean_ic",    0) or 0)
    icir = abs(row.get("icir",       0) or 0)
    wr   = row.get("win_rate",       0) or 0
    mono = abs(row.get("monotone_score", 0) or 0)
    for mic_min, icir_min, wr_min, mono_min, label in GRADE_RULES:
        if mic >= mic_min and icir >= icir_min and wr >= wr_min and mono >= mono_min:
            return label
    return "✗    不显著"


def _get_factor_list(category_filter: str | None) -> list[str]:
    """按类别筛选因子，保持注册顺序。"""
    all_names = list(BUILTIN_FACTORS.keys())
    if not category_filter:
        return all_names

    target = category_filter.upper()
    result = []
    for name in all_names:
        meta = REGISTRY.get(name)
        if meta and target in (meta.category.name.upper(), CATEGORY_LABELS.get(meta.category, "").upper()):
            result.append(name)
    if not result:
        # fallback: 子字符串匹配
        result = [n for n in all_names if target in n.upper()]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="全量因子检验：28 个内置因子 IC 分析 + 分层回测横向对比"
    )
    parser.add_argument("--config", default=None, help=" YAML ")
    parser.add_argument("--start", default=None, help=" YYYYMMDD")
    parser.add_argument("--end", default=None, help=" YYYYMMDD")
    parser.add_argument("--forward", type=int, default=None, help="（）， 21")
    parser.add_argument("--n-groups",   type=int, default=None, dest="n_groups")
    parser.add_argument("--category",   default=None,
 help="（momentum/volatility/value/size/volume/liquidity/technical）")
    parser.add_argument("--max-stocks", type=int, default=None, dest="max_stocks",
 help="（，）")
 parser.add_argument("--no-cache", action="store_true", help=" L2 ")
 parser.add_argument("--output", default=None, help="")
    parser.add_argument("--log-file",   default=None, dest="log_file",
 help="（UTF-8）")
 parser.add_argument("--show-config",action="store_true", help="")
    args = parser.parse_args()

    # ── 可选：把 stdout 同时写到 log file（Tee）────────────────────────────
    _log_fh = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(log_path, "w", encoding="utf-8")

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    s.write(data)
            def flush(self):
                for s in self._streams:
                    s.flush()

        sys.stdout = _Tee(sys.stdout, _log_fh)   # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, _log_fh)   # type: ignore[assignment]

    # ── 构建 CLI overrides ────────────────────────────────────────────────────
    overrides: dict = {}
    if args.start:    overrides["backtest.start"]    = args.start
    if args.end:      overrides["backtest.end"]      = args.end
    if args.forward:  overrides["backtest.forward"]  = args.forward
    if args.n_groups: overrides["backtest.n_groups"] = args.n_groups
    if args.output:   overrides["output.batch"]      = args.output
    if args.no_cache: overrides["cache.cache_dir"]   = None

    cfg = load_config(user_config=args.config, overrides=overrides)

    if args.show_config:
        print_config(cfg)
        return

    out_dir = Path(cfg.output.batch)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 打印标题 ──────────────────────────────────────────────────────────────
    DIVIDER = "=" * 72
    print(f"\n{DIVIDER}")
    print(" Multi-Factor ")
    print(DIVIDER)
    print(f" : {ROOT / cfg.data.stocks_dir}")
    print(f" : {cfg.backtest.start} ~ {cfg.backtest.end}")
    print(f" : {cfg.backtest.forward} （）")
    print(f" : {cfg.backtest.n_groups}")
    print(f" IC : {cfg.ic.method}")
    print(f" : winsorize={cfg.preprocess.winsorize}"
          f"  standardize={cfg.preprocess.standardize}"
          f"  neutralize={cfg.preprocess.neutralize}")
          print(f" : {cfg.backtest.periods_per_year} rf={cfg.backtest.rf}")
          print(f" : {cfg.cache.cache_dir}")
          print(f" : {out_dir}")
    print(DIVIDER + "\n")

    # ── 初始化 Pipeline ───────────────────────────────────────────────────────
    stocks_dir_path = ROOT / cfg.data.stocks_dir
    stock_basic_path = ROOT / cfg.data.stock_basic

    pipe = FactorPipeline(
        stocks_dir  = stocks_dir_path,
        stock_basic = stock_basic_path,
        verbose     = cfg.parallel.verbose,
        cache_dir   = cfg.cache.cache_dir,
    )
    pipe.register_builtins()

    # 快速测试：限制股票数量
    if args.max_stocks:
        all_csvs = sorted(stocks_dir_path.glob("*.csv"))[:args.max_stocks]
        # Override engine's stock list
        pipe.engine.stocks_dir = stocks_dir_path
        _orig_glob = stocks_dir_path.glob
        _limited   = all_csvs
        pipe.engine._stock_paths_override = _limited
        print(f" [] {len(_limited)} \n")

    # ── 确定待检验因子列表 ────────────────────────────────────────────────────
    factor_names = _get_factor_list(args.category)
    n_total      = len(factor_names)
    print(f" : {n_total} \n")

    # ── 逐因子运行 ────────────────────────────────────────────────────────────
    summaries    : list[dict] = []
    failed       : list[tuple[str, str]] = []
    t_start_all  = time.time()

    for i, name in enumerate(factor_names, 1):
        meta = REGISTRY.get(name)
        cat_label = CATEGORY_LABELS.get(meta.category, "其他") if meta else "未知"
        print(f"\n[{i:02d}/{n_total}] {name}  ({cat_label})")
        print("-" * 56)

        t0 = time.time()
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
                ic_forward_list  = cfg.ic.forward_list,
                periods_per_year = cfg.backtest.periods_per_year,
                rf               = cfg.backtest.rf,
                cost_per_side    = cfg.backtest.cost_per_side,
                resample_monthly = cfg.backtest.resample_monthly,
            )

            s = report.summary_dict.copy()
            s["factor"]        = name
            s["category"]      = cat_label
            s["display_name"]  = meta.display_name if meta else name
            s["elapsed_s"]     = round(time.time() - t0, 1)
            s["grade"]         = _grade(s)
            summaries.append(s)

            # 简洁打印关键指标
            mean_ic = s.get("mean_ic", float("nan"))
            icir    = s.get("icir",    float("nan"))
            win_r   = s.get("win_rate",float("nan"))
            ls_ret  = s.get("ls_annual_return", float("nan"))
            ls_sr   = s.get("ls_sharpe",float("nan"))
            print(f"  Mean IC={mean_ic:+.4f}  ICIR={icir:+.3f}  "
                  f"Win={win_r:.1%}  多空年化={ls_ret:+.1%}  夏普={ls_sr:.2f}")
                  print(f" : {s['grade']} : {s['elapsed_s']}s")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            print(f" [] {exc} ({elapsed}s)")
            failed.append((name, str(exc)))

    total_elapsed = time.time() - t_start_all

    # ── 汇总表输出 ────────────────────────────────────────────────────────────
    if not summaries:
        print("\n[] 。")
        return

    df = pd.DataFrame(summaries)
    df = df.set_index("factor")

    # 按 |Mean IC| 降序排列
    if "mean_ic" in df.columns:
        df = df.reindex(df["mean_ic"].abs().sort_values(ascending=False).index)

    # 重命名展示列
    col_map = {old: new for old, new in DISPLAY_COLS}
    df_display = df[[c for c, _ in DISPLAY_COLS if c in df.columns]].rename(columns=col_map)

    # 格式化数值
    fmt = {
        "Mean IC":      "{:+.4f}",
        "Std IC":       "{:.4f}",
        "ICIR":         "{:+.3f}",
        "IC 胜率":      "{:.1%}",
        "t 统计量":     "{:+.2f}",
        "多空年化收益": "{:+.1%}",
        "多空夏普":     "{:.2f}",
        "最大回撤":     "{:.1%}",
        "单调性":       "{:.3f}",
        "平均换手率":   "{:.2%}",
    }

    print(f"\n\n{'═'*72}")
    print(" （ |Mean IC| ）")
    print(f"{'═'*72}")

    # 按类别分组打印
    categories_order = ["动量 / 反转", "波动率", "估值", "规模", "量价", "流动性", "技术分析"]
    for cat in categories_order:
        cat_df = df[df["category"] == cat] if "category" in df.columns else pd.DataFrame()
        if cat_df.empty:
            continue
        print(f"\n── {cat} ──")
        cat_display = df_display.loc[cat_df.index]
        for factor_name, row in cat_display.iterrows():
            grade = df.loc[factor_name, "grade"] if "grade" in df.columns else ""
            disp  = df.loc[factor_name, "display_name"] if "display_name" in df.columns else factor_name
            ic_val  = row.get("Mean IC", float("nan"))
            icir_v  = row.get("ICIR",    float("nan"))
            wr_val  = row.get("IC 胜率", float("nan"))
            ls_ret  = row.get("多空年化收益", float("nan"))
            ls_sr   = row.get("多空夏普",     float("nan"))
            print(f"  {factor_name:<25} {disp:<20}  "
                  f"IC={ic_val:+.4f}  ICIR={icir_v:+.3f}  "
                  f"Win={wr_val:.0%}  多空={ls_ret:+.1%}  SR={ls_sr:.2f}  {grade}")

    # ── 完整数值表格打印 ──────────────────────────────────────────────────────
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 50)

    print(f"\n\n{'─'*72}")
    print(" （）")
    print(f"{'─'*72}")
    print(df_display.to_string())

    # ── 分级汇总 ──────────────────────────────────────────────────────────────
    if "grade" in df.columns:
        grade_counts = df["grade"].value_counts()
        print(f"\n\n{'─'*72}")
        print(" ")
        print(f"{'─'*72}")
        for grade_label, cnt in grade_counts.items():
            print(f" {grade_label} : {cnt} ")

    # ── 失败记录 ──────────────────────────────────────────────────────────────
    if failed:
        print(f"\n\n{'─'*72}")
        print(f" {len(failed)} ：")
        for fname, err in failed:
            print(f" [] {fname}: {err}")

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "factor_screening_summary.csv"
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n\n✓ ：{csv_path}")

    # 保存排行榜（只保留关键列）
    rank_cols = ["category", "display_name"] + [c for c in col_map if c in df.columns] + ["grade", "elapsed_s"]
    rank_df = df[[c for c in rank_cols if c in df.columns]].rename(columns=col_map)
    rank_csv = out_dir / "factor_ranking.csv"
    rank_df.to_csv(rank_csv, encoding="utf-8-sig")
    print(f"✓ ：{rank_csv}")

    print(f"\n{'='*72}")
    print(f" ！ {len(summaries)} ，{len(failed)} ")
    print(f" : {total_elapsed/60:.1f} ")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
