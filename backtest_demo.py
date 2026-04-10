"""
backtest_demo.py
================
三因子实例回测演示
------------------
因子 1: momentum_12_1  —— 12-1 月经典价格动量
因子 2: vol_20d        —— 20 日波动率（低波动异象，负向信号）
因子 3: value_pb       —— 价值因子（1/市净率）

流程（v2.8 批量流水线，一次读盘，共享缓存）：
  1. build_panel_batch  → 三因子面板一次性构建
  2. build_return_panel → 月度收益率面板（forward=21 天）
  3. build_panel        → 收盘价面板（供 IC 衰减分析）
  4. run_batch_from_panels → 截面标准化 / IC 分析 / 分层回测 / 换手率分析
  5. 打印汇总 + 保存 CSV + 绘制对比图

用法：
  & ".venv\\Scripts\\python.exe" backtest_demo.py
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# 中文字体（Windows 环境）
matplotlib.rcParams["font.family"]        = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from factor_framework.pipeline import FactorPipeline

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    stocks_dir       = "Stocks/",
    stock_basic      = "股票列表-stock_basic.csv",
    start            = "20200101",
    end              = "20251231",
    forward          = 21,          # 预测期（月度调仓）
    n_groups         = 5,
    ic_method        = "rank",      # Rank IC（Spearman）
    standardize      = "rank",      # 截面 rank 标准化
    winsorize        = True,
    neutralize       = False,
    ic_forward_list  = [1, 5, 10, 21, 60],
    periods_per_year = 12,          # v2.9: 月度重采样后用 12（而非日频 252）
    resample_monthly = True,        # v2.9: 月末重采样，消除日频换仓偏差
    rf               = 0.02,        # 年化无风险利率
    cost_per_side    = 0.002,       # 单边交易成本 0.2%
    n_jobs           = 8,
    output_dir       = Path("output/backtest_demo"),
)

# 三个示例因子
FACTORS = ["momentum_12_1", "vol_20d", "value_pb"]

# ═══════════════════════════════════════════════════════════════════════════════
# 初始化 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 64)
print("  三因子实例回测演示（v2.9 修复版：T+1滞后 + 月度重采样）")
print("=" * 64)
print(f"\n  因子  : {', '.join(FACTORS)}")
print(f"  区间  : {CFG['start']} ~ {CFG['end']}")
print(f"  预测期: {CFG['forward']} 交易日（月度）")
print(f"  分层数: {CFG['n_groups']}")

t0 = time.time()

pipe = FactorPipeline(
    stocks_dir  = CFG["stocks_dir"],
    stock_basic = CFG["stock_basic"],
    verbose     = True,
)
pipe.register_builtins(FACTORS)
engine = pipe.engine

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 批量构建三因子面板（一次遍历所有 CSV，共享 IO 缓存）
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n[Step 1/4] 批量构建因子面板（一次读盘）...")
t1 = time.time()

all_panels = engine.build_panel_batch(
    factor_names = FACTORS,
    start        = CFG["start"],
    end          = CFG["end"],
    fast_mode    = True,
    n_jobs       = CFG["n_jobs"],
)

# 过滤空面板
valid_panels = {n: p for n, p in all_panels.items() if not p.empty}
print(f"\n  有效因子面板: {list(valid_panels.keys())}")
for name, panel in valid_panels.items():
    print(f"    {name:<20}: {panel.shape[0]} 日 × {panel.shape[1]} 只股票"
          f"  (非空率 {panel.notna().mean().mean():.1%})")

print(f"\n  ✓ Step 1 完成，耗时 {time.time() - t1:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: 构建收益率面板
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n[Step 2/4] 构建月度收益率面板...")
t2 = time.time()

return_panel = engine.build_return_panel(
    forward   = CFG["forward"],
    start     = CFG["start"],
    end       = CFG["end"],
    fast_mode = True,
)
print(f"  收益率面板: {return_panel.shape[0]} 日 × {return_panel.shape[1]} 只股票")
print(f"  ✓ Step 2 完成，耗时 {time.time() - t2:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: 构建收盘价面板（IC 衰减分析用）
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n[Step 3/4] 构建收盘价面板（IC 衰减用）...")
t3 = time.time()

engine.register("__close__", lambda df: df["收盘价"])
close_panel = engine.build_panel(
    "__close__",
    start     = CFG["start"],
    end       = CFG["end"],
    fast_mode = True,
    n_jobs    = CFG["n_jobs"],
)
del engine._registry["__close__"]

print(f"  收盘价面板: {close_panel.shape[0]} 日 × {close_panel.shape[1]} 只股票")
print(f"  ✓ Step 3 完成，耗时 {time.time() - t3:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: 批量因子检验（截面处理 + IC 分析 + 分层回测 + 换手率）
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n[Step 4/4] 批量因子检验...")
t4 = time.time()

if not valid_panels:
    print("  ✗ 无有效因子面板，退出。")
    raise SystemExit(1)

reports = pipe.run_batch_from_panels(
    factor_panels    = valid_panels,
    return_panel     = return_panel,
    close_panel      = close_panel,
    forward          = CFG["forward"],
    n_groups         = CFG["n_groups"],
    ic_method        = CFG["ic_method"],
    standardize      = CFG["standardize"],
    winsorize        = CFG["winsorize"],
    neutralize       = CFG["neutralize"],
    ic_forward_list  = CFG["ic_forward_list"],
    periods_per_year = CFG["periods_per_year"],
    resample_monthly = CFG["resample_monthly"],
    rf               = CFG["rf"],
    cost_per_side    = CFG["cost_per_side"],
)

print(f"\n  ✓ Step 4 完成，耗时 {time.time() - t4:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 打印汇总报告
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 64)
print("  回测结果汇总")
print("═" * 64)

for name, rpt in reports.items():
    rpt.print_summary()

# ═══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ═══════════════════════════════════════════════════════════════════════════════

CFG["output_dir"].mkdir(parents=True, exist_ok=True)
for name, rpt in reports.items():
    rpt.save(str(CFG["output_dir"]))

# 保存汇总对比表
summary_rows = [rpt.summary_dict for rpt in reports.values()]
summary_df = pd.DataFrame(summary_rows).set_index("factor")
summary_df.to_csv(CFG["output_dir"] / "comparison_summary.csv")
print(f"\n  汇总对比表已保存至 {CFG['output_dir'] / 'comparison_summary.csv'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 绘图：IC 累积曲线 + 分层净值 + 汇总对比条形图
# ═══════════════════════════════════════════════════════════════════════════════

FACTOR_LABELS = {
    "momentum_12_1": "动量因子 (12-1月)",
    "vol_20d":        "低波动因子 (20日)",
    "value_pb":       "价值因子 (1/PB)",
}
COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]

n_factors = len(reports)
fig_rows  = n_factors + 1   # 每因子一行(IC + 净值) + 底部汇总行

fig = plt.figure(figsize=(16, 4 * fig_rows), constrained_layout=True)
gs  = fig.add_gridspec(fig_rows, 3)

fig.suptitle("三因子回测结果对比", fontsize=16, fontweight="bold", y=1.01)

for row_idx, (name, rpt) in enumerate(reports.items()):
    color = COLORS[row_idx % len(COLORS)]
    label = FACTOR_LABELS.get(name, name)

    # ── 子图 A：累积 IC 曲线 ──────────────────────────────────────────────
    ax_ic = fig.add_subplot(gs[row_idx, 0])
    cum_ic = rpt.ic_series.cumsum()
    ax_ic.plot(cum_ic.index, cum_ic.values, color=color, linewidth=1.5)
    ax_ic.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_ic.set_title(f"{label}\n累积 Rank IC", fontsize=10)
    ax_ic.set_xlabel("日期")
    ax_ic.set_ylabel("累积 IC")
    ax_ic.tick_params(axis="x", rotation=30)

    # ── 子图 B：各层累积净值 ─────────────────────────────────────────────
    ax_nav = fig.add_subplot(gs[row_idx, 1])
    layer_nav = (1 + rpt.layer_ret).cumprod()
    for col in layer_nav.columns:
        lw = 2.0 if "多空" in str(col) or "ls" in str(col).lower() else 1.0
        alpha = 1.0 if "多空" in str(col) or "ls" in str(col).lower() else 0.75
        ax_nav.plot(layer_nav.index, layer_nav[col], label=str(col), linewidth=lw, alpha=alpha)
    ax_nav.set_title(f"{label}\n分层累积净值", fontsize=10)
    ax_nav.set_xlabel("日期")
    ax_nav.set_ylabel("净值")
    ax_nav.legend(fontsize=7, ncol=2)
    ax_nav.tick_params(axis="x", rotation=30)

    # ── 子图 C：IC 衰减柱状图 ────────────────────────────────────────────
    ax_decay = fig.add_subplot(gs[row_idx, 2])
    if not rpt.ic_decay_df.empty:
        decay = rpt.ic_decay_df["mean_ic"]
        bar_colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in decay.values]
        ax_decay.bar(decay.index.astype(str), decay.values, color=bar_colors, alpha=0.8)
        ax_decay.axhline(0, color="gray", linewidth=0.8)
        ax_decay.set_title(f"{label}\nIC 衰减（不同预测期）", fontsize=10)
        ax_decay.set_xlabel("预测期（交易日）")
        ax_decay.set_ylabel("Mean IC")

# ── 底部：三因子关键指标对比条形图 ─────────────────────────────────────────
ax_cmp = fig.add_subplot(gs[n_factors, :])

metrics = {
    "Mean IC":       [reports[n].ic_stats_.get("mean_ic",      float("nan")) for n in reports],
    "ICIR":          [reports[n].ic_stats_.get("icir",         float("nan")) for n in reports],
    "年化多空收益":  [reports[n].ls_stats.get("ls_annual_return", float("nan")) for n in reports],
    "多空夏普":      [reports[n].ls_stats.get("ls_sharpe",     float("nan")) for n in reports],
    "单调性得分":    [reports[n].ls_stats.get("monotone_score", float("nan")) for n in reports],
}

factor_names_plot = [FACTOR_LABELS.get(n, n) for n in reports]
x       = range(len(metrics))
width   = 0.22
offsets = [-width, 0, width]

for i, (fname, fcolor) in enumerate(zip(factor_names_plot, COLORS)):
    vals = [list(v)[i] for v in metrics.values()]
    ax_cmp.bar(
        [xi + offsets[i] for xi in x], vals,
        width=width, label=fname, color=fcolor, alpha=0.8
    )

ax_cmp.set_xticks(list(x))
ax_cmp.set_xticklabels(list(metrics.keys()), fontsize=9)
ax_cmp.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_cmp.set_title("三因子关键指标对比", fontsize=11, fontweight="bold")
ax_cmp.legend(fontsize=8)
ax_cmp.set_ylabel("指标值")

fig_path = CFG["output_dir"] / "backtest_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  对比图已保存至 {fig_path}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 最终汇总表（终端打印）
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 80)
print("  三因子关键指标汇总")
print("═" * 80)
display_cols = ["mean_ic", "std_ic", "icir", "win_rate", "t_stat",
                "ls_annual_return", "ls_sharpe", "ls_max_drawdown",
                "ls_calmar", "monotone_score", "avg_turnover"]
display_df = summary_df.reindex(columns=[c for c in display_cols if c in summary_df.columns])
display_df.index = [FACTOR_LABELS.get(n, n) for n in display_df.index]

col_rename = {
    "mean_ic":          "Mean IC",
    "std_ic":           "Std IC",
    "icir":             "ICIR",
    "win_rate":         "IC胜率",
    "t_stat":           "t统计量",
    "ls_annual_return": "年化多空收益",
    "ls_sharpe":        "多空夏普",
    "ls_max_drawdown":  "最大回撤",
    "ls_calmar":        "Calmar",
    "monotone_score":   "单调性",
    "avg_turnover":     "平均换手率",
}
display_df = display_df.rename(columns=col_rename)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print(display_df.T.to_string())

total_time = time.time() - t0
print(f"\n\n  ✓ 三因子回测全部完成，总耗时 {total_time:.1f}s")
print(f"  输出目录: {CFG['output_dir'].resolve()}")
print("=" * 64)
