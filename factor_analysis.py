"""
factor_analysis.py
==================
对因子库中全部内置因子进行系统性分析：

(1) 单因子 IC 时序分析        —— 逐期 IC 折线 + 累积 IC 曲线
(2) 单因子核心 IC 指标汇总表  —— mean_IC / ICIR / 胜率 / t 统计量
(3) 单因子分层回测            —— 各层累积净值 + 多空组合统计
(4) 因子间截面相关性检验      —— 平均截面相关矩阵热力图
(5) 因子间聚类分析            —— 层次聚类树状图 + 聚类热力图

运行方式
--------
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"
.venv\\Scripts\\python.exe factor_analysis.py

输出
----
output/factor_analysis/
    ic_summary.csv                   核心 IC 指标汇总表
    ic_series/<factor>.csv           每个因子的 IC 时序
    layer_stats/<factor>.csv         每个因子的分层回测统计
    plots/
        01_ic_timeseries.png         所有因子 IC 时序（子图网格）
        02_cumulative_ic.png         所有因子累积 IC 曲线对比
        03_ic_summary_heatmap.png    IC 核心指标热力图（排名可视化）
        04_layer_nav/<factor>.png    各因子分层净值（每因子一张图）
        05_factor_corr_heatmap.png   因子相关性热力图
        06_cluster_dendrogram.png    因子层次聚类树状图
        07_cluster_heatmap.png       聚类排序后的相关性热力图
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"]      = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 确保项目根目录在路径中 ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_framework.pipeline    import FactorPipeline
from factor_framework.ic_analysis import cross_factor_correlation, ic_cumulative
from factor_framework.factor_zoo  import BUILTIN_FACTORS

# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数（按需调整）
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    stocks_dir    = "stocks/",
    stock_basic   = "股票列表-stock_basic.csv",
    start         = "20200101",
    end           = "20251231",
    forward       = 21,          # 预测期（交易日）
    n_groups      = 5,           # 分层数
    ic_method     = "rank",      # Rank IC
    standardize   = "rank",
    winsorize     = True,
    neutralize    = False,
    periods_per_year = 252,
    rf            = 0.0,
    cost_per_side = 0.002,
    output_dir    = Path("output/factor_analysis"),
    n_jobs        = 8,           # 并发线程数（FactorEngine 内部）
    # 聚类参数
    cluster_method  = "average", # linkage 方法
    n_clusters      = 4,         # 期望聚类数（用于着色）
)

ALL_FACTORS = list(BUILTIN_FACTORS.keys())

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_dirs(cfg: dict) -> None:
    base: Path = cfg["output_dir"]
    for sub in ["ic_series", "layer_stats",
                "plots/04_layer_nav"]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    print(f"  ✓ 图片已保存: {rel}")


def _color_by_value(val: float, vmin: float, vmax: float,
                    cmap_name: str = "RdYlGn") -> str:
    """将数值映射到颜色字符串（用于终端打印）。"""
    cmap = plt.get_cmap(cmap_name)
    t    = (val - vmin) / (vmax - vmin + 1e-12)
    rgba = cmap(np.clip(t, 0, 1))
    return mcolors.to_hex(rgba)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0：初始化 Pipeline，批量运行所有因子
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_reports(cfg: dict) -> dict:
    """
    批量对每个内置因子执行完整检验，收集 FactorReport 对象。
    返回 { factor_name: FactorReport }

    优化说明（v2.8）
    ----------------
    1. 调用 build_panel_batch() 一次性读取所有股票文件，构建全部因子面板
       （共享读盘缓存 + CSE，读盘只发生 1 次而非 N_FACTORS 次）。
    2. 收益率面板和收盘价面板也只构建一次。
    3. 每个因子的截面预处理 / IC / 回测通过 run_batch_from_panels() 串行完成。
    预期总耗时降低约 40–60%（vs. 逐因子 pipe.run()）。
    """
    pipe = FactorPipeline(
        stocks_dir  = cfg["stocks_dir"],
        stock_basic = cfg["stock_basic"],
        verbose     = True,
    )
    pipe.register_builtins()

    print(f"\n{'='*64}")
    print(f"  开始批量分析  共 {len(ALL_FACTORS)} 个因子")
    print(f"  时间范围: {cfg['start']} ~ {cfg['end']}")
    print(f"  预测期: forward={cfg['forward']} 天  分层数: {cfg['n_groups']}")
    print(f"  模式: 批量面板预构建（一次读盘 + CSE）")
    print(f"{'='*64}\n")

    engine = pipe.engine

    # ── Step 1：一次性批量构建所有因子面板（共享读盘缓存 + CSE）──────────────
    print(f"[批量] 构建 {len(ALL_FACTORS)} 个因子面板（build_panel_batch）...")
    all_panels = engine.build_panel_batch(
        factor_names = ALL_FACTORS,
        start        = cfg["start"],
        end          = cfg["end"],
        fast_mode    = True,
        n_jobs       = cfg.get("n_jobs", 8),
    )
    succeeded = [n for n, p in all_panels.items() if not p.empty]
    print(f"  ✓ 成功: {len(succeeded)}/{len(ALL_FACTORS)} 个因子面板已构建\n")

    # ── Step 2：构建收益率面板（仅一次）──────────────────────────────────────
    print(f"[批量] 构建收益率面板（forward={cfg['forward']} 天）...")
    return_panel = engine.build_return_panel(
        forward  = cfg["forward"],
        start    = cfg["start"],
        end      = cfg["end"],
        fast_mode= True,
    )
    print(f"  ✓ 收益率面板: {return_panel.shape}\n")

    # ── Step 3：构建收盘价面板（IC 衰减用，仅一次）───────────────────────────
    print(f"[批量] 构建收盘价面板（IC 衰减）...")
    engine.register("__close__", lambda df: df["收盘价"])
    close_panel = engine.build_panel(
        "__close__",
        start    = cfg["start"],
        end      = cfg["end"],
        fast_mode= True,
    )
    del engine._registry["__close__"]
    print(f"  ✓ 收盘价面板: {close_panel.shape}\n")

    # ── Step 4：逐因子执行检验（复用已有面板，无重复读盘）─────────────────────
    valid_panels = {n: p for n, p in all_panels.items() if not p.empty}
    reports = pipe.run_batch_from_panels(
        factor_panels    = valid_panels,
        return_panel     = return_panel,
        close_panel      = close_panel,
        forward          = cfg["forward"],
        n_groups         = cfg["n_groups"],
        ic_method        = cfg["ic_method"],
        standardize      = cfg["standardize"],
        winsorize        = cfg["winsorize"],
        neutralize       = cfg["neutralize"],
        periods_per_year = cfg["periods_per_year"],
        rf               = cfg["rf"],
        cost_per_side    = cfg["cost_per_side"],
    )

    failed = [n for n in ALL_FACTORS if n not in reports]
    if failed:
        print(f"\n⚠ 以下因子运行失败，已跳过：{failed}")
    print(f"\n✓ 成功完成 {len(reports)}/{len(ALL_FACTORS)} 个因子。\n")
    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# (1)  单因子 IC 时序分析
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ic_timeseries(reports: dict, cfg: dict) -> None:
    """
    为每个因子绘制：
      - 逐期 IC 柱状图（灰色）
      - 12 期滚动均值曲线（橙色）
    所有子图排列成网格，保存为一张大图。
    """
    names  = list(reports.keys())
    n      = len(names)
    ncols  = 4
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 2.8),
                             sharex=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax  = axes_flat[idx]
        ic  = reports[name].ic_series.dropna()

        # 柱状图
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in ic.values]
        ax.bar(range(len(ic)), ic.values, color=colors, alpha=0.6, width=1.0)

        # 滚动均值
        roll = ic.rolling(12, min_periods=6).mean()
        ax.plot(range(len(roll)), roll.values, color="#e67e22", lw=1.5, label="滚动12期均值")

        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylabel("IC", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")

        # x 轴标注年份
        if len(ic) > 0:
            step = max(1, len(ic) // 4)
            ticks = list(range(0, len(ic), step))
            labels = [str(ic.index[t])[:4] for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=6)

        # 保存单因子 IC 序列
        ic.to_csv(cfg["output_dir"] / "ic_series" / f"{name}.csv", header=True)

    # 隐藏多余子图
    for idx in range(len(names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"单因子 IC 时序分析（forward={cfg['forward']}天，{cfg['start']}~{cfg['end']}）",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    _save_fig(fig, cfg["output_dir"] / "plots" / "01_ic_timeseries.png", dpi=150)


# ═══════════════════════════════════════════════════════════════════════════════
# (2)  单因子核心 IC 指标汇总表
# ═══════════════════════════════════════════════════════════════════════════════

def build_ic_summary(reports: dict, cfg: dict) -> pd.DataFrame:
    """
    汇总所有因子的核心 IC 指标，保存 CSV 并打印排名表，
    同时输出热力图。
    """
    rows = []
    for name, rpt in reports.items():
        s = rpt.ic_stats_
        row = {
            "因子":        name,
            "均值IC":      s.get("mean_ic"),
            "IC标准差":    s.get("std_ic"),
            "ICIR":        s.get("icir"),
            "年化ICIR":    s.get("annualized_icir"),
            "IC胜率":      s.get("win_rate"),
            "t统计量":     s.get("t_stat"),
            "p值":         s.get("p_value"),
            "有效期数":    s.get("total_periods"),
            "NW_t统计量":  rpt.ic_nw.get("nw_t_stat"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("因子")
    df_sorted = df.sort_values("ICIR", ascending=False)

    # 保存 CSV
    df_sorted.to_csv(cfg["output_dir"] / "ic_summary.csv")
    print(f"\n  ✓ IC 汇总表已保存: output/factor_analysis/ic_summary.csv")

    # ── 终端打印排名表 ──────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  {'排名':<4} {'因子':<24} {'均值IC':>8} {'ICIR':>7} "
          f"{'年化ICIR':>9} {'IC胜率':>7} {'t统计量':>8} {'NW_t':>7}")
    print(f"{'─'*80}")
    for rank, (name, row) in enumerate(df_sorted.iterrows(), 1):
        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if pd.notna(v) else "  N/A  "
        print(f"  {rank:<4} {name:<24} "
              f"{_fmt(row['均值IC']):>8} "
              f"{_fmt(row['ICIR']):>7} "
              f"{_fmt(row['年化ICIR']):>9} "
              f"{_fmt(row['IC胜率']):>7} "
              f"{_fmt(row['t统计量']):>8} "
              f"{_fmt(row['NW_t统计量']):>7}")
    print(f"{'─'*80}\n")

    # ── 热力图 ───────────────────────────────────────────────────────────────
    display_cols = ["均值IC", "ICIR", "年化ICIR", "IC胜率", "t统计量", "NW_t统计量"]
    heat_data = df_sorted[display_cols].copy().astype(float)

    # 列归一化到 [0,1]，方便颜色对比
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.45 + 1.5)))
    im = ax.imshow(heat_norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(display_cols)))
    ax.set_xticklabels(display_cols, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels(heat_data.index.tolist(), fontsize=8)

    # 在格子内写数值
    for i in range(len(heat_data)):
        for j, col in enumerate(display_cols):
            val = heat_data.iloc[i, j]
            txt = f"{val:.3f}" if pd.notna(val) else "N/A"
            fc  = "black" if 0.2 < heat_norm.iloc[i, j] < 0.8 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=fc)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="归一化得分（列内排名）")
    ax.set_title("单因子核心 IC 指标汇总（按 ICIR 降序）",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    _save_fig(fig, cfg["output_dir"] / "plots" / "03_ic_summary_heatmap.png")

    return df_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# 累积 IC 曲线（所有因子对比）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_ic(reports: dict, cfg: dict) -> None:
    """绘制所有因子的累积 IC 曲线（同一坐标系对比）。"""
    n_colors = len(reports)
    cmap     = plt.get_cmap("tab20", n_colors)

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (name, rpt) in enumerate(reports.items()):
        cum_ic = ic_cumulative(rpt.ic_series)
        ax.plot(cum_ic.index, cum_ic.values,
                label=name, lw=1.2, alpha=0.85,
                color=cmap(idx))

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("所有因子累积 IC 曲线对比", fontsize=13, fontweight="bold")
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("累积 IC", fontsize=10)
    ax.legend(fontsize=6.5, ncol=4, loc="upper left",
              framealpha=0.7, bbox_to_anchor=(0, 1))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save_fig(fig, cfg["output_dir"] / "plots" / "02_cumulative_ic.png")


# ═══════════════════════════════════════════════════════════════════════════════
# (3)  单因子分层回测
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_backtest(reports: dict, cfg: dict) -> None:
    """
    为每个因子绘制分层累积净值曲线，并汇总多空统计到 CSV。
    """
    all_rows = []

    for name, rpt in reports.items():
        ls   = rpt.ls_stats
        lret = rpt.layer_ret

        # ── 分层净值图 ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # 左图：各层累积收益
        ax = axes[0]
        n_groups = len([c for c in lret.columns if c.startswith("G")])
        cmap     = plt.get_cmap("RdYlGn", n_groups + 2)

        nav_data = {}
        for col in lret.columns:
            if col == "L-S":
                continue
            cum = (1 + lret[col].fillna(0)).cumprod()
            nav_data[col] = cum
            color = cmap(list(lret.columns).index(col)) if col != "L-S" else "#333333"
            ax.plot(lret.index, cum.values, label=col, lw=1.4,
                    color=cmap(list(lret.columns).index(col)))

        ax.axhline(1, color="gray", lw=0.6, ls="--")
        ax.set_title(f"{name}  分层累积净值", fontsize=10, fontweight="bold")
        ax.set_xlabel("日期", fontsize=8)
        ax.set_ylabel("净值", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

        # 右图：多空净值 + 指标文字
        ax2 = axes[1]
        if "L-S" in lret.columns:
            ls_nav = (1 + lret["L-S"].fillna(0)).cumprod()
            ax2.plot(lret.index, ls_nav.values, color="#2c3e50", lw=1.8,
                     label="多空组合")
            ax2.fill_between(lret.index, 1, ls_nav.values,
                              where=(ls_nav.values >= 1),
                              alpha=0.25, color="#27ae60")
            ax2.fill_between(lret.index, 1, ls_nav.values,
                              where=(ls_nav.values < 1),
                              alpha=0.25, color="#e74c3c")
        ax2.axhline(1, color="gray", lw=0.6, ls="--")

        def _v(key):
            v = ls.get(key)
            return f"{v:.4f}" if isinstance(v, float) and pd.notna(v) else "N/A"

        info = (
            f"年化收益: {_v('ls_annual_return')}\n"
            f"夏普比率: {_v('ls_sharpe')}\n"
            f"最大回撤: {_v('ls_max_drawdown')}\n"
            f"Calmar:   {_v('ls_calmar')}\n"
            f"多空胜率: {_v('ls_win_rate')}\n"
            f"单调性:   {_v('monotone_score')}"
        )
        ax2.text(0.02, 0.97, info, transform=ax2.transAxes,
                 fontsize=8, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
        ax2.set_title(f"{name}  多空组合净值", fontsize=10, fontweight="bold")
        ax2.set_xlabel("日期", fontsize=8)
        ax2.set_ylabel("净值", fontsize=8)
        ax2.legend(fontsize=7, loc="lower left")
        ax2.tick_params(labelsize=7)

        fig.suptitle(f"因子分层回测：{name}", fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save_fig(fig, cfg["output_dir"] / "plots" / "04_layer_nav" / f"{name}.png")

        # ── 汇总行 ────────────────────────────────────────────────────────
        row = {
            "因子":         name,
            "ls_年化收益":  ls.get("ls_annual_return"),
            "ls_夏普比率":  ls.get("ls_sharpe"),
            "ls_最大回撤":  ls.get("ls_max_drawdown"),
            "ls_Calmar":    ls.get("ls_calmar"),
            "ls_胜率":      ls.get("ls_win_rate"),
            "单调性得分":   ls.get("monotone_score"),
        }
        ann = ls.get("layer_annual_return")
        if ann is not None and not (isinstance(ann, pd.Series) and ann.empty):
            for k, v in (ann.items() if isinstance(ann, (pd.Series, dict)) else {}.items()):
                row[f"年化收益_{k}"] = v
        all_rows.append(row)

    # 汇总 CSV
    df_ls = pd.DataFrame(all_rows).set_index("因子")
    df_ls_sorted = df_ls.sort_values("ls_夏普比率", ascending=False)
    df_ls_sorted.to_csv(cfg["output_dir"] / "layer_stats" / "all_factors_ls.csv")
    print(f"\n  ✓ 分层回测汇总已保存: output/factor_analysis/layer_stats/all_factors_ls.csv")

    # 终端打印分层回测排名
    print(f"\n{'─'*76}")
    print(f"  {'排名':<4} {'因子':<24} {'年化收益':>9} {'夏普':>7} {'最大回撤':>9} {'Calmar':>7} {'单调性':>7}")
    print(f"{'─'*76}")
    for rank, (name, row) in enumerate(df_ls_sorted.iterrows(), 1):
        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if pd.notna(v) else "  N/A  "
        print(f"  {rank:<4} {name:<24} "
              f"{_f(row['ls_年化收益']):>9} "
              f"{_f(row['ls_夏普比率']):>7} "
              f"{_f(row['ls_最大回撤']):>9} "
              f"{_f(row['ls_Calmar']):>7} "
              f"{_f(row['单调性得分']):>7}")
    print(f"{'─'*76}\n")

    return df_ls_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# (4)  因子间截面相关性检验
# ═══════════════════════════════════════════════════════════════════════════════

def plot_factor_correlation(reports: dict, cfg: dict) -> pd.DataFrame:
    """
    计算所有因子两两之间的平均截面 Spearman 相关性，
    绘制热力图并打印高相关因子对。
    """
    panels = {name: rpt.factor_panel for name, rpt in reports.items()}

    print(f"\n  计算因子截面相关矩阵（{len(panels)} × {len(panels)}）...")
    from factor_framework.ic_analysis import cross_factor_correlation
    corr_mat = cross_factor_correlation(panels, method="spearman")

    # 保存
    corr_mat.to_csv(cfg["output_dir"] / "factor_corr_matrix.csv")
    print(f"  ✓ 相关矩阵已保存: output/factor_analysis/factor_corr_matrix.csv")

    # ── 热力图 ───────────────────────────────────────────────────────────────
    n   = len(corr_mat)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.62), max(8, n * 0.55)))

    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="平均截面 Spearman 相关")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_mat.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_mat.index.tolist(), fontsize=8)

    # 数值注释
    for i in range(n):
        for j in range(n):
            val = corr_mat.iloc[i, j]
            if pd.notna(val):
                txt_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

    ax.set_title("因子截面相关性矩阵（平均 Spearman 相关）",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    _save_fig(fig, cfg["output_dir"] / "plots" / "05_factor_corr_heatmap.png")

    # ── 打印高相关因子对（|r| > 0.6）─────────────────────────────────────
    names = corr_mat.index.tolist()
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_mat.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.6:
                high_corr_pairs.append((names[i], names[j], r))

    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: -abs(x[2]))
        print(f"\n  ⚠ 高相关因子对（|r| > 0.6）：")
        print(f"  {'因子A':<24} {'因子B':<24} {'相关系数':>8}")
        print(f"  {'─'*58}")
        for a, b, r in high_corr_pairs:
            print(f"  {a:<24} {b:<24} {r:>8.4f}")
    else:
        print(f"\n  ✓ 无高相关因子对（|r| > 0.6）。")

    return corr_mat


# ═══════════════════════════════════════════════════════════════════════════════
# (5)  因子间聚类分析
# ═══════════════════════════════════════════════════════════════════════════════

def plot_factor_clustering(corr_mat: pd.DataFrame, cfg: dict) -> None:
    """
    基于因子相关矩阵做层次聚类：
    - 距离矩阵 = 1 - |r|（相关越高→距离越小）
    - 绘制树状图 + 聚类排序热力图
    """
    names    = corr_mat.index.tolist()
    n        = len(names)
    abs_corr = corr_mat.abs().fillna(0).values.copy()
    np.fill_diagonal(abs_corr, 1.0)

    # 距离矩阵
    dist_mat = 1.0 - abs_corr
    np.clip(dist_mat, 0, None, out=dist_mat)
    condensed = squareform(dist_mat, checks=False)

    # 层次聚类
    Z       = linkage(condensed, method=cfg["cluster_method"])
    labels  = fcluster(Z, t=cfg["n_clusters"], criterion="maxclust")
    cluster_map = {name: int(lbl) for name, lbl in zip(names, labels)}

    # ── 图1：树状图 ──────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(max(10, n * 0.55), 5.5))
    dn = dendrogram(
        Z,
        labels       = names,
        ax           = ax1,
        leaf_rotation= 45,
        leaf_font_size= 8.5,
        color_threshold= dist_mat.max() * 0.4,
    )
    ax1.set_title(
        f"因子层次聚类树状图（距离 = 1 − |Spearman相关|，linkage={cfg['cluster_method']}）",
        fontsize=11, fontweight="bold",
    )
    ax1.set_ylabel("聚类距离", fontsize=9)
    ax1.tick_params(axis="x", labelsize=8)
    fig1.tight_layout()
    _save_fig(fig1, cfg["output_dir"] / "plots" / "06_cluster_dendrogram.png")

    # ── 图2：聚类排序后的相关性热力图 ──────────────────────────────────────
    # 按树状图顺序重排
    ordered_names = [names[i] for i in dn["leaves"]]
    corr_ordered  = corr_mat.loc[ordered_names, ordered_names]

    fig2, ax2 = plt.subplots(figsize=(max(9, n * 0.62), max(8, n * 0.55)))
    im2 = ax2.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im2, ax=ax2, fraction=0.025, pad=0.02, label="Spearman 相关")

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(ordered_names, fontsize=8)

    # 数值注释
    for i in range(n):
        for j in range(n):
            val = corr_ordered.iloc[i, j]
            if pd.notna(val):
                txt_color = "white" if abs(val) > 0.6 else "black"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.5, color=txt_color)

    # 标注聚类边界
    cmap_c  = plt.get_cmap("Set1", cfg["n_clusters"])
    prev_cl = None
    boundary_positions = []
    for pos, name in enumerate(ordered_names):
        cl = cluster_map[name]
        if cl != prev_cl and pos > 0:
            boundary_positions.append(pos - 0.5)
        prev_cl = cl
    for bp in boundary_positions:
        ax2.axhline(bp, color="gold", lw=2)
        ax2.axvline(bp, color="gold", lw=2)

    ax2.set_title(
        f"因子聚类相关性热力图（聚类数={cfg['n_clusters']}，金线为聚类边界）",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig2.tight_layout()
    _save_fig(fig2, cfg["output_dir"] / "plots" / "07_cluster_heatmap.png")

    # ── 打印聚类结果 ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  因子聚类结果（共 {cfg['n_clusters']} 类）：")
    print(f"  {'─'*60}")
    cluster_groups: dict[int, list] = {}
    for name, cl in sorted(cluster_map.items(), key=lambda x: x[1]):
        cluster_groups.setdefault(cl, []).append(name)
    for cl in sorted(cluster_groups.keys()):
        members = "、".join(cluster_groups[cl])
        print(f"  第 {cl} 类（{len(cluster_groups[cl])} 个因子）：{members}")
    print(f"  {'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CFG.copy()
    cfg["output_dir"] = Path(cfg["output_dir"])
    _ensure_dirs(cfg)

    # Step 0：批量运行所有因子
    reports = build_all_reports(cfg)
    if not reports:
        print("❌ 未能生成任何因子报告，请检查数据路径和配置。")
        return

    print(f"\n{'='*64}")
    print(f"  开始输出分析结果（共 {len(reports)} 个因子）")
    print(f"{'='*64}")

    # (1) IC 时序图
    print("\n▶ (1) 绘制单因子 IC 时序图 ...")
    plot_ic_timeseries(reports, cfg)

    # (1+) 累积 IC 对比图
    print("\n▶     绘制累积 IC 对比曲线 ...")
    plot_cumulative_ic(reports, cfg)

    # (2) IC 核心指标汇总
    print("\n▶ (2) 生成核心 IC 指标汇总表 ...")
    ic_summary = build_ic_summary(reports, cfg)

    # (3) 分层回测
    print("\n▶ (3) 绘制单因子分层回测净值图 ...")
    ls_summary = plot_layer_backtest(reports, cfg)

    # (4) 截面相关性
    print("\n▶ (4) 计算因子截面相关性 ...")
    corr_mat = plot_factor_correlation(reports, cfg)

    # (5) 聚类分析
    print("\n▶ (5) 执行因子聚类分析 ...")
    plot_factor_clustering(corr_mat, cfg)

    # ── 最终汇总 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  ✅ 全部分析完成！输出目录：{cfg['output_dir'].resolve()}")
    print(f"{'='*64}")

    print(f"\n  生成的文件清单：")
    for p in sorted(cfg["output_dir"].resolve().rglob("*")):
        if p.is_file():
            try:
                rel = p.relative_to(ROOT.resolve())
            except ValueError:
                rel = p
            print(f"    {rel}")


if __name__ == "__main__":
    main()
