"""
profile_ic_diagnostics.py
=========================
IC 衰减诊断系统性能 profiling 脚本。

用法
----
    # 生成合成数据基准（无需实际股票数据）
    python scripts/profile_ic_diagnostics.py --mode synthetic --T 1500 --N 500

    # 使用真实 value_pb 数据（需先运行过一次完整流程）
    python scripts/profile_ic_diagnostics.py --mode real

    # 将结果写入 performance_report.md
    python scripts/profile_ic_diagnostics.py --mode synthetic --write-report

输出
----
    - 控制台：每模块耗时、Top-10 热点函数
    - performance_report.md：Baseline / After P0 / After P1 对比表
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── 路径设置 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from factor_framework.analytics.ic_decay_diagnostics import ICDecayDiagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# 合成数据生成
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_data(
    T: int = 1500,
    N: int = 500,
    seed: int = 42,
) -> tuple:
    """
    生成合成因子/价格/市值/行业面板用于 profiling。

    Returns
    -------
    factor_panel, price_panel, mktcap_panel, industry_map
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T, freq="B")
    stocks = [f"stock_{i:04d}" for i in range(N)]

    # 因子面板（带自相关，模拟真实 PB）
    raw = rng.standard_normal((T, N))
    factor_arr = np.zeros((T, N))
    factor_arr[0] = raw[0]
    for t in range(1, T):
        factor_arr[t] = 0.92 * factor_arr[t - 1] + raw[t]

    # 引入 5% 缺失
    miss_mask = rng.random((T, N)) < 0.05
    factor_arr[miss_mask] = np.nan

    factor_panel = pd.DataFrame(factor_arr, index=dates, columns=stocks)

    # 价格面板（几何随机游走）
    log_ret = 0.0002 + 0.015 * rng.standard_normal((T, N))
    log_price = np.cumsum(log_ret, axis=0) + 8.0   # 从 ~e^8 ≈ 3000 开始
    price_panel = pd.DataFrame(np.exp(log_price), index=dates, columns=stocks)

    # 市值面板
    mktcap_arr = np.exp(rng.normal(23, 2, (T, N)))  # log市值 ~ N(23,2)
    mktcap_panel = pd.DataFrame(mktcap_arr, index=dates, columns=stocks)

    # 行业分类（10个行业，均匀分配）
    n_inds = 10
    ind_labels = [f"industry_{i:02d}" for i in range(n_inds)]
    ind_assign = [ind_labels[i % n_inds] for i in range(N)]
    industry_map = pd.Series(ind_assign, index=stocks, name="industry")

    return factor_panel, price_panel, mktcap_panel, industry_map


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级计时
# ═══════════════════════════════════════════════════════════════════════════════

def time_modules(
    diag: ICDecayDiagnostics,
    n_splits: int = 3,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    分别计时每个诊断模块，返回 {模块名: 耗时秒} 字典。
    """
    modules = {
        "M1_time_alignment":    lambda: diag.module1_time_alignment(),
        "M2_incremental_ic":    lambda: diag.module2_incremental_ic(),
        "M3_exposure_strip":    lambda: diag.module3_exposure_strip(),
        "M4_sample_bias":       lambda: diag.module4_sample_bias(),
        "M5_factor_halflife":   lambda: diag.module5_factor_halflife(),
        "M6_robustness":        lambda: diag.module6_robustness(n_splits=n_splits),
    }

    timings: Dict[str, float] = {}
    for name, fn in modules.items():
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            if verbose:
                print(f"  [{name}] ERROR: {e}")
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        if verbose:
            print(f"  {name:30s}  {elapsed:7.2f}s")

    return timings


# ═══════════════════════════════════════════════════════════════════════════════
# cProfile 热点分析
# ═══════════════════════════════════════════════════════════════════════════════

def profile_run_all(diag: ICDecayDiagnostics, top_n: int = 15) -> str:
    """
    使用 cProfile 对 run_all() 进行全量 profiling，返回 Top-N 热点文本。
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        diag.run_all(verbose=False)
    finally:
        pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(top_n)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 数据规模统计
# ═══════════════════════════════════════════════════════════════════════════════

def data_stats(diag: ICDecayDiagnostics) -> Dict:
    T, N = diag.factor_panel.shape
    k_max = max(diag.forward_list)
    return {
        "T (交易日)": T,
        "N (股票数)": N,
        "k_max (最大 forward)": k_max,
        "forward_list": diag.forward_list,
        "industry_map_available": diag.industry_map is not None,
        "mktcap_panel_available": diag.mktcap_panel is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IC Diagnostics Performance Profiler")
    parser.add_argument("--mode",   choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--T",      type=int, default=1500, help="交易日数（synthetic 模式）")
    parser.add_argument("--N",      type=int, default=500,  help="股票数（synthetic 模式）")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--write-report", action="store_true", help="将结果写入 performance_report.md")
    parser.add_argument("--tag",    default="After_P0_P1", help="报告标签（Baseline / After_P0 / After_P1）")
    parser.add_argument("--no-profile", action="store_true", help="跳过 cProfile（仅模块计时）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"IC 衰减诊断性能 Profiling  [mode={args.mode}, tag={args.tag}]")
    print("=" * 70)

    # ── 准备数据 ──────────────────────────────────────────────────────────
    if args.mode == "synthetic":
        print(f"\n生成合成数据 T={args.T}, N={args.N}, seed={args.seed} ...")
        factor_panel, price_panel, mktcap_panel, industry_map = make_synthetic_data(
            T=args.T, N=args.N, seed=args.seed
        )
    else:
        # 读取真实数据（需要已有 artifact 缓存或直接 pickle）
        raise NotImplementedError("real 模式需手动加载因子面板，请使用 synthetic 模式进行基准测试。")

    # ── 构造诊断器（__init__ 计时）────────────────────────────────────────
    print("\n构造 ICDecayDiagnostics (含 __init__ 预计算) ...")
    t_init = time.perf_counter()
    diag = ICDecayDiagnostics(
        factor_panel = factor_panel,
        price_panel  = price_panel,
        mktcap_panel = mktcap_panel,
        industry_map = industry_map,
        forward_list = [1, 5, 10, 21, 60],
        ic_method    = "rank",
        factor_name  = "synthetic",
    )
    t_init_elapsed = time.perf_counter() - t_init
    print(f"  __init__ 耗时: {t_init_elapsed:.2f}s")

    # ── 数据规模 ──────────────────────────────────────────────────────────
    stats = data_stats(diag)
    print("\n数据规模:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── 模块级计时 ────────────────────────────────────────────────────────
    print("\n各模块耗时:")
    timings = time_modules(diag, verbose=True)
    total_elapsed = sum(timings.values())
    print(f"\n  总计 (M1~M6): {total_elapsed:.2f}s")
    print(f"  含 __init__:  {total_elapsed + t_init_elapsed:.2f}s")

    # ── cProfile 热点 ────────────────────────────────────────────────────
    profile_text = ""
    if not args.no_profile:
        print("\n运行 cProfile (run_all 完整流程) ...")
        # 需要第二次构造（清空缓存，保证 M3 中性化重新计算）
        diag2 = ICDecayDiagnostics(
            factor_panel = factor_panel,
            price_panel  = price_panel,
            mktcap_panel = mktcap_panel,
            industry_map = industry_map,
            forward_list = [1, 5, 10, 21, 60],
            ic_method    = "rank",
            factor_name  = "synthetic",
        )
        profile_text = profile_run_all(diag2, top_n=15)
        print("\nTop-15 热点函数 (cumulative time):")
        # 只打印关键列
        for line in profile_text.split("\n")[:35]:
            print(line)

    # ── 写入 performance_report.md ─────────────────────────────────────
    if args.write_report:
        report_path = ROOT / "performance_report.md"
        _write_report(
            report_path  = report_path,
            tag          = args.tag,
            stats        = stats,
            t_init       = t_init_elapsed,
            timings      = timings,
            total        = total_elapsed,
            profile_text = profile_text,
        )
        print(f"\n报告已写入: {report_path}")


def _write_report(
    report_path: Path,
    tag:         str,
    stats:       Dict,
    t_init:      float,
    timings:     Dict[str, float],
    total:       float,
    profile_text: str,
) -> None:
    """追加写入（或创建）performance_report.md。"""
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    block = f"""
## {tag}

**数据规模**: T={stats['T (交易日)']}, N={stats['N (股票数)']}, k_max={stats['k_max (最大 forward)']}, forward={stats['forward_list']}

### 各模块耗时

| 模块 | 耗时 (s) | 占比 |
|------|----------|------|
"""
    for name, t in timings.items():
        pct = t / total * 100 if total > 0 else 0
        block += f"| {name} | {t:.2f} | {pct:.1f}% |\n"

    block += f"| **__init__ 预计算** | {t_init:.2f} | — |\n"
    block += f"| **合计 (M1~M6)** | {total:.2f} | 100% |\n"
    block += f"| **含 __init__** | {total + t_init:.2f} | — |\n"

    if profile_text:
        # 保留前 30 行 profile 摘要
        prof_lines = "\n".join(profile_text.split("\n")[:32])
        block += f"\n### Top-15 热点函数 (cProfile, cumulative)\n\n```\n{prof_lines}\n```\n"

    if existing.strip() == "":
        header = "# IC 衰减诊断系统 Performance Report\n\n"
        header += "记录每次优化前后的耗时基准。\n"
        header += "\n---\n"
        content = header + block
    else:
        content = existing + "\n---\n" + block

    report_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
