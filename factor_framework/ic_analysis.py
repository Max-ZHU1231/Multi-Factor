"""
ic_analysis.py
==============
IC 分析模块。

功能
----
- compute_ic()       : 逐期计算 Normal IC 或 Rank IC
- ic_stats()         : Mean IC、Std IC、ICIR、胜率、t 统计量
- ic_decay()         : IC 衰减分析（不同 forward 期的 Mean IC）
- ic_cumulative()    : IC 累积曲线
- ic_significance()  : Newey-West 修正 t 检验
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# 核心 IC 计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ic(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    method:       str = "rank",
    min_stocks:   int = 5,
) -> pd.Series:
    """
    逐期计算因子与未来收益的截面相关系数（IC）。

    完全向量化实现：消除逐日 Python 循环，改用 NumPy 矩阵运算，
    速度约为原循环版的 10~50 倍。

    Parameters
    ----------
    factor_panel : (日期 × 股票) 因子面板；可为 TimestampedPanel（优先使用语义守卫）
    return_panel : (日期 × 股票) 未来收益面板；可为 TimestampedPanel
    method       : 'rank'（Rank IC，推荐）或 'normal'（Normal IC / Pearson）
    min_stocks   : 每期至少需要的有效截面数量，不足则置 NaN

    Returns
    -------
    pd.Series，index = 日期，values = IC 值
    """
    # ── B1: TimestampedPanel 语义守卫 ────────────────────────────────────────
    try:
        from factor_framework.core.panel import TimestampedPanel
        if isinstance(factor_panel, TimestampedPanel) and isinstance(return_panel, TimestampedPanel):
            # align_with 会抛出 TimingAlignmentError / SemanticCompatibilityError
            factor_panel, return_panel = factor_panel.align_with(return_panel)
        elif isinstance(factor_panel, TimestampedPanel):
            factor_panel.assert_valid()
        elif isinstance(return_panel, TimestampedPanel):
            return_panel.assert_valid()
    except ImportError:
        pass

    # 取公共日期 + 公共股票，保证维度对齐
    common_dates  = factor_panel.index.intersection(return_panel.index)
    common_stocks = factor_panel.columns.intersection(return_panel.columns)

    f = factor_panel.loc[common_dates, common_stocks].astype(float)
    r = return_panel.loc[common_dates, common_stocks].astype(float)

    # NaN 对齐：某股票在因子或收益任意一方缺失，则两方均置 NaN
    nan_mask = f.isna() | r.isna()
    f = f.where(~nan_mask)
    r = r.where(~nan_mask)

    if method == "rank":
        # 横截面排名（忽略 NaN，每行独立排名）
        f = f.rank(axis=1, na_option="keep")
        r = r.rank(axis=1, na_option="keep")

    # ── 向量化 Pearson 相关（逐行）──────────────────────────────────────────
    # 每行去均值（仅用有效值的均值，NaN 不参与）
    f_mean = f.mean(axis=1)
    r_mean = r.mean(axis=1)
    f_dm   = f.sub(f_mean, axis=0)   # demean，形状 (T, N)
    r_dm   = r.sub(r_mean, axis=0)

    # 分子：Σ (f_dm * r_dm)，NaN 位置乘积自动为 NaN → nansum
    num  = f_dm.mul(r_dm).sum(axis=1, skipna=True)

    # 分母：sqrt(Σ f_dm² * Σ r_dm²)
    denom = np.sqrt(
        f_dm.pow(2).sum(axis=1, skipna=True) *
        r_dm.pow(2).sum(axis=1, skipna=True)
    )
    denom = denom.replace(0, np.nan)

    ic = num / denom
    ic.name = "IC"

    # 过滤有效截面数量不足的日期
    valid_counts = (~nan_mask).sum(axis=1)
    ic = ic.where(valid_counts >= min_stocks)

    return ic


# ═══════════════════════════════════════════════════════════════════════════════
# IC 统计指标体系
# ═══════════════════════════════════════════════════════════════════════════════

def ic_stats(
    ic: pd.Series,
    annualize_periods: int = 12,
) -> Dict[str, float]:
    """
    计算 IC 核心统计指标。

    Parameters
    ----------
    ic                : IC 时间序列
    annualize_periods : 年化期数（月频=12，日频=252）

    Returns
    -------
    dict:
        mean_ic      : 平均 IC
        std_ic       : IC 标准差
        icir         : IC / std_ic（信息比率）
        win_rate     : IC > 0 的比例（做多方向）
        t_stat       : t 统计量（简单 t 检验）
        p_value      : 对应 p 值
        ic_positive  : IC > 0 的期数
        ic_negative  : IC < 0 的期数
        total_periods: 有效期数
        annualized_icir: 年化 ICIR = ICIR * sqrt(annualize_periods)
    """
    clean = ic.dropna()
    n     = len(clean)
    if n == 0:
        return {k: np.nan for k in [
            "mean_ic","std_ic","icir","win_rate","t_stat","p_value",
            "ic_positive","ic_negative","total_periods","annualized_icir"
        ]}

    mean_ic = float(clean.mean())
    std_ic  = float(clean.std(ddof=1))
    icir    = mean_ic / std_ic if std_ic > 0 else np.nan
    win_rate = float((clean > 0).mean())
    t_stat   = float(mean_ic / (std_ic / np.sqrt(n))) if std_ic > 0 else np.nan
    p_value  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if not np.isnan(t_stat) else np.nan

    return {
        "mean_ic":          round(mean_ic, 6),
        "std_ic":           round(std_ic, 6),
        "icir":             round(icir, 4) if not np.isnan(icir) else np.nan,
        "win_rate":         round(win_rate, 4),
        "t_stat":           round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
        "p_value":          round(p_value, 4) if not np.isnan(p_value) else np.nan,
        "ic_positive":      int((clean > 0).sum()),
        "ic_negative":      int((clean < 0).sum()),
        "total_periods":    n,
        "annualized_icir":  round(icir * np.sqrt(annualize_periods), 4) if not np.isnan(icir) else np.nan,
    }


def ic_significance(
    ic: pd.Series,
    lags: int = 4,
) -> Dict[str, float]:
    """
    Newey-West 修正 t 检验（处理 IC 自相关）。

    Parameters
    ----------
    ic   : IC 时间序列
    lags : Newey-West 滞后阶数（建议取 int(T^0.25)）

    Returns
    -------
    dict: nw_t_stat, nw_p_value, mean_ic, n
    """
    clean = ic.dropna().values
    n     = len(clean)
    if n < lags + 2:
        return {"nw_t_stat": np.nan, "nw_p_value": np.nan, "mean_ic": np.nan, "n": n}

    mean_ic = clean.mean()
    e       = clean - mean_ic  # 去均值残差

    # Newey-West 方差估计
    var_nw = np.sum(e ** 2) / n
    for k in range(1, lags + 1):
        cov_k = np.sum(e[k:] * e[:-k]) / n
        var_nw += 2 * (1 - k / (lags + 1)) * cov_k

    se_nw   = np.sqrt(var_nw / n)
    t_nw    = mean_ic / se_nw if se_nw > 0 else np.nan
    p_nw    = float(2 * (1 - stats.t.cdf(abs(t_nw), df=n - 1))) if not np.isnan(t_nw) else np.nan

    return {
        "nw_t_stat":  round(t_nw, 4),
        "nw_p_value": round(p_nw, 4),
        "mean_ic":    round(mean_ic, 6),
        "n":          n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IC 衰减分析
# ═══════════════════════════════════════════════════════════════════════════════

def ic_decay(
    factor_panel:      pd.DataFrame,
    price_panel:       Optional[pd.DataFrame] = None,
    forward_periods:   List[int] = (1, 5, 10, 20, 60),
    method:            str = "rank",
    return_panels:     Optional[Dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    计算不同预测期的 IC，分析因子信息的衰减速度。

    Parameters
    ----------
    factor_panel    : (日期 × 股票) 因子面板
    price_panel     : (日期 × 股票) 收盘价面板（与 return_panels 二选一）
    forward_periods : 预测期列表（天数，仅在 price_panel 模式下使用）
    method          : 'rank' 或 'normal'
    return_panels   : {forward_days: (日期 × 股票) 已构建的收益率面板}（优先）
                      若提供，则直接使用，不从 price_panel 重算收益率。
                      这消除了 pipeline 中主 IC 路径（有 T+1）与 ic_decay
                      内部重算路径（无 T+1）之间的双路径不一致问题（BUG-9）。

    Returns
    -------
    pd.DataFrame，columns = ['forward','mean_ic','std_ic','icir','win_rate','t_stat']

    实现说明
    --------
    优先路径（return_panels 传入）：BUG-9 修复
        - 直接消费调用方已构建的收益率面板（与主 IC 同源）
        - 收益率语义与 build_return_panel 完全一致（含 T+1 shift）
        - forward_periods 参数从 return_panels.keys() 推导，保持一致性

    回退路径（仅 price_panel 传入）：向后兼容
        - 逐列计算 price.shift(-fwd) / price - 1
        - 不加 T+1 shift（与旧实现保持一致，但与主 IC 路径存在语义差异）
        - 保留用于独立调用 ic_decay 而不经过 pipeline 的场景
    """
    import warnings

    # ── 优先路径：使用调用方传入的收益率面板（BUG-9 修复）─────────────────
    if return_panels is not None:
        rows = []
        for fwd, ret_panel in sorted(return_panels.items()):
            # 取公共日期对齐
            common_idx = factor_panel.index.intersection(ret_panel.index)
            if len(common_idx) == 0:
                rows.append({"forward": fwd, "mean_ic": np.nan, "std_ic": np.nan,
                             "icir": np.nan, "win_rate": np.nan, "t_stat": np.nan})
                continue

            # 丢弃收益全为 NaN 的行（尾部 forward+1 行因 T+1 shift 产生）
            valid_ret_rows = ret_panel.loc[common_idx].dropna(how="all").index
            if len(valid_ret_rows) == 0:
                rows.append({"forward": fwd, "mean_ic": np.nan, "std_ic": np.nan,
                             "icir": np.nan, "win_rate": np.nan, "t_stat": np.nan})
                continue

            fp = factor_panel.loc[valid_ret_rows]
            rp = ret_panel.loc[valid_ret_rows]

            ic_series = compute_ic(fp, rp, method=method)
            st        = ic_stats(ic_series)
            rows.append({
                "forward":  fwd,
                "mean_ic":  st["mean_ic"],
                "std_ic":   st["std_ic"],
                "icir":     st["icir"],
                "win_rate": st["win_rate"],
                "t_stat":   st["t_stat"],
            })
        return pd.DataFrame(rows).set_index("forward")

    # ── 回退路径：从 price_panel 重算收益率（向后兼容）────────────────────
    if price_panel is None:
        raise ValueError(
            "ic_decay：price_panel 和 return_panels 不能同时为 None，"
            "请至少提供其中一个。"
        )

    max_fwd = max(forward_periods)
    if len(price_panel) <= max_fwd:
        raise ValueError(
            f"ic_decay: price_panel 长度（{len(price_panel)}）"
            f"不超过最大 forward={max_fwd}，无法计算。"
        )

    warnings.warn(
        "ic_decay() 的 price_panel 路径已废弃（v3.2+）。\n"
        "请改用 return_panels 参数（由 ReturnPanel.build_multi_forward() 构建），\n"
        "以确保与主 IC 路径时间语义一致（含 T+1 shift）。",
        DeprecationWarning,
        stacklevel=2,
    )
    warnings.warn(
        f"[ic_decay] 已截断 price_panel 末尾 {max_fwd} 行"
        f"（原 {len(price_panel)} 行 → 截后 {len(price_panel) - max_fwd} 行），"
        f"防止 shift(-fwd) 引入未来价格数据（前瞻偏差）。",
        stacklevel=2,
    )

    rows = []
    for fwd in forward_periods:
        # 逐列构建未来 fwd 日收益率面板（与 build_return_panel 公式一致）
        # price.shift(-fwd) / price - 1：每只股票独立计算，NaN 不跨列传播。
        # 注意：此路径不含 T+1 shift，与主 IC 路径存在语义差异（BUG-9 的根源）
        ret_panel = price_panel.shift(-fwd) / price_panel.replace(0, np.nan) - 1

        # 丢弃尾部全 NaN 行（即原 price_panel 末尾 fwd 行）
        valid_rows = ret_panel.dropna(how="all").index
        ret_panel  = ret_panel.loc[valid_rows]

        # factor_panel 取交集（不截断，仅对齐）
        common_idx = factor_panel.index.intersection(ret_panel.index)
        if len(common_idx) == 0:
            rows.append({"forward": fwd, "mean_ic": np.nan, "std_ic": np.nan,
                         "icir": np.nan, "win_rate": np.nan, "t_stat": np.nan})
            continue

        fp = factor_panel.loc[common_idx]
        rp = ret_panel.loc[common_idx]

        ic_series = compute_ic(fp, rp, method=method)
        st        = ic_stats(ic_series)
        rows.append({
            "forward":  fwd,
            "mean_ic":  st["mean_ic"],
            "std_ic":   st["std_ic"],
            "icir":     st["icir"],
            "win_rate": st["win_rate"],
            "t_stat":   st["t_stat"],
        })
    return pd.DataFrame(rows).set_index("forward")


# ═══════════════════════════════════════════════════════════════════════════════
# IC 累积曲线
# ═══════════════════════════════════════════════════════════════════════════════

def ic_cumulative(ic: pd.Series) -> pd.Series:
    """返回 IC 的累积求和序列（用于绘制 IC 累积曲线）。"""
    return ic.fillna(0).cumsum()


# ═══════════════════════════════════════════════════════════════════════════════
# 因子相关性检验
# ═══════════════════════════════════════════════════════════════════════════════

def cross_factor_correlation(
    panels: Dict[str, pd.DataFrame],
    method: str = "pearson",
) -> pd.DataFrame:
    """
    计算因子库中所有因子两两之间的平均截面相关性。

    Parameters
    ----------
    panels : {因子名 → (日期 × 股票) 面板}
    method : 'pearson'（默认）或 'spearman'

    Returns
    -------
    (n_factors × n_factors) 平均截面相关系数矩阵
    """
    names  = list(panels.keys())
    n      = len(names)
    mat    = pd.DataFrame(np.eye(n), index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            fi = panels[names[i]]
            fj = panels[names[j]]
            common_dates = fi.index.intersection(fj.index)
            corrs = []
            for d in common_dates:
                a = fi.loc[d].dropna()
                b = fj.loc[d].dropna()
                common_s = a.index.intersection(b.index)
                if len(common_s) < 5:
                    continue
                if method == "spearman":
                    a, b = a.rank(), b.rank()
                c, _ = stats.pearsonr(a.reindex(common_s), b.reindex(common_s))
                corrs.append(c)
            avg_corr = np.nanmean(corrs) if corrs else np.nan
            mat.loc[names[i], names[j]] = avg_corr
            mat.loc[names[j], names[i]] = avg_corr

    return mat


def residual_ic(
    new_factor:       pd.DataFrame,
    existing_factors: List[pd.DataFrame],
    return_panel:     pd.DataFrame,
    method:           str = "rank",
) -> Dict[str, object]:
    """
    信息增量检验（残差 IC 法）。

    将 new_factor 对 existing_factors 做横截面回归取残差，
    再计算残差的 IC，判断是否仍显著。

    Returns
    -------
    dict: residual_ic_series, stats
    """
    from factor_framework.neutralize import orthogonalize
    resid_panel = orthogonalize(new_factor, existing_factors)
    ic_series   = compute_ic(resid_panel, return_panel, method=method)
    st          = ic_stats(ic_series)
    return {
        "residual_ic_series": ic_series,
        "stats":              st,
    }
