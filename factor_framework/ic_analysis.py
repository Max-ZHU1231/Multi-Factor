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
) -> pd.Series:
    """
    逐期计算因子与未来收益的截面相关系数（IC）。

    Parameters
    ----------
    factor_panel : (日期 × 股票) 因子面板
    return_panel : (日期 × 股票) 未来收益面板（已与因子对齐到同一日期，即因子日→未来收益）
    method       : 'rank'（Rank IC，推荐）或 'normal'（Normal IC / Pearson）

    Returns
    -------
    pd.Series，index = 日期，values = IC 值
    """
    common_dates = factor_panel.index.intersection(return_panel.index)
    ic_series = {}

    for date in common_dates:
        f = factor_panel.loc[date].dropna()
        r = return_panel.loc[date].dropna()
        common_stocks = f.index.intersection(r.index)
        if len(common_stocks) < 5:
            ic_series[date] = np.nan
            continue

        f_val = f.reindex(common_stocks)
        r_val = r.reindex(common_stocks)

        if method == "rank":
            f_val = f_val.rank()
            r_val = r_val.rank()

        corr, _ = stats.pearsonr(f_val, r_val)
        ic_series[date] = float(corr)

    return pd.Series(ic_series, name="IC")


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
    price_panel:       pd.DataFrame,
    forward_periods:   List[int] = (1, 5, 10, 20, 60),
    method:            str = "rank",
) -> pd.DataFrame:
    """
    计算不同预测期的 IC，分析因子信息的衰减速度。

    Parameters
    ----------
    factor_panel    : (日期 × 股票) 因子面板
    price_panel     : (日期 × 股票) 收盘价面板
    forward_periods : 预测期列表（天数）
    method          : 'rank' 或 'normal'

    Returns
    -------
    pd.DataFrame，columns = ['forward','mean_ic','std_ic','icir','win_rate','t_stat']
    """
    rows = []
    for fwd in forward_periods:
        # 构建未来 fwd 日收益率面板
        ret_panel = price_panel.pct_change(fwd, axis=0).shift(-fwd)
        ic_series = compute_ic(factor_panel, ret_panel, method=method)
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
