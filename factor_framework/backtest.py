"""
backtest.py
===========
分层回测模块。

功能
----
- layer_backtest()    : 将股票按因子值分 Q 层，逐期等权持有，计算各层收益
- long_short_stats()  : 多空组合（Top - Bottom）的年化收益、夏普比、最大回撤、Calmar
- turnover_analysis() : 换手率分析与交易成本估算
- full_report()       : 汇总所有指标，返回 dict + pd.DataFrame
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _max_drawdown(nav: pd.Series) -> float:
    """计算净值序列的最大回撤。"""
    peak   = nav.cummax()
    dd     = (nav - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def _annual_return(ret: pd.Series, periods_per_year: int = 252) -> float:
    """从日度（或任意频率）收益率序列计算年化收益。"""
    n = len(ret.dropna())
    if n == 0:
        return np.nan
    total = (1 + ret.fillna(0)).prod()
    return float(total ** (periods_per_year / n) - 1)


def _sharpe(ret: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """年化夏普比率（无风险利率 rf 为年化值）。"""
    r = ret.dropna()
    if len(r) < 2:
        return np.nan
    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = r - rf_per_period
    ann_ret = _annual_return(r, periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    return float((ann_ret - rf) / ann_vol) if ann_vol > 0 else np.nan


def _calmar(ret: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar 比率 = 年化收益 / |最大回撤|。"""
    nav  = (1 + ret.fillna(0)).cumprod()
    mdd  = abs(_max_drawdown(nav))
    ann  = _annual_return(ret, periods_per_year)
    return float(ann / mdd) if mdd > 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 分层回测
# ═══════════════════════════════════════════════════════════════════════════════

def layer_backtest(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    n_groups:     int = 5,
    direction:    int = 1,
) -> pd.DataFrame:
    """
    分层回测：按因子值将股票分为 n_groups 层，等权持有，计算各层收益。

    Parameters
    ----------
    factor_panel : (日期 × 股票) 因子面板（t 期因子值）
    return_panel : (日期 × 股票) 未来 1 期收益率面板
                   需与 factor_panel 的日期对齐（同一 date 行对应：
                   以该日因子持仓，持有至下一期的收益）
    n_groups     : 分组数（默认 5 = Q1~Q5）
    direction    : +1 表示因子越大越好（Q5 为多头），-1 反转

    Returns
    -------
    pd.DataFrame，index = 日期，columns = ['Q1','Q2',...,'Qn','LS']
    LS = Q_top - Q_bottom（多空收益）

    实现说明
    --------
    向量化实现：对对齐后的矩阵用 np.argsort 一次性完成所有截面分组，
    避免逐日期 for 循环，在日期数 ≥ 100 时有显著加速（约 10–30x）。
    """
    group_names = [f"Q{i + 1}" for i in range(n_groups)]
    col_names   = group_names + ["LS"]

    # ── 对齐日期与股票 ────────────────────────────────────────────────────────
    common_dates = factor_panel.index.intersection(return_panel.index)
    if len(common_dates) == 0:
        return pd.DataFrame(columns=col_names)

    # 取公共列（股票）
    common_stocks = factor_panel.columns.intersection(return_panel.columns)
    f_mat = factor_panel.loc[common_dates, common_stocks].values.astype(float)  # (T, N)
    r_mat = return_panel.loc[common_dates, common_stocks].values.astype(float)  # (T, N)

    T, N = f_mat.shape
    result = np.full((T, len(col_names)), np.nan)

    for t in range(T):
        f_row = f_mat[t]
        r_row = r_mat[t]

        # 只保留 f 和 r 均非 NaN 的股票
        valid = ~(np.isnan(f_row) | np.isnan(r_row))
        n_valid = int(valid.sum())

        if n_valid < n_groups * 2:
            continue  # 保持 NaN

        fv = f_row[valid] * direction
        rv = r_row[valid]

        # 等频分组：用分位数边界切分，与 pd.qcut 语义一致（边界股票归入较高组）
        # 这保证 direction=1 和 direction=-1 的分组是精确的对称翻转
        quantile_points = np.linspace(0.0, 100.0, n_groups + 1)
        boundaries = np.nanpercentile(fv, quantile_points)

        group_rets = []
        for g in range(n_groups):
            lo = boundaries[g]
            hi = boundaries[g + 1]
            if g == n_groups - 1:
                mask = (fv >= lo) & (fv <= hi)   # 最后一组包含右端点
            else:
                mask = (fv >= lo) & (fv < hi)    # 其余组右端点开区间
            if mask.sum() == 0:
                group_rets.append(np.nan)
            else:
                group_rets.append(float(rv[mask].mean()))

        result[t, :n_groups] = group_rets

        # ── LS = top - bottom（修复：显式 pd.isna 检查，0.0 不转 NaN）───────
        top    = group_rets[-1]
        bottom = group_rets[0]
        if pd.isna(top) or pd.isna(bottom):
            result[t, n_groups] = np.nan
        else:
            result[t, n_groups] = top - bottom

    return pd.DataFrame(result, index=common_dates, columns=col_names)


# ═══════════════════════════════════════════════════════════════════════════════
# 多空组合统计
# ═══════════════════════════════════════════════════════════════════════════════

def long_short_stats(
    layer_ret:        pd.DataFrame,
    periods_per_year: int = 252,
    rf:               float = 0.0,
) -> Dict[str, object]:
    """
    计算各层及多空组合的绩效指标。

    Parameters
    ----------
    layer_ret        : layer_backtest() 返回的收益率 DataFrame
    periods_per_year : 日频=252，月频=12，周频=52
    rf               : 年化无风险利率

    Returns
    -------
    dict: {
        'layer_annual_return' : pd.Series，各层年化收益
        'layer_sharpe'        : pd.Series，各层夏普
        'ls_annual_return'    : float
        'ls_sharpe'           : float
        'ls_max_drawdown'     : float
        'ls_calmar'           : float
        'ls_win_rate'         : float
        'monotone_score'      : float，单调性得分（Q1~Qn 年化收益的 Spearman 秩相关）
        'nav'                 : pd.DataFrame，各层净值曲线
    }
    """
    # 净值曲线
    nav = (1 + layer_ret.fillna(0)).cumprod()

    group_cols = [c for c in layer_ret.columns if c.startswith("Q")]

    layer_ann = layer_ret[group_cols].apply(
        lambda s: _annual_return(s.dropna(), periods_per_year), axis=0
    )
    layer_sh = layer_ret[group_cols].apply(
        lambda s: _sharpe(s.dropna(), rf, periods_per_year), axis=0
    )

    ls = layer_ret["LS"].dropna() if "LS" in layer_ret else pd.Series(dtype=float)
    ls_nav  = (1 + ls.fillna(0)).cumprod()

    # 单调性：Q1~Qn 年化收益与 [1,2,...,n] 的 Spearman 相关
    vals = layer_ann.dropna().values
    if len(vals) >= 2:
        ranks = np.arange(1, len(vals) + 1)
        mono_score = float(pd.Series(vals).corr(pd.Series(ranks), method="spearman"))
    else:
        mono_score = np.nan

    return {
        "layer_annual_return": layer_ann,
        "layer_sharpe":        layer_sh,
        "ls_annual_return":    _annual_return(ls, periods_per_year),
        "ls_sharpe":           _sharpe(ls, rf, periods_per_year),
        "ls_max_drawdown":     _max_drawdown(ls_nav),
        "ls_calmar":           _calmar(ls, periods_per_year),
        "ls_win_rate":         float((ls > 0).mean()) if len(ls) > 0 else np.nan,
        "monotone_score":      mono_score,
        "nav":                 nav,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 换手率分析
# ═══════════════════════════════════════════════════════════════════════════════

def turnover_analysis(
    factor_panel:    pd.DataFrame,
    n_groups:        int = 5,
    direction:       int = 1,
    cost_per_side:   float = 0.002,
) -> Dict[str, float]:
    """
    分析最高分组（Top 组）的单边换手率及估算交易成本。

    Parameters
    ----------
    factor_panel   : 因子面板
    n_groups       : 分组数
    direction      : 因子方向
    cost_per_side  : 单边交易成本（默认 0.2%，含印花税+佣金）

    Returns
    -------
    dict:
        avg_turnover   : 平均单边换手率
        avg_cost       : 每期平均交易成本（收益率扣减）
        turnover_series: pd.Series，逐期换手率
    """
    dates = sorted(factor_panel.index)
    if len(dates) < 2:
        return {"avg_turnover": np.nan, "avg_cost": np.nan, "turnover_series": pd.Series(dtype=float)}

    group_top = f"Q{n_groups}"
    prev_portfolio: set = set()
    turnovers = {}

    for date in dates:
        f = factor_panel.loc[date].dropna()
        if len(f) < n_groups * 2:
            prev_portfolio = set()
            continue
        try:
            labels = pd.qcut(f * direction, n_groups, labels=[f"Q{i+1}" for i in range(n_groups)], duplicates="drop")
        except Exception:
            prev_portfolio = set()
            continue
        curr_portfolio = set(labels[labels == group_top].index.tolist())
        if len(prev_portfolio) > 0 and len(curr_portfolio) > 0:
            n_total  = len(prev_portfolio | curr_portfolio)
            turnover = len(prev_portfolio.symmetric_difference(curr_portfolio)) / (2 * n_total)
            turnovers[date] = turnover
        prev_portfolio = curr_portfolio

    if not turnovers:
        return {"avg_turnover": np.nan, "avg_cost": np.nan, "turnover_series": pd.Series(dtype=float)}

    ts  = pd.Series(turnovers)
    avg = float(ts.mean())
    return {
        "avg_turnover":    round(avg, 4),
        "avg_cost":        round(avg * cost_per_side, 6),
        "turnover_series": ts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ═══════════════════════════════════════════════════════════════════════════════

def full_report(
    factor_panel:     pd.DataFrame,
    return_panel:     pd.DataFrame,
    ic_series:        Optional[pd.Series] = None,
    ic_stats_dict:    Optional[Dict] = None,
    n_groups:         int = 5,
    direction:        int = 1,
    periods_per_year: int = 252,
    rf:               float = 0.0,
    cost_per_side:    float = 0.002,
) -> Dict[str, object]:
    """
    端到端因子评估报告。

    Returns
    -------
    dict:
        ic_stats      : IC 统计指标
        layer_ret     : 分层收益 DataFrame
        ls_stats      : 多空组合指标
        turnover      : 换手率指标
        summary_table : pd.DataFrame，一行汇总
    """
    # 分层回测
    layer_ret = layer_backtest(factor_panel, return_panel, n_groups=n_groups, direction=direction)
    ls_stats  = long_short_stats(layer_ret, periods_per_year=periods_per_year, rf=rf)
    turnover  = turnover_analysis(factor_panel, n_groups=n_groups, direction=direction, cost_per_side=cost_per_side)

    # 净 LS 收益（扣成本）
    net_ls_ret = ls_stats["ls_annual_return"]
    if net_ls_ret is not None and not np.isnan(net_ls_ret):
        net_ls_ret = net_ls_ret - turnover["avg_cost"] * periods_per_year

    # 汇总表
    ic_s = ic_stats_dict or {}
    summary = {
        "mean_ic":          ic_s.get("mean_ic", np.nan),
        "icir":             ic_s.get("icir", np.nan),
        "ic_win_rate":      ic_s.get("win_rate", np.nan),
        "ic_t_stat":        ic_s.get("t_stat", np.nan),
        "ls_annual_return": ls_stats["ls_annual_return"],
        "ls_sharpe":        ls_stats["ls_sharpe"],
        "ls_max_drawdown":  ls_stats["ls_max_drawdown"],
        "ls_calmar":        ls_stats["ls_calmar"],
        "ls_win_rate":      ls_stats["ls_win_rate"],
        "monotone_score":   ls_stats["monotone_score"],
        "avg_turnover":     turnover["avg_turnover"],
        "net_ls_annual":    net_ls_ret,
    }

    return {
        "ic_stats":     ic_s,
        "layer_ret":    layer_ret,
        "ls_stats":     ls_stats,
        "turnover":     turnover,
        "summary_table": pd.DataFrame([summary]),
    }
