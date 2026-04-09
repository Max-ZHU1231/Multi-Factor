"""
neutralize.py
=============
因子中性化模块。

实现三种方法：
1. 回归法中性化（推荐）: 市值 + 行业 + 波动率 联合中性化
2. 分组标准化法（简易）: 行业内 Z-Score
3. 正交化处理        : Gram-Schmidt 正交化，消除与已有因子的相关性

输入/输出均为 pd.DataFrame（index=日期，columns=ts_code）。
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_residual(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """
    OLS 残差：y ~ X，返回残差 Series（保持原 index）。
    X 应已添加截距列（常数列 1）。
    丢弃含 NaN 的行后回归，其余位置填 NaN。
    """
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < X.shape[1] + 2:
        return pd.Series(np.nan, index=y.index, name=y.name)
    y_  = df.iloc[:, 0].values
    X_  = df.iloc[:, 1:].values
    try:
        coef, _, _, _ = np.linalg.lstsq(X_, y_, rcond=None)
        fitted = X_ @ coef
        resid  = y_ - fitted
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=y.index, name=y.name)

    result = pd.Series(np.nan, index=y.index, name=y.name)
    result.loc[df.index] = resid
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 回归法中性化
# ═══════════════════════════════════════════════════════════════════════════════

def neutralize_regression(
    factor_panel:   pd.DataFrame,
    mktcap_panel:   pd.DataFrame,
    industry_map:   Optional[pd.Series] = None,
    vol_panel:      Optional[pd.DataFrame] = None,
    use_log_mktcap: bool = True,
) -> pd.DataFrame:
    """
    截面回归法因子中性化（逐日处理）。

    模型：
        f_i = α + β·ln(MktCap_i) + Σγ_k·D_ik + δ·σ_i + ε_i

    Parameters
    ----------
    factor_panel  : (日期 × 股票) 原始因子面板
    mktcap_panel  : (日期 × 股票) 总市值面板（万元）
    industry_map  : ts_code → industry 的 Series（None 则跳过行业哑变量）
    vol_panel     : (日期 × 股票) 波动率面板（None 则跳过波动率控制）
    use_log_mktcap: True 则对市值取自然对数

    Returns
    -------
    残差面板（中性化后因子），与 factor_panel 等形状
    """
    result = pd.DataFrame(np.nan, index=factor_panel.index, columns=factor_panel.columns)

    for date in factor_panel.index:
        y = factor_panel.loc[date].dropna()
        if len(y) < 10:
            continue

        common = y.index.tolist()

        # 市值列
        mkts = mktcap_panel.loc[date].reindex(common) if date in mktcap_panel.index else pd.Series(np.nan, index=common)
        if use_log_mktcap:
            mkts = np.log(mkts.replace(0, np.nan).clip(lower=1e-6))

        X_parts = {
            "const":    pd.Series(1.0, index=common),
            "ln_mktcap": mkts,
        }

        # 行业哑变量
        if industry_map is not None:
            ind = industry_map.reindex(common).fillna("Unknown")
            dummies = pd.get_dummies(ind, prefix="ind", drop_first=True).astype(float)
            for col in dummies.columns:
                X_parts[col] = dummies[col]

        # 波动率列
        if vol_panel is not None and date in vol_panel.index:
            X_parts["vol"] = vol_panel.loc[date].reindex(common)

        X = pd.DataFrame(X_parts, index=common)
        resid = _ols_residual(y, X)
        result.loc[date, resid.index] = resid.values

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 分组标准化法（行业内 Z-Score）
# ═══════════════════════════════════════════════════════════════════════════════

def neutralize_industry_zscore(
    factor_panel: pd.DataFrame,
    industry_map: pd.Series,
) -> pd.DataFrame:
    """
    行业内 Z-Score 标准化（逐日处理）。

    Parameters
    ----------
    factor_panel : (日期 × 股票) 因子面板
    industry_map : ts_code → industry

    Returns
    -------
    行业内 Z-Score 后的面板
    """
    result = pd.DataFrame(np.nan, index=factor_panel.index, columns=factor_panel.columns)

    for date in factor_panel.index:
        row = factor_panel.loc[date].dropna()
        if len(row) < 5:
            continue
        ind = industry_map.reindex(row.index).fillna("Unknown")

        def _zscore(s: pd.Series) -> pd.Series:
            if len(s) < 2:
                return s - s
            sig = s.std(ddof=1)
            return (s - s.mean()) / (sig if sig > 0 else 1.0)

        neutralized = row.groupby(ind).transform(_zscore)
        result.loc[date, neutralized.index] = neutralized.values

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 正交化（Gram-Schmidt）
# ═══════════════════════════════════════════════════════════════════════════════

def orthogonalize(
    new_factor:       pd.DataFrame,
    existing_factors: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    将 new_factor 正交化到 existing_factors 张成空间的正交补空间。

    对每个截面日：
        new = new - Σ (new·f_k / f_k·f_k) * f_k

    等价于：对 existing_factors 做多元回归，取残差。

    Parameters
    ----------
    new_factor       : 待正交化的 (日期 × 股票) 面板
    existing_factors : 已有因子面板列表

    Returns
    -------
    正交化后的因子面板
    """
    result = pd.DataFrame(np.nan, index=new_factor.index, columns=new_factor.columns)

    for date in new_factor.index:
        y = new_factor.loc[date].dropna()
        if len(y) < 10:
            continue
        common = y.index.tolist()

        X_parts = {"const": pd.Series(1.0, index=common)}
        for i, ef in enumerate(existing_factors):
            if date in ef.index:
                X_parts[f"f{i}"] = ef.loc[date].reindex(common)

        X = pd.DataFrame(X_parts, index=common)
        resid = _ols_residual(y, X)
        result.loc[date, resid.index] = resid.values

    return result
