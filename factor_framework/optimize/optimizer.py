"""
optimizer.py
============
因子组合与权重优化模块（§2.4）

提供将多个单因子合成为最终组合信号的各类方法。

已实现
------
- 等权组合      (equal_weight)        §2.4.1
- ICIR 加权     (icir_weight)         §2.4.2

函数约定
--------
所有函数接收：
    factor_panels : Dict[str, pd.DataFrame]
        键为因子名称，值为 (日期 × 股票) 因子面板（已经过截面标准化）
    ic_series_dict: Dict[str, pd.Series]
        键为因子名称，值为对应的 IC 时间序列（compute_ic 输出）

返回：
    pd.DataFrame  (日期 × 股票) 合成因子面板
    Dict[str, float]  各因子权重

示例
----
from factor_framework.optimizer import equal_weight, icir_weight

composite, weights = equal_weight(factor_panels)
composite, weights = icir_weight(
    factor_panels,
    ic_series_dict,
    window=12,          # 滚动窗口（期数）
)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _align_panels(
    factor_panels: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    将所有因子面板对齐到公共日期 × 公共股票的交集，
    确保加权合成时维度一致。
    """
    # 取日期交集
    dates = None
    for panel in factor_panels.values():
        d = panel.index
        dates = d if dates is None else dates.intersection(d)

    # 取股票交集
    stocks = None
    for panel in factor_panels.values():
        s = panel.columns
        stocks = s if stocks is None else stocks.intersection(s)

    aligned = {
        name: panel.loc[dates, stocks]
        for name, panel in factor_panels.items()
    }
    return aligned


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    权重归一化：确保所有权重之和为 1，且权重 >= 0。
    若所有权重均为 0（或负），退回等权。
    """
    # 截断负权重为 0（因子合成只做多因子信号，不做空）
    clipped = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _weighted_combine(
    aligned_panels: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    给定各因子（已对齐的）面板和权重，逐格加权求和生成合成面板。
    每个截面独立处理 NaN：某只股票在某因子缺失时，用其余有效因子的
    归一化权重加权（局部归一化），避免该股票因少一个因子就变成 NaN。
    """
    # 取任意面板的维度作为骨架
    sample = next(iter(aligned_panels.values()))
    result = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)
    weight_sum = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)

    for name, panel in aligned_panels.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        valid_mask = panel.notna()
        result     = result.add(panel.fillna(0.0) * w)
        weight_sum = weight_sum.add(valid_mask.astype(float) * w)

    # 局部归一化：除以实际有效权重之和
    composite = result.div(weight_sum.replace(0, np.nan))
    return composite


# ═══════════════════════════════════════════════════════════════════════════════
# §2.4.1  等权组合
# ═══════════════════════════════════════════════════════════════════════════════

def equal_weight(
    factor_panels: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    等权组合：所有因子权重相等。

    公式：F = (1/N) * Σ fᵢ

    优点：简单、稳健、避免过拟合。
    适用场景：因子数量少、各因子 IC 相近时。

    Parameters
    ----------
    factor_panels : Dict[因子名, (日期×股票) 面板]
        各单因子面板（建议已做截面标准化）

    Returns
    -------
    composite : pd.DataFrame  合成因子面板（日期×股票）
    weights   : Dict[str, float]  各因子权重（等权为 1/N）
    """
    if not factor_panels:
        raise ValueError("[ERROR] factor_panels cannot be empty.")

    n = len(factor_panels)
    weights = {name: 1.0 / n for name in factor_panels}

    aligned = _align_panels(factor_panels)
    composite = _weighted_combine(aligned, weights)
    return composite, weights


# ═══════════════════════════════════════════════════════════════════════════════
# §2.4.2  ICIR 加权
# ═══════════════════════════════════════════════════════════════════════════════

def icir_weight(
    factor_panels:  Dict[str, pd.DataFrame],
    ic_series_dict: Dict[str, pd.Series],
    window:         Optional[int] = 12,
    min_periods:    int = 6,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    ICIR 加权组合：按各因子的 ICIR 比例分配权重。

    公式：wᵢ = |ICIR_i| / Σⱼ |ICIR_j|

    说明
    ----
    - ICIR = Mean(IC) / Std(IC)，衡量因子预测能力的稳定性。
    - 若 window=None，使用全样本 ICIR；否则使用最近 window 期滚动 ICIR。
    - 权重取 |ICIR| 的绝对值，方向信息已通过面板标准化反映。
    - 归一化处理确保权重之和为 1，且均 >= 0。

    Parameters
    ----------
    factor_panels  : Dict[因子名, (日期×股票) 面板]
    ic_series_dict : Dict[因子名, IC 时间序列]  ← compute_ic() 的输出
    window         : 滚动计算 ICIR 的窗口期数（None = 全样本）
    min_periods    : 滚动窗口最少有效期数（不足时该因子权重设为 0）

    Returns
    -------
    composite : pd.DataFrame  合成因子面板（日期×股票）
    weights   : Dict[str, float]  各因子权重
    """
    if not factor_panels:
        raise ValueError("[ERROR] factor_panels cannot be empty.")

    names = list(factor_panels.keys())

    # 计算每个因子的 ICIR
    icir_values: Dict[str, float] = {}
    missing_ic = []

    for name in names:
        if name not in ic_series_dict:
            missing_ic.append(name)
            icir_values[name] = 0.0
            continue

        ic = ic_series_dict[name].dropna()

        if window is not None:
            # 取最近 window 期
            ic = ic.iloc[-window:] if len(ic) >= window else ic

        if len(ic) < min_periods:
            icir_values[name] = 0.0
            continue

        mean_ic = float(ic.mean())
        std_ic  = float(ic.std(ddof=1))
        icir    = mean_ic / std_ic if std_ic > 0 else 0.0
        icir_values[name] = icir

    if missing_ic:
        import warnings
        warnings.warn(
            f"[WARN] The following factors are missing IC series; weights are set to 0: {missing_ic}\n"
            "Ensure ic_series_dict contains corresponding keys."
        )

    # 取 |ICIR|，负 ICIR 的因子同样贡献权重（合成层面方向已处理）
    abs_icir = {k: abs(v) for k, v in icir_values.items()}
    weights  = _normalize_weights(abs_icir)

    aligned   = _align_panels(factor_panels)
    composite = _weighted_combine(aligned, weights)
    return composite, weights


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总打印工具
# ═══════════════════════════════════════════════════════════════════════════════

def print_weights(
    weights:    Dict[str, float],
    method:     str = "Equal Weight",
    icir_dict:  Optional[Dict[str, float]] = None,
) -> None:
    """
    格式化打印因子权重表。

    Parameters
    ----------
    weights   : 各因子权重 Dict
    method    : 权重方法名称（用于标题）
    icir_dict : 可选，各因子 ICIR 值（ICIR 加权时显示）
    """
    sep = "─" * 52
    print(f"\n{'='*52}")
    print(f"[INFO] Factor Combination Weights  [{method}]")
    print(f"{'='*52}")
    print(f"[INFO] {'Factor':<24}  {'Weight':>7}  {'ICIR':>8}")
    print(sep)
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        icir_str = ""
        if icir_dict and name in icir_dict:
            icir_str = f"{icir_dict[name]:>8.4f}"
        bar = "█" * max(0, int(w * 40))
        print(f"[INFO] {name:<24}  {w:>6.2%}  {icir_str}  {bar}")
    print(sep)
    print(f"[INFO] Total                       {sum(weights.values()):>6.2%}")
    print(f"{'='*52}\n")
