"""
operators.py
============
因子算子库：时间序列 / 横截面 / 数学逻辑 / 跨资产

所有时间序列算子作用于 pd.Series（单只股票历史序列）。
所有横截面算子作用于 pd.Series（某截面日所有股票的因子值），index 为 ts_code。
面板运算通过 factor_engine.apply_cross_section() 调度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2.1  时间序列算子（Time-Series Operators）
# ═══════════════════════════════════════════════════════════════════════════════

def ts_sum(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动累加和。"""
    return x.rolling(d, min_periods=d).sum()


def ts_mean(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动算术均值。"""
    return x.rolling(d, min_periods=d).mean()


def ts_stddev(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动标准差（样本标准差，ddof=1）。"""
    return x.rolling(d, min_periods=d).std(ddof=1)


def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 与 y 的滚动皮尔逊相关系数。"""
    return x.rolling(d, min_periods=d).corr(y)


def delay(x: pd.Series, d: int) -> pd.Series:
    """x 向后平移 d 天（d 天前的值）。"""
    return x.shift(d)


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动最大值。"""
    return x.rolling(d, min_periods=d).max()


def ts_min(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动最小值。"""
    return x.rolling(d, min_periods=d).min()


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """
    当日 x 在过去 d 天中的排名分位（0~1）。
    返回值为当日值在窗口内的百分比排名（1 = 最高）。
    """
    def _rank_last(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        return float(pd.Series(window).rank(pct=True).iloc[-1])

    return x.rolling(d, min_periods=d).apply(_rank_last, raw=True)


def ts_delta(x: pd.Series, d: int) -> pd.Series:
    """x 当日值 - d 天前的值（变化量）。"""
    return x - x.shift(d)


def ts_wma(x: pd.Series, d: int) -> pd.Series:
    """
    过去 d 天 x 的线性加权移动均值（近期权重更高）。
    权重为 1, 2, …, d（归一化后）。
    """
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()

    def _wma(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        return float(np.dot(window, weights))

    return x.rolling(d, min_periods=d).apply(_wma, raw=True)


def ts_zscore(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动 Z-Score（(x - mean) / std）。"""
    mu  = ts_mean(x, d)
    sig = ts_stddev(x, d)
    return (x - mu) / sig.replace(0, np.nan)


def ts_skew(x: pd.Series, d: int) -> pd.Series:
    """过去 d 天 x 的滚动偏度。"""
    return x.rolling(d, min_periods=d).skew()


def ts_autocorr(x: pd.Series, d: int, lag: int = 1) -> pd.Series:
    """过去 d 天 x 的滚动自相关系数（lag 阶）。"""
    return x.rolling(d, min_periods=d).apply(
        lambda w: pd.Series(w).autocorr(lag=lag), raw=False
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2.2  横截面算子（Cross-Section Operators）
# 输入 x：pd.Series，index = ts_code；group：pd.Series，index = ts_code（行业等）
# ═══════════════════════════════════════════════════════════════════════════════

def cs_rank(x: pd.Series) -> pd.Series:
    """
    对 x 在全部股票中排序，输出 [0, 1] 分位（pct rank）。
    NaN 不参与排名，结果保持 NaN。
    """
    return x.rank(pct=True)


def cs_zscore(x: pd.Series) -> pd.Series:
    """(x - 均值) / 标准差，横截面标准化。"""
    mu  = x.mean()
    sig = x.std(ddof=1)
    if sig == 0 or np.isnan(sig):
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sig


def cs_demean(x: pd.Series) -> pd.Series:
    """x 减去截面均值。"""
    return x - x.mean()


def cs_scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    """将 x 线性映射到 [0, a]。"""
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return pd.Series(a / 2, index=x.index)
    return (x - x_min) / (x_max - x_min) * a


def cs_industry_neutral(x: pd.Series, group: pd.Series) -> pd.Series:
    """
    x 减去所在行业均值（行业内去均值）。
    group：与 x 同 index 的行业标签 Series。
    """
    group_mean = x.groupby(group).transform("mean")
    return x - group_mean


def cs_industry_zscore(x: pd.Series, group: pd.Series) -> pd.Series:
    """在每个行业内对 x 做 Z-Score 标准化（简易行业中性化）。"""
    def _zscore(s: pd.Series) -> pd.Series:
        sig = s.std(ddof=1)
        if sig == 0 or np.isnan(sig):
            return s - s.mean()
        return (s - s.mean()) / sig
    return x.groupby(group).transform(_zscore)


def cs_winsorize(x: pd.Series, n_std: float = 3.0) -> pd.Series:
    """横截面 MAD Winsorize（复用 data_cleaner 逻辑）。"""
    s = x.dropna()
    if len(s) < 4:
        return x
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return x
    lower = med - n_std * mad
    upper = med + n_std * mad
    return x.clip(lower, upper)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2.3  数学与逻辑算子
# ═══════════════════════════════════════════════════════════════════════════════

def log(x: pd.Series) -> pd.Series:
    """自然对数（x > 0）。"""
    return np.log(x.replace(0, np.nan))


def sqrt(x: pd.Series) -> pd.Series:
    """平方根（x >= 0）。"""
    return np.sqrt(x.clip(lower=0))


def absx(x: pd.Series) -> pd.Series:
    """绝对值。"""
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    """符号函数：+1 / 0 / -1。"""
    return np.sign(x)


def if_else(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """条件选择：cond 为 True 时取 a，否则取 b。"""
    return pd.Series(
        np.where(cond.values, a.values if isinstance(a, pd.Series) else a,
                 b.values if isinstance(b, pd.Series) else b),
        index=cond.index,
    )


def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """将 x 截断到 [lo, hi]。"""
    return x.clip(lower=lo, upper=hi)


def power(x: pd.Series, n: float) -> pd.Series:
    """幂运算 x^n。"""
    return x ** n


def cs_min(x: pd.Series, y: pd.Series) -> pd.Series:
    """逐元素取 min(x, y)。"""
    return pd.concat([x, y], axis=1).min(axis=1)


def cs_max_pair(x: pd.Series, y: pd.Series) -> pd.Series:
    """逐元素取 max(x, y)。"""
    return pd.concat([x, y], axis=1).max(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2.4  跨资产算子（Cross-Asset Operators）
# ═══════════════════════════════════════════════════════════════════════════════

def relative_strength(stock_price: pd.Series, index_price: pd.Series) -> pd.Series:
    """个股价格除以指数价格，得到相对强度（RS）。"""
    return stock_price / index_price.replace(0, np.nan)


def idiosyncratic_return(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    d: int = 60,
) -> pd.Series:
    """
    特质收益：stock_ret - beta * market_ret。
    beta 由滚动 OLS 估计（窗口 d 天）。
    """
    beta = ts_corr(stock_ret, market_ret, d) * (
        ts_stddev(stock_ret, d) / ts_stddev(market_ret, d).replace(0, np.nan)
    )
    return stock_ret - beta * market_ret


def market_beta(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    d: int = 60,
) -> pd.Series:
    """滚动 OLS beta：cov(stock, market) / var(market)，窗口 d 天。"""
    corr = ts_corr(stock_ret, market_ret, d)
    beta = corr * ts_stddev(stock_ret, d) / ts_stddev(market_ret, d).replace(0, np.nan)
    return beta


def group_relative(x: pd.Series, group: pd.Series) -> pd.Series:
    """x - group_mean(x, group)：个股因子值减去所属行业均值。"""
    return cs_industry_neutral(x, group)
