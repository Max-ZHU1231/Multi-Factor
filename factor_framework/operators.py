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


def ts_ema(x: pd.Series, d: int) -> pd.Series:
    """
    指数加权移动均值（EWMA），span = d。
    相比 WMA 更重视近期数据，是 MACD、RSI 等技术指标的基础算子。
    前 d-1 个位置输出 NaN（与其他滚动算子保持一致）。
    """
    ema = x.ewm(span=d, min_periods=d, adjust=False).mean()
    return ema


def ts_slope(x: pd.Series, d: int) -> pd.Series:
    """
    对时间（0, 1, …, d-1）做线性回归，返回归一化斜率：slope / mean(x)。
    衡量序列在过去 d 天内的趋势方向与强度（如价格趋势、成交量趋势）。
    均值为 0 时直接返回原始斜率。
    """
    t = np.arange(d, dtype=float)
    t -= t.mean()
    t_var = (t ** 2).sum()

    def _slope(w: np.ndarray) -> float:
        if np.isnan(w).any():
            return np.nan
        w_dm = w - w.mean()
        slope = float(np.dot(t, w_dm) / t_var)
        mean_abs = abs(w.mean())
        return slope / mean_abs if mean_abs > 1e-10 else slope

    return x.rolling(d, min_periods=d).apply(_slope, raw=True)


def ts_rsi(x: pd.Series, d: int) -> pd.Series:
    """
    滚动 RSI（Relative Strength Index）：
        RS   = avg_gain / avg_loss  (过去 d 天)
        RSI  = 100 - 100 / (1 + RS)
    x 通常为价格或收益率序列。
    值域 [0, 100]：>70 超买，<30 超卖。
    """
    delta = x.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(d, min_periods=d).mean()
    avg_loss = loss.rolling(d, min_periods=d).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def ts_drawdown(x: pd.Series, d: int) -> pd.Series:
    """
    过去 d 天内的最大回撤（Max Drawdown）。
    定义：(peak - trough) / peak，值域 [0, 1]。
    通常取负值作为下行风险因子（越小越好）。
    """
    def _mdd(w: np.ndarray) -> float:
        if np.isnan(w).any():
            return np.nan
        cummax = np.maximum.accumulate(w)
        dd = (cummax - w) / np.where(cummax == 0, np.nan, cummax)
        return float(np.nanmax(dd))

    return x.rolling(d, min_periods=d).apply(_mdd, raw=True)


def ts_beta(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """
    对基准序列 y 做滚动 OLS Beta（窗口 d 天）：
        beta = cov(x, y) / var(y)
    常用于滚动市场 Beta 估计；y 通常为市场指数收益率。
    """
    cov = x.rolling(d, min_periods=d).cov(y)
    var = y.rolling(d, min_periods=d).var(ddof=1)
    return cov / var.replace(0, np.nan)


def ts_regression_residual(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """
    滚动回归残差：x ~ α + β·y （窗口 d 天）。
    残差 = x - α̂ - β̂·y，提取 x 中无法被 y 解释的特质成分。
    常用于特质动量（Alpha 动量）因子构建。
    """
    beta = ts_beta(x, y, d)
    mu_x = ts_mean(x, d)
    mu_y = ts_mean(y, d)
    alpha = mu_x - beta * mu_y
    return x - alpha - beta * y


def ts_decay_linear(x: pd.Series, d: int) -> pd.Series:
    """
    线性衰减加权和（WorldQuant 101 常用算子）：
        weights = [1, 2, …, d]，归一化后对窗口内数值加权求和。
    与 ts_wma 同义，但此处独立实现以与 WQ 风格对齐（ts_wma 为归一化均值，
    ts_decay_linear 输出为加权 **和** 的归一化值，含义相同）。
    """
    return ts_wma(x, d)


def ts_prod(x: pd.Series, d: int) -> pd.Series:
    """
    滚动连乘（d 天窗口）。适合收益率序列的累乘（复利增长）。
    例：ts_prod(1 + ret, 21) - 1 = 21 日复合收益率。
    """
    return x.rolling(d, min_periods=d).apply(np.prod, raw=True)


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


def cs_rank_by_group(x: pd.Series, group: pd.Series) -> pd.Series:
    """
    行业内百分比排名（0~1）。
    在每个行业（group）内独立对 x 做 pct rank，剔除行业效应后的相对排名。
    NaN 不参与排名，保持 NaN。

    Parameters
    ----------
    x     : 因子值 Series，index = ts_code
    group : 行业标签 Series，index = ts_code
    """
    return x.groupby(group).rank(pct=True)


def cs_neutralize(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    单变量快速截面中性化：x 对 y 做 OLS 回归取残差。
    等价于消除 x 中与 y 线性相关的部分（如去除市值暴露）。
    NaN 行自动跳过。

    Parameters
    ----------
    x : 原始因子值，index = ts_code
    y : 控制变量（如 ln_mktcap），index = ts_code
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 4:
        return pd.Series(np.nan, index=x.index)
    y_  = df.iloc[:, 0].values
    X_  = np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    try:
        coef, _, _, _ = np.linalg.lstsq(X_, y_, rcond=None)
        fitted = X_ @ coef
        resid  = y_ - fitted
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=x.index)
    result = pd.Series(np.nan, index=x.index)
    result.loc[df.index] = resid
    return result


def cs_top_n(x: pd.Series, n: int) -> pd.Series:
    """
    返回 Top-N 的布尔 mask（True = 进入 Top-N 名）。
    NaN 值永远不进入 Top-N。用于构建选股信号池。

    Parameters
    ----------
    x : 因子值，值越大排名越靠前
    n : 保留的股票数量
    """
    threshold = x.nlargest(n).min()
    return (x >= threshold) & x.notna()


def cs_quantile(x: pd.Series, q: float) -> float:
    """
    返回 x（截面）的第 q 分位数值。
    常用于设置分位数阈值，配合 cs_top_n 或 if_else 使用。

    Parameters
    ----------
    q : 分位数，取值 [0, 1]，如 0.8 表示第 80 百分位
    """
    return float(x.dropna().quantile(q))


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
