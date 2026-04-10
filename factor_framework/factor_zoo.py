"""
factor_zoo.py
=============
内置预定义因子库（示例 + 常用因子）。

每个函数均遵循签名：(df: pd.DataFrame) -> pd.Series
df 为单只股票的日频 DataFrame（已清洗，按交易日升序）。

分类
----
动量因子：momentum_*
反转因子：reversal_*
波动率因子：vol_*
估值因子：value_*
成长因子：growth_*
量价因子：amihud_*, turnover_*
规模因子：size_*
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_framework.operators import (
    delay, ts_mean, ts_stddev, ts_sum, ts_corr,
    ts_max, ts_min, ts_rank, ts_delta, ts_wma,
    ts_ema, ts_slope, ts_rsi,
    log, absx, sign, ts_zscore,
)

# ── 列名常量 ────────────────────────────────────────────────────────────────
_C  = "收盘价"
_O  = "开盘价"
_H  = "最高价"
_L  = "最低价"
_V  = "成交量（手）"
_A  = "成交额（千元）"
_PC = "前收盘价"
_MK = "总市值（万元）"
_FM = "流通市值（万元）"
_PB = "市净率"
_PE = "市盈率（TTM，亏损为空）"
_PS = "市销率（TTM）"
_TR = "换手率（%）"
_ADJ = "复权因子"
_RET = "_ret"    # 引擎内部日收益率列


def _ret(df: pd.DataFrame) -> pd.Series:
    """日收益率（优先使用引擎预算列，否则自行计算）。"""
    if _RET in df.columns:
        return df[_RET]
    return df[_C].pct_change()


def _hfq_close(df: pd.DataFrame) -> pd.Series:
    """
    后复权收盘价（v2.9.1）：收盘价 × 复权因子。

    动机
    ----
    前复权（qfq）价格在每次除权时回溯修改所有历史价格，
    导致今天看到的历史价格包含了未来的分红/除权信息（前瞻偏差）。
    例如：2020 年底的价格会被 2023 年的分红事件修改。

    后复权（hfq）价格从 IPO 起始基准向前累积，不修改历史价格，
    是计算动量/反转等历史收益类因子的正确选择。

    实现
    ----
    若 CSV 中包含「复权因子」列（_ADJ），则：
        hfq = 收盘价 × 复权因子  （该列为当日相对 IPO 日的累计复权因子）
    否则退化为直接使用收盘价（前复权，有偏但无替代数据时可接受）。

    Returns
    -------
    pd.Series，与 df 等长，index 与 df.index 对齐
    """
    if _ADJ in df.columns:
        adj = df[_ADJ].replace(0, np.nan).ffill()
        return df[_C].replace(0, np.nan) * adj
    # 无复权因子列：退化到原始收盘价（前复权，记录 warning 但不中断）
    return df[_C].replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 动量因子
# ═══════════════════════════════════════════════════════════════════════════════

def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """
    经典 12-1 月动量：过去 252 个交易日收益率 - 过去 21 个交易日收益率
    （跳过最近 1 个月，规避短期反转）。

    实现（v2.9.1）：
    - 使用后复权价格（_hfq_close）避免前复权价格的前瞻偏差
    - 用对数价格差分替代 rolling.apply(np.prod)（精度更高、速度更快）
    """
    log_p  = np.log(_hfq_close(df))
    mom_12 = np.exp(log_p - log_p.shift(252)) - 1
    mom_1  = np.exp(log_p - log_p.shift(21))  - 1
    # 仅保留 warm-up 期足够的有效行
    valid = log_p.notna() & log_p.shift(252).notna()
    result = mom_12 - mom_1
    result[~valid] = np.nan
    return result


def momentum_6_1(df: pd.DataFrame) -> pd.Series:
    """6-1 月中期动量（v2.9.1：后复权 + 对数价格差分）。"""
    log_p  = np.log(_hfq_close(df))
    mom_6  = np.exp(log_p - log_p.shift(126)) - 1
    mom_1  = np.exp(log_p - log_p.shift(21))  - 1
    valid  = log_p.notna() & log_p.shift(126).notna()
    result = mom_6 - mom_1
    result[~valid] = np.nan
    return result


def momentum_1m(df: pd.DataFrame) -> pd.Series:
    """短期 1 月动量（v2.9.1：后复权 + 对数价格差分）。"""
    log_p  = np.log(_hfq_close(df))
    result = np.exp(log_p - log_p.shift(21)) - 1
    result[log_p.shift(21).isna()] = np.nan
    return result


def momentum_52w_high(df: pd.DataFrame) -> pd.Series:
    """
    52 周高点动量：收盘价 / 过去 252 日最高价。
    接近高点 → 正向信号。使用后复权价格保证历史高点可比性。
    """
    hfq = _hfq_close(df)
    high_52w = ts_max(hfq, 252)
    return hfq / high_52w.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 反转因子
# ═══════════════════════════════════════════════════════════════════════════════

def reversal_1w(df: pd.DataFrame) -> pd.Series:
    """短期 1 周反转：过去 5 日收益率的负值（反转信号，v2.9.1：后复权 + 对数差分）。"""
    log_p  = np.log(_hfq_close(df))
    ret_5  = np.exp(log_p - log_p.shift(5)) - 1
    return -ret_5


def reversal_1m(df: pd.DataFrame) -> pd.Series:
    """短期 1 月反转：过去 21 日收益率的负值（v2.9.1：后复权 + 对数差分）。"""
    log_p  = np.log(_hfq_close(df))
    ret_21 = np.exp(log_p - log_p.shift(21)) - 1
    return -ret_21


# ═══════════════════════════════════════════════════════════════════════════════
# 波动率因子
# ═══════════════════════════════════════════════════════════════════════════════

def vol_20d(df: pd.DataFrame) -> pd.Series:
    """20 日历史波动率（负向因子，低波动有超额）。"""
    return -ts_stddev(_ret(df), 20)


def vol_60d(df: pd.DataFrame) -> pd.Series:
    """60 日历史波动率。"""
    return -ts_stddev(_ret(df), 60)


def vol_skew(df: pd.DataFrame) -> pd.Series:
    """
    收益率偏度（负向，偏度越大 → 彩票效应 → 预期收益越低）。
    使用 20 日滚动偏度，取负。
    """
    return -_ret(df).rolling(20, min_periods=15).skew()


def downside_vol(df: pd.DataFrame, d: int = 60) -> pd.Series:
    """
    下行波动率：仅用负收益计算标准差（d 天窗口）。
    """
    r = _ret(df)
    def _down_std(w: np.ndarray) -> float:
        neg = w[w < 0]
        return float(np.std(neg, ddof=1)) if len(neg) >= 2 else np.nan
    return r.rolling(d, min_periods=d // 2).apply(_down_std, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 估值因子
# ═══════════════════════════════════════════════════════════════════════════════

def value_pb(df: pd.DataFrame, lag_days: int = 20) -> pd.Series:
    """
    市净率倒数（BP = 1/PB，值越大→估值越低→正向）。

    Parameters
    ----------
    lag_days : 财务数据滞后天数（默认 20 个交易日 ≈ 1 个月）。
               用于规避尚未公告的财报被提前使用（前瞻偏差）。
               若数据源已按公告日严格对齐，可设为 0。
    """
    pb = df[_PB].replace(0, np.nan)
    if lag_days > 0:
        pb = pb.shift(lag_days)
    return 1.0 / pb


def value_pe_ttm(df: pd.DataFrame, lag_days: int = 20) -> pd.Series:
    """
    市盈率（TTM）倒数（EP，值越大→估值越低→正向）。

    Parameters
    ----------
    lag_days : 财务数据滞后天数（默认 20 个交易日）。
               防止使用尚未披露的盈利数据（前瞻偏差）。
    """
    pe = df[_PE].replace(0, np.nan)
    if lag_days > 0:
        pe = pe.shift(lag_days)
    return 1.0 / pe


def value_ps_ttm(df: pd.DataFrame, lag_days: int = 20) -> pd.Series:
    """
    市销率（TTM）倒数（SP，值越大→越便宜）。

    Parameters
    ----------
    lag_days : 财务数据滞后天数（默认 20 个交易日）。
               防止使用尚未披露的营收数据（前瞻偏差）。
    """
    ps = df[_PS].replace(0, np.nan)
    if lag_days > 0:
        ps = ps.shift(lag_days)
    return 1.0 / ps


# ═══════════════════════════════════════════════════════════════════════════════
# 规模因子
# ═══════════════════════════════════════════════════════════════════════════════

def size_log_mktcap(df: pd.DataFrame) -> pd.Series:
    """对数总市值（负向因子，小盘股溢价→取负）。"""
    return -log(df[_MK])


def size_log_free_cap(df: pd.DataFrame) -> pd.Series:
    """对数流通市值（负向）。"""
    return -log(df[_FM])


# ═══════════════════════════════════════════════════════════════════════════════
# 量价因子
# ═══════════════════════════════════════════════════════════════════════════════

def amihud_illiquidity(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """
    Amihud 非流动性因子：|日收益率| / 成交额。
    值越大→流动性越差（一般为负向，流动性越差 → 风险溢价越高）。
    """
    r       = _ret(df).abs()
    amount  = df[_A].replace(0, np.nan)  # 千元单位
    daily   = r / amount
    return ts_mean(daily, d)


def turnover_rate(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """d 日平均换手率（高换手 → 短期反转信号）。"""
    return -ts_mean(df[_TR], d)   # 取负：换手越高 → 预期收益越低


def vol_price_corr(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """量价相关性：成交量与收盘价的滚动相关（负向信号）。"""
    return -ts_corr(df[_V], df[_C], d)


def vwap_deviation(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """
    VWAP 偏离度：(收盘价 - VWAP) / VWAP。
    VWAP ≈ 成交额 / (成交量 * 100)（千元 / (手*100股) ≈ 元/股）。
    """
    # 成交额（千元）/ 成交量（手）* 1000 / 100 = 元/股
    vwap = (df[_A] * 10.0) / df[_V].replace(0, np.nan)
    return (df[_C] - vwap) / vwap.replace(0, np.nan)


def price_strength(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """
    价格强度：当前收盘价在过去 d 天区间内的位置 [0,1]。
    类似 KD 指标中的 K 值。
    """
    lo  = ts_min(df[_L], d)
    hi  = ts_max(df[_H], d)
    rng = (hi - lo).replace(0, np.nan)
    return (df[_C] - lo) / rng


# ═══════════════════════════════════════════════════════════════════════════════
# 流动性质量因子（Liquidity / Market-Microstructure）
# ═══════════════════════════════════════════════════════════════════════════════

def bid_ask_spread_proxy(df: pd.DataFrame, d: int = 21) -> pd.Series:
    """
    买卖价差隐性代理：(High - Low) / Close 的 d 日均值。
    来源：Corwin & Schultz (2012) 的简化版 Kyle Lambda 近似。
    值越大 → 隐性摩擦成本越高 → 流动性越差（负向因子）。
    """
    hl_ratio = (df[_H] - df[_L]) / df[_C].replace(0, np.nan)
    return -ts_mean(hl_ratio, d)   # 取负：值越大流动性越好


def zero_return_ratio(df: pd.DataFrame, d: int = 21) -> pd.Series:
    """
    零收益日占比（过去 d 天收益率 = 0 的天数 / d）。
    来源：Lesmond, Ogden & Trzcinka (1999)。
    值越高 → 股票越不活跃 → 流动性越差（负向因子）。
    """
    ret = _ret(df)
    is_zero = (ret.abs() < 1e-8).astype(float)
    return -ts_mean(is_zero, d)   # 取负：占比越高流动性越差


def pastor_stambaugh(df: pd.DataFrame, d: int = 21) -> pd.Series:
    """
    Pastor-Stambaugh (2003) 流动性因子（简化版）：
        收益率对 sign(ret_{t-1}) * volume_{t-1} * ret_{t-1} 做滚动回归的斜率。
    斜率越负（绝对值越大）→ 价格冲击越大 → 流动性越差（取负后正向）。
    使用 ts_corr 近似实现：corr(ret_t, signed_vol_{t-1})。
    """
    ret = _ret(df)
    signed_vol = np.sign(ret.shift(1)) * df[_V].shift(1) * ret.shift(1)
    # 相关系数负值越大说明价格冲击越明显；取负使其成为正向流动性代理
    corr = ts_corr(ret, signed_vol, d)
    return -corr


def order_imbalance(df: pd.DataFrame, d: int = 21) -> pd.Series:
    """
    订单不平衡持续性：委比（买盘 / (买盘 + 卖盘)）的滚动均值。
    使用价格变动方向近似委比：ret > 0 视为买压，ret < 0 视为卖压。
        order_imbalance_proxy_t = (ret_t > 0) * 1 - (ret_t < 0) * 1  ∈ {-1, 0, 1}
    正值 → 买压主导 → 正向动量信号。
    """
    ret = _ret(df)
    buy_pressure = (ret > 0).astype(float) - (ret < 0).astype(float)
    return ts_mean(buy_pressure, d)


# ═══════════════════════════════════════════════════════════════════════════════
# 技术分析因子（Technical Analysis Factors）
# ═══════════════════════════════════════════════════════════════════════════════

def rsi_14(df: pd.DataFrame) -> pd.Series:
    """
    14 日 RSI（Relative Strength Index）。
    值 > 70 超买（负向信号），值 < 30 超卖（正向信号）。
    作为反转因子使用：返回 (50 - RSI) 使其与超卖正相关。
    """
    rsi = ts_rsi(df[_C], 14)
    return 50.0 - rsi   # 超卖时值为正，超买时值为负


def macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    MACD 差值（DIF）：12 日 EMA - 26 日 EMA。
    值为正 → 短期动量强于长期 → 动量正向信号。
    """
    ema12 = ts_ema(df[_C], 12)
    ema26 = ts_ema(df[_C], 26)
    dif   = ema12 - ema26
    # 归一化：除以收盘价避免量纲差异
    return dif / df[_C].replace(0, np.nan)


def bb_position(df: pd.DataFrame, d: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    布林带位置（Bollinger Band Position）：
        BB% = (Close - Lower) / (Upper - Lower)
    Upper = MA + n_std * σ，Lower = MA - n_std * σ。
    值 ∈ [0, 1]；接近 1 → 接近上轨（超买），接近 0 → 接近下轨（超卖）。
    作为反转因子：返回 (0.5 - BB%) 使超卖时为正值。
    """
    ma    = ts_mean(df[_C], d)
    sigma = ts_stddev(df[_C], d)
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    band  = (upper - lower).replace(0, np.nan)
    bb_pct = (df[_C] - lower) / band
    return 0.5 - bb_pct   # 反转方向：超卖（低位）为正信号


def volume_trend(df: pd.DataFrame, d: int = 20) -> pd.Series:
    """
    成交量趋势：对过去 d 日成交量做线性回归，返回归一化斜率。
    斜率为正 → 量能放大 → 量价配合（配合动量使用）。
    直接复用 ts_slope 算子。
    """
    return ts_slope(df[_V], d)


# ═══════════════════════════════════════════════════════════════════════════════
# 因子注册辅助：一次性注册所有内置因子
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_FACTORS = {
    # 动量
    "momentum_12_1":     momentum_12_1,
    "momentum_6_1":      momentum_6_1,
    "momentum_1m":       momentum_1m,
    "momentum_52w_high": momentum_52w_high,
    # 反转
    "reversal_1w":       reversal_1w,
    "reversal_1m":       reversal_1m,
    # 波动率
    "vol_20d":           vol_20d,
    "vol_60d":           vol_60d,
    "vol_skew":          vol_skew,
    "downside_vol":      downside_vol,
    # 估值
    "value_pb":          value_pb,
    "value_pe_ttm":      value_pe_ttm,
    "value_ps_ttm":      value_ps_ttm,
    # 规模
    "size_log_mktcap":   size_log_mktcap,
    "size_log_free_cap": size_log_free_cap,
    # 量价
    "amihud_illiquidity":amihud_illiquidity,
    "turnover_rate":     turnover_rate,
    "vol_price_corr":    vol_price_corr,
    "vwap_deviation":    vwap_deviation,
    "price_strength":    price_strength,
    # 流动性质量
    "bid_ask_spread_proxy": bid_ask_spread_proxy,
    "zero_return_ratio":    zero_return_ratio,
    "pastor_stambaugh":     pastor_stambaugh,
    "order_imbalance":      order_imbalance,
    # 技术分析
    "rsi_14":            rsi_14,
    "macd_signal":       macd_signal,
    "bb_position":       bb_position,
    "volume_trend":      volume_trend,
}


def register_all(engine) -> None:
    """将所有内置因子注册到 FactorEngine 实例。"""
    for name, fn in BUILTIN_FACTORS.items():
        engine.register(name, fn)
