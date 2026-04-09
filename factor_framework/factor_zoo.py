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


# ═══════════════════════════════════════════════════════════════════════════════
# 动量因子
# ═══════════════════════════════════════════════════════════════════════════════

def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """
    经典 12-1 月动量：过去 252 个交易日收益率 - 过去 21 个交易日收益率
    （跳过最近 1 个月，规避短期反转）。
    """
    r = _ret(df)
    mom_12 = (1 + r).rolling(252, min_periods=200).apply(np.prod, raw=True) - 1
    mom_1  = (1 + r).rolling(21,  min_periods=15).apply(np.prod, raw=True) - 1
    return mom_12 - mom_1


def momentum_6_1(df: pd.DataFrame) -> pd.Series:
    """6-1 月中期动量。"""
    r      = _ret(df)
    mom_6  = (1 + r).rolling(126, min_periods=100).apply(np.prod, raw=True) - 1
    mom_1  = (1 + r).rolling(21,  min_periods=15).apply(np.prod, raw=True) - 1
    return mom_6 - mom_1


def momentum_1m(df: pd.DataFrame) -> pd.Series:
    """短期 1 月动量（包含最近 1 月，可用于短期策略）。"""
    r = _ret(df)
    return (1 + r).rolling(21, min_periods=15).apply(np.prod, raw=True) - 1


def momentum_52w_high(df: pd.DataFrame) -> pd.Series:
    """
    52 周高点动量：收盘价 / 过去 252 日最高价。
    接近高点 → 正向信号。
    """
    high_52w = ts_max(df[_C], 252)
    return df[_C] / high_52w.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════════
# 反转因子
# ═══════════════════════════════════════════════════════════════════════════════

def reversal_1w(df: pd.DataFrame) -> pd.Series:
    """短期 1 周反转：过去 5 日收益率的负值（反转信号）。"""
    r = _ret(df)
    return -(1 + r).rolling(5, min_periods=3).apply(np.prod, raw=True) + 1


def reversal_1m(df: pd.DataFrame) -> pd.Series:
    """短期 1 月反转：过去 21 日收益率的负值。"""
    r = _ret(df)
    return -(1 + r).rolling(21, min_periods=15).apply(np.prod, raw=True) + 1


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
# 因子注册辅助：一次性注册所有内置因子
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_FACTORS = {
    "momentum_12_1":     momentum_12_1,
    "momentum_6_1":      momentum_6_1,
    "momentum_1m":       momentum_1m,
    "momentum_52w_high": momentum_52w_high,
    "reversal_1w":       reversal_1w,
    "reversal_1m":       reversal_1m,
    "vol_20d":           vol_20d,
    "vol_60d":           vol_60d,
    "vol_skew":          vol_skew,
    "downside_vol":      downside_vol,
    "value_pb":          value_pb,
    "value_pe_ttm":      value_pe_ttm,
    "value_ps_ttm":      value_ps_ttm,
    "size_log_mktcap":   size_log_mktcap,
    "size_log_free_cap": size_log_free_cap,
    "amihud_illiquidity":amihud_illiquidity,
    "turnover_rate":     turnover_rate,
    "vol_price_corr":    vol_price_corr,
    "vwap_deviation":    vwap_deviation,
    "price_strength":    price_strength,
}


def register_all(engine) -> None:
    """将所有内置因子注册到 FactorEngine 实例。"""
    for name, fn in BUILTIN_FACTORS.items():
        engine.register(name, fn)
