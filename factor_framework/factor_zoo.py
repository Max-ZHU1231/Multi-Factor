"""
factor_zoo.py
=============
内置预定义因子库（示例 + 常用因子）。

.. deprecated::
    **兼容层（v3.3+）**：直接导入本模块是旧路径。
    新代码推荐使用按类别拆分的子模块：

    - `factor_framework.factors.momentum`  — 动量 / 反转因子
    - `factor_framework.factors.volatility` — 波动率因子
    - `factor_framework.factors.value`      — 估值 / 规模因子
    - `factor_framework.factors.volume`     — 量价 / 流动性 / 技术因子

    本模块作为向后兼容层保留至 v4.0，届时将移除。

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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from factor_framework.factors.registry import _CompatDict

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

    v2.9.1 修复：min_periods 从 d//2 改为 d，要求满窗口数据。
    半窗口计算会在回测期初引入大量噪声（用 30 天数据估计 60 天波动率），
    使因子值在冷启动期严重失真。满窗口后行数会减少，但质量更可靠。
    """
    r = _ret(df)
    def _down_std(w: np.ndarray) -> float:
        neg = w[w < 0]
        return float(np.std(neg, ddof=1)) if len(neg) >= 2 else np.nan
    return r.rolling(d, min_periods=d).apply(_down_std, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 估值因子
# ═══════════════════════════════════════════════════════════════════════════════

def value_pb(df: pd.DataFrame, lag_days: int = 0) -> pd.Series:
    """
    市净率倒数（BP = 1/PB，值越大→估值越低→正向）。

    数据说明（v2.9.1）
    ------------------
    市净率列来自 ak.stock_zh_valuation_baidu，返回日频市场实时估值：
    每天根据当日股价 ÷ 最新财报净资产重新计算，属于市场数据每日更新，
    不存在财报公告日前瞻问题，因此 lag_days 默认改为 0。

    若使用其他数据源（按报告期而非公告日存储财务数据），
    可手动设置 lag_days > 0（如 lag_days=60）以规避前瞻偏差。
    """
    pb = df[_PB].replace(0, np.nan)
    if lag_days > 0:
        pb = pb.shift(lag_days)
    return 1.0 / pb


def value_pe_ttm(df: pd.DataFrame, lag_days: int = 0) -> pd.Series:
    """
    市盈率（TTM）倒数（EP，值越大→估值越低→正向）。

    数据说明（v2.9.1）
    ------------------
    市盈率（TTM）列为日频市场实时估值数据（同 value_pb），
    每天根据当日股价和滚动 12 个月盈利重新计算，lag_days 默认改为 0。
    """
    pe = df[_PE].replace(0, np.nan)
    if lag_days > 0:
        pe = pe.shift(lag_days)
    return 1.0 / pe


def value_ps_ttm(df: pd.DataFrame, lag_days: int = 0) -> pd.Series:
    """
    市销率（TTM）倒数（SP，值越大→越便宜）。

    数据说明（v2.9.1）
    ------------------
    市销率（TTM）列为日频市场实时估值数据（同 value_pb），lag_days 默认改为 0。
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

    v2.9.1 修复：改用后复权价格（_hfq_close）衍生的日收益率，
    消除前复权数据回溯修改历史价格带来的隐式前瞻偏差（同 BUG 4 修复）。
    shift(1) 使用的是 t-1 期数据，执行方向正确，无未来函数。
    """
    log_p  = np.log(_hfq_close(df))
    ret    = (log_p - log_p.shift(1))          # 对数日收益率（后复权）
    signed_vol = np.sign(ret.shift(1)) * df[_V].shift(1) * ret.shift(1)
    corr   = ts_corr(ret, signed_vol, d)
    return -corr


def order_imbalance(df: pd.DataFrame, d: int = 21) -> pd.Series:
    """
    订单不平衡持续性：d 日收益方向均值（买压代理）。

    实现说明
    --------
    使用价格变动方向近似委比：ret > 0 视为买压，ret < 0 视为卖压。
        order_imbalance_proxy_t = (ret_t > 0) * 1 - (ret_t < 0) * 1  ∈ {-1, 0, 1}
    正值 → 买压主导 → 正向动量信号。

    无未来函数：ret_t 是 t 日收盘价相对 t-1 日的变动，收盘后才能知道，
    T+1 滞后由 pipeline 的 return_panel 统一处理，此处无需额外 shift。

    局限性说明（v2.9.1）
    --------------------
    用价格变动方向代理委比是极粗糙的近似，实际意义有限。
    若有实际委比/成交量分类（如 Lee-Ready 算法结果），应替换此因子。
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

# ═══════════════════════════════════════════════════════════════════════════════
# 因子注册辅助：一次性注册所有内置因子
# ═══════════════════════════════════════════════════════════════════════════════

def _register_builtins() -> "_CompatDict":
    """
    将所有内置因子注册到全局 REGISTRY，并返回 _CompatDict（向后兼容）。

    依赖方向：factor_zoo → factors.registry（单向，无循环）。
    必须在所有因子函数定义完成后调用（模块尾部）。
    """
    # Import sub-modules directly (not the factors package) to avoid
    # triggering factors/__init__.py while factor_zoo is mid-initialisation.
    from factor_framework.factors.meta import FactorMeta, FactorCategory, FactorStatus
    from factor_framework.factors.registry import REGISTRY, _CompatDict

    C  = FactorCategory
    St = FactorStatus
    V  = "2.9.1"   # 当前内置因子实现版本

    _specs = [
        # ── 动量因子 ────────────────────────────────────────────────────────
        dict(name="momentum_12_1",     fn=momentum_12_1,     display_name="经典 12-1 月动量",
             category=C.MOMENTUM,  warmup_days=252, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=stronger_momentum",
             forward_safe=True, tags=("hfq", "log_return", "skip_1m"),
             description="过去 252 日收益 - 过去 21 日收益（跳过最近 1 月）。后复权 + 对数差分。"),
        dict(name="momentum_6_1",      fn=momentum_6_1,      display_name="6-1 月中期动量",
             category=C.MOMENTUM,  warmup_days=126, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=stronger_momentum",
             forward_safe=True, tags=("hfq", "log_return", "skip_1m"),
             description="过去 126 日收益 - 过去 21 日收益。后复权 + 对数差分。"),
        dict(name="momentum_1m",       fn=momentum_1m,       display_name="短期 1 月动量",
             category=C.MOMENTUM,  warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=stronger_short_term_momentum",
             forward_safe=True, tags=("hfq", "log_return"),
             description="过去 21 日对数收益率。后复权 + 对数差分。"),
        dict(name="momentum_52w_high", fn=momentum_52w_high, display_name="52 周高点动量",
             category=C.MOMENTUM,  warmup_days=252, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=closer_to_52w_high",
             forward_safe=True, tags=("hfq", "52w_high"),
             description="收盘价 / 过去 252 日最高价。接近高点 → 正向信号。"),
        # ── 反转因子 ────────────────────────────────────────────────────────
        dict(name="reversal_1w",       fn=reversal_1w,       display_name="1 周短期反转",
             category=C.REVERSAL,  warmup_days=5, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=stronger_reversal_signal",
             forward_safe=True, tags=("hfq", "log_return", "negated"),
             description="过去 5 日收益率的负值。函数内部已取负，direction=+1。"),
        dict(name="reversal_1m",       fn=reversal_1m,       display_name="1 月短期反转",
             category=C.REVERSAL,  warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子"),
             output_semantic="higher=stronger_reversal_signal",
             forward_safe=True, tags=("hfq", "log_return", "negated"),
             description="过去 21 日收益率的负值。函数内部已取负，direction=+1。"),
        # ── 波动率因子 ──────────────────────────────────────────────────────
        dict(name="vol_20d",           fn=vol_20d,           display_name="20 日历史波动率",
             category=C.VOLATILITY, warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=lower_realized_vol",
             forward_safe=True, tags=("daily_ret", "negated"),
             description="20 日日收益率标准差的负值。函数内部已取负，direction=+1。"),
        dict(name="vol_60d",           fn=vol_60d,           display_name="60 日历史波动率",
             category=C.VOLATILITY, warmup_days=60, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=lower_realized_vol",
             forward_safe=True, tags=("daily_ret", "negated"),
             description="60 日日收益率标准差的负值。函数内部已取负，direction=+1。"),
        dict(name="vol_skew",          fn=vol_skew,          display_name="收益率偏度",
             category=C.VOLATILITY, warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=less_lottery_characteristic",
             forward_safe=True, tags=("daily_ret", "skewness", "negated"),
             description="20 日滚动偏度的负值（彩票效应：偏度大 → 预期收益低）。"),
        dict(name="downside_vol",      fn=downside_vol,      display_name="下行波动率",
             category=C.VOLATILITY, warmup_days=60, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=lower_downside_risk",
             forward_safe=True, tags=("daily_ret", "downside_risk", "full_window"),
             description="仅用负收益计算的 60 日下行标准差（满窗口 min_periods=60）。"),
        # ── 估值因子 ────────────────────────────────────────────────────────
        dict(name="value_pb",          fn=value_pb,          display_name="账面市值比（BP）",
             category=C.VALUE,     warmup_days=1, status=St.ACTIVE, version=V,
             inputs=("市净率",),
             output_semantic="higher=cheaper_valuation_pb",
             forward_safe=True, tags=("valuation", "daily_market_data"),
             description="1 / PB，值越大估值越低，正向因子。日频市场实时估值，lag_days=0。"),
        dict(name="value_pe_ttm",      fn=value_pe_ttm,      display_name="盈利市值比（EP）",
             category=C.VALUE,     warmup_days=1, status=St.ACTIVE, version=V,
             inputs=("市盈率（TTM，亏损为空）",),
             output_semantic="higher=cheaper_valuation_pe",
             forward_safe=True, tags=("valuation", "daily_market_data", "ttm"),
             description="1 / PE(TTM)，值越大估值越低，正向因子。日频市场实时估值，lag_days=0。"),
        dict(name="value_ps_ttm",      fn=value_ps_ttm,      display_name="营收市值比（SP）",
             category=C.VALUE,     warmup_days=1, status=St.ACTIVE, version=V,
             inputs=("市销率（TTM）",),
             output_semantic="higher=cheaper_valuation_ps",
             forward_safe=True, tags=("valuation", "daily_market_data", "ttm"),
             description="1 / PS(TTM)，值越大越便宜，正向因子。日频市场实时估值，lag_days=0。"),
        # ── 规模因子 ────────────────────────────────────────────────────────
        dict(name="size_log_mktcap",   fn=size_log_mktcap,   display_name="对数总市值（小盘溢价）",
             category=C.SIZE,      warmup_days=1, status=St.ACTIVE, version=V,
             neutral_by_default=False, skip_neutralize_cols=("市值",),
             inputs=("总市值（万元）",),
             output_semantic="higher=smaller_cap",
             forward_safe=True, tags=("size", "negated", "no_neutralize"),
             description="负对数总市值。函数内部取负，小盘→正向。不参与中性化。"),
        dict(name="size_log_free_cap", fn=size_log_free_cap, display_name="对数流通市值（小盘溢价）",
             category=C.SIZE,      warmup_days=1, status=St.ACTIVE, version=V,
             neutral_by_default=False, skip_neutralize_cols=("市值", "流通市值"),
             inputs=("流通市值（万元）",),
             output_semantic="higher=smaller_free_float_cap",
             forward_safe=True, tags=("size", "negated", "no_neutralize"),
             description="负对数流通市值。函数内部取负，小盘→正向。不参与中性化。"),
        # ── 量价因子 ────────────────────────────────────────────────────────
        dict(name="amihud_illiquidity",fn=amihud_illiquidity, display_name="Amihud 非流动性",
             category=C.VOLUME,    warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("_ret", "成交额（千元）"),
             output_semantic="higher=more_illiquid_risk_premium",
             forward_safe=True, tags=("liquidity_risk", "amihud"),
             description="|日收益率| / 成交额，20 日均值。值越大流动性越差，正向风险溢价。"),
        dict(name="turnover_rate",     fn=turnover_rate,     display_name="平均换手率（反转）",
             category=C.VOLUME,    warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("换手率（%）",),
             output_semantic="higher=lower_turnover",
             forward_safe=True, tags=("turnover", "negated"),
             description="负 20 日平均换手率。函数内部取负：高换手 → 预期收益低。"),
        dict(name="vol_price_corr",    fn=vol_price_corr,    display_name="量价相关性（负向）",
             category=C.VOLUME,    warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("成交量（手）", "收盘价"),
             output_semantic="higher=lower_vol_price_correlation",
             forward_safe=True, tags=("volume_price", "correlation", "negated"),
             description="成交量与收盘价的负 20 日滚动相关。函数内部取负。"),
        dict(name="vwap_deviation",    fn=vwap_deviation,    display_name="VWAP 偏离度",
             category=C.VOLUME,    warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("收盘价", "成交额（千元）", "成交量（手）"),
             output_semantic="higher=close_above_vwap",
             forward_safe=True, tags=("vwap", "intraday_proxy"),
             description="(Close - VWAP) / VWAP，20 日内日内均价偏离。"),
        dict(name="price_strength",    fn=price_strength,    display_name="价格强度（区间位置）",
             category=C.VOLUME,    warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("收盘价", "最高价", "最低价"),
             output_semantic="higher=stronger_price_position_in_range",
             forward_safe=True, tags=("price_range", "kd_proxy"),
             description="收盘价在过去 20 日高低区间中的位置 [0,1]，类似 KD 的 K 值。"),
        # ── 流动性质量因子 ──────────────────────────────────────────────────
        dict(name="bid_ask_spread_proxy", fn=bid_ask_spread_proxy, display_name="买卖价差代理（流动性）",
             category=C.LIQUIDITY, warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("最高价", "最低价", "收盘价"),
             output_semantic="higher=better_liquidity",
             forward_safe=True, tags=("spread_proxy", "corwin_schultz", "negated"),
             description="负 (H-L)/C 均值（Corwin & Schultz 简化版）。值越大流动性越好。"),
        dict(name="zero_return_ratio", fn=zero_return_ratio, display_name="零收益日占比（非活跃度）",
             category=C.LIQUIDITY, warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=more_active_trading",
             forward_safe=True, tags=("zero_return", "lesmond_1999", "negated"),
             description="负 21 日内零收益日占比（Lesmond 1999）。占比低→流动性好→正向。"),
        dict(name="pastor_stambaugh",  fn=pastor_stambaugh,  display_name="Pastor-Stambaugh 流动性",
             category=C.LIQUIDITY, warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("收盘价", "复权因子", "成交量（手）"),
             output_semantic="higher=better_liquidity_ps",
             forward_safe=True, tags=("ps_2003", "hfq", "negated"),
             description="负成交量加权收益率自相关（PS 2003 简化版）。后复权日收益率。"),
        dict(name="order_imbalance",   fn=order_imbalance,   display_name="订单不平衡（买压代理）",
             category=C.LIQUIDITY, warmup_days=21, status=St.ACTIVE, version=V,
             inputs=("_ret",),
             output_semantic="higher=stronger_buy_pressure",
             forward_safe=True, tags=("order_flow", "direction_proxy"),
             description="21 日价格变动方向均值（买压代理，动量信号）。"),
        # ── 技术分析因子 ────────────────────────────────────────────────────
        dict(name="rsi_14",            fn=rsi_14,            display_name="RSI-14 反转",
             category=C.TECHNICAL, warmup_days=14, status=St.ACTIVE, version=V,
             inputs=("收盘价",),
             output_semantic="higher=more_oversold",
             forward_safe=True, tags=("rsi", "overbought_oversold", "reversal"),
             description="50 - RSI(14)。超卖时为正，超买时为负，作为反转因子使用。"),
        dict(name="macd_signal",       fn=macd_signal,       display_name="MACD 差值（动量）",
             category=C.TECHNICAL, warmup_days=26, status=St.ACTIVE, version=V,
             inputs=("收盘价",),
             output_semantic="higher=stronger_short_term_trend",
             forward_safe=True, tags=("macd", "ema", "trend_following"),
             description="(EMA12 - EMA26) / Close，归一化 DIF，正向动量信号。"),
        dict(name="bb_position",       fn=bb_position,       display_name="布林带位置（反转）",
             category=C.TECHNICAL, warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("收盘价",),
             output_semantic="higher=closer_to_lower_band",
             forward_safe=True, tags=("bollinger", "mean_reversion", "reversal"),
             description="0.5 - BB%，接近下轨（超卖）为正值，反转因子。"),
        dict(name="volume_trend",      fn=volume_trend,      display_name="成交量趋势（线性斜率）",
             category=C.TECHNICAL, warmup_days=20, status=St.ACTIVE, version=V,
             inputs=("成交量（手）",),
             output_semantic="higher=increasing_volume_trend",
             forward_safe=True, tags=("volume", "linear_slope"),
             description="过去 20 日成交量线性回归归一化斜率，正值→量能放大。"),
    ]

    for spec in _specs:
        meta = FactorMeta(
            name                 = spec["name"],
            fn                   = spec["fn"],
            display_name         = spec["display_name"],
            category             = spec["category"],
            direction            = spec.get("direction", +1),
            warmup_days          = spec.get("warmup_days", 252),
            description          = spec.get("description", ""),
            neutral_by_default   = spec.get("neutral_by_default", True),
            skip_neutralize_cols = spec.get("skip_neutralize_cols", ()),
            # Phase E1 fields
            inputs               = tuple(spec.get("inputs", ())),
            output_semantic      = spec.get("output_semantic", ""),
            forward_safe         = spec.get("forward_safe", None),
            version              = spec.get("version", ""),
            tags                 = tuple(spec.get("tags", ())),
            status               = spec.get("status", St.ACTIVE),
        )
        REGISTRY.register(meta)

    return REGISTRY.to_compat_dict()


BUILTIN_FACTORS = _register_builtins()


def register_all(engine) -> None:
    """将所有内置因子注册到 FactorEngine 实例。"""
    for name, fn in BUILTIN_FACTORS.items():
        engine.register(name, fn)
