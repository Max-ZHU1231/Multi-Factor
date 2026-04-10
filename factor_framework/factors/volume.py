"""
factor_framework.factors.volume
=================================
量价 / 流动性质量 / 技术分析类内置因子函数（re-export from factor_zoo）。
"""
from factor_framework.factor_zoo import (   # noqa: F401
    amihud_illiquidity,
    turnover_rate,
    vol_price_corr,
    vwap_deviation,
    price_strength,
    bid_ask_spread_proxy,
    zero_return_ratio,
    pastor_stambaugh,
    order_imbalance,
    rsi_14,
    macd_signal,
    bb_position,
    volume_trend,
)

__all__ = [
    "amihud_illiquidity", "turnover_rate", "vol_price_corr",
    "vwap_deviation", "price_strength",
    "bid_ask_spread_proxy", "zero_return_ratio", "pastor_stambaugh", "order_imbalance",
    "rsi_14", "macd_signal", "bb_position", "volume_trend",
]
