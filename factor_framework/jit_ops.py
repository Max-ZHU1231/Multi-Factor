"""
jit_ops.py
==========
因子表达式编译引擎 — 三层加速策略

层级 1：Numba JIT（纯数值滚动算子）
    适用：ts_mean / ts_std / ts_sum / ts_corr / ts_rank / ts_wma /
          ts_drawdown / ts_slope / ts_prod 等含循环的时间序列算子。
    原理：将 NumPy 数组操作用 @numba.njit(cache=True) 装饰，编译为
          LLVM 机器码。首次调用触发编译（约 3~15 秒，有磁盘缓存后
          重启无需重编译）；后续调用为原生速度，比 Pandas rolling
          快 5~10×。

层级 2：Numexpr（逐元素数学运算）
    适用：log / sqrt / power / if_else / clip 等无滚动窗口的纯数学
          表达式，以及多因子线性合成（A * w1 + B * w2）。
    原理：numexpr 将字符串表达式编译为字节码，自动多核并行 + 内存
          分块，避免 NumPy 产生的临时大数组。比 NumPy 快 2~4×。

层级 3：fallback（原生 Pandas/NumPy）
    适用：所有其他算子（ts_autocorr / ts_skew 等调用 Pandas 内置方
          法，已足够快；或 numba/numexpr 不可用时）。

编译元数据
----------
算子分类表 COMPILE_TARGET：
    'numba'   → 层级 1
    'numexpr' → 层级 2
    'numpy'   → 层级 3（向量化，无需额外编译）
    'pandas'  → 层级 3（直接使用 Pandas API）

可用性检测
----------
_NUMBA_OK    : bool，numba 是否可用
_NUMEXPR_OK  : bool，numexpr 是否可用
两者均通过 import 探测，不可用时自动退化到 Pandas 路径，不影响功能。

使用方法
--------
>>> from factor_framework.jit_ops import ts_mean_fast, ts_std_fast, ne_eval
>>> # 在自定义因子中替换 Pandas rolling：
>>> def my_factor(df):
...     arr = df["收盘价"].to_numpy(dtype=np.float64)
...     return pd.Series(ts_mean_fast(arr, 20), index=df.index)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# 可用性探测
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import numba
    from numba import njit, prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    warnings.warn(
        "[jit_ops] numba 未安装，时间序列算子将退化到 Pandas 路径。\n"
        "安装方法：pip install numba",
        stacklevel=2,
    )

try:
    import numexpr as ne
    _NUMEXPR_OK = True
except ImportError:
    _NUMEXPR_OK = False
    warnings.warn(
        "[jit_ops] numexpr 未安装，数学算子将退化到 NumPy 路径。\n"
        "安装方法：pip install numexpr",
        stacklevel=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 算子分类注册表
# ═══════════════════════════════════════════════════════════════════════════════

COMPILE_TARGET: dict[str, str] = {
    # 时间序列滚动算子 → Numba JIT
    "ts_mean":               "numba",
    "ts_stddev":             "numba",
    "ts_sum":                "numba",
    "ts_corr":               "numba",
    "ts_rank":               "numba",
    "ts_wma":                "numba",
    "ts_drawdown":           "numba",
    "ts_slope":              "numba",
    "ts_prod":               "numba",
    "ts_max":                "numba",
    "ts_min":                "numba",
    "ts_beta":               "numba",
    # 逐元素数学算子 → Numexpr
    "log":                   "numexpr",
    "sqrt":                  "numexpr",
    "absx":                  "numexpr",
    "power":                 "numexpr",
    "if_else":               "numexpr",
    "clip":                  "numexpr",
    # 已向量化（NumPy/Pandas 已足够快）
    "cs_rank":               "numpy",
    "cs_zscore":             "numpy",
    "cs_demean":             "numpy",
    "cs_scale":              "numpy",
    "cs_rank_by_group":      "numpy",
    "cs_neutralize":         "numpy",
    "cs_top_n":              "numpy",
    "cs_quantile":           "numpy",
    # 复杂 Pandas 算子（保持原路径）
    "ts_skew":               "pandas",
    "ts_autocorr":           "pandas",
    "ts_ema":                "pandas",
    "ts_rsi":                "pandas",
    "ts_regression_residual":"pandas",
    "ts_decay_linear":       "pandas",
    "delay":                 "pandas",
    "ts_delta":              "pandas",
    "ts_zscore":             "pandas",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 层级 1：Numba JIT 核心实现
# ═══════════════════════════════════════════════════════════════════════════════
#
# 所有 _nb_* 函数均以 @numba.njit(cache=True) 装饰，接受 float64 数组，
# 返回 float64 数组（长度与输入相同，前 d-1 个元素为 NaN）。
#
# Pandas 包装函数（公开 API）：ts_*_fast(x: pd.Series, d: int) -> pd.Series
# ─────────────────────────────────────────────────────────────────────────────

if _NUMBA_OK:
    # ── ts_sum ──────────────────────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_sum(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            out[i] = np.sum(w)
        return out

    # ── ts_mean ─────────────────────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_mean(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            out[i] = np.sum(w) / d
        return out

    # ── ts_std (sample, ddof=1) ─────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_std(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            mu  = np.sum(w) / d
            var = np.sum((w - mu) ** 2) / (d - 1)
            out[i] = np.sqrt(var)
        return out

    # ── ts_max / ts_min ──────────────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_max(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            out[i] = np.max(w)
        return out

    @njit(cache=True)
    def _nb_ts_min(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            out[i] = np.min(w)
        return out

    # ── ts_corr (Pearson) ────────────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        n   = len(x)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            xw = x[i - d + 1: i + 1]
            yw = y[i - d + 1: i + 1]
            if np.any(np.isnan(xw)) or np.any(np.isnan(yw)):
                continue
            mx = np.sum(xw) / d
            my = np.sum(yw) / d
            xd = xw - mx
            yd = yw - my
            num   = np.sum(xd * yd)
            denom = np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2))
            if denom > 1e-14:
                out[i] = num / denom
        return out

    # ── ts_wma（线性加权移动均值）───────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_wma(arr: np.ndarray, d: int) -> np.ndarray:
        n       = len(arr)
        out     = np.full(n, np.nan)
        w_sum   = d * (d + 1) / 2.0
        for i in range(d - 1, n):
            window = arr[i - d + 1: i + 1]
            if np.any(np.isnan(window)):
                continue
            val = 0.0
            for k in range(d):
                val += (k + 1) * window[k]
            out[i] = val / w_sum
        return out

    # ── ts_rank（窗口内分位排名）────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_rank(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            window = arr[i - d + 1: i + 1]
            if np.any(np.isnan(window)):
                continue
            last  = window[-1]
            count = 0
            for v in window:
                if v <= last:
                    count += 1
            out[i] = count / d
        return out

    # ── ts_prod（滚动连乘）──────────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_prod(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            p = 1.0
            for v in w:
                p *= v
            out[i] = p
        return out

    # ── ts_drawdown（最大回撤）──────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_drawdown(arr: np.ndarray, d: int) -> np.ndarray:
        n   = len(arr)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            peak = w[0]
            mdd  = 0.0
            for v in w:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak != 0.0 else 0.0
                if dd > mdd:
                    mdd = dd
            out[i] = mdd
        return out

    # ── ts_slope（归一化线性回归斜率）──────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_slope(arr: np.ndarray, d: int) -> np.ndarray:
        n    = len(arr)
        out  = np.full(n, np.nan)
        # 预计算时间轴（去均值）
        t    = np.arange(d, dtype=np.float64)
        t   -= t.mean()
        t_var = np.sum(t ** 2)
        if t_var < 1e-14:
            return out
        for i in range(d - 1, n):
            w = arr[i - d + 1: i + 1]
            if np.any(np.isnan(w)):
                continue
            mu    = np.sum(w) / d
            w_dm  = w - mu
            slope = np.sum(t * w_dm) / t_var
            mean_abs = abs(mu)
            out[i] = slope / mean_abs if mean_abs > 1e-10 else slope
        return out

    # ── ts_beta（滚动 OLS beta）─────────────────────────────────────────────
    @njit(cache=True)
    def _nb_ts_beta(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        n   = len(x)
        out = np.full(n, np.nan)
        for i in range(d - 1, n):
            xw = x[i - d + 1: i + 1]
            yw = y[i - d + 1: i + 1]
            if np.any(np.isnan(xw)) or np.any(np.isnan(yw)):
                continue
            mx = np.sum(xw) / d
            my = np.sum(yw) / d
            cov = np.sum((xw - mx) * (yw - my)) / (d - 1)
            var = np.sum((yw - my) ** 2) / (d - 1)
            if var > 1e-14:
                out[i] = cov / var
        return out

else:
    # ── Numba 不可用：定义 Python 等价体（保持接口一致）───────────────────

    def _nb_ts_sum(arr, d):
        return _pandas_rolling(pd.Series(arr), d, "sum")

    def _nb_ts_mean(arr, d):
        return _pandas_rolling(pd.Series(arr), d, "mean")

    def _nb_ts_std(arr, d):
        return _pandas_rolling(pd.Series(arr), d, "std")

    def _nb_ts_max(arr, d):
        return _pandas_rolling(pd.Series(arr), d, "max")

    def _nb_ts_min(arr, d):
        return _pandas_rolling(pd.Series(arr), d, "min")

    def _nb_ts_corr(x, y, d):
        return pd.Series(x).rolling(d, min_periods=d).corr(pd.Series(y)).to_numpy()

    def _nb_ts_wma(arr, d):
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        s = pd.Series(arr)
        return s.rolling(d, min_periods=d).apply(
            lambda w: np.dot(w, weights) if not np.isnan(w).any() else np.nan,
            raw=True,
        ).to_numpy()

    def _nb_ts_rank(arr, d):
        def _r(w):
            if np.isnan(w).any():
                return np.nan
            return float(pd.Series(w).rank(pct=True).iloc[-1])
        return pd.Series(arr).rolling(d, min_periods=d).apply(_r, raw=True).to_numpy()

    def _nb_ts_prod(arr, d):
        return pd.Series(arr).rolling(d, min_periods=d).apply(np.prod, raw=True).to_numpy()

    def _nb_ts_drawdown(arr, d):
        def _mdd(w):
            if np.isnan(w).any():
                return np.nan
            cummax = np.maximum.accumulate(w)
            dd = (cummax - w) / np.where(cummax == 0, np.nan, cummax)
            return float(np.nanmax(dd))
        return pd.Series(arr).rolling(d, min_periods=d).apply(_mdd, raw=True).to_numpy()

    def _nb_ts_slope(arr, d):
        t = np.arange(d, dtype=float)
        t -= t.mean()
        t_var = (t ** 2).sum()
        def _slope(w):
            if np.isnan(w).any() or t_var < 1e-14:
                return np.nan
            w_dm = w - w.mean()
            slope = float(np.dot(t, w_dm) / t_var)
            ma = abs(w.mean())
            return slope / ma if ma > 1e-10 else slope
        return pd.Series(arr).rolling(d, min_periods=d).apply(_slope, raw=True).to_numpy()

    def _nb_ts_beta(x, y, d):
        sx, sy = pd.Series(x), pd.Series(y)
        cov = sx.rolling(d, min_periods=d).cov(sy)
        var = sy.rolling(d, min_periods=d).var(ddof=1)
        return (cov / var.replace(0, np.nan)).to_numpy()


def _pandas_rolling(s: pd.Series, d: int, method: str) -> np.ndarray:
    """Helper for the no-numba fallback path."""
    return getattr(s.rolling(d, min_periods=d), method)().to_numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 公开 Pandas 包装函数（层级 1 API）
# ═══════════════════════════════════════════════════════════════════════════════
# 命名规则：ts_*_fast(x: pd.Series, d: int) -> pd.Series
# 与 operators.py 中的 ts_* 函数签名兼容。

def _wrap(arr_fn, x: pd.Series, *args) -> pd.Series:
    """将 ndarray 函数的结果包装回 pd.Series（保留 index 和 name）。"""
    arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    out = arr_fn(arr, *args)
    return pd.Series(out, index=x.index, name=x.name)


def ts_sum_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动累加和（numba 可用时比 Pandas 快 5~8×）。"""
    return _wrap(_nb_ts_sum, x, d)


def ts_mean_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动均值。"""
    return _wrap(_nb_ts_mean, x, d)


def ts_std_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动标准差（ddof=1）。"""
    return _wrap(_nb_ts_std, x, d)


def ts_max_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动最大值。"""
    return _wrap(_nb_ts_max, x, d)


def ts_min_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动最小值。"""
    return _wrap(_nb_ts_min, x, d)


def ts_corr_fast(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动 Pearson 相关系数。"""
    xa = x.to_numpy(dtype=np.float64, na_value=np.nan)
    ya = y.to_numpy(dtype=np.float64, na_value=np.nan)
    out = _nb_ts_corr(xa, ya, d)
    return pd.Series(out, index=x.index, name=x.name)


def ts_wma_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的线性加权移动均值。"""
    return _wrap(_nb_ts_wma, x, d)


def ts_rank_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的窗口内分位排名（0~1）。"""
    return _wrap(_nb_ts_rank, x, d)


def ts_prod_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动连乘。"""
    return _wrap(_nb_ts_prod, x, d)


def ts_drawdown_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动最大回撤（0~1）。"""
    return _wrap(_nb_ts_drawdown, x, d)


def ts_slope_fast(x: pd.Series, d: int) -> pd.Series:
    """JIT 加速的归一化线性回归斜率。"""
    return _wrap(_nb_ts_slope, x, d)


def ts_beta_fast(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """JIT 加速的滚动 OLS Beta。"""
    xa = x.to_numpy(dtype=np.float64, na_value=np.nan)
    ya = y.to_numpy(dtype=np.float64, na_value=np.nan)
    out = _nb_ts_beta(xa, ya, d)
    return pd.Series(out, index=x.index, name=x.name)


# ═══════════════════════════════════════════════════════════════════════════════
# 层级 2：Numexpr 表达式求值
# ═══════════════════════════════════════════════════════════════════════════════

def ne_eval(expr: str, local_dict: dict) -> np.ndarray:
    """
    使用 numexpr 对字符串表达式求值。

    Parameters
    ----------
    expr       : 数学表达式字符串，如 "log(x + 1e-9)" 或 "a * 0.5 + b * 0.5"
    local_dict : 变量字典，值为 np.ndarray（自动转换 pd.Series）

    Returns
    -------
    np.ndarray，与输入等长

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> ne_eval("log(x)", {"x": x})
    array([0.       , 0.6931472, 1.0986123])
    """
    if not _NUMEXPR_OK:
        # 降级：用 Python eval + numpy 命名空间
        env = {k: (v.to_numpy() if isinstance(v, pd.Series) else v)
               for k, v in local_dict.items()}
        env["log"]  = np.log
        env["sqrt"] = np.sqrt
        env["abs"]  = np.abs
        env["exp"]  = np.exp
        return eval(expr, {"__builtins__": {}}, env)  # noqa: S307

    clean = {k: (v.to_numpy(dtype=np.float64) if isinstance(v, pd.Series) else
                 np.asarray(v, dtype=np.float64))
             for k, v in local_dict.items()}
    return ne.evaluate(expr, local_dict=clean)


def ne_log(x: pd.Series) -> pd.Series:
    """Numexpr 加速的自然对数（x ≤ 0 → NaN）。"""
    arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    if _NUMEXPR_OK:
        # numexpr log of non-positive → nan via where
        out = ne.evaluate("where(x > 0, log(x), nan)", local_dict={"x": arr, "nan": np.nan})
    else:
        out = np.where(arr > 0, np.log(np.where(arr > 0, arr, 1.0)), np.nan)
    return pd.Series(out, index=x.index, name=x.name)


def ne_sqrt(x: pd.Series) -> pd.Series:
    """Numexpr 加速的平方根（x < 0 → 0）。"""
    arr = np.clip(x.to_numpy(dtype=np.float64, na_value=np.nan), 0, None)
    if _NUMEXPR_OK:
        out = ne.evaluate("sqrt(x)", local_dict={"x": arr})
    else:
        out = np.sqrt(arr)
    return pd.Series(out, index=x.index, name=x.name)


def ne_combine(panels: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """
    多因子线性合成（Numexpr 多核加速）。

    Parameters
    ----------
    panels  : {factor_name: panel_DataFrame}
    weights : {factor_name: weight_float}

    Returns
    -------
    合成后的面板 DataFrame（与输入面板等形状）
    """
    # 取公共日期 × 股票
    names = list(panels.keys())
    ref   = panels[names[0]]
    result = pd.DataFrame(np.zeros(ref.shape), index=ref.index, columns=ref.columns)

    # 对齐所有面板
    aligned = {n: panels[n].reindex(index=ref.index, columns=ref.columns)
               for n in names}

    if _NUMEXPR_OK:
        # 构造表达式：a * wa + b * wb + ...
        vars_   = {f"p{i}": aligned[n].to_numpy(dtype=np.float64)
                   for i, n in enumerate(names)}
        vars_   .update({f"w{i}": np.float64(weights[n])
                         for i, n in enumerate(names)})
        expr    = " + ".join(f"p{i} * w{i}" for i in range(len(names)))
        out_arr = ne.evaluate(expr, local_dict=vars_)
        result  = pd.DataFrame(out_arr, index=ref.index, columns=ref.columns)
    else:
        out = np.zeros(ref.shape)
        for n in names:
            out += aligned[n].to_numpy(dtype=np.float64) * weights[n]
        result = pd.DataFrame(out, index=ref.index, columns=ref.columns)

    # 任意面板位置为 NaN → 合成结果也为 NaN
    any_nan = np.zeros(ref.shape, dtype=bool)
    for n in names:
        any_nan |= np.isnan(aligned[n].to_numpy(dtype=np.float64))
    result[any_nan] = np.nan
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 预热 / 编译触发（可选）
# ═══════════════════════════════════════════════════════════════════════════════

def warmup(verbose: bool = True) -> dict[str, float]:
    """
    触发所有 Numba JIT 函数的首次编译（预热）。
    在程序启动时调用可避免第一次因子计算时的编译延迟。

    Returns
    -------
    dict：各函数名 → 编译耗时（秒）
    """
    import time

    dummy = np.random.default_rng(0).normal(0, 1, 60).astype(np.float64)
    dummy2 = np.random.default_rng(1).normal(0, 1, 60).astype(np.float64)

    funcs = [
        ("ts_sum",      lambda: _nb_ts_sum(dummy, 5)),
        ("ts_mean",     lambda: _nb_ts_mean(dummy, 5)),
        ("ts_std",      lambda: _nb_ts_std(dummy, 5)),
        ("ts_max",      lambda: _nb_ts_max(dummy, 5)),
        ("ts_min",      lambda: _nb_ts_min(dummy, 5)),
        ("ts_corr",     lambda: _nb_ts_corr(dummy, dummy2, 5)),
        ("ts_wma",      lambda: _nb_ts_wma(dummy, 5)),
        ("ts_rank",     lambda: _nb_ts_rank(dummy, 5)),
        ("ts_prod",     lambda: _nb_ts_prod(dummy, 5)),
        ("ts_drawdown", lambda: _nb_ts_drawdown(dummy, 5)),
        ("ts_slope",    lambda: _nb_ts_slope(dummy, 5)),
        ("ts_beta",     lambda: _nb_ts_beta(dummy, dummy2, 5)),
    ]

    times = {}
    for name, fn in funcs:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times[name] = elapsed
        if verbose:
            status = "(JIT)" if _NUMBA_OK else "(fallback)"
            print(f"  [warmup] {name:<18s} {status}  {elapsed:.3f}s")

    return times
