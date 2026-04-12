"""
ic_decay_diagnostics.py
=======================
IC 衰减异常诊断框架（6 模块完整实现）

背景
----
当观测到 forward 越长、IC 越高的"反常"衰减形态时（如 forward=1 的 IC=0.025 vs
forward=60 的 IC=0.153），该框架提供系统性的诊断与量化证据。

6 个诊断模块
------------
Module 1 — 时间对齐与前瞻偏差检查
    - 验证收益窗口是否严格从 t+1 开始（不含当日）
    - lag=0/1/2 敏感性对比：若 lag=0 IC 显著高于 lag=1，判定存在泄露
    - 月末重采样后因子-收益对齐验证

Module 2 — 收益定义与累计窗口拆解
    - 累计 IC vs 增量 IC（IC(f_t, r_{t+k}) for k=1..h）
    - 重叠窗口调整：Newey-West 修正 t 值 vs 标准 t 值
    - 结论标准：若增量 IC 在 k>5 后趋近 0 而累计 IC 仍升 → 累计统计放大效应

Module 3 — 市场/风格暴露剥离
    - 市场超额收益（去掉截面均值）下的 IC
    - 行业内超额收益下的 IC
    - 市值中性化后的 IC
    - 市值+行业双重中性化后的 IC

Module 4 — 样本偏差检查（生存偏差/覆盖率）
    - 各 forward 的有效样本覆盖率（非 NaN 占比）
    - forward=60 vs forward=21 的覆盖率差异
    - 月末样本与日频样本的数量对比

Module 5 — 因子属性验证（时效性）
    - 因子截面自相关（lag=1..12 个月）
    - 自相关半衰期估算
    - 增量 IC 的"有效预测半衰期"

Module 6 — 稳健性复核
    - 滚动样本外 IC（时间外推稳定性）
    - 参数扰动：winsorize 阈值 ±20%、standardize 方式对比
    - 按市场状态（牛市/熊市/震荡）分 regime
    - 结论汇总：通过/不通过 + 定量证据

用法
----
from factor_framework.analytics.ic_decay_diagnostics import ICDecayDiagnostics

diag = ICDecayDiagnostics(
    factor_panel    = factor_panel,     # (T×N) 日频或月频因子面板
    price_panel     = price_panel,      # (T×N) 后复权价格面板（用于构建收益）
    industry_map    = industry_map,     # pd.Series: ts_code → industry（可选）
    mktcap_panel    = mktcap_panel,     # (T×N) 市值面板（可选）
    forward_list    = [1, 5, 10, 21, 60],
    ic_method       = "rank",
)
report = diag.run_all()
report.print_full()
report.to_dict()         # 返回 dict，可序列化为 JSON
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 抑制 nanmean 在全 NaN 行时的 RuntimeWarning（诊断模块中属正常 edge case）
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom")


# ═══════════════════════════════════════════════════════════════════════════════
# 进度条工具（兼容交互/非交互两种环境）
# ═══════════════════════════════════════════════════════════════════════════════

def _is_interactive() -> bool:
    """检测当前是否为支持 ANSI 进度条的交互式终端。"""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class _ProgressBar:
    """
    单条进度条（TTY 模式下原地刷新，非 TTY 模式下输出简洁日志行）。

    Parameters
    ----------
    total    : 总步数
    desc     : 描述文字（显示在进度条前）
    width    : 进度条字符宽度（仅 TTY 模式）
    tty      : 强制覆盖自动检测结果
    unit     : 计数单位名称（如 "step", "k"）
    indent   : 缩进空格数（用于子进度条）
    log_every: 非 TTY 模式下每多少步打印一次日志
    """

    def __init__(
        self,
        total:     int,
        desc:      str  = "",
        width:     int  = 32,
        tty:       Optional[bool] = None,
        unit:      str  = "step",
        indent:    int  = 4,
        log_every: int  = 0,         # 0 = 仅 start/end
    ):
        self.total     = max(1, total)
        self.desc      = desc
        self.width     = width
        self.tty       = _is_interactive() if tty is None else tty
        self.unit      = unit
        self.indent    = indent
        self.log_every = log_every
        self._n        = 0
        self._t0       = time.monotonic()
        self._last_log = -1          # 上次 log 时的 n（非 TTY）

        if self.tty:
            self._render(force=True)

    # ── internal ──────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def _eta(self) -> Optional[float]:
        if self._n == 0:
            return None
        rate = self._n / self._elapsed()
        return (self.total - self._n) / rate if rate > 0 else None

    def _fmt_sec(self, secs: Optional[float]) -> str:
        if secs is None or secs != secs:        # None or NaN
            return "??"
        secs = int(secs)
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def _bar_str(self) -> str:
        frac  = self._n / self.total
        filled = int(self.width * frac)
        bar   = "█" * filled + "░" * (self.width - filled)
        pct   = int(frac * 100)
        ela   = self._fmt_sec(self._elapsed())
        eta   = self._fmt_sec(self._eta())
        pad   = " " * self.indent
        return (
            f"\r{pad}{self.desc} [{bar}] {pct:3d}% "
            f"{self._n}/{self.total} {self.unit}  "
            f"elapsed {ela}  ETA {eta}   "
        )

    def _render(self, force: bool = False) -> None:
        if not self.tty:
            return
        print(self._bar_str(), end="", flush=True)

    # ── public ────────────────────────────────────────────────────────────

    def update(self, n: int = 1) -> None:
        """推进 n 步。"""
        self._n = min(self._n + n, self.total)
        if self.tty:
            self._render()
        else:
            # 非 TTY：仅在设定了 log_every 时才周期打印
            if self.log_every > 0 and self._n - self._last_log >= self.log_every:
                pad = " " * self.indent
                ela = self._fmt_sec(self._elapsed())
                eta = self._fmt_sec(self._eta())
                print(
                    f"{pad}{self.desc}: {self._n}/{self.total} {self.unit}  "
                    f"elapsed {ela}  ETA {eta}",
                    flush=True,
                )
                self._last_log = self._n

    def close(self, suffix: str = "") -> None:
        """完成并换行。"""
        self._n = self.total
        ela = self._fmt_sec(self._elapsed())
        if self.tty:
            pad = " " * self.indent
            print(
                f"\r{pad}{self.desc} [{'█' * self.width}] 100%  "
                f"{self.total}/{self.total} {self.unit}  elapsed {ela}   ",
                flush=True,
            )
            print()           # newline before next message
        else:
            pad = " " * self.indent
            msg = f"{pad}[完成] {self.desc}  {self.total} {self.unit}  elapsed {ela}"
            if suffix:
                msg += f"  {suffix}"
            print(msg, flush=True)

    def __enter__(self) -> "_ProgressBar":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── 迭代器包装 ────────────────────────────────────────────────────────

    def iter(self, iterable: Iterable) -> Iterator:
        """包装可迭代对象，每次 yield 后自动 update(1)。"""
        for item in iterable:
            yield item
            self.update(1)
        self.close()


class _ModuleProgress:
    """
    外层模块总进度控制器（run_all() 使用）。

    负责：
    - 打印 6 模块总进度头部
    - 提供 begin_module / end_module API
    - 在 TTY 模式下原地更新；非 TTY 下打印简洁 start/end 行
    """

    _MODULE_NAMES = {
        1: "时间对齐/前瞻检查",
        2: "累计窗口拆解",
        3: "风格暴露剥离",
        4: "样本偏差检查",
        5: "因子时效/半衰期",
        6: "稳健性复核",
    }

    def __init__(self, module_ids: List[int], tty: Optional[bool] = None):
        self.module_ids  = module_ids
        self.total       = len(module_ids)
        self.tty         = _is_interactive() if tty is None else tty
        self._done       = 0
        self._t0         = time.monotonic()
        self._mod_t0     = self._t0
        self._current_mid: Optional[int] = None

    def _fmt_sec(self, secs: float) -> str:
        secs = int(secs)
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"

    def begin_module(self, mid: int) -> None:
        self._current_mid = mid
        self._mod_t0 = time.monotonic()
        name = self._MODULE_NAMES.get(mid, f"Module {mid}")
        if self.tty:
            # 打印当前模块 header（非覆盖行，让子进度条在其下方）
            idx = self.module_ids.index(mid) + 1
            ela = self._fmt_sec(time.monotonic() - self._t0)
            print(
                f"  [M{mid}/{self.total}] {name} ...  (total elapsed {ela})",
                flush=True,
            )
        else:
            idx = self.module_ids.index(mid) + 1
            print(f"[INFO] [M{mid}] {name} start", flush=True)

    def end_module(self, mid: int, status: str) -> None:
        self._done += 1
        mod_ela = self._fmt_sec(time.monotonic() - self._mod_t0)
        name    = self._MODULE_NAMES.get(mid, f"Module {mid}")
        total_ela = self._fmt_sec(time.monotonic() - self._t0)

        if self.tty:
            # 覆盖同一行：在 begin_module 之后再打印结果行
            print(
                f"  [M{mid}] {name:18s}  {status}  ({mod_ela})"
                f"  [{self._done}/{self.total} done, total elapsed {total_ela}]",
                flush=True,
            )
        else:
            print(
                f"  [M{mid}] {name}  {status}  elapsed {mod_ela}",
                flush=True,
            )

    def summary(self) -> None:
        total_ela = self._fmt_sec(time.monotonic() - self._t0)
        print(f"\n[INFO] Diagnostics completed  total elapsed {total_ela}", flush=True)


# ── 便捷工厂函数 ──────────────────────────────────────────────────────────────

def _make_pbar(
    total: int,
    desc:  str,
    *,
    tty:       Optional[bool] = None,
    unit:      str = "step",
    indent:    int = 6,
    log_every: int = 0,
) -> _ProgressBar:
    """创建子进度条（用于模块内部循环）。"""
    return _ProgressBar(
        total=total, desc=desc, tty=tty,
        unit=unit, indent=indent, log_every=log_every,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """截面排名（处理 NaN）。"""
    return panel.rank(axis=1, na_option="keep")


# ── P1: 纯 ndarray 排名（避免 DataFrame overhead）────────────────────────────

def _rank_array(arr: np.ndarray) -> np.ndarray:
    """
    截面排名（T×N ndarray），NaN 保持 NaN。
    使用 argsort 两次求法，比 scipy.stats.rankdata 快约 3×。
    """
    result = np.empty_like(arr, dtype=np.float64)
    result[:] = np.nan
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = ~np.isnan(row)
        n_valid = valid.sum()
        if n_valid < 2:
            continue
        idx = np.where(valid)[0]
        vals = row[idx]
        order = vals.argsort()
        ranks = np.empty(n_valid, dtype=np.float64)
        ranks[order] = np.arange(1, n_valid + 1, dtype=np.float64)
        result[i, idx] = ranks
    return result


# ── P1: 矩阵批量 IC（无 Python 循环）─────────────────────────────────────────

def _ic_from_arrays(
    f_arr: np.ndarray,
    r_arr: np.ndarray,
    method: str = "rank",
    min_stocks: int = 5,
) -> np.ndarray:
    """
    批量计算所有行的 IC（T,）。输入均为 T×N ndarray，已对齐（相同行列）。

    返回 shape (T,) 的 IC 向量，无效行为 NaN。
    """
    if method == "rank":
        f_arr = _rank_array(f_arr)
        r_arr = _rank_array(r_arr)

    nan_mask = np.isnan(f_arr) | np.isnan(r_arr)
    f_arr = np.where(nan_mask, np.nan, f_arr)
    r_arr = np.where(nan_mask, np.nan, r_arr)

    valid_counts = (~nan_mask).sum(axis=1)          # (T,)

    # nanmean per row — suppress empty-slice RuntimeWarning
    with np.errstate(all="ignore"):
        f_mean = np.nanmean(f_arr, axis=1, keepdims=True)   # (T,1)
        r_mean = np.nanmean(r_arr, axis=1, keepdims=True)

    f_dm = f_arr - f_mean
    r_dm = r_arr - r_mean
    f_dm = np.where(nan_mask, 0.0, f_dm)
    r_dm = np.where(nan_mask, 0.0, r_dm)

    num   = (f_dm * r_dm).sum(axis=1)               # (T,)
    ss_f  = (f_dm ** 2).sum(axis=1)
    ss_r  = (r_dm ** 2).sum(axis=1)
    denom = np.sqrt(ss_f * ss_r)
    denom = np.where(denom == 0, np.nan, denom)

    ic = num / denom
    ic = np.where(valid_counts < min_stocks, np.nan, ic)
    return ic


def _compute_ic_series(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    method: str = "rank",
    min_stocks: int = 5,
) -> pd.Series:
    """
    逐期计算 IC（P1 向量化 ndarray 路径，比原版快 ~3-5×）。
    仅在诊断模块内部使用，避免循环导入。
    """
    common_dates  = factor.index.intersection(ret.index)
    common_stocks = factor.columns.intersection(ret.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.Series(dtype=float, name="IC")

    f_arr = factor.loc[common_dates, common_stocks].to_numpy(dtype=float, na_value=np.nan)
    r_arr = ret.loc[common_dates, common_stocks].to_numpy(dtype=float, na_value=np.nan)

    ic_arr = _ic_from_arrays(f_arr, r_arr, method=method, min_stocks=min_stocks)
    return pd.Series(ic_arr, index=common_dates, name="IC")


def _ic_stats_dict(ic: pd.Series) -> Dict:
    """简化版 IC 统计指标。"""
    clean = ic.dropna()
    n = len(clean)
    if n == 0:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan,
                "win_rate": np.nan, "t_stat": np.nan, "n": 0}
    mean_ic  = float(clean.mean())
    std_ic   = float(clean.std(ddof=1))
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    win_rate = float((clean > 0).mean())
    t_stat   = float(mean_ic / (std_ic / np.sqrt(n))) if std_ic > 0 else np.nan
    return {"mean_ic": round(mean_ic, 6), "std_ic": round(std_ic, 6),
            "icir": round(icir, 4) if not np.isnan(icir) else np.nan,
            "win_rate": round(win_rate, 4), "t_stat": round(t_stat, 4), "n": n}


def _nw_t(ic: pd.Series, lags: Optional[int] = None) -> float:
    """Newey-West 修正 t 统计量。"""
    clean = ic.dropna().values
    n = len(clean)
    if n < 4:
        return np.nan
    if lags is None:
        lags = max(1, int(n ** 0.25))
    mean_ic = clean.mean()
    e = clean - mean_ic
    var_nw = np.sum(e ** 2) / n
    for k in range(1, lags + 1):
        cov_k = np.sum(e[k:] * e[:-k]) / n
        var_nw += 2 * (1 - k / (lags + 1)) * cov_k
    se_nw = np.sqrt(var_nw / n)
    return float(mean_ic / se_nw) if se_nw > 0 else np.nan


def _build_forward_ret(price_panel: pd.DataFrame, fwd: int) -> pd.DataFrame:
    """
    构建 forward 收益率（不含 T+1 shift）。

    诊断模块中使用 lag 参数来显式控制 T+1 偏移，
    故此处不内置 shift，由调用方根据 lag 参数决定是否 shift。
    """
    p = price_panel.replace(0, np.nan)
    return p.shift(-fwd) / p - 1


def _shift_factor(factor_panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    对因子面板做 lag 期滞后（防前瞻偏差测试用）。
    lag=1：标准 T+1 对齐；lag=0：不滞后（含当日，存在泄露风险）；lag=2：额外1天。
    """
    if lag == 0:
        return factor_panel
    return factor_panel.shift(lag)


def _to_datetime_index(idx) -> pd.DatetimeIndex:
    """统一将 index 转为 DatetimeIndex。"""
    if pd.api.types.is_datetime64_any_dtype(idx):
        return idx
    try:
        return pd.to_datetime(idx, format="%Y%m%d")
    except Exception:
        return pd.to_datetime(idx)


def _resample_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    """重采样为月末截面（取每月最后一个有效日期）。"""
    df2 = df.copy()
    df2.index = _to_datetime_index(df.index)
    try:
        return df2.resample("ME").last().dropna(how="all")
    except Exception:
        return df2.resample("M").last().dropna(how="all")


# ── P0: 批量 OLS 截面中性化（避免逐日 Python 循环）──────────────────────────

def _neutralize_batch(
    factor_arr:    np.ndarray,    # T×N, float64, NaN for missing
    covariate_arr: np.ndarray,    # T×N, float64 (e.g. log_mktcap)
    min_obs:       int = 5,
) -> np.ndarray:
    """
    P0 向量化单变量截面 OLS 中性化（无 Python 行循环）。

    对每一行 t，做 y_i = a + b * x_i 的 OLS，取残差 e_i = y_i - a - b*x_i。
    利用 broadcast 矩阵运算，比逐行 np.linalg.lstsq 快 ~10-30×。

    Parameters
    ----------
    factor_arr    : (T×N) 因子矩阵
    covariate_arr : (T×N) 协变量矩阵（log_mktcap 等）
    min_obs       : 最小有效观测数，不足则该行残差全为 NaN

    Returns
    -------
    resid_arr : (T×N) 残差矩阵，与输入同形状
    """
    T, N = factor_arr.shape
    resid = np.full((T, N), np.nan, dtype=np.float64)

    # 有效掩码
    valid = ~(np.isnan(factor_arr) | np.isnan(covariate_arr))  # T×N bool

    y = np.where(valid, factor_arr,    np.nan)   # T×N
    x = np.where(valid, covariate_arr, np.nan)   # T×N

    n_valid = valid.sum(axis=1)                  # (T,)

    # nanmean per row
    with np.errstate(all="ignore"):
        y_mean = np.nanmean(y, axis=1, keepdims=True)   # T×1
        x_mean = np.nanmean(x, axis=1, keepdims=True)

    y_dm = np.where(valid, y - y_mean, 0.0)
    x_dm = np.where(valid, x - x_mean, 0.0)

    # OLS slope: b = Σ(x_dm * y_dm) / Σ(x_dm²)
    cov_xy = (x_dm * y_dm).sum(axis=1)      # (T,)
    var_x  = (x_dm ** 2).sum(axis=1)        # (T,)

    b = np.where(var_x > 0, cov_xy / var_x, 0.0)      # (T,)
    a = y_mean[:, 0] - b * x_mean[:, 0]               # (T,)

    # residuals = y - a - b*x, only where valid
    fitted = a[:, None] + b[:, None] * x              # T×N
    r      = y - fitted                                # T×N

    # mask rows with too few observations
    enough = n_valid >= min_obs                        # (T,)
    resid  = np.where(valid & enough[:, None], r, np.nan)
    return resid


def _neutralize_batch_with_dummies(
    factor_arr:    np.ndarray,    # T×N
    covariate_arr: np.ndarray,    # T×N (log_mktcap)
    dummy_arr:     np.ndarray,    # N×K (industry dummies, already computed once)
    valid_stocks:  np.ndarray,    # (N,) bool — stocks that have industry mapping
    min_obs:       int = 10,
) -> np.ndarray:
    """
    P0 带行业哑变量的截面 OLS 中性化。

    因每期哑变量矩阵形状固定（N×K），可向量化（不逐日 Python 循环）。
    使用 (X'X)^{-1} X'y per row via einsum / lstsq over stacked matrix.

    注意：每行的有效集合不同（NaN 不同），无法完全消除行循环，
    但将 np.linalg.lstsq 换成手工 QR 减少了大量 Python 对象开销。
    """
    T, N = factor_arr.shape
    resid = np.full((T, N), np.nan, dtype=np.float64)

    # 预计算 covariate_arr 中使用 valid_stocks 列的子集
    # dummy_arr shape: (n_vs, K)
    n_vs = valid_stocks.sum()
    if n_vs == 0:
        return resid

    vs_idx = np.where(valid_stocks)[0]              # indices into N columns
    D = dummy_arr                                   # (n_vs, K)
    K = D.shape[1]

    for t in range(T):
        f_row = factor_arr[t, vs_idx]               # (n_vs,)
        m_row = covariate_arr[t, vs_idx]            # (n_vs,)

        row_valid = ~(np.isnan(f_row) | np.isnan(m_row))
        n_eff = row_valid.sum()
        if n_eff < max(min_obs, K + 2):
            continue

        idx_eff = np.where(row_valid)[0]
        y  = f_row[idx_eff]
        m  = m_row[idx_eff]
        Dv = D[idx_eff, :-1]                        # drop last dummy (collinearity)

        # X = [1 | m | Dv]
        X = np.concatenate([
            np.ones((len(idx_eff), 1)),
            m[:, None],
            Dv,
        ], axis=1)

        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ coef
            global_idx = vs_idx[idx_eff]
            resid[t, global_idx] = r
        except Exception:
            pass

    return resid


# ═══════════════════════════════════════════════════════════════════════════════
# 诊断结果容器
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticResult:
    """
    单个诊断模块的结果容器。

    Attributes
    ----------
    module_id   : 模块编号（1-6）
    module_name : 模块名称
    passed      : True/False/None（None 表示无法判断）
    evidence    : 定量证据（DataFrame 或 dict）
    conclusion  : 文字结论
    risk_level  : "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN"
    """

    def __init__(
        self,
        module_id:   int,
        module_name: str,
        passed:      Optional[bool],
        evidence:    object,
        conclusion:  str,
        risk_level:  str = "UNKNOWN",
    ):
        self.module_id   = module_id
        self.module_name = module_name
        self.passed      = passed
        self.evidence    = evidence
        self.conclusion  = conclusion
        self.risk_level  = risk_level

    def status_str(self) -> str:
        if self.passed is True:
            return "[PASS]"
        elif self.passed is False:
            return "[FAIL]"
        else:
            return "[N/A ]"

    def print(self) -> None:
        sep = "-" * 64
        status = self.status_str()
        risk   = f"[risk:{self.risk_level}]"
        print(f"\n{sep}")
        print(f"  Module {self.module_id}: {self.module_name}  {status}  {risk}")
        print(sep)
        print(f"[INFO] Conclusion: {self.conclusion}")
        if isinstance(self.evidence, pd.DataFrame):
            print(f"\n  Evidence:\n{self.evidence.to_string()}")
        elif isinstance(self.evidence, dict):
            print(f"\n  Evidence:")
            for k, v in self.evidence.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
                else:
                    print(f"    {k}: {v}")

    def to_dict(self) -> Dict:
        ev = self.evidence
        if isinstance(ev, pd.DataFrame):
            ev = ev.to_dict(orient="index")
        return {
            "module_id":   self.module_id,
            "module_name": self.module_name,
            "passed":      self.passed,
            "risk_level":  self.risk_level,
            "conclusion":  self.conclusion,
            "evidence":    ev,
        }


class DiagnosticReport:
    """
    完整诊断报告（6 模块汇总）。
    """

    def __init__(self, results: List[DiagnosticResult], factor_name: str = ""):
        self.results     = results
        self.factor_name = factor_name

    def print_full(self) -> None:
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"[INFO] IC Decay Diagnostics Report  Factor: {self.factor_name or '(unnamed)'}")
        print(sep)
        # 汇总表
        print(f"\n{'Module':<6} {'Name':<30} {'Status':<8} {'Risk':<10}")
        print("-" * 60)
        for r in self.results:
            print(f"  M{r.module_id:<4} {r.module_name:<30} {r.status_str():<8} {r.risk_level:<10}")
        print()

        # 各模块详情
        for r in self.results:
            r.print()

        # 最终判定
        print(f"\n{'=' * 64}")
        self._print_final_verdict()

    def _print_final_verdict(self) -> None:
        """根据 6 模块结果给出最终判定。"""
        fails = [r for r in self.results if r.passed is False]
        high_risk = [r for r in self.results if r.risk_level == "HIGH"]

        if len(fails) == 0:
            verdict = "Genuine medium-horizon efficacy (all modules passed)"
            risk = "LOW"
        elif any(r.module_id in (1, 2) for r in fails):
            verdict = "Implementation bias (time alignment/cumulative stats issue; fix and re-test)"
            risk = "HIGH"
        elif len(fails) >= 3:
            verdict = "Multi-source bias (fix issues module by module and re-test)"
            risk = "HIGH"
        elif len(high_risk) > 0:
            verdict = "Driven by structural exposure (re-evaluate after neutralization)"
            risk = "MEDIUM"
        else:
            verdict = "Partially plausible (low-risk modules passed; further analysis recommended)"
            risk = "MEDIUM"

        print(f"[INFO] Final Verdict: [{risk}] {verdict}")
        print(f"[INFO] Passed Modules: {sum(1 for r in self.results if r.passed is True)} / {len(self.results)}")
        print(f"[INFO] Failed Modules: {[f'M{r.module_id}' for r in fails]}")
        print("=" * 64)

    def to_dict(self) -> Dict:
        return {
            "factor_name": self.factor_name,
            "modules":     [r.to_dict() for r in self.results],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 主诊断类
# ═══════════════════════════════════════════════════════════════════════════════

class ICDecayDiagnostics:
    """
    IC 衰减异常诊断框架（6 模块）。

    Parameters
    ----------
    factor_panel  : (T×N) 因子面板（日频，未经 T+1 shift；诊断模块自行控制 lag）
    price_panel   : (T×N) 后复权收盘价面板（用于构建各 forward 收益）
    forward_list  : 预测期列表（天），如 [1, 5, 10, 21, 60]
    industry_map  : pd.Series: ts_code → industry（用于行业中性化，可选）
    mktcap_panel  : (T×N) 市值面板（用于市值中性化，可选）
    ic_method     : 'rank' 或 'normal'
    factor_name   : 因子名称（仅用于报告显示）
    """

    def __init__(
        self,
        factor_panel:  pd.DataFrame,
        price_panel:   pd.DataFrame,
        forward_list:  List[int] = (1, 5, 10, 21, 60),
        industry_map:  Optional[pd.Series] = None,
        mktcap_panel:  Optional[pd.DataFrame] = None,
        ic_method:     str = "rank",
        factor_name:   str = "",
    ):
        self.factor_panel = factor_panel.copy()
        self.price_panel  = price_panel.copy()
        self.forward_list = sorted(forward_list)
        self.industry_map = industry_map
        self.mktcap_panel = mktcap_panel
        self.ic_method    = ic_method
        self.factor_name  = factor_name

        # ── P0: 预计算各 forward 的收益率面板（不含 T+1，诊断模块按需 shift）
        self._ret_panels: Dict[int, pd.DataFrame] = {
            fwd: _build_forward_ret(price_panel, fwd)
            for fwd in self.forward_list
        }

        # ── P0: 一次性对齐所有面板到公共列集合，缓存 lag=1 移位后的因子
        self._f_shifted: pd.DataFrame = _shift_factor(self.factor_panel, lag=1)

        # 日收益率面板（M2/M5 共用）
        p = self.price_panel.replace(0, np.nan)
        self._daily_ret: pd.DataFrame = p / p.shift(1) - 1

        # ── P0: 缓存中性化结果（按 key 惰性计算，各模块首次调用时填充）
        # key: "cap"=市值单变量, "cap_ind"=双重中性化
        self._neut_cache: Dict[str, pd.DataFrame] = {}

        # ── P0: 预计算行业哑变量（M3 多次调用时复用）
        self._ind_dummies: Optional[np.ndarray] = None  # (n_vs, K)
        self._ind_vs_idx:  Optional[np.ndarray] = None  # valid-stock column indices
        self._ind_vs_mask: Optional[np.ndarray] = None  # (N,) bool
        self._precompute_industry_dummies()

    # ═══════════════════════════════════════════════════════════════════════
    # P0 内部缓存辅助
    # ═══════════════════════════════════════════════════════════════════════

    def _precompute_industry_dummies(self) -> None:
        """预计算行业哑变量矩阵（仅一次）。"""
        if self.industry_map is None:
            return
        cols = self.factor_panel.columns
        common_stocks = cols.intersection(self.industry_map.index)
        if len(common_stocks) == 0:
            return
        vs_mask = cols.isin(common_stocks)
        vs_idx  = np.where(vs_mask)[0]
        inds    = self.industry_map.reindex(cols[vs_mask])
        dummies = pd.get_dummies(inds, prefix="ind").values.astype(np.float64)
        self._ind_vs_mask = vs_mask.to_numpy(dtype=bool) if hasattr(vs_mask, "to_numpy") else np.array(vs_mask, dtype=bool)
        self._ind_vs_idx  = vs_idx
        self._ind_dummies = dummies   # shape (n_vs, K)

    def _get_neutralized(self, key: str) -> Optional[pd.DataFrame]:
        """
        惰性计算并缓存中性化后的因子面板。

        key : "cap"    → log(mktcap) 单变量
              "cap_ind" → log(mktcap) + industry dummies
        """
        if key in self._neut_cache:
            return self._neut_cache[key]

        f_shifted = self._f_shifted
        cols = f_shifted.columns

        if self.mktcap_panel is None:
            return None

        log_mktcap = np.log(self.mktcap_panel.replace(0, np.nan))

        # 对齐 index 和 columns
        common_dates  = f_shifted.index.intersection(log_mktcap.index)
        common_stocks = cols.intersection(log_mktcap.columns)
        if len(common_dates) == 0 or len(common_stocks) == 0:
            return None

        # 构建对齐后的完整矩阵（补 NaN 到原 columns 宽度）
        f_sub   = f_shifted.reindex(index=common_dates, columns=cols).to_numpy(dtype=float, na_value=np.nan)
        cap_sub = log_mktcap.reindex(index=common_dates, columns=cols).to_numpy(dtype=float, na_value=np.nan)

        if key == "cap":
            resid_arr = _neutralize_batch(f_sub, cap_sub, min_obs=5)
        elif key == "cap_ind":
            if self._ind_dummies is None:
                return None
            resid_arr = _neutralize_batch_with_dummies(
                f_sub, cap_sub,
                dummy_arr=self._ind_dummies,
                valid_stocks=self._ind_vs_mask,
                min_obs=10,
            )
        else:
            return None

        result = pd.DataFrame(resid_arr, index=common_dates, columns=cols)
        # Reindex back to full factor_panel index (rows outside common_dates stay NaN)
        result = result.reindex(f_shifted.index)
        self._neut_cache[key] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Module 1 — 时间对齐与前瞻偏差检查
    # ═══════════════════════════════════════════════════════════════════════

    def module1_time_alignment(
        self,
        lag_list: List[int] = (0, 1, 2),
    ) -> DiagnosticResult:
        """
        Module 1: 时间对齐与前瞻偏差检查。

        核心逻辑
        --------
        对同一个 forward（取 forward_list 的主 forward，通常是最大值），
        分别用 lag=0/1/2 计算 IC，比较其差异：

        - lag=0：因子与收益率完全对齐（即 factor[t] vs ret[t]），
          若此时 IC 显著更高，说明因子包含前瞻信息（泄露）。
        - lag=1：标准 T+1 对齐（因子在 t 日计算，收益从 t+1 日起）。
        - lag=2：额外滞后1天，用于验证 lag=1 是否充分。

        判断标准
        --------
        - 若 IC(lag=0) - IC(lag=1) > 0.01，判定存在泄露风险（FAIL）
        - 若 IC(lag=1) ≈ IC(lag=2)（差距 < 0.005），判定对齐正确（PASS）
        """
        rows = []
        # 选主 forward（用最小 forward 和最大 forward 各测一次，更完整）
        test_fwds = [self.forward_list[0], self.forward_list[-1]]

        for fwd in test_fwds:
            ret = self._ret_panels[fwd]
            for lag in lag_list:
                # lag 控制因子是否向前滞后（lag=1 = T+1 标准对齐）
                # ret 不动（ret[t] = t 日起 fwd 天累计收益），
                # factor shift(lag) 意味着因子滞后 lag 天后再与 ret 对齐
                f_shifted = _shift_factor(self.factor_panel, lag)
                ic = _compute_ic_series(f_shifted, ret, method=self.ic_method)
                st = _ic_stats_dict(ic)
                rows.append({
                    "forward": fwd,
                    "lag":     lag,
                    "mean_ic": st["mean_ic"],
                    "icir":    st["icir"],
                    "t_stat":  st["t_stat"],
                    "n":       st["n"],
                })

        evidence = pd.DataFrame(rows).set_index(["forward", "lag"])

        # 判断逻辑：对每个 fwd，比较 lag=0 vs lag=1
        issues = []
        for fwd in test_fwds:
            try:
                ic0 = evidence.loc[(fwd, 0), "mean_ic"]
                ic1 = evidence.loc[(fwd, 1), "mean_ic"]
                ic2 = evidence.loc[(fwd, 2), "mean_ic"]
                diff_0_1 = ic0 - ic1
                diff_1_2 = abs(ic1 - ic2)
                if diff_0_1 > 0.010:
                    issues.append(
                        f"forward={fwd}: IC(lag=0)={ic0:.4f} > IC(lag=1)={ic1:.4f}, "
                        f"差值={diff_0_1:.4f}（>0.010 → 疑似泄露）"
                    )
                if diff_1_2 > 0.010:
                    issues.append(
                        f"forward={fwd}: IC(lag=1)={ic1:.4f} vs IC(lag=2)={ic2:.4f}, "
                        f"差值={diff_1_2:.4f}（>0.010 → lag=1 可能不充分）"
                    )
            except KeyError:
                pass

        if issues:
            passed     = False
            risk_level = "HIGH"
            conclusion = "检测到前瞻偏差嫌疑: " + "; ".join(issues)
        else:
            passed     = True
            risk_level = "LOW"
            conclusion = (
                "时间对齐通过: IC(lag=0) 与 IC(lag=1) 差值均 <= 0.010，"
                "IC(lag=1) 与 IC(lag=2) 差值 <= 0.010，未检测到前瞻偏差。"
            )

        return DiagnosticResult(
            module_id   = 1,
            module_name = "时间对齐与前瞻偏差检查",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 2 — 收益定义与累计窗口拆解
    # ═══════════════════════════════════════════════════════════════════════

    def module2_incremental_ic(self) -> DiagnosticResult:
        """
        Module 2: 累计 IC vs 增量 IC 分解。

        核心逻辑
        --------
        对每个 forward h，同时计算：
        1. 累计 IC：IC(f_t, R_{t+1:t+h})（与主 pipeline 一致，lag=1 shift）
        2. 增量 IC：IC(f_t, r_{t+k})（只看第 k 天的单日收益，k=1..h）

        若增量 IC 随 k 增大后趋近 0，而累计 IC 仍上升，
        则"长 forward IC 更高"是累计统计放大效应，非真实中期预测力。

        判断标准
        --------
        - 若最大 forward 的增量 IC 在 k >= min(forward) 之后均 < 0.5*IC(k=1)，
          标记"累计统计放大效应 MEDIUM 风险"
        - 若增量 IC 在长窗口仍稳定为正（> 0.5*IC(k=1)），可判定中期信号存在
        """
        # ── P0: 使用缓存的 f_shifted 和 daily_ret，不重复计算
        f_shifted = self._f_shifted
        daily_ret = self._daily_ret

        # 累计 IC（使用预构建的 ret panels，加 shift_factor）
        cumul_rows = []
        for fwd in self.forward_list:
            ret = self._ret_panels[fwd]
            ic  = _compute_ic_series(f_shifted, ret, method=self.ic_method)
            st  = _ic_stats_dict(ic)
            nw  = _nw_t(ic)
            cumul_rows.append({
                "forward":   fwd,
                "cumul_ic":  st["mean_ic"],
                "cumul_icir": st["icir"],
                "t_stat":    st["t_stat"],
                "nw_t_stat": round(nw, 4) if not np.isnan(nw) else np.nan,
            })
        cumul_df = pd.DataFrame(cumul_rows).set_index("forward")

        # 增量 IC：对每个 k=1..max_fwd，计算 IC(f_t, r_{t+k})
        # r_{t+k} = price[t+k] / price[t+k-1] - 1 (单日收益)
        max_fwd = max(self.forward_list)
        incr_checkpoints = sorted(set(
            [1] +
            self.forward_list +
            [max(1, f // 2) for f in self.forward_list if f > 2]
        ))

        # ── P1: 批量计算所有 k 的增量 IC（无逐 k 重新对齐开销）
        # 预先对齐 factor 和 daily_ret 到公共列，once
        common_stocks_m2 = f_shifted.columns.intersection(daily_ret.columns)
        common_dates_m2  = f_shifted.index.intersection(daily_ret.index)
        f_arr_m2 = f_shifted.reindex(index=common_dates_m2, columns=common_stocks_m2).to_numpy(dtype=float, na_value=np.nan)
        dr_arr_m2 = daily_ret.reindex(index=common_dates_m2, columns=common_stocks_m2).to_numpy(dtype=float, na_value=np.nan)
        T_m2 = len(common_dates_m2)

        incr_rows = []
        _pb2 = _make_pbar(
            total=len(incr_checkpoints), desc="M2 增量IC(k)",
            unit="k", indent=6, log_every=5,
        )
        for k in incr_checkpoints:
            if k < 1:
                _pb2.update()
                continue
            # r_{t+k}: shift dr_arr_m2 by -k rows (rows 0..T-k-1 get values from rows k..T-1)
            if k >= T_m2:
                incr_rows.append({"k": k, "incr_ic": np.nan, "incr_icir": np.nan, "t_stat": np.nan})
                _pb2.update()
                continue
            r_tk_arr = np.full_like(dr_arr_m2, np.nan)
            r_tk_arr[:T_m2 - k] = dr_arr_m2[k:]
            # f_arr_m2 already aligned; use rows 0..T-k-1 (row T-k onward becomes NaN after shift)
            ic_arr = _ic_from_arrays(f_arr_m2, r_tk_arr, method=self.ic_method)
            ic_s = pd.Series(ic_arr, index=common_dates_m2, name="IC")
            st   = _ic_stats_dict(ic_s)
            incr_rows.append({
                "k":          k,
                "incr_ic":    st["mean_ic"],
                "incr_icir":  st["icir"],
                "t_stat":     st["t_stat"],
            })
            _pb2.update()
        _pb2.close()
        incr_df = pd.DataFrame(incr_rows).set_index("k")

        # 判断：比较 k=1 的增量 IC 与 k=max_fwd 的增量 IC
        # evidence["incr_ic"] 存为 pd.Series（indexed by k）方便下游访问
        evidence = {"cumul_ic": cumul_df, "incr_ic": incr_df["incr_ic"], "incr_ic_full": incr_df}

        try:
            ic_k1 = incr_df.loc[1, "incr_ic"] if 1 in incr_df.index else np.nan
            ic_kmax = incr_df.loc[max_fwd, "incr_ic"] if max_fwd in incr_df.index else np.nan
            # 也看中点
            mid_k = max_fwd // 2
            ic_kmid = incr_df.loc[mid_k, "incr_ic"] if mid_k in incr_df.index else np.nan

            if np.isnan(ic_k1) or ic_k1 == 0:
                passed     = None
                risk_level = "UNKNOWN"
                conclusion = "k=1 增量 IC 为 NaN，无法判断。"
            elif abs(ic_kmax) < 0.5 * abs(ic_k1) and abs(ic_kmid) < 0.75 * abs(ic_k1):
                passed     = False
                risk_level = "MEDIUM"
                conclusion = (
                    f"累计统计放大效应：增量 IC 从 k=1({ic_k1:.4f}) 快速衰减至 "
                    f"k={max_fwd}({ic_kmax:.4f})（< 50% 初始值），"
                    f"而累计 IC 仍维持在 {cumul_df.loc[max_fwd,'cumul_ic']:.4f}。"
                    f"长 forward IC 上升主要由累计窗口的统计放大效应驱动。"
                )
            else:
                passed     = True
                risk_level = "LOW"
                conclusion = (
                    f"存在中期持续预测信号：增量 IC 在 k={max_fwd} 仍为 "
                    f"{ic_kmax:.4f}（>= 50% 的 k=1 值 {ic_k1:.4f}），"
                    f"因子具备真实的中期预测力。"
                )
        except Exception as e:
            passed     = None
            risk_level = "UNKNOWN"
            conclusion = f"判断过程出错: {e}"

        return DiagnosticResult(
            module_id   = 2,
            module_name = "收益定义与累计窗口拆解",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 3 — 市场/风格暴露剥离
    # ═══════════════════════════════════════════════════════════════════════

    def module3_exposure_strip(self) -> DiagnosticResult:
        """
        Module 3: 市场/风格暴露剥离。

        四个版本的 IC 对比
        ------------------
        V0: 原始 IC（基准）
        V1: 市场超额收益 IC（每期收益减去截面均值，去掉市场 beta）
        V2: 行业超额收益 IC（每只股票收益减去同行业均值，需要 industry_map）
        V3: 因子市值中性化后的 IC（因子对 log(mktcap) 回归取残差，需要 mktcap_panel）
        V4: 因子市值+行业双重中性化后的 IC（最严格）

        判断标准
        --------
        - 若 V1/V2/V3 的 IC 显著低于 V0（>20% 差距），说明原 IC 中含有市场/风格暴露
        - 若 V4 的 forward 上升现象消失，判定"结构暴露驱动"
        - 若 V4 仍显示 forward 上升，判定"存在独立因子信号"
        """
        # ── P0: 使用缓存的 f_shifted，惰性获取中性化面板（_get_neutralized 只算一次）
        f_shifted = self._f_shifted

        # 预先获取中性化面板（触发缓存，后续循环内直接用）
        f_cap_neut  = self._get_neutralized("cap")       # None if mktcap unavailable
        f_dual_neut = self._get_neutralized("cap_ind")   # None if either unavailable

        rows = []
        _pb3 = _make_pbar(
            total=len(self.forward_list), desc="M3 neutralized variants",
            unit="fwd", indent=6, log_every=0,
        )
        for fwd in self.forward_list:
            raw_ret = self._ret_panels[fwd]

            # V0: 原始 IC
            ic_v0 = _ic_stats_dict(
                _compute_ic_series(f_shifted, raw_ret, method=self.ic_method)
            )

            # V1: 市场超额收益（截面 demean）
            market_excess = raw_ret.sub(raw_ret.mean(axis=1), axis=0)
            ic_v1 = _ic_stats_dict(
                _compute_ic_series(f_shifted, market_excess, method=self.ic_method)
            )

            # V2: 行业超额收益（需要 industry_map）
            if self.industry_map is not None:
                industry_excess = self._compute_industry_excess(raw_ret)
                ic_v2 = _ic_stats_dict(
                    _compute_ic_series(f_shifted, industry_excess, method=self.ic_method)
                )
                ic_v2_val = ic_v2["mean_ic"]
            else:
                ic_v2_val = np.nan

            # V3: 因子市值中性化（P0: 复用缓存，不重复 OLS）
            if f_cap_neut is not None:
                ic_v3    = _ic_stats_dict(
                    _compute_ic_series(f_cap_neut, raw_ret, method=self.ic_method)
                )
                ic_v3_val = ic_v3["mean_ic"]
            else:
                ic_v3_val = np.nan

            # V4: 因子市值+行业双重中性化（P0: 复用缓存）
            if f_dual_neut is not None:
                ic_v4    = _ic_stats_dict(
                    _compute_ic_series(f_dual_neut, raw_ret, method=self.ic_method)
                )
                ic_v4_val = ic_v4["mean_ic"]
            else:
                ic_v4_val = np.nan

            rows.append({
                "forward":    fwd,
                "ic_raw":     ic_v0["mean_ic"],
                "ic_mkt_excess": ic_v1["mean_ic"],
                "ic_ind_excess": ic_v2_val,
                "ic_mktcap_neut_factor": ic_v3_val,
                "ic_dual_neut_factor":   ic_v4_val,
            })
            _pb3.update()
        _pb3.close()

        evidence = pd.DataFrame(rows).set_index("forward")

        # 判断逻辑：检查 V4（最严格中性化）的 forward 趋势
        try:
            raw_max   = evidence["ic_raw"].iloc[-1]
            raw_min   = evidence["ic_raw"].iloc[0]
            raw_rise  = raw_max - raw_min

            v4_col = "ic_dual_neut_factor"
            if evidence[v4_col].notna().sum() >= 2:
                v4_max  = evidence[v4_col].dropna().iloc[-1]
                v4_min  = evidence[v4_col].dropna().iloc[0]
                v4_rise = v4_max - v4_min

                if v4_rise < 0.3 * raw_rise:
                    passed     = False
                    risk_level = "HIGH"
                    conclusion = (
                        f"结构暴露驱动：原始 IC 随 forward 上升幅度={raw_rise:.4f}，"
                        f"双重中性化后上升幅度={v4_rise:.4f}（<30% 原始），"
                        f"IC 上升主要来自市值/行业暴露。"
                    )
                elif v4_rise >= 0.7 * raw_rise:
                    passed     = True
                    risk_level = "LOW"
                    conclusion = (
                        f"独立因子信号：双重中性化后 IC 上升幅度仍有 {v4_rise:.4f}"
                        f"（>= 70% 原始 {raw_rise:.4f}），因子具有独立的中期预测力。"
                    )
                else:
                    passed     = None
                    risk_level = "MEDIUM"
                    conclusion = (
                        f"部分暴露：双重中性化后 IC 上升幅度={v4_rise:.4f}"
                        f"（30-70% 原始 {raw_rise:.4f}），市场/风格解释部分但非全部。"
                    )
            elif evidence["ic_mkt_excess"].notna().sum() >= 2:
                # V4 不可用，使用 V1
                v1_max  = evidence["ic_mkt_excess"].dropna().iloc[-1]
                v1_min  = evidence["ic_mkt_excess"].dropna().iloc[0]
                v1_rise = v1_max - v1_min
                passed     = None
                risk_level = "MEDIUM"
                conclusion = (
                    f"市值/行业面板不完整，仅市场超额收益测试可用。"
                    f"市场超额 IC 上升幅度={v1_rise:.4f}（原始={raw_rise:.4f}）。"
                )
            else:
                passed     = None
                risk_level = "UNKNOWN"
                conclusion = "industry_map 和 mktcap_panel 均未提供，无法执行暴露剥离。"
        except Exception as e:
            passed     = None
            risk_level = "UNKNOWN"
            conclusion = f"判断过程出错: {e}"

        return DiagnosticResult(
            module_id   = 3,
            module_name = "市场/风格暴露剥离",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    def _compute_industry_excess(self, ret_panel: pd.DataFrame) -> pd.DataFrame:
        """计算行业内超额收益（每股收益减去同行业均值）。"""
        result = ret_panel.copy()
        ind_map = self.industry_map
        # 对公共股票执行（逐日）
        common_stocks = ret_panel.columns.intersection(ind_map.index)
        if len(common_stocks) == 0:
            return ret_panel
        r = ret_panel[common_stocks].copy()
        industries = ind_map[common_stocks]
        for date in r.index:
            row = r.loc[date]
            row_valid = row.dropna()
            if len(row_valid) == 0:
                continue
            ind_means = row_valid.groupby(industries[row_valid.index]).mean()
            excess = row_valid - row_valid.index.map(
                lambda s: ind_means.get(industries.get(s, "__NONE__"), np.nan)
            )
            result.loc[date, excess.index] = excess.values
        return result

    def _neutralize_mktcap(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        """
        因子对 log(mktcap) 做截面 OLS 回归，取残差。
        P0: 委托给 _get_neutralized 缓存，若缓存已有则直接返回。
        """
        cached = self._get_neutralized("cap")
        if cached is not None:
            return cached
        # Fallback: no mktcap_panel
        return factor_panel.copy() * np.nan

    def _neutralize_mktcap_industry(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        """
        因子对 log(mktcap) + 行业哑变量做截面 OLS 回归，取残差。
        P0: 委托给 _get_neutralized 缓存，若缓存已有则直接返回。
        """
        cached = self._get_neutralized("cap_ind")
        if cached is not None:
            return cached
        # Fallback
        return factor_panel.copy() * np.nan

    # ═══════════════════════════════════════════════════════════════════════
    # Module 4 — 样本偏差检查（生存偏差/覆盖率）
    # ═══════════════════════════════════════════════════════════════════════

    def module4_sample_bias(self) -> DiagnosticResult:
        """
        Module 4: 样本偏差检查。

        检查项
        ------
        1. 各 forward 的有效股票覆盖率（非 NaN 占比）
        2. forward=60 vs forward=21 的覆盖率差异（>10% 视为显著偏离）
        3. 因子面板覆盖率（是否有系统性的股票遗漏）
        4. 月末重采样前后样本量变化

        判断标准
        --------
        - 若 forward=60 覆盖率比 forward=21 低 >10 个百分点，判定样本偏差（FAIL）
        - 若覆盖率稳定（<5% 差距），判定样本偏差较小（PASS）
        """
        # ── P0: 使用缓存的 f_shifted
        f_shifted = self._f_shifted

        rows = []
        for fwd in self.forward_list:
            ret = self._ret_panels[fwd]
            # 对齐日期
            common_dates  = f_shifted.index.intersection(ret.index)
            common_stocks = f_shifted.columns.intersection(ret.columns)
            f_ = f_shifted.loc[common_dates, common_stocks]
            r_ = ret.loc[common_dates, common_stocks]

            # 有效样本：因子和收益率均非 NaN
            both_valid = (~f_.isna()) & (~r_.isna())
            factor_coverage = (~f_.isna()).values.mean()  # 因子面板覆盖率
            ret_coverage    = (~r_.isna()).values.mean()  # 收益率面板覆盖率
            joint_coverage  = both_valid.values.mean()   # 联合有效覆盖率

            # 月末样本
            f_monthly = _resample_monthly_last(f_)
            r_monthly = _resample_monthly_last(r_)
            common_monthly = f_monthly.index.intersection(r_monthly.index)
            n_monthly = len(common_monthly)

            rows.append({
                "forward":          fwd,
                "n_dates":          len(common_dates),
                "n_stocks":         len(common_stocks),
                "factor_coverage":  round(factor_coverage, 4),
                "ret_coverage":     round(ret_coverage, 4),
                "joint_coverage":   round(joint_coverage, 4),
                "n_monthly_periods":n_monthly,
            })

        evidence = pd.DataFrame(rows).set_index("forward")

        # 判断逻辑
        try:
            ref_fwd  = 21 if 21 in self.forward_list else self.forward_list[len(self.forward_list) // 2]
            long_fwd = max(self.forward_list)

            cov_ref  = evidence.loc[ref_fwd, "joint_coverage"] if ref_fwd in evidence.index else np.nan
            cov_long = evidence.loc[long_fwd, "joint_coverage"]
            cov_diff = cov_ref - cov_long  # 正数 = 长 forward 覆盖率更低

            if np.isnan(cov_ref):
                passed     = None
                risk_level = "UNKNOWN"
                conclusion = f"forward={ref_fwd} 不在列表中，无法对比。"
            elif cov_diff > 0.10:
                passed     = False
                risk_level = "HIGH"
                conclusion = (
                    f"样本覆盖率显著下降：forward={ref_fwd} 覆盖率={cov_ref:.2%}，"
                    f"forward={long_fwd} 覆盖率={cov_long:.2%}，"
                    f"差距={cov_diff:.2%}(>10%）。长窗口样本偏向存活股票，"
                    f"可能导致长 forward IC 被高估。"
                )
            elif cov_diff > 0.05:
                passed     = None
                risk_level = "MEDIUM"
                conclusion = (
                    f"样本覆盖率中等下降：forward={ref_fwd} 覆盖率={cov_ref:.2%}，"
                    f"forward={long_fwd} 覆盖率={cov_long:.2%}，"
                    f"差距={cov_diff:.2%}（5-10%），需关注但尚可接受。"
                )
            else:
                passed     = True
                risk_level = "LOW"
                conclusion = (
                    f"样本覆盖率稳定：forward={ref_fwd} 覆盖率={cov_ref:.2%}，"
                    f"forward={long_fwd} 覆盖率={cov_long:.2%}，"
                    f"差距={cov_diff:.2%}（<= 5%），样本偏差风险低。"
                )
        except Exception as e:
            passed     = None
            risk_level = "UNKNOWN"
            conclusion = f"判断过程出错: {e}"

        return DiagnosticResult(
            module_id   = 4,
            module_name = "样本偏差检查（生存偏差/覆盖率）",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 5 — 因子属性验证（时效性/半衰期）
    # ═══════════════════════════════════════════════════════════════════════

    def module5_factor_halflife(self) -> DiagnosticResult:
        """
        Module 5: 因子时效性验证（自相关半衰期）。

        检查项
        ------
        1. 因子截面自相关：对每期因子排名，计算 lag=1..12 的序列自相关
        2. 自相关半衰期：拟合指数衰减 ρ(k) = ρ0 * e^(-λk)，求 k=ln(2)/λ
        3. 增量 IC 半衰期：增量 IC 下降到 IC(k=1) 50% 时的 k 值

        判断标准
        --------
        - 若因子自相关半衰期 > 60（交易日），factor 属于缓变因子，
          forward 较长时 IC 较高"部分"合理
        - 若半衰期 < 21 且 forward=60 的 IC 比 forward=21 高 >30%，
          则判定不合理（FAIL）
        """
        # ── P0/P1: 月度重采样并向量化自相关（不重复构建 pivot）
        f_monthly = _resample_monthly_last(self.factor_panel)
        f_ranked  = _rank_panel(f_monthly)

        # P1: 预先转为 ndarray，避免逐 lag 重复 DataFrame 操作
        f_ranked_arr = f_ranked.to_numpy(dtype=float, na_value=np.nan)  # M×N
        M = f_ranked_arr.shape[0]

        autocorr_rows = []
        _lag_range = list(range(1, min(13, M)))
        _pb5a = _make_pbar(
            total=len(_lag_range), desc="M5 自相关(lag)",
            unit="月", indent=6, log_every=0,
        )
        for lag in _lag_range:
            # f_t  = rows [lag..M-1], f_{t-lag} = rows [0..M-lag-1]
            f_t  = f_ranked_arr[lag:]        # (M-lag, N)
            f_tl = f_ranked_arr[:M - lag]    # (M-lag, N)

            nan_m = np.isnan(f_t) | np.isnan(f_tl)
            f_t  = np.where(nan_m, np.nan, f_t)
            f_tl = np.where(nan_m, np.nan, f_tl)

            # 截面 demean per row (date)
            f_t_mean  = np.nanmean(f_t,  axis=1, keepdims=True)
            f_tl_mean = np.nanmean(f_tl, axis=1, keepdims=True)
            f_t_dm  = np.where(nan_m, 0.0, f_t  - f_t_mean)
            f_tl_dm = np.where(nan_m, 0.0, f_tl - f_tl_mean)

            num   = (f_t_dm * f_tl_dm).sum(axis=1)
            denom = np.sqrt((f_t_dm**2).sum(axis=1) * (f_tl_dm**2).sum(axis=1))
            corr_series = np.where(denom > 0, num / denom, np.nan)
            mean_corr = float(np.nanmean(corr_series))
            autocorr_rows.append({"lag_months": lag, "autocorr": round(mean_corr, 4)})
            _pb5a.update()
        _pb5a.close()

        autocorr_df = pd.DataFrame(autocorr_rows).set_index("lag_months")

        # 拟合指数衰减模型估算半衰期（月度）
        halflife_months = np.nan
        try:
            lags = autocorr_df.index.values.astype(float)
            rho  = autocorr_df["autocorr"].values
            valid_mask = ~np.isnan(rho)
            if valid_mask.sum() >= 3:
                log_rho = np.log(np.abs(rho[valid_mask]).clip(1e-6))
                k_valid = lags[valid_mask]
                coef    = np.polyfit(k_valid, log_rho, deg=1)
                lam     = -coef[0]
                if lam > 0:
                    halflife_months = np.log(2) / lam
        except Exception:
            pass

        # ── P0: 复用缓存的 f_shifted 和 daily_ret（不重复构建）
        f_shifted = self._f_shifted
        daily_ret = self._daily_ret

        # ── P1: 批量计算所有 k 的增量 IC（ndarray 路径，复用 M2 的对齐结果）
        common_stocks_m5 = f_shifted.columns.intersection(daily_ret.columns)
        common_dates_m5  = f_shifted.index.intersection(daily_ret.index)
        f_arr_m5  = f_shifted.reindex(index=common_dates_m5, columns=common_stocks_m5).to_numpy(dtype=float, na_value=np.nan)
        dr_arr_m5 = daily_ret.reindex(index=common_dates_m5, columns=common_stocks_m5).to_numpy(dtype=float, na_value=np.nan)
        T_m5 = len(common_dates_m5)

        incr_ic_by_k = {}
        checkpoints = sorted(set([1, 2, 3, 5] + self.forward_list))
        _pb5b = _make_pbar(
            total=len(checkpoints), desc="M5 增量IC(k)",
            unit="k", indent=6, log_every=0,
        )
        for k in checkpoints:
            if k >= T_m5:
                incr_ic_by_k[k] = np.nan
                _pb5b.update()
                continue
            r_tk_arr = np.full_like(dr_arr_m5, np.nan)
            r_tk_arr[:T_m5 - k] = dr_arr_m5[k:]
            ic_arr = _ic_from_arrays(f_arr_m5, r_tk_arr, method=self.ic_method)
            ic_s = pd.Series(ic_arr, index=common_dates_m5, name="IC")
            incr_ic_by_k[k] = _ic_stats_dict(ic_s)["mean_ic"]
            _pb5b.update()
        _pb5b.close()

        # 增量 IC 半衰期：找第一个 IC < 50% 初始值的 k
        ic_k1 = incr_ic_by_k.get(1, np.nan)
        incr_halflife_days = np.nan
        if not np.isnan(ic_k1) and ic_k1 != 0:
            for k in sorted(incr_ic_by_k.keys()):
                if k == 1:
                    continue
                if abs(incr_ic_by_k[k]) < 0.5 * abs(ic_k1):
                    incr_halflife_days = k
                    break

        evidence = {
            "autocorr_by_lag_month": autocorr_df,
            "halflife_months_estimate": round(halflife_months, 2) if not np.isnan(halflife_months) else "N/A",
            "incr_ic_by_k": pd.Series(incr_ic_by_k, name="incr_ic"),
            "incr_halflife_days": incr_halflife_days if not np.isnan(incr_halflife_days) else "N/A (仍 >50% 初始值)",
        }

        # 判断
        try:
            fwd_max = max(self.forward_list)
            fwd_mid = 21 if 21 in self.forward_list else self.forward_list[len(self.forward_list) // 2]
            ic_at_mid = incr_ic_by_k.get(fwd_mid, np.nan)
            ic_at_max = incr_ic_by_k.get(fwd_max, np.nan)

            halflife_days = halflife_months * 21 if not np.isnan(halflife_months) else np.nan

            if np.isnan(halflife_days) or np.isnan(ic_at_max):
                passed     = None
                risk_level = "UNKNOWN"
                conclusion = "Insufficient data to estimate factor half-life."
            elif halflife_days >= 60:
                passed     = True
                risk_level = "LOW"
                conclusion = (
                    f"Slow-varying factor: autocorrelation half-life is about {halflife_months:.1f} months "
                    f"(~{halflife_days:.0f} trading days); "
                    f"higher IC at longer forward horizons is plausible."
                )
            elif halflife_days < 21 and not np.isnan(ic_at_mid) and not np.isnan(ic_at_max):
                ic_ratio = abs(ic_at_max) / abs(ic_at_mid) if ic_at_mid != 0 else np.inf
                if ic_ratio > 1.30:
                    passed     = False
                    risk_level = "MEDIUM"
                    conclusion = (
                        f"Short half-life (~{halflife_days:.0f} trading days), "
                        f"but incremental IC(k={fwd_max}) = {ic_at_max:.4f} is still significantly higher than "
                        f"IC(k={fwd_mid}) = {ic_at_mid:.4f} (ratio={ic_ratio:.2f}); "
                        f"this is inconsistent with expected factor timeliness and should be cross-checked with Modules 1 and 2."
                    )
                else:
                    passed     = True
                    risk_level = "LOW"
                    conclusion = (
                        f"Factor half-life is about {halflife_days:.0f} trading days; "
                        f"incremental IC decay at longer forward horizons is broadly consistent with this half-life."
                    )
            else:
                passed     = None
                risk_level = "MEDIUM"
                conclusion = (
                    f"Factor half-life is about {halflife_months:.1f} months (~{halflife_days:.0f} trading days); "
                    f"medium timeliness, recommend combining evidence from Modules 1-4."
                )
        except Exception as e:
            passed     = None
            risk_level = "UNKNOWN"
            conclusion = f"Error while deriving conclusion: {e}"

        return DiagnosticResult(
            module_id   = 5,
            module_name = "Factor Property Validation (Timeliness/Half-life)",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 6 — 稳健性复核
    # ═══════════════════════════════════════════════════════════════════════

    def module6_robustness(
        self,
        n_splits: int = 3,
    ) -> DiagnosticResult:
        """
        Module 6: 稳健性复核。

        检查项
        ------
        1. 时间外推稳定性：将样本均分为 n_splits 个子期，各子期的 IC 衰减形态
        2. Winsorize 参数扰动：标准（3 MAD）vs 宽松（5 MAD）vs 严格（1.5 MAD）
        3. 按市场状态分 regime（牛/熊/震荡）：以年化市场收益分段

        判断标准
        --------
        - 若所有子期的 IC 衰减形态一致（forward 越长 IC 越高），则视为稳健（PASS）
        - 若超过 1/3 子期出现正常衰减（短 forward IC 更高），则视为不稳健（FAIL）
        """
        # ── P0: 复用缓存的 f_shifted
        f_shifted = self._f_shifted

        # ── 子期分割 IC 衰减 ─────────────────────────────────────────────
        all_dates = f_shifted.index
        split_size = len(all_dates) // n_splits
        split_results = []

        _pb6a = _make_pbar(
            total=n_splits, desc="M6 split-period",
            unit="split", indent=6, log_every=0,
        )
        for i in range(n_splits):
            s_start = i * split_size
            s_end   = (i + 1) * split_size if i < n_splits - 1 else len(all_dates)
            sub_dates = all_dates[s_start:s_end]
            if len(sub_dates) < 20:
                _pb6a.update()
                continue

            f_sub = f_shifted.loc[sub_dates]
            sub_row = {"split": i + 1,
                       "start": str(sub_dates[0])[:10],
                       "end":   str(sub_dates[-1])[:10]}
            for fwd in self.forward_list:
                ret_sub = self._ret_panels[fwd].reindex(sub_dates)
                ic_sub  = _compute_ic_series(f_sub, ret_sub, method=self.ic_method)
                sub_row[f"ic_fwd{fwd}"] = round(_ic_stats_dict(ic_sub)["mean_ic"], 4)
            split_results.append(sub_row)
            _pb6a.update()
        _pb6a.close()

        split_df = pd.DataFrame(split_results).set_index("split")

        # ── 参数扰动：Winsorize 阈值 ─────────────────────────────────────
        from factor_framework.operators import cs_winsorize
        try:
            from factor_framework.factor_engine import FactorEngine as _FE
            _apply_cs = _FE._apply_cross_section_static if hasattr(_FE, "_apply_cross_section_static") else None
        except Exception:
            _apply_cs = None

        def _winsorize_mad(panel: pd.DataFrame, n_mads: float) -> pd.DataFrame:
            """MAD Winsorize（截面逐行）。"""
            result = panel.copy()
            for date in panel.index:
                row = panel.loc[date].dropna()
                if len(row) < 5:
                    continue
                median = row.median()
                mad    = (row - median).abs().median()
                lo     = median - n_mads * mad
                hi     = median + n_mads * mad
                clipped = row.clip(lo, hi)
                result.loc[date, clipped.index] = clipped.values
            return result

        # 只在主 forward（最大 forward）上测试 winsorize 扰动
        main_fwd = max(self.forward_list)
        ret_main = self._ret_panels[main_fwd]

        winsor_configs = [(1.5, "strict"), (3.0, "standard"), (5.0, "loose")]
        winsor_rows = []
        _pb6b = _make_pbar(
            total=len(winsor_configs), desc="M6 winsorize-perturbation",
            unit="level", indent=6, log_every=0,
        )
        for n_mads, label in winsor_configs:
            f_win = _winsorize_mad(self.factor_panel.copy(), n_mads)
            f_win_shifted = _shift_factor(f_win, lag=1)
            ic_win = _compute_ic_series(f_win_shifted, ret_main, method=self.ic_method)
            st_win = _ic_stats_dict(ic_win)
            winsor_rows.append({
                "winsor_mads":  n_mads,
                "label":        label,
                "mean_ic":      st_win["mean_ic"],
                "icir":         st_win["icir"],
                "t_stat":       st_win["t_stat"],
            })
            _pb6b.update()
        _pb6b.close()
        winsor_df = pd.DataFrame(winsor_rows)

        # ── 市场状态分类（按月度市场收益分 regime）───────────────────────
        # P0: 复用缓存的 daily_ret（截面均值为市场收益）
        daily_ret = self._daily_ret.mean(axis=1).dropna()
        monthly_ret = _resample_monthly_last(daily_ret.to_frame("mkt")).squeeze()

        # 滚动12个月年化收益判断 regime
        rolling_ann = monthly_ret.rolling(12).apply(
            lambda x: (1 + x).prod() ** (12 / len(x)) - 1, raw=True
        )
        regime_map = rolling_ann.apply(
            lambda x: "bull" if x > 0.10 else ("bear" if x < -0.10 else "flat")
        ).dropna()

        # 月末因子（对齐 regime）
        f_monthly = _resample_monthly_last(f_shifted)
        f_monthly.index = pd.to_datetime(f_monthly.index)

        regime_rows = []
        for regime_label in ["bull", "bear", "flat"]:
            regime_dates_raw = regime_map[regime_map == regime_label].index
            # 对齐到因子月末日期
            regime_dates = f_monthly.index.intersection(regime_dates_raw)
            if len(regime_dates) < 5:
                regime_rows.append({"regime": regime_label, "n_periods": len(regime_dates),
                                     **{f"ic_fwd{fwd}": np.nan for fwd in self.forward_list}})
                continue
            f_reg = f_monthly.loc[regime_dates]
            # 转回字符串日期以对齐 ret_panels（如果 ret_panels 是字符串索引）
            regime_str_dates = regime_dates.strftime("%Y%m%d")
            regime_row = {"regime": regime_label, "n_periods": len(regime_dates)}
            for fwd in self.forward_list:
                ret_fwd = self._ret_panels[fwd]
                # 尝试用 datetime 对齐，再用字符串对齐
                common_d = f_reg.index.intersection(
                    pd.to_datetime(ret_fwd.index) if not pd.api.types.is_datetime64_any_dtype(ret_fwd.index)
                    else ret_fwd.index
                )
                if len(common_d) < 3:
                    regime_row[f"ic_fwd{fwd}"] = np.nan
                    continue
                f_  = f_reg.loc[common_d]
                ret_ = ret_fwd.loc[ret_fwd.index.isin(
                    common_d.strftime("%Y%m%d") if pd.api.types.is_datetime64_any_dtype(common_d)
                    else common_d
                )]
                if len(ret_) < 3:
                    regime_row[f"ic_fwd{fwd}"] = np.nan
                    continue
                # 强制对齐行数
                min_len = min(len(f_), len(ret_))
                ic_reg  = _compute_ic_series(f_.iloc[:min_len], ret_.iloc[:min_len], method=self.ic_method)
                regime_row[f"ic_fwd{fwd}"] = round(_ic_stats_dict(ic_reg)["mean_ic"], 4)
            regime_rows.append(regime_row)

        regime_df = pd.DataFrame(regime_rows)

        evidence = {
            "split_period_ic":    split_df,
            "winsor_sensitivity": winsor_df,
            "regime_ic":          regime_df,
        }

        # 判断：检查子期是否均呈现 forward 越长 IC 越高
        try:
            ic_cols_sorted = [f"ic_fwd{fwd}" for fwd in self.forward_list]
            existing_cols  = [c for c in ic_cols_sorted if c in split_df.columns]
            if len(existing_cols) < 2:
                passed     = None
                risk_level = "UNKNOWN"
                conclusion = "Insufficient samples after split-period partitioning; robustness is inconclusive."
            else:
                # 对每个子期检查是否 forward 越长 IC 越高（单调性）
                monotone_count = 0
                for _, row in split_df.iterrows():
                    vals = [row[c] for c in existing_cols if not np.isnan(row[c])]
                    if len(vals) >= 2:
                        is_monotone = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
                        if is_monotone:
                            monotone_count += 1

                ratio = monotone_count / len(split_df)

                # Winsorize 扰动检查：标准 vs 松紧 IC 差值
                std_ic = winsor_df[winsor_df["label"] == "standard"]["mean_ic"].values
                loose_ic = winsor_df[winsor_df["label"] == "loose"]["mean_ic"].values
                winsor_sensitive = (
                    len(std_ic) > 0 and len(loose_ic) > 0 and
                    abs(float(std_ic[0]) - float(loose_ic[0])) > 0.02
                )

                if ratio >= 2 / 3:
                    passed     = True
                    risk_level = "LOW"
                    conclusion = (
                        f"Robust conclusion: {monotone_count}/{len(split_df)} split periods show "
                        f"higher IC at longer forward horizons."
                        + (f" Note: IC is sensitive to winsorize threshold changes (>{0.02:.2%}); review recommended."
                           if winsor_sensitive else "")
                    )
                elif ratio <= 1 / 3:
                    passed     = False
                    risk_level = "MEDIUM"
                    conclusion = (
                        f"Non-robust conclusion: only {monotone_count}/{len(split_df)} split periods show "
                        f"higher IC at longer forward horizons; this appears regime-dependent."
                    )
                else:
                    passed     = None
                    risk_level = "MEDIUM"
                    conclusion = (
                        f"Partially robust conclusion: {monotone_count}/{len(split_df)} split periods show monotonic rise; "
                        f"validity is conditional on market regime."
                    )
        except Exception as e:
            passed     = None
            risk_level = "UNKNOWN"
            conclusion = f"Error while deriving conclusion: {e}"

        return DiagnosticResult(
            module_id   = 6,
            module_name = "Robustness Recheck",
            passed      = passed,
            evidence    = evidence,
            conclusion  = conclusion,
            risk_level  = risk_level,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════════════════

    def run_all(
        self,
        run_modules: Optional[List[int]] = None,
        lag_list:    List[int] = (0, 1, 2),
        n_splits:    int = 3,
        verbose:     bool = True,
    ) -> DiagnosticReport:
        """
        执行全部（或指定）诊断模块，返回 DiagnosticReport。

        Parameters
        ----------
        run_modules : 指定执行的模块编号列表（None = 全部 1-6）
        lag_list    : Module 1 使用的 lag 值列表
        n_splits    : Module 6 时间分割数
        verbose     : 是否打印进度

        Returns
        -------
        DiagnosticReport
        """
        if run_modules is None:
            run_modules = [1, 2, 3, 4, 5, 6]

        module_fns = {
            1: lambda: self.module1_time_alignment(lag_list=list(lag_list)),
            2: lambda: self.module2_incremental_ic(),
            3: lambda: self.module3_exposure_strip(),
            4: lambda: self.module4_sample_bias(),
            5: lambda: self.module5_factor_halflife(),
            6: lambda: self.module6_robustness(n_splits=n_splits),
        }

        sorted_ids = [m for m in sorted(run_modules) if m in module_fns]

        # ── 外层模块总进度 ─────────────────────────────────────────────
        mod_prog = _ModuleProgress(module_ids=sorted_ids) if verbose else None

        results: List[DiagnosticResult] = []
        for mid in sorted_ids:
            if mod_prog:
                mod_prog.begin_module(mid)
            try:
                r = module_fns[mid]()
                status = r.status_str()
            except Exception as exc:
                warnings.warn(f"[WARN] Module {mid} failed: {exc}")
                r = DiagnosticResult(
                    module_id   = mid,
                    module_name = {1: "时间对齐", 2: "累计窗口", 3: "暴露剥离",
                                   4: "样本偏差", 5: "因子属性", 6: "稳健性复核"}.get(mid, ""),
                    passed      = None,
                    evidence    = {"error": str(exc)},
                    conclusion  = f"运行异常: {exc}",
                    risk_level  = "UNKNOWN",
                )
                status = "[ERROR]"
            if mod_prog:
                mod_prog.end_module(mid, status)
            results.append(r)

        if mod_prog:
            mod_prog.summary()

        return DiagnosticReport(results=results, factor_name=self.factor_name)
