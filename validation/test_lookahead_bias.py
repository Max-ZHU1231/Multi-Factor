"""
test_lookahead_bias.py
======================
前视偏差（Look-ahead Bias）检测测试套件

设计原则
--------
若程序无前视偏差，则对历史日期 D 的决策（因子值、持仓分组）与数据截止于
D 之后任意时刻重新运行的结果必须完全一致。

三个检测层次
-----------
- Layer 1 (L1)：算子/因子函数层 —— 单股维度，直接对比 factor_zoo 函数输出
- Layer 2 (L2)：因子面板层     —— 跨股票截面（Winsorize/中性化/标准化）
- Layer 3 (L3)：持仓分组层     —— layer_backtest 每日分组标签的前后一致性

专项检测
--------
- S1：前复权价格系统性偏差（数据层，不可通过修改代码修复）
- S2：build_panel_batch vs build_panel 行为一致性
- S3：ic_decay 路径 vs build_return_panel 路径的一致性（含停牌股）

参数规范
--------
- N_SHORT  = 63  （约 3 个月）：短周期因子
- N_MEDIUM = 130 （约 6 个月）：中周期因子
- N_LONG   = 270 （约 13 个月）：长周期因子（主要测试档）
- WARMUP   = 270 ：比较区间开头排除的 warm-up 行数（覆盖最长回看窗口 252+lag20）
- TOL      = 1e-8：浮点比较阈值

运行方式
--------
    python -m pytest test_lookahead_bias.py -v --tb=short

或仅运行某一层次：
    python -m pytest test_lookahead_bias.py -k "L1" -v
    python -m pytest test_lookahead_bias.py -k "L2" -v
    python -m pytest test_lookahead_bias.py -k "L3" -v
    python -m pytest test_lookahead_bias.py -k "S1 or S2 or S3" -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_framework.factor_engine import FactorEngine

# ── 确保项目根目录在 sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # validation/ -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 截断参数常量 ──────────────────────────────────────────────────────────────
N_SHORT  = 63    # 约 3 个月，覆盖短周期因子
N_MEDIUM = 130   # 约 6 个月，覆盖中周期因子
N_LONG   = 270   # 约 13 个月，覆盖长周期因子（主要测试档）
WARMUP   = 270   # 比较区间前部排除的行数（warm-up 期）
TOL      = 1e-8  # 浮点比较阈值

# ── 待检测的内置因子列表 ──────────────────────────────────────────────────────
ALL_BUILTIN_FACTORS = [
    "momentum_12_1", "momentum_6_1", "momentum_1m", "momentum_52w_high",
    "reversal_1w", "reversal_1m",
    "vol_20d", "vol_60d", "vol_skew", "downside_vol",
    "value_pb", "value_pe_ttm", "value_ps_ttm",
    "size_log_mktcap", "size_log_free_cap",
    "amihud_illiquidity", "turnover_rate", "vol_price_corr",
    "vwap_deviation", "price_strength",
    "bid_ask_spread_proxy", "zero_return_ratio",
    "pastor_stambaugh", "order_imbalance",
    "rsi_14", "macd_signal", "bb_position", "volume_trend",
]

# 按最长回看窗口分类（决定使用哪个截断参数）
_FACTORS_SHORT  = [  # 回看 ≤ 21 天
    "reversal_1w", "reversal_1m", "vol_20d", "vol_skew", "rsi_14",
    "bb_position", "vwap_deviation", "price_strength",
    "bid_ask_spread_proxy", "zero_return_ratio", "order_imbalance",
    "volume_trend", "value_pb", "value_pe_ttm", "value_ps_ttm",
    "size_log_mktcap", "size_log_free_cap",
]
_FACTORS_MEDIUM = [  # 回看 22–130 天
    "momentum_1m", "amihud_illiquidity", "turnover_rate", "vol_price_corr",
    "pastor_stambaugh", "macd_signal", "vol_60d", "downside_vol",
]
_FACTORS_LONG   = [  # 回看 131–270 天
    "momentum_12_1", "momentum_6_1", "momentum_52w_high",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 测试夹具与辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _make_stock_csv(
    tmpdir: str,
    symbol: str,
    n_rows: int,
    seed: int = 42,
    start_date: str = "20150101",
    with_valuation: bool = True,
    with_suspension: bool = False,
    suspension_start: int = 200,
    suspension_len:   int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    生成合成股票 CSV 文件，返回 (完整 DataFrame, 所有日期列表)。

    参数
    ----
    n_rows          : 总行数（交易日数）
    with_valuation  : 是否生成市净率/市盈率/市销率列（测试估值因子与 ffill）
    with_suspension : 是否模拟停牌（停牌期间收盘价/成交量置 NaN）
    suspension_start: 停牌起始行（行索引）
    suspension_len  : 停牌天数
    """
    rng = np.random.default_rng(seed)
    dates_dt = pd.bdate_range(start=start_date, periods=n_rows)
    dates    = [d.strftime("%Y%m%d") for d in dates_dt]

    # 模拟随机游走价格
    log_returns = rng.normal(0.0005, 0.015, n_rows)
    prices = np.exp(np.cumsum(log_returns)) * 10.0

    # 复权因子（模拟两次除权事件）
    adj_factor = np.ones(n_rows)
    split1 = n_rows // 3
    split2 = 2 * n_rows // 3
    adj_factor[split1:] *= 0.9     # 第一次除权
    adj_factor[split2:] *= 0.95    # 第二次除权

    volumes  = rng.integers(100, 10000, n_rows).astype(float)
    turnovers = rng.uniform(0.1, 5.0, n_rows)
    amounts  = prices * volumes * 100 / 1000   # 千元
    mktcap   = prices * rng.integers(5000, 50000, n_rows) / 100  # 万元
    free_cap = mktcap * rng.uniform(0.3, 0.9, n_rows)

    rows = []
    for i, d in enumerate(dates):
        p = prices[i]
        high = p * rng.uniform(1.0, 1.05)
        low  = p * rng.uniform(0.95, 1.0)
        row = {
            "交易日":         d,
            "股票代码":        symbol,
            "收盘价":          round(p, 4),
            "开盘价":          round(p * rng.uniform(0.98, 1.02), 4),
            "最高价":          round(high, 4),
            "最低价":          round(low, 4),
            "成交量（手）":    round(volumes[i], 0),
            "成交额（千元）":  round(amounts[i], 2),
            "换手率（%）":     round(turnovers[i], 4),
            "总市值（万元）":  round(mktcap[i], 2),
            "流通市值（万元）":round(free_cap[i], 2),
            "复权因子":        round(adj_factor[i], 8),
        }
        if with_valuation:
            pb  = rng.uniform(0.5, 5.0)
            pe  = rng.uniform(5.0, 100.0) if rng.random() > 0.05 else np.nan
            ps  = rng.uniform(0.2, 10.0)
            row["市净率"]                    = round(pb, 4)
            row["市盈率（TTM，亏损为空）"]   = round(pe, 4) if not np.isnan(pe) else np.nan
            row["市销率（TTM）"]             = round(ps, 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 模拟停牌（收盘价 / 成交量 置 NaN，复权因子和估值通过 ffill 填充）
    if with_suspension:
        idx = slice(suspension_start, suspension_start + suspension_len)
        df.loc[idx, "收盘价"]     = np.nan
        df.loc[idx, "成交量（手）"] = np.nan
        df.loc[idx, "成交额（千元）"] = np.nan

    path = os.path.join(tmpdir, f"{symbol}.csv")
    df.to_csv(path, index=False)
    return df, dates


def _make_engine(tmpdir: str, n_stocks: int = 8, n_rows: int = 800,
                 seed_base: int = 0) -> "FactorEngine":
    """构建含 n_stocks 只合成股票的 FactorEngine（不注册因子）。"""
    from factor_framework.factor_engine import FactorEngine
    from factor_framework.factor_zoo import register_all

    for i in range(n_stocks):
        _make_stock_csv(tmpdir, f"SYN{i:03d}", n_rows=n_rows, seed=seed_base + i)

    engine = FactorEngine(tmpdir, verbose=False, min_rows=30)
    register_all(engine)
    return engine


def _truncate_csvs(src_dir: str, dst_dir: str, cut_rows: int) -> None:
    """
    将 src_dir 中所有 CSV 文件截断（去掉最后 cut_rows 行），写入 dst_dir。
    这是 Look-ahead Bias 检测的核心操作：模拟"删掉未来数据再重跑"。
    """
    import glob
    os.makedirs(dst_dir, exist_ok=True)
    for path in glob.glob(os.path.join(src_dir, "*.csv")):
        df = pd.read_csv(path, dtype={"交易日": str, "股票代码": str})
        df_cut = df.iloc[:-cut_rows] if cut_rows < len(df) else df.iloc[:0]
        dst_path = os.path.join(dst_dir, os.path.basename(path))
        df_cut.to_csv(dst_path, index=False)


def _compare_series(
    full: pd.Series,
    cut:  pd.Series,
    label: str,
    tol: float = TOL,
    warmup: int = WARMUP,
) -> Dict:
    """
    比较两条序列（完整数据截断版 vs 截断数据重算版）在 warm-up 期后的差异。

    Returns
    -------
    dict with keys:
        n_compared    : 比较的时间点总数
        n_diff        : 差异点数
        max_diff      : 最大绝对差值
        diff_ratio    : 差异比例
        first_diff_idx: 第一个差异点的 index 值（或 None）
        passed        : bool，是否通过（diff_ratio ≈ 0 且 max_diff ≤ tol）
        details       : 差异点详情 DataFrame（date, full_val, cut_val, abs_diff）
    """
    # 对齐公共 index
    common = full.index.intersection(cut.index)
    if len(common) == 0:
        return dict(n_compared=0, n_diff=0, max_diff=np.nan,
                    diff_ratio=np.nan, first_diff_idx=None, passed=True,
                    details=pd.DataFrame())

    # 排除 warm-up 期（按位置排除前 warmup 行）
    sorted_common = sorted(common)
    compare_idx   = sorted_common[warmup:]  # 跳过前 warmup 个时间点
    if len(compare_idx) == 0:
        return dict(n_compared=0, n_diff=0, max_diff=np.nan,
                    diff_ratio=np.nan, first_diff_idx=None, passed=True,
                    details=pd.DataFrame())

    a = full.reindex(compare_idx).values
    b = cut.reindex(compare_idx).values

    # NaN 对 NaN 视为相同；一个 NaN 一个非 NaN 视为差异
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan  = np.isnan(a) ^ np.isnan(b)
    diff     = np.where(both_nan | one_nan, np.where(one_nan, np.inf, 0.0),
                        np.abs(a - b))
    diff_mask = (diff > tol)

    n_compared   = len(compare_idx)
    n_diff       = int(diff_mask.sum())
    max_diff     = float(np.nanmax(diff)) if n_compared > 0 else np.nan
    diff_ratio   = n_diff / n_compared if n_compared > 0 else 0.0
    first_diff   = compare_idx[np.argmax(diff_mask)] if n_diff > 0 else None

    details_rows = []
    for i, idx in enumerate(compare_idx):
        if diff_mask[i]:
            details_rows.append({
                "date":     idx,
                "full_val": a[i],
                "cut_val":  b[i],
                "abs_diff": diff[i],
            })
    details = pd.DataFrame(details_rows)

    passed = (n_diff == 0) or (max_diff <= tol)

    return dict(
        n_compared=n_compared, n_diff=n_diff,
        max_diff=max_diff, diff_ratio=diff_ratio,
        first_diff_idx=first_diff, passed=passed,
        details=details,
    )


def _compare_panels(
    full: pd.DataFrame,
    cut:  pd.DataFrame,
    label: str,
    tol: float = TOL,
    warmup: int = WARMUP,
) -> Dict:
    """
    比较两个面板（完整数据截断版 vs 截断数据重算版）的所有单元格差异。

    Returns
    -------
    dict: n_cells, n_diff_cells, max_diff, diff_cell_ratio, passed,
          per_stock_max_diff (Series), per_date_diff_count (Series)
    """
    common_dates  = sorted(full.index.intersection(cut.index))
    common_stocks = sorted(full.columns.intersection(cut.columns))

    # 排除 warm-up 期
    compare_dates = common_dates[warmup:]
    if len(compare_dates) == 0 or len(common_stocks) == 0:
        return dict(n_cells=0, n_diff_cells=0, max_diff=np.nan,
                    diff_cell_ratio=0.0, passed=True,
                    per_stock_max_diff=pd.Series(dtype=float),
                    per_date_diff_count=pd.Series(dtype=int))

    a = full.loc[compare_dates, common_stocks].values.astype(float)
    b = cut.loc[compare_dates,  common_stocks].values.astype(float)

    both_nan = np.isnan(a) & np.isnan(b)
    one_nan  = np.isnan(a) ^ np.isnan(b)
    diff     = np.where(both_nan, 0.0, np.where(one_nan, np.inf, np.abs(a - b)))
    diff_mask = (diff > tol)

    n_cells         = diff.size
    n_diff_cells    = int(diff_mask.sum())
    max_diff        = float(np.nanmax(diff)) if n_cells > 0 else np.nan
    diff_cell_ratio = n_diff_cells / n_cells if n_cells > 0 else 0.0

    per_stock_max   = pd.Series(
        np.nanmax(diff, axis=0), index=common_stocks, name="max_diff"
    )
    per_date_count  = pd.Series(
        diff_mask.sum(axis=1), index=compare_dates, name="n_diff_stocks"
    )

    passed = (n_diff_cells == 0) or (max_diff <= tol)

    return dict(
        n_cells=n_cells, n_diff_cells=n_diff_cells, max_diff=max_diff,
        diff_cell_ratio=diff_cell_ratio, passed=passed,
        per_stock_max_diff=per_stock_max,
        per_date_diff_count=per_date_count,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 共享 pytest fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def stock_dirs():
    """
    模块级 fixture：创建一次合成数据目录（full），再派生截断版（cut_*）。
    全部使用 tempfile，测试结束后自动清理。
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="lab_full_")
    # 8 只股票，1200 行（约 5 年），保证 warm-up 期（270）+ 比较期充足
    for i in range(8):
        _make_stock_csv(tmp, f"SYN{i:03d}", n_rows=1200, seed=i * 7)
    # 1 只含停牌的股票（用于 S3 测试）
    _make_stock_csv(tmp, "SUSP000", n_rows=1200, seed=99,
                    with_suspension=True, suspension_start=400, suspension_len=15)

    dirs = {"full": tmp}
    for n in (N_SHORT, N_MEDIUM, N_LONG):
        cut_dir = tempfile.mkdtemp(prefix=f"lab_cut{n}_")
        _truncate_csvs(tmp, cut_dir, n)
        dirs[f"cut_{n}"] = cut_dir

    yield dirs

    # 清理（Windows 上 tempfile 有时需要手动清理）
    import shutil
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def engines(stock_dirs):
    """为每个目录（full + 三档截断）构建一个 FactorEngine（含所有内置因子）。"""
    from factor_framework.factor_engine import FactorEngine
    from factor_framework.factor_zoo import register_all

    result = {}
    for key, d in stock_dirs.items():
        eng = FactorEngine(d, verbose=False, min_rows=30)
        register_all(eng)
        result[key] = eng
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1：算子 / 因子函数层检测
# ═══════════════════════════════════════════════════════════════════════════════

class TestL1FactorFunctionLookahead:
    """
    Layer 1：逐个检测每个内置因子函数在单股维度的前视偏差。

    方法
    ----
    1. 用完整数据调用因子函数，取前 len(Full)-N 行 → A_trimmed
    2. 用截断数据（前 len(Full)-N 行）调用同一因子函数 → B
    3. 在 warm-up 期之后逐点比较 A_trimmed 与 B
    """

    def _run_l1_check(
        self,
        factor_name: str,
        cut_n: int,
        seed: int = 42,
        n_rows: int = 1000,
    ) -> Dict:
        """
        通用 Layer 1 检测逻辑。

        Returns
        -------
        比较结果 dict（见 _compare_series）
        """
        from factor_framework.factor_zoo import BUILTIN_FACTORS

        fn = BUILTIN_FACTORS[factor_name]

        with tempfile.TemporaryDirectory() as tmp:
            full_df, dates = _make_stock_csv(tmp, "TEST", n_rows=n_rows, seed=seed)

            # 用 _fast_load 模拟引擎加载（含 ffill 和 _ret 列）
            from factor_framework.factor_engine import _fast_load
            from pathlib import Path

            full_path = Path(tmp) / "TEST.csv"
            df_full = _fast_load(full_path)
            assert df_full is not None, "_fast_load 返回 None，CSV 格式有误"

            # Run A：完整数据 → 截断到 T-N
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                series_full = fn(df_full)
            series_full.index = df_full["交易日"].values
            a_trimmed = series_full.iloc[: len(series_full) - cut_n]

            # 截断 df：去掉最后 cut_n 行
            df_cut = df_full.iloc[: len(df_full) - cut_n].copy()

            # Run B：截断数据
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                series_cut = fn(df_cut)
            series_cut.index = df_cut["交易日"].values

            return _compare_series(a_trimmed, series_cut, label=factor_name)

    # ── 短周期因子（N_SHORT = 63） ────────────────────────────────────────────

    @pytest.mark.parametrize("factor_name", _FACTORS_SHORT)
    def test_l1_short_factors_no_lookahead(self, factor_name):
        """Layer 1 短周期因子（回看 ≤ 21 天），截断 63 天后结果一致。"""
        result = self._run_l1_check(factor_name, cut_n=N_SHORT)
        assert result["passed"], (
            f"[L1] {factor_name}（N={N_SHORT}）存在前视偏差：\n"
            f"  比较点数={result['n_compared']}，差异点数={result['n_diff']}，"
            f"最大差值={result['max_diff']:.2e}，差异比例={result['diff_ratio']:.2%}\n"
            f"  首个差异日期: {result['first_diff_idx']}\n"
            f"{result['details'].head(5).to_string()}"
        )

    # ── 中周期因子（N_MEDIUM = 130） ──────────────────────────────────────────

    @pytest.mark.parametrize("factor_name", _FACTORS_MEDIUM)
    def test_l1_medium_factors_no_lookahead(self, factor_name):
        """Layer 1 中周期因子（回看 22–130 天），截断 130 天后结果一致。"""
        result = self._run_l1_check(factor_name, cut_n=N_MEDIUM)
        assert result["passed"], (
            f"[L1] {factor_name}（N={N_MEDIUM}）存在前视偏差：\n"
            f"  比较点数={result['n_compared']}，差异点数={result['n_diff']}，"
            f"最大差值={result['max_diff']:.2e}，差异比例={result['diff_ratio']:.2%}\n"
            f"  首个差异日期: {result['first_diff_idx']}\n"
            f"{result['details'].head(5).to_string()}"
        )

    # ── 长周期因子（N_LONG = 270） ────────────────────────────────────────────

    @pytest.mark.parametrize("factor_name", _FACTORS_LONG)
    def test_l1_long_factors_no_lookahead(self, factor_name):
        """Layer 1 长周期因子（回看 131–270 天），截断 270 天后结果一致。"""
        result = self._run_l1_check(factor_name, cut_n=N_LONG, n_rows=1500)
        assert result["passed"], (
            f"[L1] {factor_name}（N={N_LONG}）存在前视偏差：\n"
            f"  比较点数={result['n_compared']}，差异点数={result['n_diff']}，"
            f"最大差值={result['max_diff']:.2e}，差异比例={result['diff_ratio']:.2%}\n"
            f"  首个差异日期: {result['first_diff_idx']}\n"
            f"{result['details'].head(5).to_string()}"
        )

    def test_l1_all_factors_with_long_cutoff(self):
        """
        Layer 1 综合测试：对所有 28 个内置因子均使用 N_LONG=270 截断，
        逐一检测，汇总失败因子列表。
        """
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        from factor_framework.factor_engine import _fast_load
        from pathlib import Path

        failed = []
        with tempfile.TemporaryDirectory() as tmp:
            _make_stock_csv(tmp, "TEST", n_rows=1500, seed=77)
            full_path = Path(tmp) / "TEST.csv"
            df_full = _fast_load(full_path)
            assert df_full is not None
            df_cut = df_full.iloc[: len(df_full) - N_LONG].copy()

            for fname, fn in BUILTIN_FACTORS.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        sa = fn(df_full)
                        sa.index = df_full["交易日"].values
                        a_trimmed = sa.iloc[: len(sa) - N_LONG]

                        sb = fn(df_cut)
                        sb.index = df_cut["交易日"].values

                        res = _compare_series(a_trimmed, sb, label=fname)
                        if not res["passed"]:
                            failed.append({
                                "factor": fname,
                                "n_diff": res["n_diff"],
                                "max_diff": res["max_diff"],
                                "diff_ratio": res["diff_ratio"],
                                "first_diff_idx": res["first_diff_idx"],
                            })
                    except Exception as e:
                        failed.append({"factor": fname, "error": str(e)})

        if failed:
            summary = pd.DataFrame(failed).to_string()
            pytest.fail(
                f"[L1] 以下因子存在前视偏差（N={N_LONG}）：\n{summary}"
            )

    def test_l1_ffill_valuation_columns_no_future_leak(self):
        """
        Layer 1 专项：验证 _fast_load 中估值列（市净率等）的 ffill 不跨越截断点。

        若估值列每日都有值（日频市场数据），则 ffill 不会把 T-N 之后的值
        填回到 T-N 之前，因此不存在前视偏差。
        若估值列是季报数据（稀疏），则 ffill 会把截断点之后的数据填回，
        导致前视偏差（本测试标记为 xfail 并记录）。
        """
        from factor_framework.factor_engine import _fast_load
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            # 构建含每日估值数据的 CSV
            full_df, dates = _make_stock_csv(
                tmp, "VALTEST", n_rows=800, seed=12, with_valuation=True)
            full_path = Path(tmp) / "VALTEST.csv"
            df_full = _fast_load(full_path)
            assert df_full is not None

            cut_n = N_LONG
            df_cut = _fast_load(full_path)
            # 重新写入截断版 CSV
            cut_csv_path = Path(tmp) / "VALTEST_cut.csv"
            raw_df = pd.read_csv(full_path, dtype={"交易日": str})
            raw_df.iloc[: len(raw_df) - cut_n].to_csv(cut_csv_path, index=False)
            df_cut_loaded = _fast_load(cut_csv_path)
            assert df_cut_loaded is not None

            # 比较市净率列（在 warm-up 期后）
            pb_full = pd.Series(
                df_full["市净率"].values
                if "市净率" in df_full.columns else np.full(len(df_full), np.nan),
                index=df_full["交易日"].values,
            )
            pb_cut = pd.Series(
                df_cut_loaded["市净率"].values
                if "市净率" in df_cut_loaded.columns else np.full(len(df_cut_loaded), np.nan),
                index=df_cut_loaded["交易日"].values,
            )
            pb_full_trimmed = pb_full.iloc[: len(pb_full) - cut_n]
            res = _compare_series(pb_full_trimmed, pb_cut, label="市净率_ffill", warmup=0)

            # 合成数据中估值列是每日有值的，预期通过
            assert res["passed"], (
                f"[L1] 估值列 ffill 存在前视偏差（市净率，N={cut_n}）：\n"
                f"  差异点数={res['n_diff']}，最大差值={res['max_diff']:.2e}\n"
                f"  ⚠️  若使用季报数据源，此处会失败（设计如此），需改用后复权价格"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2：因子面板层检测
# ═══════════════════════════════════════════════════════════════════════════════

class TestL2FactorPanelLookahead:
    """
    Layer 2：在面板维度（所有股票 × 全部交易日）验证截断前后因子值矩阵一致。
    
    重点检测跨股票操作（Winsorize / 中性化 / 标准化）是否引入前视偏差。
    """

    def _build_panels(
        self,
        stock_dirs: Dict,
        engines: Dict,
        factor_name: str,
        cut_n: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建完整数据因子面板（截断到 T-N）和截断数据因子面板。

        Returns
        -------
        (panel_a_trimmed, panel_b)
        """
        eng_full = engines["full"]
        eng_cut  = engines[f"cut_{cut_n}"]

        syms = [f"SYN{i:03d}" for i in range(8)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panel_full = eng_full.build_panel(factor_name, symbols=syms)
            panel_cut  = eng_cut.build_panel(factor_name, symbols=syms)

        # 截断完整面板的最后 cut_n 行
        panel_a_trimmed = panel_full.iloc[: len(panel_full) - cut_n]
        return panel_a_trimmed, panel_cut

    @pytest.mark.parametrize("factor_name", [
        "reversal_1w", "vol_20d", "value_pb", "size_log_mktcap",
        "amihud_illiquidity", "macd_signal",
        "momentum_12_1", "momentum_6_1",
    ])
    def test_l2_raw_panel_no_lookahead(self, stock_dirs, engines, factor_name):
        """
        Layer 2 基础：原始因子面板（无截面预处理）在截断前后应保持一致。
        """
        cut_n = N_LONG
        panel_a, panel_b = self._build_panels(stock_dirs, engines, factor_name, cut_n)
        res = _compare_panels(panel_a, panel_b, label=factor_name)
        assert res["passed"], (
            f"[L2-raw] {factor_name}（N={cut_n}）因子面板存在前视偏差：\n"
            f"  总单元格={res['n_cells']}，差异单元格={res['n_diff_cells']}，"
            f"最大差值={res['max_diff']:.2e}，差异比例={res['diff_cell_ratio']:.2%}\n"
            f"  各股票最大差值:\n{res['per_stock_max_diff'].sort_values(ascending=False).head()}"
        )

    def test_l2_winsorize_cross_section_no_lookahead(self, stock_dirs, engines):
        """
        Layer 2 专项：横截面 MAD Winsorize 不应引入前视偏差。

        Winsorize 是逐日独立截面操作，理论上截断未来数据不影响历史截面结果，
        除非截断导致某只股票被排除出截面（边界情况）。
        """
        from factor_framework.operators import cs_winsorize
        from factor_framework.factor_engine import FactorEngine

        factor_name = "vol_20d"
        cut_n       = N_LONG
        syms        = [f"SYN{i:03d}" for i in range(8)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = engines["full"].build_panel(factor_name, symbols=syms)
            pc = engines[f"cut_{cut_n}"].build_panel(factor_name, symbols=syms)

        # apply_cross_section 是 @staticmethod
        pf_w = FactorEngine.apply_cross_section(pf, cs_winsorize)
        pc_w = FactorEngine.apply_cross_section(pc, cs_winsorize)

        pa_trimmed = pf_w.iloc[: len(pf_w) - cut_n]
        res = _compare_panels(pa_trimmed, pc_w, label="vol_20d_winsorize")
        assert res["passed"], (
            f"[L2-winsorize] vol_20d Winsorize 存在前视偏差（N={cut_n}）：\n"
            f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}"
        )

    def test_l2_standardize_rank_no_lookahead(self, stock_dirs, engines):
        """
        Layer 2 专项：横截面 rank 标准化不应引入前视偏差。
        Rank 是截面百分位，逐日独立，不依赖历史分布。
        """
        from factor_framework.operators import cs_rank
        from factor_framework.factor_engine import FactorEngine

        factor_name = "momentum_6_1"
        cut_n       = N_LONG
        syms        = [f"SYN{i:03d}" for i in range(8)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = engines["full"].build_panel(factor_name, symbols=syms)
            pc = engines[f"cut_{cut_n}"].build_panel(factor_name, symbols=syms)

        # apply_cross_section 是 @staticmethod
        pa_trimmed = FactorEngine.apply_cross_section(pf, cs_rank).iloc[: len(pf) - cut_n]
        pb         = FactorEngine.apply_cross_section(pc, cs_rank)

        res = _compare_panels(pa_trimmed, pb, label="momentum_6_1_rank")
        assert res["passed"], (
            f"[L2-rank] momentum_6_1 rank 标准化存在前视偏差（N={cut_n}）：\n"
            f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}"
        )

    def test_l2_neutralize_regression_no_lookahead(self, stock_dirs, engines):
        """
        Layer 2 专项：回归中性化（neutralize_regression）不应引入前视偏差。
        中性化是截面 OLS，逐日独立，理论上不依赖未来数据。
        """
        from factor_framework.neutralize import neutralize_regression

        factor_name = "momentum_6_1"
        cut_n       = N_LONG
        syms        = [f"SYN{i:03d}" for i in range(8)]

        # 简单行业映射（合成数据无真实行业，用偶奇分组代替）
        ind_map = pd.Series({s: f"IND{i % 3}" for i, s in enumerate(syms)})

        def _get_neutralized(eng_key: str) -> pd.DataFrame:
            eng = engines[eng_key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp = eng.build_panel(factor_name, symbols=syms)
                # 用 __mktcap__ 临时因子获取市值面板
                fn_mktcap = lambda df: df["总市值（万元）"]
                fn_mktcap.__name__ = "__mktcap_temp__"
                eng.register("__mktcap_temp__", fn_mktcap)
                mp = eng.build_panel("__mktcap_temp__", symbols=syms)
                del eng._registry["__mktcap_temp__"]

                mp_aligned = mp.reindex(fp.index)
                result = neutralize_regression(fp, mp_aligned, industry_map=ind_map)
            return result

        pf_neut = _get_neutralized("full")
        pc_neut = _get_neutralized(f"cut_{cut_n}")

        pa_trimmed = pf_neut.iloc[: len(pf_neut) - cut_n]
        res = _compare_panels(pa_trimmed, pc_neut, label="momentum_6_1_neutralize")
        assert res["passed"], (
            f"[L2-neutralize] momentum_6_1 中性化存在前视偏差（N={cut_n}）：\n"
            f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}"
        )

    def test_l2_cross_section_on_specific_date_consistent(self, stock_dirs, engines):
        """
        Layer 2 截面单日验证：取 warm-up 期后某一具体历史日期 D，
        完整数据与截断数据在该日的截面因子值应完全一致。
        """
        factor_name = "vol_60d"
        cut_n       = N_LONG
        syms        = [f"SYN{i:03d}" for i in range(8)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = engines["full"].build_panel(factor_name, symbols=syms)
            pc = engines[f"cut_{cut_n}"].build_panel(factor_name, symbols=syms)

        # 在公共有效日期中选取一个位于 warm-up 期后、截断点前的日期
        common = sorted(pf.index.intersection(pc.index))
        target_dates = common[WARMUP:]
        assert len(target_dates) > 0, "无可比较日期"

        # 取第一个有效比较日期
        D = target_dates[0]
        row_full = pf.loc[D, syms] if D in pf.index else pd.Series(np.nan, index=syms)
        row_cut  = pc.loc[D, syms] if D in pc.index else pd.Series(np.nan, index=syms)

        for stock in syms:
            a_val = row_full[stock]
            b_val = row_cut[stock]
            if pd.isna(a_val) and pd.isna(b_val):
                continue
            diff = abs(a_val - b_val) if not (pd.isna(a_val) or pd.isna(b_val)) else np.inf
            assert diff <= TOL, (
                f"[L2-date] {factor_name} 在日期 {D} 股票 {stock} 截面值不一致：\n"
                f"  完整数据={a_val:.8f}，截断数据={b_val:.8f}，差={diff:.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3：持仓分组层检测（核心测试）
# ═══════════════════════════════════════════════════════════════════════════════

def _get_holdings(
    factor_panel: pd.DataFrame,
    n_groups: int = 5,
    direction: int = 1,
    warmup: int = WARMUP,
) -> pd.DataFrame:
    """
    从因子面板生成每日每股票的持仓分组记录。

    Returns
    -------
    pd.DataFrame，columns: [date, ts_code, factor_value, group]
    """
    rows = []
    dates = sorted(factor_panel.index)[warmup:]   # 跳过 warm-up 期

    for date in dates:
        row_vals = factor_panel.loc[date]
        valid = row_vals.dropna()
        if len(valid) < n_groups * 2:
            # 有效股票数不足，分组无意义，跳过
            continue

        fv = valid.values * direction
        try:
            # 用分位数边界分组（与 layer_backtest 内部逻辑一致）
            quantile_points = np.linspace(0.0, 100.0, n_groups + 1)
            boundaries = np.nanpercentile(fv, quantile_points)

            for j, stock in enumerate(valid.index):
                v = fv[j]
                group = None
                for g in range(n_groups):
                    lo, hi = boundaries[g], boundaries[g + 1]
                    if g == n_groups - 1:
                        in_group = (v >= lo) and (v <= hi)
                    else:
                        in_group = (v >= lo) and (v < hi)
                    if in_group:
                        group = f"Q{g+1}"
                        break
                rows.append({
                    "date":         str(date),
                    "ts_code":      stock,
                    "factor_value": float(valid.iloc[j]),
                    "group":        group or "NaN",
                })
        except Exception:
            continue

    _EMPTY_HOLDINGS = pd.DataFrame(columns=["date", "ts_code", "factor_value", "group"])
    return pd.DataFrame(rows) if rows else _EMPTY_HOLDINGS


def _compare_holdings(hold_a: pd.DataFrame, hold_b: pd.DataFrame) -> Dict:
    """
    对比两份持仓记录（文件 A / 文件 B）的逐日一致性。

    Returns
    -------
    dict:
        n_dates_compared   : 比较日期数
        n_dates_diff_stocks: 股票池不一致的日期数
        n_dates_diff_group : 分组不一致的日期数
        n_dates_diff_value : 因子值不一致的日期数（差 > TOL）
        max_value_diff     : 最大因子值差
        diff_type          : 差异主类型 ('A'/'B'/'C'/'none')
        passed             : bool
        date_summary       : DataFrame 按日期汇总差异
    """
    dates_a = set(hold_a["date"].unique())
    dates_b = set(hold_b["date"].unique())
    common_dates = sorted(dates_a & dates_b)

    if not common_dates:
        return dict(n_dates_compared=0, n_dates_diff_stocks=0,
                    n_dates_diff_group=0, n_dates_diff_value=0,
                    max_value_diff=np.nan, diff_type="none", passed=True,
                    date_summary=pd.DataFrame())

    diff_rows = []
    n_diff_stocks = 0
    n_diff_group  = 0
    n_diff_value  = 0
    max_val_diff  = 0.0

    for d in common_dates:
        sub_a = hold_a[hold_a["date"] == d].set_index("ts_code")
        sub_b = hold_b[hold_b["date"] == d].set_index("ts_code")

        stocks_a = set(sub_a.index)
        stocks_b = set(sub_b.index)
        common_stocks = stocks_a & stocks_b

        # 股票池差异
        stocks_only_a = stocks_a - stocks_b
        stocks_only_b = stocks_b - stocks_a
        has_stock_diff = bool(stocks_only_a or stocks_only_b)
        if has_stock_diff:
            n_diff_stocks += 1

        # 共同股票的分组 & 因子值差异
        date_diff_group = 0
        date_diff_val   = 0.0
        for s in common_stocks:
            ga = sub_a.loc[s, "group"]
            gb = sub_b.loc[s, "group"]
            va = sub_a.loc[s, "factor_value"]
            vb = sub_b.loc[s, "factor_value"]

            if ga != gb:
                date_diff_group += 1

            val_diff = abs(va - vb) if not (pd.isna(va) or pd.isna(vb)) else np.inf
            if val_diff > TOL:
                date_diff_val = max(date_diff_val, val_diff)
                max_val_diff  = max(max_val_diff, val_diff)

        if date_diff_group > 0:
            n_diff_group += 1
        if date_diff_val > TOL:
            n_diff_value += 1

        if has_stock_diff or date_diff_group > 0 or date_diff_val > TOL:
            diff_rows.append({
                "date":           d,
                "diff_stocks_A_only": len(stocks_only_a),
                "diff_stocks_B_only": len(stocks_only_b),
                "diff_group_count":   date_diff_group,
                "max_value_diff":     date_diff_val,
            })

    n_dates  = len(common_dates)
    n_any    = len(diff_rows)
    date_sum = pd.DataFrame(diff_rows)

    # 判定差异类型
    if n_any == 0:
        diff_type = "none"
    elif n_any < n_dates * 0.2:
        diff_type = "A"   # 局部，靠近截断点
    elif n_diff_stocks > n_dates * 0.5:
        diff_type = "C"   # 特定股票池变化
    else:
        diff_type = "B"   # 全局均匀分布

    passed = (n_diff_group == 0) and (max_val_diff <= TOL)

    return dict(
        n_dates_compared=n_dates,
        n_dates_diff_stocks=n_diff_stocks,
        n_dates_diff_group=n_diff_group,
        n_dates_diff_value=n_diff_value,
        max_value_diff=max_val_diff,
        diff_type=diff_type,
        passed=passed,
        date_summary=date_sum,
    )


class TestL3HoldingsLookahead:
    """
    Layer 3：持仓分组层前视偏差检测（最顶层，综合性最强）。

    对每个因子：
    1. 完整数据 → 因子面板 → 持仓分组（文件 A）
    2. 截断数据 → 因子面板 → 持仓分组（文件 B）
    3. 对齐时间区间（去掉 warm-up + 截断后日期），逐日比较
    """

    def _get_factor_holdings(
        self,
        engines: Dict,
        eng_key: str,
        factor_name: str,
        cut_n: int,
        n_groups: int = 5,
    ) -> pd.DataFrame:
        """从指定引擎构建因子面板并生成持仓记录。"""
        eng  = engines[eng_key]
        syms = [f"SYN{i:03d}" for i in range(8)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp = eng.build_panel(factor_name, symbols=syms)

        # 对 full 引擎截断最后 cut_n 行，模拟"文件 A"
        if eng_key == "full":
            fp = fp.iloc[: len(fp) - cut_n]

        return _get_holdings(fp, n_groups=n_groups, warmup=WARMUP)

    @pytest.mark.parametrize("factor_name,cut_n", [
        ("reversal_1w",    N_SHORT),
        ("vol_20d",        N_SHORT),
        ("value_pb",       N_SHORT),
        ("amihud_illiquidity", N_MEDIUM),
        ("vol_60d",        N_MEDIUM),
        ("macd_signal",    N_MEDIUM),
        ("momentum_12_1",  N_LONG),
        ("momentum_6_1",   N_LONG),
        ("momentum_52w_high", N_LONG),
    ])
    def test_l3_holdings_consistent_after_truncation(
        self, stock_dirs, engines, factor_name, cut_n
    ):
        """
        Layer 3 核心测试：截断数据后，历史持仓分组应与完整数据结果完全一致。

        若分组不一致，说明因子计算使用了被截掉的未来数据（前视偏差）。
        """
        hold_a = self._get_factor_holdings(engines, "full", factor_name, cut_n)
        hold_b = self._get_factor_holdings(engines, f"cut_{cut_n}", factor_name, cut_n)

        res = _compare_holdings(hold_a, hold_b)

        assert res["passed"], (
            f"[L3] {factor_name}（N={cut_n}）持仓分组存在前视偏差：\n"
            f"  比较日期数={res['n_dates_compared']}，"
            f"分组不一致日期数={res['n_dates_diff_group']}，"
            f"最大因子值差={res['max_value_diff']:.2e}\n"
            f"  差异类型={res['diff_type']}\n"
            f"  按日期差异摘要（前5行）:\n"
            f"{res['date_summary'].head(5).to_string()}"
        )

    def test_l3_all_factors_long_cutoff_summary(self, stock_dirs, engines):
        """
        Layer 3 综合汇总测试：对所有内置因子使用 N_LONG 截断，
        输出完整的测试记录表（类似文档中表 7.1）。
        """
        from factor_framework.factor_zoo import BUILTIN_FACTORS

        records = []
        syms = [f"SYN{i:03d}" for i in range(8)]

        for fname in BUILTIN_FACTORS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fp_full = engines["full"].build_panel(fname, symbols=syms)
                    fp_cut  = engines[f"cut_{N_LONG}"].build_panel(fname, symbols=syms)

                fp_a = fp_full.iloc[: len(fp_full) - N_LONG]
                hold_a = _get_holdings(fp_a,   warmup=WARMUP)
                hold_b = _get_holdings(fp_cut,  warmup=WARMUP)

                res = _compare_holdings(hold_a, hold_b)
                records.append({
                    "factor":           fname,
                    "n_dates":          res["n_dates_compared"],
                    "n_diff_group":     res["n_dates_diff_group"],
                    "max_value_diff":   f"{res['max_value_diff']:.2e}",
                    "diff_type":        res["diff_type"],
                    "passed":           "✓" if res["passed"] else "✗",
                })
            except Exception as e:
                records.append({
                    "factor": fname, "n_dates": 0, "n_diff_group": 0,
                    "max_value_diff": "ERROR", "diff_type": "-",
                    "passed": f"✗ ({e})",
                })

        table = pd.DataFrame(records)
        failed = table[table["passed"] == "✗"]

        # 打印完整表格（无论通过与否，方便查阅）
        print("\n\n" + "="*80)
        print("Layer 3 前视偏差检测结果汇总（N_LONG=270）")
        print("="*80)
        print(table.to_string(index=False))
        print("="*80 + "\n")

        assert len(failed) == 0, (
            f"[L3] 以下 {len(failed)} 个因子存在持仓分组前视偏差：\n"
            f"{failed.to_string(index=False)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 专项检测 S1：前复权价格系统性偏差
# ═══════════════════════════════════════════════════════════════════════════════

class TestS1ForwardAdjustBias:
    """
    专项 S1：前复权价格引发的系统性偏差检测。

    前复权（qfq）机制：每次除权时回溯修改所有历史价格。
    如果 T-N 到 T 之间有除权事件，截断后加载的历史价格与原版不同，
    导致所有基于价格的因子在这些股票上有系统性偏差。

    注意：这是数据层问题，不能通过修改代码逻辑修复。
    本框架使用后复权（_hfq_close = 收盘价 × 复权因子）规避了这一问题。
    """

    def test_s1_hfq_close_uses_adj_factor(self):
        """
        S1 源码验证：_hfq_close 使用复权因子构建后复权价格，
        而非直接使用前复权收盘价。
        这是规避前复权系统性偏差的关键修复。
        """
        import inspect
        from factor_framework.factor_zoo import _hfq_close
        src = inspect.getsource(_hfq_close)
        assert "_ADJ" in src or "复权因子" in src, \
            "_hfq_close 应使用 '复权因子' 列（后复权）而非直接使用前复权收盘价"
        assert "df[_C]" in src and "adj" in src.lower(), \
            "_hfq_close 应将收盘价乘以复权因子"

    def test_s1_momentum_uses_hfq_close(self):
        """
        S1 源码验证：momentum_12_1 / momentum_6_1 / momentum_1m 均使用 _hfq_close，
        不直接使用 df['收盘价']（前复权价格）。
        """
        import inspect
        from factor_framework.factor_zoo import (
            momentum_12_1, momentum_6_1, momentum_1m, momentum_52w_high
        )
        for fn_name, fn in [
            ("momentum_12_1", momentum_12_1),
            ("momentum_6_1",  momentum_6_1),
            ("momentum_1m",   momentum_1m),
            ("momentum_52w_high", momentum_52w_high),
        ]:
            src = inspect.getsource(fn)
            assert "_hfq_close" in src, \
                f"{fn_name} 应使用 _hfq_close()（后复权），当前可能直接使用前复权收盘价"

    def test_s1_reversal_uses_hfq_close(self):
        """S1 源码验证：reversal_1w / reversal_1m 使用后复权价格。"""
        import inspect
        from factor_framework.factor_zoo import reversal_1w, reversal_1m
        for fn_name, fn in [("reversal_1w", reversal_1w), ("reversal_1m", reversal_1m)]:
            src = inspect.getsource(fn)
            assert "_hfq_close" in src, \
                f"{fn_name} 应使用 _hfq_close()（后复权）"

    def test_s1_pastor_stambaugh_uses_hfq_close(self):
        """S1 源码验证：pastor_stambaugh 使用后复权价格衍生的日收益率。"""
        import inspect
        from factor_framework.factor_zoo import pastor_stambaugh
        src = inspect.getsource(pastor_stambaugh)
        assert "_hfq_close" in src, \
            "pastor_stambaugh 应使用 _hfq_close()（后复权），避免前复权回溯修改"

    def test_s1_adj_factor_consistency_after_truncation(self):
        """
        S1 功能验证：后复权价格（收盘价 × 复权因子）在截断前后保持一致。

        合成数据中的复权因子是固定的（在生成时已设定），
        截断最后 N_LONG 天不影响历史复权因子的值，
        因此 _hfq_close 结果在截断前后应完全一致。
        """
        from factor_framework.factor_engine import _fast_load
        from factor_framework.factor_zoo import _hfq_close
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            _make_stock_csv(tmp, "ADJTEST", n_rows=800, seed=55)
            path = Path(tmp) / "ADJTEST.csv"
            df_full = _fast_load(path)
            assert df_full is not None

            df_cut = df_full.iloc[: len(df_full) - N_LONG].copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hfq_full = _hfq_close(df_full)
                hfq_cut  = _hfq_close(df_cut)

            hfq_full_trimmed = hfq_full.iloc[: len(hfq_full) - N_LONG]
            diff = (hfq_full_trimmed.values - hfq_cut.values)
            max_diff = float(np.nanmax(np.abs(diff)))
            assert max_diff <= TOL, (
                f"[S1] 后复权价格在截断前后不一致（最大差={max_diff:.2e}）。\n"
                f"若使用 AKShare 前复权实时下载，此处会失败（前复权回溯修改历史）。\n"
                f"本框架使用本地复权因子 × 收盘价，应通过此测试。"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 专项检测 S2：build_panel_batch vs build_panel 行为一致性
# ═══════════════════════════════════════════════════════════════════════════════

class TestS2BatchVsSingleConsistency:
    """
    专项 S2：build_panel_batch 与 build_panel（compute_single）路径的一致性。

    背景：两条路径对 warm-up 期的处理方式不同：
    - build_panel → compute_single：先在完整 df 上计算，再按 start 切片（正确）
    - build_panel_batch：v2.9.1 修复后也在完整 df 上计算（修复前存在截断问题）

    检测：同一因子、同一日期范围，两条路径的结果在 warm-up 期后应完全一致。
    """

    @pytest.mark.parametrize("factor_name", [
        "momentum_12_1",   # 最长回看 252 天
        "momentum_6_1",
        "vol_60d",
        "downside_vol",
        "amihud_illiquidity",
        "reversal_1w",
    ])
    def test_s2_batch_matches_single(self, stock_dirs, engines, factor_name):
        """
        build_panel_batch 与 build_panel 在 warm-up 期后结果完全一致。
        """
        syms = [f"SYN{i:03d}" for i in range(8)]
        eng  = engines["full"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 路径一：逐因子 build_panel（compute_single 内部实现）
            panel_single = eng.build_panel(factor_name, symbols=syms)
            # 路径二：批量 build_panel_batch
            panel_batch_dict = eng.build_panel_batch([factor_name], symbols=syms)
            panel_batch = panel_batch_dict.get(factor_name, pd.DataFrame())

        assert not panel_single.empty, f"build_panel 返回空 DataFrame ({factor_name})"
        assert not panel_batch.empty,  f"build_panel_batch 返回空 DataFrame ({factor_name})"

        res = _compare_panels(panel_single, panel_batch, label=f"{factor_name}_s2")
        assert res["passed"], (
            f"[S2] {factor_name}：build_panel_batch 与 build_panel 结果不一致。\n"
            f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}\n"
            f"  各股票最大差值:\n{res['per_stock_max_diff'].sort_values(ascending=False).head()}"
        )

    def test_s2_batch_multiple_factors_consistent(self, stock_dirs, engines):
        """
        S2 多因子批量场景：同时计算多个因子，批量路径与逐一路径结果应一致。
        （验证 DAG CSE 共享中间节点不影响正确性）
        """
        syms         = [f"SYN{i:03d}" for i in range(8)]
        factor_names = ["momentum_12_1", "momentum_6_1", "reversal_1m",
                        "vol_20d", "vol_60d"]
        eng = engines["full"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch_dict = eng.build_panel_batch(factor_names, symbols=syms)
            for fname in factor_names:
                single = eng.build_panel(fname, symbols=syms)
                batch  = batch_dict.get(fname, pd.DataFrame())

                res = _compare_panels(single, batch, label=f"{fname}_s2_multi")
                assert res["passed"], (
                    f"[S2-multi] {fname}：批量与单独计算结果不一致。\n"
                    f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}"
                )

    def test_s2_batch_no_lookahead_after_date_filter(self, stock_dirs, engines):
        """
        S2 前视偏差验证：build_panel_batch 指定 start 日期后，
        warm-up 期后的因子值不应因 start 截断而改变。

        验证 v2.9.1 修复：批量路径应先在完整 df 上计算，再按日期切片。
        """
        factor_name = "momentum_12_1"
        syms = [f"SYN{i:03d}" for i in range(8)]
        eng  = engines["full"]

        # 获取完整面板（无 start 限制）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panel_full = eng.build_panel(factor_name, symbols=syms)

        # 取一个 start 日期（确保有充足 warm-up 历史）
        all_dates = sorted(panel_full.index)
        if len(all_dates) < WARMUP + 100:
            pytest.skip("数据不足，跳过 S2 日期筛选测试")

        start_idx = WARMUP  # warm-up 期末
        start_date = str(all_dates[start_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 路径一：无 start 限制，计算完整面板，再手动切片
            panel_no_start = eng.build_panel(factor_name, symbols=syms)
            panel_a = panel_no_start.loc[panel_no_start.index >= start_date]

            # 路径二：指定 start（批量路径，内部会做日期过滤）
            batch_dict = eng.build_panel_batch(
                [factor_name], symbols=syms, start=start_date
            )
            panel_b = batch_dict.get(factor_name, pd.DataFrame())

        res = _compare_panels(panel_a, panel_b, label=f"{factor_name}_s2_start",
                              warmup=0)  # 已手动排除 warm-up
        assert res["passed"], (
            f"[S2-start] {factor_name}：指定 start={start_date} 后批量路径结果异常。\n"
            f"  差异单元格={res['n_diff_cells']}，最大差值={res['max_diff']:.2e}\n"
            f"  ⚠️  可能是 warm-up 期截断引起（参考 v2.9.1 BUG FIX）"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 专项检测 S3：ic_decay 路径 vs build_return_panel 路径一致性
# ═══════════════════════════════════════════════════════════════════════════════

class TestS3IcDecayPathConsistency:
    """
    专项 S3：ic_decay 路径与 build_return_panel 路径在收益率计算上的一致性。

    背景：
    - build_return_panel（factor_engine.py）：逐股 shift(-fwd)/price-1，再 .shift(1)
    - ic_decay（ic_analysis.py）：面板 price_panel.pct_change(fwd).shift(-fwd)

    两条路径对停牌日（NaN）的处理方式不同，可能产生系统性差异。
    本测试量化差异幅度，识别是否为代码逻辑缺陷。
    """

    def test_s3_ic_decay_vs_return_panel_no_suspension(self, stock_dirs, engines):
        """
        S3 基础：无停牌股票下，ic_decay 内部收益率应与 build_return_panel 基本一致。

        两条路径理论上对相同数据应产生相同结果（无停牌时 pct_change 行为一致）。
        """
        from factor_framework.factor_engine import FactorEngine
        from factor_framework.ic_analysis import ic_decay

        syms = [f"SYN{i:03d}" for i in range(4)]   # 用前 4 只股票（无停牌）
        eng  = engines["full"]

        fwd = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 路径一：build_return_panel（逐股计算）
            ret_panel = eng.build_return_panel(forward=fwd, symbols=syms)

            # 路径二：ic_decay 内部使用的收益率（通过 price_panel）
            eng.register("__close__", lambda df: df["收盘价"])
            close_panel = eng.build_panel("__close__", symbols=syms)
            del eng._registry["__close__"]

            # 模拟 ic_decay 内部的收益率计算
            # ic_decay 在内部做：price_panel.pct_change(fwd, axis=0).shift(-fwd)
            ret_icdecay = close_panel.pct_change(fwd, axis=0).shift(-fwd)

        # 对齐并比较（两者 index 应相同）
        common = ret_panel.index.intersection(ret_icdecay.index)
        compare_dates = sorted(common)[WARMUP:]

        if not compare_dates:
            pytest.skip("无可比较日期，跳过 S3 基础测试")

        ra = ret_panel.loc[compare_dates, syms].values.astype(float)
        rb = ret_icdecay.loc[compare_dates, syms].values.astype(float)

        # 注意：两条路径的定义本身有差异（build_return_panel 含 T+1 shift）
        # 所以这里不要求完全一致，而是检查差值分布
        both_nan = np.isnan(ra) & np.isnan(rb)
        diff = np.where(both_nan, 0.0, np.abs(ra - rb))
        non_nan_mask = ~np.isnan(diff)

        if non_nan_mask.sum() == 0:
            pytest.skip("所有单元格均为 NaN，跳过比较")

        mean_diff = float(np.mean(diff[non_nan_mask]))
        max_diff  = float(np.max(diff[non_nan_mask]))

        # 记录差异（S3 是记录性测试，差异存在时不强制失败，而是量化）
        print(f"\n[S3] ic_decay vs build_return_panel（fwd={fwd}，无停牌）：")
        print(f"     均值差={mean_diff:.4f}，最大差={max_diff:.4f}")
        print(f"     注：build_return_panel 含 T+1 shift，ic_decay 不含，故有系统性偏移")

        # 两条路径定义不同（T+1 vs 无 T+1），因此允许较大差异，
        # 但差异应有规律（约等于一期收益率的量级，而非随机噪声）
        # 此处记录，不强制 assert（符合文档中"系统性差异"的处理方式）

    def test_s3_ic_decay_suspension_quantifies_difference(self, stock_dirs, engines):
        """
        S3 停牌股：量化含停牌股票在停牌期前后的两路径差异幅度。

        停牌期间收盘价为 NaN，pct_change 的 axis=0（面板）和逐股行为不同：
        - 逐股 pct_change：NaN 期间跳过，可能将停牌前后隔开
        - 面板 pct_change(fwd, axis=0)：跨 NaN 计算，结果不同
        """
        from factor_framework.ic_analysis import ic_decay

        susp_sym = "SUSP000"
        syms     = [susp_sym]
        eng      = engines["full"]

        fwd = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret_panel = eng.build_return_panel(forward=fwd, symbols=syms)

            eng.register("__close__", lambda df: df["收盘价"])
            close_panel = eng.build_panel("__close__", symbols=syms)
            del eng._registry["__close__"]

            ret_icdecay = close_panel.pct_change(fwd, axis=0).shift(-fwd)

        common = sorted(ret_panel.index.intersection(ret_icdecay.index))[WARMUP:]
        if not common:
            pytest.skip("含停牌股票无可比较日期")

        ra = ret_panel.reindex(common)[susp_sym].values if susp_sym in ret_panel.columns else None
        rb = ret_icdecay.reindex(common)[susp_sym].values if susp_sym in ret_icdecay.columns else None

        if ra is None or rb is None:
            pytest.skip(f"停牌股票 {susp_sym} 在面板中不存在")

        diff = np.abs(ra - rb)
        valid = ~np.isnan(diff)
        if valid.sum() == 0:
            pytest.skip("全为 NaN，跳过")

        mean_diff = float(np.mean(diff[valid]))
        max_diff  = float(np.max(diff[valid]))
        n_diff_above_tol = int((diff[valid] > TOL).sum())

        print(f"\n[S3] 停牌股 {susp_sym}（fwd={fwd}）：")
        print(f"     均值差={mean_diff:.4f}，最大差={max_diff:.4f}")
        print(f"     差值 > {TOL:.0e} 的日期数={n_diff_above_tol}/{valid.sum()}")
        print(f"     结论：{'存在停牌处理差异（预期中，记录差异幅度）' if n_diff_above_tol > 0 else '停牌处理一致'}")

        # 记录性测试：不强制失败，但差异幅度不应超过 5 倍收益率量级
        # （超过则说明有算法错误，而不仅是定义差异）
        assert max_diff < 2.0, (
            f"[S3] 停牌股 {susp_sym} 两路径最大差={max_diff:.4f} 超过阈值 2.0，\n"
            f"可能存在算法错误（非定义差异）。请检查 ic_decay 的 pct_change 处理。"
        )

    def test_s3_ic_decay_path_independent_of_future_data(self, stock_dirs, engines):
        """
        S3 前视偏差：ic_decay 本身（在日频面板上调用）不应有前视偏差。

        ic_decay 内部做 shift(-fwd)，这是向未来看，但这是 IC 计算的合理设计
        （用未来收益评价当前因子），不是前视偏差。
        真正的前视偏差是：用截断后的数据重算，同一历史日期的因子值改变。

        注意：ic_decay 返回的是各 forward_period 的聚合统计量
        （均值 IC、ICIR 等），每个 forward_period 一行。截断 N_LONG 行后，
        统计量会有所不同（更少样本），但差异应在可接受范围内
        （不是由前视偏差导致的系统性偏移）。
        """
        from factor_framework.ic_analysis import ic_decay

        factor_name = "vol_20d"
        syms = [f"SYN{i:03d}" for i in range(8)]

        def _run_icdecay(eng_key: str) -> pd.DataFrame:
            eng = engines[eng_key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp = eng.build_panel(factor_name, symbols=syms)
                eng.register("__close__", lambda df: df["收盘价"])
                cp = eng.build_panel("__close__", symbols=syms)
                del eng._registry["__close__"]
                # 使用较短的 forward_periods 以适应截断后的数据量
                icd = ic_decay(fp, cp, forward_periods=[1, 5, 10], method="rank")
            return icd

        icd_full = _run_icdecay("full")
        icd_cut  = _run_icdecay(f"cut_{N_LONG}")

        # ic_decay 返回的是 (forward → stats) 的汇总表（每个 forward 一行）
        # 两者应有相同的 forward 行索引（1, 5, 10）
        assert not icd_full.empty, "ic_decay（full）返回空 DataFrame"
        assert not icd_cut.empty,  "ic_decay（cut）返回空 DataFrame"

        common_fwds = sorted(icd_full.index.intersection(icd_cut.index))
        assert len(common_fwds) > 0, "ic_decay 输出无公共 forward 行"

        # 检查两版本统计量的差异
        # 截断减少了样本数，允许最多 ±0.5 的绝对差（统计波动）
        # 若因子函数有前视偏差，ic_decay 会系统性地偏高/偏低
        for col in ["mean_ic", "icir"]:
            if col not in icd_full.columns or col not in icd_cut.columns:
                continue
            for fwd in common_fwds:
                a_val = icd_full.loc[fwd, col]
                b_val = icd_cut.loc[fwd, col]
                if pd.isna(a_val) or pd.isna(b_val):
                    continue
                diff = abs(a_val - b_val)
                assert diff < 0.5, (
                    f"[S3] ic_decay {col} forward={fwd}：差异={diff:.4f} 超过阈值 0.5\n"
                    f"  full={a_val:.4f}, cut={b_val:.4f}\n"
                    f"  可能存在前视偏差导致 ic_decay 统计量系统性偏移"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 综合汇总报告
# ═══════════════════════════════════════════════════════════════════════════════

class TestLookaheadBiasReport:
    """
    综合汇总：打印结构化测试报告，格式对应文档 §7.1 测试记录表。
    """

    def test_print_full_test_table(self, stock_dirs, engines):
        """
        生成完整测试记录表（对应文档表 7.1），打印到 stdout。
        本测试本身不会失败，仅用于生成可视化报告。
        """
        from factor_framework.factor_zoo import BUILTIN_FACTORS
        from factor_framework.factor_engine import _fast_load
        from pathlib import Path

        print("\n\n" + "=" * 100)
        print("前视偏差（Look-ahead Bias）完整检测报告")
        print(f"截断参数: N_SHORT={N_SHORT}, N_MEDIUM={N_MEDIUM}, N_LONG={N_LONG}")
        print(f"比较阈值: TOL={TOL}, WARMUP={WARMUP}")
        print("=" * 100)

        records = []

        # ── Layer 1 汇总 ──
        with tempfile.TemporaryDirectory() as tmp:
            _make_stock_csv(tmp, "REPORT", n_rows=1500, seed=88)
            path = Path(tmp) / "REPORT.csv"
            df_full = _fast_load(path)
            df_cut  = df_full.iloc[: len(df_full) - N_LONG].copy()

            for fname, fn in BUILTIN_FACTORS.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sa = fn(df_full)
                        sa.index = df_full["交易日"].values
                        a_trimmed = sa.iloc[: len(sa) - N_LONG]
                        sb = fn(df_cut)
                        sb.index = df_cut["交易日"].values
                    res = _compare_series(a_trimmed, sb, label=fname)
                    records.append({
                        "测试ID":   f"L1-{fname[:20]}",
                        "测试层次": "Layer 1",
                        "测试因子": fname,
                        "截断N":    N_LONG,
                        "差异点数": res["n_diff"],
                        "总点数":   res["n_compared"],
                        "最大差值": f"{res['max_diff']:.2e}" if res["n_compared"] > 0 else "-",
                        "是否通过": "✓" if res["passed"] else "✗",
                    })
                except Exception as e:
                    records.append({
                        "测试ID": f"L1-{fname[:20]}", "测试层次": "Layer 1",
                        "测试因子": fname, "截断N": N_LONG,
                        "差异点数": 0, "总点数": 0, "最大差值": "ERROR",
                        "是否通过": f"✗ ({e})",
                    })

        # ── Layer 2 & 3 汇总（使用引擎面板结果）──
        syms = [f"SYN{i:03d}" for i in range(8)]
        for fname in list(BUILTIN_FACTORS.keys())[:8]:   # 取前 8 个以节省时间
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pf = engines["full"].build_panel(fname, symbols=syms)
                    pc = engines[f"cut_{N_LONG}"].build_panel(fname, symbols=syms)
                pa = pf.iloc[: len(pf) - N_LONG]
                res_l2 = _compare_panels(pa, pc, label=fname)
                records.append({
                    "测试ID":   f"L2-{fname[:20]}",
                    "测试层次": "Layer 2",
                    "测试因子": fname,
                    "截断N":    N_LONG,
                    "差异点数": res_l2["n_diff_cells"],
                    "总点数":   res_l2["n_cells"],
                    "最大差值": f"{res_l2['max_diff']:.2e}",
                    "是否通过": "✓" if res_l2["passed"] else "✗",
                })
                # Layer 3
                hold_a = _get_holdings(pa,  warmup=WARMUP)
                hold_b = _get_holdings(pc,  warmup=WARMUP)
                res_l3 = _compare_holdings(hold_a, hold_b)
                records.append({
                    "测试ID":   f"L3-{fname[:20]}",
                    "测试层次": "Layer 3",
                    "测试因子": fname,
                    "截断N":    N_LONG,
                    "差异点数": res_l3["n_dates_diff_group"],
                    "总点数":   res_l3["n_dates_compared"],
                    "最大差值": f"{res_l3['max_value_diff']:.2e}",
                    "是否通过": "✓" if res_l3["passed"] else "✗",
                })
            except Exception as e:
                records.append({
                    "测试ID": f"L2/L3-{fname[:20]}", "测试层次": "Layer 2+3",
                    "测试因子": fname, "截断N": N_LONG,
                    "差异点数": 0, "总点数": 0, "最大差值": "ERROR",
                    "是否通过": f"✗ ({e})",
                })

        table = pd.DataFrame(records)
        print(table.to_string(index=False))
        print("\n" + "=" * 100)

        n_pass = (table["是否通过"] == "✓").sum()
        n_fail = (table["是否通过"] != "✓").sum()
        print(f"总测试项: {len(table)}，通过: {n_pass}，失败: {n_fail}")
        print("=" * 100 + "\n")
        # 报告测试本身不 fail（仅汇总展示）
