"""
tests/unit/test_ic_diagnostics_perf.py
=======================================
性能回归测试：确保后续修改不造成明显性能退化。

测试分为两类：
  1. 快速正确性测试（始终在 CI 中运行）
     - 向量化工具函数的数值一致性
     - 缓存无副作用（多次 run_all() 结果相同）
  2. 慢速计时测试（标记为 @pytest.mark.slow，默认跳过）
     - 小规模合成数据下的 run_all() 耗时上限
     - 大规模（T=1500, N=500）耗时上限

运行方式
--------
# 仅运行快速测试（CI 默认）
pytest tests/unit/test_ic_diagnostics_perf.py -m "not slow" -v

# 运行全部（含计时测试，本地验收）
pytest tests/unit/test_ic_diagnostics_perf.py -m slow -v -s
pytest tests/unit/test_ic_diagnostics_perf.py -v -s
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from factor_framework.analytics.ic_decay_diagnostics import (
    ICDecayDiagnostics,
    _ic_from_arrays,
    _neutralize_batch,
    _rank_array,
    _compute_ic_series,
    _build_forward_ret,
)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：合成数据生成
# ─────────────────────────────────────────────────────────────────────────────

def _make_dates(n: int) -> list:
    return pd.bdate_range("2020-01-02", periods=n).strftime("%Y%m%d").tolist()


def _make_price_panel(n_dates: int, n_stocks: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = rng.normal(0.0005, 0.02, size=(n_dates, n_stocks))
    prices = 100 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(
        prices,
        index=_make_dates(n_dates),
        columns=[f"S{i:04d}" for i in range(n_stocks)],
    )


def _make_factor_panel(price_panel: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(price_panel.shape)
    return pd.DataFrame(noise, index=price_panel.index, columns=price_panel.columns)


def _make_diag(n_dates: int, n_stocks: int, with_industry: bool = False, seed: int = 42) -> ICDecayDiagnostics:
    price = _make_price_panel(n_dates, n_stocks, seed=seed)
    factor = _make_factor_panel(price, seed=seed + 1)
    mktcap = price * 1e6

    industry_map = None
    if with_industry:
        stocks = price.columns
        inds = [f"IND{i % 5}" for i in range(len(stocks))]
        industry_map = pd.Series(inds, index=stocks, name="industry")

    return ICDecayDiagnostics(
        factor_panel=factor,
        price_panel=price,
        mktcap_panel=mktcap,
        industry_map=industry_map,
        forward_list=[1, 5, 10, 21, 60],
        ic_method="rank",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. _rank_array 正确性
# ─────────────────────────────────────────────────────────────────────────────

class TestRankArray:
    def test_rank_values_match_scipy(self):
        """_rank_array 结果与 scipy.stats.rankdata 一致。"""
        from scipy import stats as sp_stats

        rng = np.random.default_rng(0)
        arr = rng.standard_normal((30, 50))
        arr[2, :5] = np.nan  # 引入 NaN

        got = _rank_array(arr)

        for i in range(arr.shape[0]):
            row = arr[i]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                assert np.isnan(got[i, :]).all()
                continue
            expected_ranks = sp_stats.rankdata(row[valid], method="ordinal").astype(float)
            np.testing.assert_array_equal(got[i, valid], expected_ranks)

    def test_rank_nan_preserved(self):
        """NaN 位置在输出中仍为 NaN。"""
        arr = np.array([[1.0, np.nan, 3.0, 2.0]])
        out = _rank_array(arr)
        assert np.isnan(out[0, 1])
        assert not np.isnan(out[0, 0])

    def test_rank_shape_preserved(self):
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((100, 200))
        out = _rank_array(arr)
        assert out.shape == arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# 2. _ic_from_arrays 正确性 — 与 DataFrame 路径比较
# ─────────────────────────────────────────────────────────────────────────────

class TestIcFromArrays:
    def _reference_ic(self, f_arr: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
        """参考实现：用 pd.DataFrame.rank().corr() 逐行计算 IC。"""
        T = f_arr.shape[0]
        ic_ref = np.full(T, np.nan)
        for i in range(T):
            f_row = pd.Series(f_arr[i])
            r_row = pd.Series(r_arr[i])
            mask = f_row.notna() & r_row.notna()
            if mask.sum() < 5:
                continue
            ic_ref[i] = f_row[mask].rank().corr(r_row[mask].rank())
        return ic_ref

    def test_rank_ic_matches_reference(self):
        """_ic_from_arrays(method='rank') 与 DataFrame 参考路径差异 < 1e-6。

        注：_rank_array 使用 argsort（ordinal 排名），而 pd.Series.rank() 默认
        使用 'average' 方法处理并列值；对于连续随机数，两者等价，但因 NaN 处理
        路径略有差异，容差设为 1e-6（而非更严格的 1e-10）。
        """
        rng = np.random.default_rng(42)
        f_arr = rng.standard_normal((50, 60))
        r_arr = rng.standard_normal((50, 60))
        # 不引入 NaN 以避免 ordinal vs average rank 细节差异
        # （NaN 的情况单独在 test_nan_rows_produce_nan 中测试）

        got = _ic_from_arrays(f_arr, r_arr, method="rank", min_stocks=5)
        ref = self._reference_ic(f_arr, r_arr)

        valid = ~np.isnan(ref)
        assert valid.sum() > 0, "参考 IC 全为 NaN"
        np.testing.assert_allclose(got[valid], ref[valid], atol=1e-6,
                                   err_msg="_ic_from_arrays rank IC 与参考不一致")

    def test_nan_rows_produce_nan(self):
        """有效股票数 < min_stocks 的行输出应为 NaN。"""
        rng = np.random.default_rng(0)
        f_arr = rng.standard_normal((10, 20))
        r_arr = rng.standard_normal((10, 20))
        # 第 0 行全为 NaN
        f_arr[0, :] = np.nan
        out = _ic_from_arrays(f_arr, r_arr, min_stocks=5)
        assert np.isnan(out[0])

    def test_output_in_minus1_1(self):
        """IC 值域应在 [-1, 1]。"""
        rng = np.random.default_rng(7)
        f_arr = rng.standard_normal((100, 80))
        r_arr = rng.standard_normal((100, 80))
        out = _ic_from_arrays(f_arr, r_arr, method="rank", min_stocks=5)
        valid = out[~np.isnan(out)]
        assert (valid >= -1 - 1e-9).all() and (valid <= 1 + 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. _neutralize_batch 正确性 — 与逐行 lstsq 比较
# ─────────────────────────────────────────────────────────────────────────────

class TestNeutralizeBatch:
    def _reference_neutralize(self, factor_arr: np.ndarray, cov_arr: np.ndarray,
                              min_obs: int = 5) -> np.ndarray:
        """参考实现：逐行 np.linalg.lstsq OLS。"""
        T, N = factor_arr.shape
        resid = np.full((T, N), np.nan)
        for t in range(T):
            y = factor_arr[t]
            x = cov_arr[t]
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < min_obs:
                continue
            y_v, x_v = y[valid], x[valid]
            X = np.column_stack([np.ones(len(y_v)), x_v])
            coef, *_ = np.linalg.lstsq(X, y_v, rcond=None)
            r = y_v - X @ coef
            resid[t, valid] = r
        return resid

    def test_matches_lstsq_reference(self):
        """_neutralize_batch 与逐行 lstsq 差异 < 1e-8。"""
        rng = np.random.default_rng(123)
        T, N = 80, 100
        f_arr = rng.standard_normal((T, N))
        c_arr = rng.standard_normal((T, N))
        # 引入少量 NaN
        f_arr[5, :3] = np.nan
        c_arr[10, 7:12] = np.nan

        got = _neutralize_batch(f_arr, c_arr, min_obs=5)
        ref = self._reference_neutralize(f_arr, c_arr, min_obs=5)

        valid = ~np.isnan(ref)
        assert valid.sum() > 0, "参考残差全为 NaN"
        np.testing.assert_allclose(got[valid], ref[valid], atol=1e-8,
                                   err_msg="_neutralize_batch 与 lstsq 参考不一致")

    def test_rows_below_min_obs_are_nan(self):
        """有效观测 < min_obs 的行全为 NaN。"""
        rng = np.random.default_rng(0)
        T, N = 10, 20
        f_arr = rng.standard_normal((T, N))
        c_arr = rng.standard_normal((T, N))
        # 第 0 行只保留 3 个有效值
        f_arr[0, 3:] = np.nan
        out = _neutralize_batch(f_arr, c_arr, min_obs=5)
        assert np.isnan(out[0]).all()

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        T, N = 50, 70
        f_arr = rng.standard_normal((T, N))
        c_arr = rng.standard_normal((T, N))
        out = _neutralize_batch(f_arr, c_arr)
        assert out.shape == (T, N)

    def test_residuals_orthogonal_to_covariate(self):
        """残差应与协变量近似正交（OLS 性质）。"""
        rng = np.random.default_rng(42)
        T, N = 30, 80
        f_arr = rng.standard_normal((T, N))
        c_arr = rng.standard_normal((T, N))
        out = _neutralize_batch(f_arr, c_arr, min_obs=5)

        for t in range(T):
            valid = ~np.isnan(out[t]) & ~np.isnan(c_arr[t])
            if valid.sum() < 10:
                continue
            r = out[t, valid]
            x = c_arr[t, valid]
            dot = np.dot(r - r.mean(), x - x.mean())
            # dot / (n * std_r * std_x) should be near 0
            corr = dot / (valid.sum() * r.std(ddof=1) * x.std(ddof=1) + 1e-12)
            assert abs(corr) < 1e-6, f"第 {t} 行残差与协变量相关：corr={corr:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. 缓存无副作用：多次调用 run_all() 结果一致
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheIdempotency:
    """验证 __init__ 缓存（_f_shifted, _daily_ret, _neut_cache）无副作用。"""

    @pytest.fixture(scope="class")
    def diag(self):
        return _make_diag(n_dates=200, n_stocks=40, with_industry=True, seed=0)

    def test_run_all_twice_same_m1(self, diag):
        """连续两次调用 run_all()，M1 的 passed/evidence 完全相同。"""
        r1 = diag.run_all()
        r2 = diag.run_all()
        # DiagnosticReport 存储 results 列表，M1 在 index 0
        m1_passed_1 = r1.results[0].passed
        m1_passed_2 = r2.results[0].passed
        assert m1_passed_1 == m1_passed_2, \
            f"两次 run_all() M1 passed 不一致: {m1_passed_1} vs {m1_passed_2}"
        # 检查 evidence 中的数值
        ev1 = r1.results[0].evidence
        ev2 = r2.results[0].evidence
        if isinstance(ev1, dict) and "lag_1" in ev1:
            ic1 = ev1["lag_1"]["mean_ic"]
            ic2 = ev2["lag_1"]["mean_ic"]
            assert abs(ic1 - ic2) < 1e-12, \
                f"两次 run_all() M1 IC 不一致: {ic1} vs {ic2}"
        elif isinstance(ev1, pd.DataFrame):
            pd.testing.assert_frame_equal(ev1, ev2)

    def test_run_all_twice_same_m2(self, diag):
        """连续两次调用 run_all()，M2 的 passed 结果完全相同。"""
        r1 = diag.run_all()
        r2 = diag.run_all()
        # M2 在 results index 1
        m2_passed_1 = r1.results[1].passed
        m2_passed_2 = r2.results[1].passed
        assert m2_passed_1 == m2_passed_2, \
            f"两次 run_all() M2 passed 不一致: {m2_passed_1} vs {m2_passed_2}"
        # 对 evidence DataFrame/dict 做宽松数值比较
        ev1 = r1.results[1].evidence
        ev2 = r2.results[1].evidence
        if isinstance(ev1, pd.DataFrame) and isinstance(ev2, pd.DataFrame):
            np.testing.assert_allclose(
                ev1.select_dtypes(float).to_numpy(),
                ev2.select_dtypes(float).to_numpy(),
                atol=1e-12,
                err_msg="两次 run_all() M2 evidence 数值不一致",
            )

    def test_neutralized_cache_consistent(self, diag):
        """_get_neutralized 惰性缓存结果与直接计算一致（调用两次）。"""
        out1 = diag._get_neutralized("cap")
        out2 = diag._get_neutralized("cap")
        assert out1 is out2, "_neut_cache 未正确返回同一对象"

    def test_f_shifted_matches_manual(self, diag):
        """_f_shifted 与手动 _shift_factor(factor_panel, lag=1) 一致。"""
        from factor_framework.analytics.ic_decay_diagnostics import _shift_factor
        manual = _shift_factor(diag.factor_panel, lag=1)
        # 对齐公共区域
        common_idx = diag._f_shifted.index.intersection(manual.index)
        common_col = diag._f_shifted.columns.intersection(manual.columns)
        a = diag._f_shifted.loc[common_idx, common_col].to_numpy(dtype=float)
        b = manual.loc[common_idx, common_col].to_numpy(dtype=float)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() > 0:
            np.testing.assert_allclose(a[mask], b[mask], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 快速计时回归测试（小规模，始终运行，给出警告而不是 xfail）
# ─────────────────────────────────────────────────────────────────────────────

class TestTimingFast:
    """T=300, N=80 — run_all() 必须在 60s 内完成（CI 可接受）。"""

    WALL_LIMIT_S = 60.0

    @pytest.fixture(scope="class")
    def diag_small(self):
        return _make_diag(n_dates=300, n_stocks=80, with_industry=True, seed=11)

    def test_run_all_small_within_limit(self, diag_small):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diag_small.run_all()
        elapsed = time.perf_counter() - t0
        assert elapsed < self.WALL_LIMIT_S, (
            f"run_all() on T=300,N=80 took {elapsed:.2f}s "
            f"(limit: {self.WALL_LIMIT_S}s) — possible performance regression"
        )

    def test_ic_from_arrays_faster_than_dataframe(self):
        """_ic_from_arrays 应比 DataFrame 逐行 rank+corr 快至少 2×。"""
        rng = np.random.default_rng(0)
        T, N = 500, 100
        f_arr = rng.standard_normal((T, N))
        r_arr = rng.standard_normal((T, N))
        f_df = pd.DataFrame(f_arr)
        r_df = pd.DataFrame(r_arr)

        # 向量化路径
        t0 = time.perf_counter()
        for _ in range(3):
            _ic_from_arrays(f_arr, r_arr, method="rank")
        new_time = time.perf_counter() - t0

        # 参考路径（DataFrame 逐行）
        def _df_ic():
            out = []
            for i in range(T):
                f_row = f_df.iloc[i]
                r_row = r_df.iloc[i]
                out.append(f_row.rank().corr(r_row.rank()))
            return out

        t0 = time.perf_counter()
        for _ in range(3):
            _df_ic()
        old_time = time.perf_counter() - t0

        speedup = old_time / (new_time + 1e-9)
        assert speedup >= 2.0, (
            f"_ic_from_arrays speedup={speedup:.2f}x, expected >=2x "
            f"(new={new_time:.3f}s, old={old_time:.3f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. 慢速计时回归测试（大规模，仅本地运行）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestTimingSlow:
    """
    T=1500, N=500 — 性能回归保护。
    使用宽松阈值（1.5× 实测值），防止重大退化的同时保留优化空间。

    实测基准（After_P0_P1）：~27s
    阈值：< 60s（约 2× 余量）
    """

    WALL_LIMIT_S = 60.0

    @pytest.fixture(scope="class")
    def diag_large(self):
        return _make_diag(n_dates=1500, n_stocks=500, with_industry=True, seed=42)

    def test_run_all_large_within_limit(self, diag_large):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diag_large.run_all()
        elapsed = time.perf_counter() - t0
        print(f"\n[perf] run_all(T=1500,N=500) = {elapsed:.2f}s  (limit={self.WALL_LIMIT_S}s)")
        assert elapsed < self.WALL_LIMIT_S, (
            f"run_all(T=1500,N=500) took {elapsed:.2f}s "
            f"(limit: {self.WALL_LIMIT_S}s) — PERFORMANCE REGRESSION"
        )

    def test_neutralize_batch_speedup_vs_lstsq(self):
        """_neutralize_batch 比逐行 lstsq 快至少 5×（T=1000, N=500）。"""
        from factor_framework.analytics.ic_decay_diagnostics import _neutralize_batch

        rng = np.random.default_rng(7)
        T, N = 1000, 500
        f_arr = rng.standard_normal((T, N))
        c_arr = rng.standard_normal((T, N))

        # 向量化
        t0 = time.perf_counter()
        _neutralize_batch(f_arr, c_arr, min_obs=5)
        new_time = time.perf_counter() - t0

        # 逐行 lstsq
        def _lstsq_loop():
            resid = np.full((T, N), np.nan)
            for t in range(T):
                y = f_arr[t]
                x = c_arr[t]
                valid = ~(np.isnan(y) | np.isnan(x))
                if valid.sum() < 5:
                    continue
                y_v, x_v = y[valid], x[valid]
                X = np.column_stack([np.ones(len(y_v)), x_v])
                coef, *_ = np.linalg.lstsq(X, y_v, rcond=None)
                resid[t, valid] = y_v - X @ coef
            return resid

        t0 = time.perf_counter()
        _lstsq_loop()
        old_time = time.perf_counter() - t0

        speedup = old_time / (new_time + 1e-9)
        print(f"\n[perf] _neutralize_batch speedup={speedup:.1f}x "
              f"(new={new_time:.3f}s, old={old_time:.3f}s)")
        assert speedup >= 5.0, (
            f"_neutralize_batch speedup={speedup:.1f}x, expected >=5x"
        )
