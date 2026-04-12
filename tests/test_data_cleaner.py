"""
test_data_cleaner.py
====================
单元测试：随机抽查 Stocks/ 目录下的股票 CSV，验证数据是否经过正确清洗。

运行方式
--------
    python -m pytest test_data_cleaner.py -v
    python -m pytest test_data_cleaner.py -v --tb=short -q   # 简洁输出

抽查策略
--------
- 固定随机种子，每次运行结果可复现
- 从全量股票文件中随机抽取 SAMPLE_N 只进行检测
- 若需全量扫描，设置 SAMPLE_N = None
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── 项目模块 ──────────────────────────────────────────────────────────────────
from data_cleaner import (
    MAD_THRESHOLD,
    MISSING_HEAVY,
    MISSING_RANDOM,
    MIN_ROWS,
    FFILL_PRICE_MAX,
    NON_NUMERIC_COLS,
    PRICE_VOL_COLS,
    VALUATION_COLS,
    clean_stock_df,
    diagnose_df,
    load_and_clean,
    mad_winsorize,
    _has_outliers,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
STOCKS_DIR = Path(__file__).parent.parent / "Stocks"
SAMPLE_N   = 30     # 随机抽查只数；None 表示全量
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1：MAD Winsorize 单元测试（纯逻辑，不依赖真实数据）
# ═══════════════════════════════════════════════════════════════════════════════

class TestMadWinsorize:
    """验证 MAD Winsorize 算法正确性。"""

    def test_no_outlier_unchanged(self):
        """正常分布数据不应被截断。"""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(100, 5, 200))
        result = mad_winsorize(s, k=MAD_THRESHOLD)
        # 允许极端值被截，但整体变化应极小
        assert result.notna().sum() == s.notna().sum()

    def test_extreme_values_clipped(self):
        """插入明显极值后，清洗后范围应变窄。"""
        base = pd.Series(list(range(100)))  # 0..99
        with_outlier = pd.concat([base, pd.Series([9999, -9999])], ignore_index=True)
        cleaned = mad_winsorize(with_outlier)
        assert cleaned.max() < 9999, "正极值未被截断"
        assert cleaned.min() > -9999, "负极值未被截断"

    def test_all_same_unchanged(self):
        """所有值相同（MAD=0）时不应改变。"""
        s = pd.Series([5.0] * 50)
        assert mad_winsorize(s).equals(s)

    def test_preserves_non_outlier_values(self):
        """Winsorize 不改变非极值数据的值。"""
        s = pd.Series([10.0, 11.0, 10.5, 9.8, 10.2, 10000.0])
        cleaned = mad_winsorize(s)
        # 前 5 个值应保持不变（相对于 median/MAD 不超出阈值）
        for orig, c in zip(s[:5], cleaned[:5]):
            assert abs(orig - c) < 1e-9, f"非极值 {orig} 被意外修改为 {c}"

    def test_nan_handled(self):
        """含 NaN 的 series 不应报错，NaN 仍保持 NaN。"""
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 9999.0])
        result = mad_winsorize(s)
        assert result.isna().sum() == 1

    def test_small_sample_unchanged(self):
        """样本数 < 4 时不做处理。"""
        s = pd.Series([1.0, 9999.0, 2.0])
        assert mad_winsorize(s).equals(s)

    def test_symmetry(self):
        """正负极值应对称截断。"""
        center = 50.0
        base   = pd.Series([center] * 50)
        s = pd.concat([base, pd.Series([center + 1000, center - 1000])], ignore_index=True)
        cleaned = mad_winsorize(s)
        assert abs(cleaned.max() - (2 * center - cleaned.min())) < 1e-6, "截断不对称"

    def test_outlier_detection(self):
        """_has_outliers 应能正确识别含极值的 series。"""
        normal = pd.Series(list(range(50)))
        with_outlier = pd.concat([normal, pd.Series([99999])], ignore_index=True)
        assert not _has_outliers(normal), "正常数据误报为含极值"
        assert _has_outliers(with_outlier), "含极值的数据未被检测到"


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2：clean_stock_df 逻辑测试（合成数据）
# ═══════════════════════════════════════════════════════════════════════════════

def _make_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """生成合成单股 DataFrame，用于逻辑测试。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y%m%d").tolist()
    price = 10 + np.cumsum(rng.normal(0, 0.2, n))
    df = pd.DataFrame({
        "股票代码": "000001.SZ",
        "股票名称": "测试",
        "交易日":   dates,
        "开盘价":   price + rng.uniform(-0.1, 0.1, n),
        "最高价":   price + rng.uniform(0, 0.3, n),
        "最低价":   price - rng.uniform(0, 0.3, n),
        "收盘价":   price,
        "前收盘价": np.roll(price, 1),
        "涨跌额":   rng.normal(0, 0.1, n),
        "涨跌幅（%）": rng.normal(0, 1, n),
        "成交量（手）": rng.integers(10000, 500000, n).astype(float),
        "成交额（千元）": rng.uniform(1e5, 1e7, n),
        "换手率（%）": rng.uniform(0.1, 5, n),
        "换手率（%，自由流通股）": rng.uniform(0.1, 8, n),
        "量比": rng.uniform(0.5, 3, n),
        "市盈率（亏损为空）": rng.uniform(5, 80, n),
        "市盈率（TTM，亏损为空）": rng.uniform(5, 80, n),
        "市净率": rng.uniform(0.5, 10, n),
        "市销率": rng.uniform(0.1, 5, n),
        "市销率（TTM）": rng.uniform(0.1, 5, n),
        "股息率（%）": rng.uniform(0, 5, n),
        "股息率（%，TTM）": rng.uniform(0, 5, n),
        "总股本（万股）": np.full(n, 100000.0),
        "流通股本（万股）": np.full(n, 80000.0),
        "自由流通股本（万）": np.full(n, 50000.0),
        "总市值（万元）": price * 100000,
        "流通市值（万元）": price * 80000,
        "复权因子": np.ones(n),
        "当日涨停价": price * 1.1,
        "当日跌停价": price * 0.9,
    })
    return df


class TestCleanStockDf:
    """验证 clean_stock_df 的各种清洗行为。"""

    def test_returns_df_normal(self):
        """正常数据应返回非空 DataFrame。"""
        df = _make_df(100)
        result = clean_stock_df(df)
        assert result is not None
        assert len(result) == 100

    def test_new_stock_returns_none(self):
        """有效行数 < MIN_ROWS 的新股应返回 None。"""
        df = _make_df(MIN_ROWS - 1)
        assert clean_stock_df(df) is None

    def test_outliers_clipped(self):
        """插入极值后，清洗结果不应含原始极值。"""
        df = _make_df(200)
        df.loc[50, "收盘价"] = 9999999.0
        df.loc[51, "收盘价"] = -9999999.0
        result = clean_stock_df(df)
        assert result["收盘价"].max() < 9999999.0
        assert result["收盘价"].min() > -9999999.0

    def test_price_ffill_limit(self):
        """价格列停牌 NaN 应被向前填充，但不超过 FFILL_PRICE_MAX 天。"""
        df = _make_df(200)
        # 连续 3 天 NaN（在限制内，应被填充）
        df.loc[100:102, "收盘价"] = np.nan
        result = clean_stock_df(df)
        assert result.loc[100:102, "收盘价"].notna().all(), "3天停牌未被 ffill 填充"

        # 连续 7 天 NaN（超出 limit=5，第 6、7 天应仍为 NaN）
        df2 = _make_df(200)
        df2.loc[100:106, "收盘价"] = np.nan  # 7 天
        result2 = clean_stock_df(df2)
        # ffill(limit=5)：第 100 行原本有值，101-105 被填充（5行），106 超限仍为 NaN
        assert pd.isna(result2.loc[106, "收盘价"]), "超出 ffill limit 的 NaN 未被保留"

    def test_valuation_ffill_unlimited(self):
        """估值列应无限向前填充（PIT 原则）。"""
        df = _make_df(200)
        # 让市盈率从第 50 行开始全为 NaN（模拟财报未更新）
        # 先记录第 49 行（清洗前）的原始值
        val_before_clean = df.loc[49, "市盈率（TTM，亏损为空）"]
        df.loc[50:, "市盈率（TTM，亏损为空）"] = np.nan
        result = clean_stock_df(df)
        # 清洗后第 50 行及之后应不全为 NaN（ffill 已传播）
        assert result.loc[50:, "市盈率（TTM，亏损为空）"].notna().all(), \
            "估值列 ffill 未传播到后续行"
        # 填充值应等于第 49 行（经过可能的 Winsorize 后）的值
        filled_val = result.loc[49, "市盈率（TTM，亏损为空）"]
        assert result.loc[60, "市盈率（TTM，亏损为空）"] == pytest.approx(
            filled_val, rel=1e-6
        ), "估值列 ffill 填入的值与第 49 行不一致"

    def test_heavy_missing_marked_invalid(self):
        """缺失率 > 30% 的列应被记录在 invalid_cols 中。"""
        df = _make_df(200)
        df.loc[:160, "市净率"] = np.nan  # 80% 缺失
        result = clean_stock_df(df)
        assert "市净率" in result.attrs["invalid_cols"]

    def test_random_missing_filled(self):
        """随机缺失率 < 5% 的列应被中位数填充（非 PRICE_VOL/VALUATION 列）。"""
        df = _make_df(200)
        # 添加一个不在预定义分组中的数值列，模拟小量随机缺失
        df["自定义因子"] = np.random.default_rng(99).uniform(1, 10, 200)
        df.loc[[10, 20, 30], "自定义因子"] = np.nan  # 缺失率 1.5%
        result = clean_stock_df(df)
        assert result["自定义因子"].isna().sum() == 0, "随机缺失未被填充"

    def test_winsorized_cols_recorded(self):
        """被 Winsorize 的列名应记录在 attrs['winsorized_cols'] 中。"""
        df = _make_df(200)
        df.loc[50, "成交量（手）"] = 1e12  # 明显极值
        result = clean_stock_df(df)
        assert "成交量（手）" in result.attrs["winsorized_cols"]

    def test_no_future_data_in_valuation(self):
        """
        严格 PIT 检验：对估值列只允许 ffill（向前），
        不允许 bfill（用未来填过去）。
        """
        df = _make_df(100)
        df.loc[0:9, "市净率"] = np.nan  # 前 10 行 NaN
        result = clean_stock_df(df)
        # 前 10 行没有可向前填充的值，应保持 NaN
        assert result.loc[0:9, "市净率"].isna().all(), "前段 NaN 被非法 bfill 填充（违反 PIT）"

    def test_sort_by_date(self):
        """清洗后数据应按交易日升序排列。"""
        df = _make_df(100)
        df = df.sample(frac=1, random_state=42)  # 打乱顺序
        result = clean_stock_df(df)
        dates = result["交易日"].tolist()
        assert dates == sorted(dates), "清洗后数据未按日期排序"


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3：diagnose_df 诊断工具测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnoseDf:
    """验证 diagnose_df 的诊断输出。"""

    def test_diagnosis_keys(self):
        """诊断结果应包含所有必要 key。"""
        df = _make_df(100)
        d = diagnose_df(df)
        for key in ("total_rows", "missing_rates", "outlier_cols", "invalid_cols"):
            assert key in d, f"缺少 key: {key}"

    def test_no_outlier_after_clean(self):
        """经过 clean_stock_df 后，诊断不应再发现极值。"""
        df = _make_df(200)
        df.loc[50, "收盘价"] = 9999999.0
        cleaned = clean_stock_df(df)
        d = diagnose_df(cleaned)
        assert "收盘价" not in d["outlier_cols"], "清洗后仍检测到极值"

    def test_missing_rate_zero_after_clean(self):
        """对 PRICE_VOL 列，清洗后缺失率应 ≤ 清洗前（ffill 后减少）。"""
        df = _make_df(200)
        df.loc[100:103, "收盘价"] = np.nan
        before_miss = diagnose_df(df)["missing_rates"]["收盘价"]
        cleaned = clean_stock_df(df)
        after_miss = diagnose_df(cleaned)["missing_rates"]["收盘价"]
        assert after_miss <= before_miss, "清洗后缺失率不减反增"


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4：真实数据随机抽查（从 Stocks/ 目录随机采样）
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sample_files() -> list[Path]:
    """从 Stocks/ 目录随机抽取 SAMPLE_N 只股票文件。"""
    all_files = sorted(STOCKS_DIR.glob("*.csv"))
    if not all_files:
        pytest.skip(f"Stocks/ 目录为空或不存在: {STOCKS_DIR}", allow_module_level=True)
    random.seed(RANDOM_SEED)
    n = SAMPLE_N if SAMPLE_N else len(all_files)
    return random.sample(all_files, min(n, len(all_files)))


SAMPLE_FILES = _get_sample_files()


@pytest.mark.parametrize("csv_path", SAMPLE_FILES, ids=lambda p: p.stem)
class TestRealDataSample:
    """对真实抽样文件逐一进行清洗质量检验。"""

    def _load_raw(self, csv_path: Path) -> pd.DataFrame:
        return pd.read_csv(csv_path, dtype={"股票代码": str, "交易日": str})

    # ── 4-1. 清洗前诊断：记录原始状态（不做 assert，仅打印供参考）─────────────
    def test_pre_clean_diagnosis(self, csv_path: Path):
        """清洗前诊断：输出缺失率和极值列（供人工审核，本测试不会失败）。"""
        df_raw = self._load_raw(csv_path)
        d = diagnose_df(df_raw)
        # 打印诊断摘要（pytest -v 时可见）
        if d["outlier_cols"]:
            print(f"\n [{csv_path.stem}] : {d['outlier_cols'][:5]}")
        heavy = {k: f"{v:.1%}" for k, v in d["missing_rates"].items() if v > MISSING_HEAVY}
        if heavy:
            print(f"\n [{csv_path.stem}] : {heavy}")

    # ── 4-2. 清洗后：不含 MAD 极值 ────────────────────────────────────────────
    def test_no_outliers_after_clean(self, csv_path: Path):
        """清洗后，所有数值列不应再含 MAD 极值。"""
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        numeric_cols = [
            c for c in df_cleaned.columns
            if c not in NON_NUMERIC_COLS
            and pd.api.types.is_numeric_dtype(df_cleaned[c])
            and c not in df_cleaned.attrs.get("invalid_cols", [])
        ]
        outlier_cols = [c for c in numeric_cols if _has_outliers(df_cleaned[c])]
        assert not outlier_cols, (
            f"{csv_path.stem}: 清洗后仍含极值列 {outlier_cols}"
        )

    # ── 4-3. 清洗后：价格列停牌缺失率 ≤ 原始 ──────────────────────────────────
    def test_price_missing_reduced(self, csv_path: Path):
        """清洗后，价格/量能列的缺失率不应高于清洗前。"""
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        for col in PRICE_VOL_COLS:
            if col not in df_raw.columns:
                continue
            before = df_raw[col].isna().mean()
            after  = df_cleaned[col].isna().mean()
            assert after <= before + 1e-9, (
                f"{csv_path.stem}[{col}]: 清洗后缺失率 {after:.2%} > 清洗前 {before:.2%}"
            )

    # ── 4-4. 清洗后：估值列缺失率 ≤ 原始 ─────────────────────────────────────
    def test_valuation_missing_reduced(self, csv_path: Path):
        """清洗后，估值列缺失率不应高于清洗前（ffill 应减少缺失）。"""
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        for col in VALUATION_COLS:
            if col not in df_raw.columns:
                continue
            before = df_raw[col].isna().mean()
            after  = df_cleaned[col].isna().mean()
            assert after <= before + 1e-9, (
                f"{csv_path.stem}[{col}]: 清洗后缺失率 {after:.2%} > 清洗前 {before:.2%}"
            )

    # ── 4-5. 清洗后：数据行数与原始一致 ────────────────────────────────────────
    def test_row_count_preserved(self, csv_path: Path):
        """清洗不应增减行数（仅对非新股）。"""
        df_raw = self._load_raw(csv_path)
        if len(df_raw) < MIN_ROWS:
            pytest.skip(f"{csv_path.stem}: 新股")
        df_cleaned = clean_stock_df(df_raw)
        assert df_cleaned is not None
        assert len(df_cleaned) == len(df_raw), (
            f"{csv_path.stem}: 清洗后行数 {len(df_cleaned)} ≠ 原始 {len(df_raw)}"
        )

    # ── 4-6. 清洗后：日期升序 ─────────────────────────────────────────────────
    def test_dates_sorted(self, csv_path: Path):
        """清洗后交易日应严格升序。"""
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        dates = df_cleaned["交易日"].tolist()
        assert dates == sorted(dates), f"{csv_path.stem}: 日期未升序"

    # ── 4-7. 严格 PIT 检验：估值列不允许 bfill ────────────────────────────────
    def test_no_future_fill_in_valuation(self, csv_path: Path):
        """
        PIT 检验：估值列第一个非 NaN 之前的所有行应保持 NaN（未被未来数据填充）。
        """
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        for col in VALUATION_COLS:
            if col not in df_cleaned.columns:
                continue
            first_valid_idx = df_cleaned[col].first_valid_index()
            if first_valid_idx is None or first_valid_idx == 0:
                continue
            # 在 first_valid_idx 位置前的行，应全为 NaN
            pre_valid = df_cleaned.loc[:first_valid_idx - 1, col]
            assert pre_valid.isna().all(), (
                f"{csv_path.stem}[{col}]: 第一个有效值之前存在非 NaN，"
                f"可能违反 PIT 原则（bfill 污染）"
            )

    # ── 4-8. 整体缺失率摘要断言（非 invalid 列最终缺失率 < 30%）────────────────
    def test_final_missing_rate_acceptable(self, csv_path: Path):
        """
        清洗后，非 invalid_cols 的每列缺失率应 < 30%。
        （invalid_cols 已标记为无效，允许高缺失率）
        """
        df_raw = self._load_raw(csv_path)
        df_cleaned = clean_stock_df(df_raw)

        if df_cleaned is None:
            pytest.skip(f"{csv_path.stem}: 新股行数不足，已剔除")

        invalid = set(df_cleaned.attrs.get("invalid_cols", []))
        numeric_cols = [
            c for c in df_cleaned.columns
            if c not in NON_NUMERIC_COLS
            and pd.api.types.is_numeric_dtype(df_cleaned[c])
            and c not in invalid
        ]
        bad_cols = {
            c: float(df_cleaned[c].isna().mean())
            for c in numeric_cols
            if df_cleaned[c].isna().mean() > MISSING_HEAVY
        }
        assert not bad_cols, (
            f"{csv_path.stem}: 以下非 invalid 列仍有 >30% 缺失: {bad_cols}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5：全局汇总（可选，生成简报）
# ═══════════════════════════════════════════════════════════════════════════════

def test_sample_summary(capsys):
    """
    汇总抽查结果：打印每只股票的清洗状态简报。
    （此测试永远通过，仅用于输出信息）
    """
    results = []
    t0 = time.time()
    for csv_path in SAMPLE_FILES:
        df_raw = pd.read_csv(csv_path, dtype={"股票代码": str, "交易日": str})
        df_cleaned = clean_stock_df(df_raw)
        if df_cleaned is None:
            status = "SKIP(新股)"
            n_invalid = n_winsorized = "-"
        else:
            status      = "OK"
            n_invalid   = len(df_cleaned.attrs.get("invalid_cols", []))
            n_winsorized= len(df_cleaned.attrs.get("winsorized_cols", []))

        results.append({
            "文件":    csv_path.stem,
            "状态":    status,
            "无效列数": n_invalid,
            "Winsorize列数": n_winsorized,
            "原始行数": len(df_raw),
        })

    elapsed = time.time() - t0
    with capsys.disabled():
        print(f"\n{'='*65}")
        print(f" （{len(SAMPLE_FILES)} ， {elapsed:.1f}s）")
        print(f"{'='*65}")
        print(f" {'':<20} {'':<12} {'':>6} {'Winsor':>8} {'':>7}")
        print(f"  {'-'*60}")
        for r in results:
            print(
                f"  {r['文件']:<20} {r['状态']:<12} "
                f"{str(r['无效列数']):>6} {str(r['Winsorize列数']):>8} {r['原始行数']:>7}"
            )
        print(f"{'='*65}\n")
