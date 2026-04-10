#!/usr/bin/env python
"""
scripts/run_validation.py
=========================
一键验证阶段二 DoD（Definition of Done）收敛状态。

用法
----
    python scripts/run_validation.py

功能
----
运行以下五个 DoD 验证维度，打印摘要并在任意失败时以非零退出码退出：

    B1  TimestampedPanel 语义守卫（compute_ic / layer_backtest 入口）
    B2  DataStore → PanelBuilder → FactorPipeline 接线
    B3  ic_decay price_panel 回退路径 DeprecationWarning
    B4  cache_dir 默认值 'cache/' / CacheLayer 开箱启用
    C1  FactorEngine 直接实例化 DeprecationWarning

退出码
------
    0  全部通过
    1  有失败项（控制台输出失败详情）
"""

from __future__ import annotations

import sys
import traceback
import warnings
import pathlib

# ── 确保从任意目录运行时都能找到 factor_framework ────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════════

_results: list[tuple[str, bool, str]] = []


def _check(label: str, fn):
    """运行单个检查，捕获异常并记录结果。"""
    try:
        fn()
        _results.append((label, True, ""))
        print(f"  ✓  {label}")
    except Exception as e:
        tb = traceback.format_exc()
        _results.append((label, False, tb))
        print(f"  ✗  {label}")
        print(f"     {e}")


def _make_panels(n=60, n_stocks=15, seed=0):
    rng = np.random.default_rng(seed)
    dates   = pd.date_range("20200101", periods=n, freq="B").strftime("%Y%m%d")
    stocks  = [f"S{i:02d}" for i in range(n_stocks)]
    factor  = pd.DataFrame(rng.standard_normal((n, n_stocks)), index=dates, columns=stocks)
    returns = pd.DataFrame(rng.standard_normal((n, n_stocks)) * 0.01, index=dates, columns=stocks)
    return factor, returns


# ═══════════════════════════════════════════════════════════════════════════════
# B1: TimestampedPanel 守卫
# ═══════════════════════════════════════════════════════════════════════════════

def _check_b1_guard_compute_ic():
    from factor_framework.core.panel import TimestampedPanel, TimingAlignmentError
    from factor_framework.ic_analysis import compute_ic

    fp, rp = _make_panels()
    fp_ts = TimestampedPanel.from_dataframe(fp, semantic="factor_observation", factor_name="f")
    rp_ts = TimestampedPanel.from_dataframe(rp, semantic="forward_return", forward_days=21)
    # 未 T+1 shift → 应抛出 TimingAlignmentError
    raised = False
    try:
        compute_ic(fp_ts, rp_ts)
    except TimingAlignmentError:
        raised = True
    assert raised, "compute_ic 未对 unshifted factor×forward_return 抛出 TimingAlignmentError"


def _check_b1_guard_layer_backtest():
    from factor_framework.core.panel import TimestampedPanel, TimingAlignmentError
    from factor_framework.backtest import layer_backtest

    fp, rp = _make_panels()
    fp_ts = TimestampedPanel.from_dataframe(fp, semantic="factor_observation", factor_name="f")
    rp_ts = TimestampedPanel.from_dataframe(rp, semantic="forward_return", forward_days=21)
    raised = False
    try:
        layer_backtest(fp_ts, rp_ts)
    except TimingAlignmentError:
        raised = True
    assert raised, "layer_backtest 未对 unshifted factor×forward_return 抛出 TimingAlignmentError"


def _check_b1_valid_pair_passes():
    from factor_framework.core.panel import TimestampedPanel
    from factor_framework.ic_analysis import compute_ic

    fp, rp = _make_panels()
    fp_ts = TimestampedPanel.from_dataframe(fp, semantic="factor_observation", factor_name="f")
    fp_ts = fp_ts.shift_to_t1()
    rp_ts = TimestampedPanel.from_dataframe(rp, semantic="forward_return", forward_days=21)
    ic = compute_ic(fp_ts, rp_ts)
    assert isinstance(ic, pd.Series), "compute_ic 未返回 pd.Series"


# ═══════════════════════════════════════════════════════════════════════════════
# B2: DataStore 接线
# ═══════════════════════════════════════════════════════════════════════════════

def _check_b2_panel_builder_accepts_store():
    import tempfile, pathlib
    from factor_framework.engine.panel_builder import PanelBuilder
    from factor_framework.data.store import CSVDataStore

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        store = CSVDataStore(stocks_dir=str(stocks_dir))
        builder = PanelBuilder(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            store       = store,
        )
        assert builder.store is store, "PanelBuilder.store 未保存传入的 store"


def _check_b2_pipeline_auto_constructs_csvdatastore():
    import tempfile, pathlib
    from factor_framework.pipeline import FactorPipeline
    from factor_framework.data.store import CSVDataStore

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            verbose     = False,
            cache_dir   = None,
        )
        assert pipe._builder.store is not None, "FactorPipeline 未自动构造 CSVDataStore"
        assert isinstance(pipe._builder.store, CSVDataStore)


def _check_b2_pipeline_accepts_custom_store():
    import tempfile, pathlib
    from factor_framework.pipeline import FactorPipeline
    from factor_framework.data.store import CSVDataStore

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        custom = CSVDataStore(stocks_dir=str(stocks_dir))
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            verbose     = False,
            cache_dir   = None,
            store       = custom,
        )
        assert pipe._builder.store is custom, "FactorPipeline 未将 custom store 传给 PanelBuilder"


# ═══════════════════════════════════════════════════════════════════════════════
# B3: ic_decay 回退路径 DeprecationWarning
# ═══════════════════════════════════════════════════════════════════════════════

def _check_b3_price_panel_deprecation_warning():
    from factor_framework.ic_analysis import ic_decay
    fp, price = _make_panels()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ic_decay(fp, price_panel=price, forward_periods=[1, 5])

    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) >= 1, "ic_decay price_panel 路径未发出 DeprecationWarning"
    msg = str(dep[0].message)
    assert "price_panel" in msg or "废弃" in msg, f"DeprecationWarning 消息不含关键词: {msg!r}"


def _check_b3_return_panels_no_deprecation():
    from factor_framework.ic_analysis import ic_decay
    fp, price = _make_panels()
    ret = price.shift(-5) / price.replace(0, np.nan) - 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ic_decay(fp, return_panels={5: ret})

    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 0, f"ic_decay return_panels 路径不应发出 DeprecationWarning，但收到 {len(dep)} 个"


# ═══════════════════════════════════════════════════════════════════════════════
# B4: cache_dir 默认值
# ═══════════════════════════════════════════════════════════════════════════════

def _check_b4_default_cache_dir_creates_cache_layer():
    import tempfile, pathlib
    from factor_framework.pipeline import FactorPipeline
    from factor_framework.engine.cache import CacheLayer

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            verbose     = False,
            # cache_dir 使用默认值 "cache/"
        )
        assert pipe._builder.cache is not None, "默认 cache_dir 未创建 CacheLayer（期望自动启用）"
        assert isinstance(pipe._builder.cache, CacheLayer)


def _check_b4_explicit_none_disables_cache():
    import tempfile, pathlib
    from factor_framework.pipeline import FactorPipeline

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        pipe = FactorPipeline(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            verbose     = False,
            cache_dir   = None,
        )
        assert pipe._builder.cache is None, "cache_dir=None 时 _builder.cache 应为 None"


# ═══════════════════════════════════════════════════════════════════════════════
# C1: FactorEngine 直接实例化 DeprecationWarning
# ═══════════════════════════════════════════════════════════════════════════════

def _check_c1_factor_engine_deprecation():
    import tempfile, pathlib
    from factor_framework.factor_engine import FactorEngine

    FactorEngine._deprecation_warned = False  # 重置 flag

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FactorEngine(stocks_dir=str(tmp), stock_basic=str(tmp / "none.csv"), verbose=False)

    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) >= 1, "直接实例化 FactorEngine 未发出 DeprecationWarning"


def _check_c1_internal_flag_suppresses():
    import tempfile, pathlib
    from factor_framework.factor_engine import FactorEngine

    FactorEngine._deprecation_warned = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FactorEngine(
                stocks_dir  = str(tmp),
                stock_basic = str(tmp / "none.csv"),
                verbose     = False,
                _internal   = True,
            )

    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 0, "_internal=True 时不应发出 DeprecationWarning"


def _check_c1_panel_builder_no_deprecation():
    import tempfile, pathlib
    from factor_framework.engine.panel_builder import PanelBuilder
    from factor_framework.factor_engine import FactorEngine

    FactorEngine._deprecation_warned = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        stocks_dir = tmp / "Stocks"
        stocks_dir.mkdir()
        builder = PanelBuilder(
            stocks_dir  = str(stocks_dir),
            stock_basic = str(tmp / "none.csv"),
            verbose     = False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = builder.engine  # 触发延迟初始化

    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 0, "PanelBuilder.engine 不应触发 DeprecationWarning"


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 64)
    print("  阶段二 DoD 验证")
    print("=" * 64)

    print("\n【B1】TimestampedPanel 语义守卫")
    _check("B1-1  compute_ic 拦截 unshifted forward_return",  _check_b1_guard_compute_ic)
    _check("B1-2  layer_backtest 拦截 unshifted forward_return", _check_b1_guard_layer_backtest)
    _check("B1-3  合法 T+1 shifted 配对正常通过",              _check_b1_valid_pair_passes)

    print("\n【B2】DataStore 接线")
    _check("B2-1  PanelBuilder 接受 store 参数",               _check_b2_panel_builder_accepts_store)
    _check("B2-2  FactorPipeline 自动构造 CSVDataStore",       _check_b2_pipeline_auto_constructs_csvdatastore)
    _check("B2-3  FactorPipeline 接受并透传 custom store",     _check_b2_pipeline_accepts_custom_store)

    print("\n【B3】ic_decay DeprecationWarning")
    _check("B3-1  price_panel 回退路径发出 DeprecationWarning", _check_b3_price_panel_deprecation_warning)
    _check("B3-2  return_panels 主路径无 DeprecationWarning",   _check_b3_return_panels_no_deprecation)

    print("\n【B4】cache_dir 默认值")
    _check("B4-1  默认 cache_dir='cache/' 自动创建 CacheLayer", _check_b4_default_cache_dir_creates_cache_layer)
    _check("B4-2  cache_dir=None 禁用缓存",                    _check_b4_explicit_none_disables_cache)

    print("\n【C1】FactorEngine 废弃警告")
    _check("C1-1  直接实例化发出 DeprecationWarning",           _check_c1_factor_engine_deprecation)
    _check("C1-2  _internal=True 抑制 DeprecationWarning",     _check_c1_internal_flag_suppresses)
    _check("C1-3  PanelBuilder.engine 不触发 DeprecationWarning", _check_c1_panel_builder_no_deprecation)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    print(f"\n{'=' * 64}")
    print(f"  结果: {passed}/{total} 通过")
    print("=" * 64)

    if passed < total:
        print("\n【失败详情】")
        for label, ok, tb in _results:
            if not ok:
                print(f"\n  ✗ {label}")
                print(tb)
        sys.exit(1)
    else:
        print("\n  ✓ 全部通过 — 阶段二 DoD 收敛！")
        sys.exit(0)


if __name__ == "__main__":
    main()
