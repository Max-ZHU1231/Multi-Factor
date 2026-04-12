"""
factor_framework.factors.transform  [v4.0 COMPATIBILITY SHIM]
===============================================================
⚠️  TransformPipeline 已迁移至 factor_framework.transform（v4.0）。
    旧路径将在 v4.2 移除，请更新 import：

    旧：from factor_framework.factors.transform import TransformPipeline
    新：from factor_framework.transform import TransformPipeline

TransformPipeline —— 可组合的横截面变换管道。
支持链式调用：pipe.winsorize().neutralize(...).standardize('rank')
"""
from __future__ import annotations

import warnings
import warnings as _warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

_warnings.warn(
    "factor_framework.factors.transform has moved to factor_framework.transform. "
    "The legacy import path will be removed in v4.2.",
    DeprecationWarning,
    stacklevel=2,
)

import pandas as pd


# 步骤类型别名
_StepFn = Callable[[pd.DataFrame], pd.DataFrame]


class TransformPipeline:
    """
    横截面变换管道：winsorize → neutralize → standardize。

    Parameters
    ----------
    engine : FactorEngine 实例（用于 apply_cross_section 和 build_panel）。
             若仅使用 winsorize / standardize，engine 可为 None。
    """

    def __init__(self, engine=None) -> None:
        self._engine = engine
        self._steps:  List[Tuple[str, _StepFn]] = []

    # ── 步骤注册（流式 API）──────────────────────────────────────────────────

    def winsorize(self, n_std: float = 3.0) -> "TransformPipeline":
        """
        添加截面 MAD Winsorize 步骤（逐行 ±n_std×MAD 截尾）。

        Parameters
        ----------
        n_std : 截尾阈值（MAD 倍数，默认 3.0）
        """
        from factor_framework.operators import cs_winsorize

        def _step(panel: pd.DataFrame) -> pd.DataFrame:
            if self._engine is not None:
                return self._engine.apply_cross_section(panel, cs_winsorize)
            # 无 engine：纯向量化回退
            import numpy as np
            arr  = panel.values.astype(float)
            med  = pd.DataFrame(arr, index=panel.index, columns=panel.columns).median(axis=1)
            mad_ = (panel.sub(med, axis=0)).abs().median(axis=1)
            lower = med - n_std * mad_
            upper = med + n_std * mad_
            clipped = panel.clip(lower=lower, upper=upper, axis=0)
            return clipped

        self._steps.append(("winsorize", _step))
        return self

    def neutralize(
        self,
        mktcap_panel:  Optional[pd.DataFrame] = None,
        industry_map:  Optional[pd.Series] = None,
    ) -> "TransformPipeline":
        """
        添加市值+行业中性化步骤（OLS 残差）。

        Parameters
        ----------
        mktcap_panel : (日期 × 股票) 市值面板；若为 None 则仅做行业中性化
        industry_map : ts_code → industry 的 Series；若为 None 则仅做市值中性化
        """
        if mktcap_panel is None and industry_map is None:
            warnings.warn(
                "[TransformPipeline.neutralize] mktcap_panel 和 industry_map 均为 None，"
                "中性化步骤无效，已忽略。",
                stacklevel=2,
            )
            return self

        from factor_framework.neutralize import neutralize_regression

        def _step(panel: pd.DataFrame) -> pd.DataFrame:
            mp = mktcap_panel
            if mp is not None:
                mp = mp.reindex(panel.index)
            return neutralize_regression(panel, mp, industry_map=industry_map)

        self._steps.append(("neutralize", _step))
        return self

    def standardize(self, method: str = "rank") -> "TransformPipeline":
        """
        添加横截面标准化步骤。

        Parameters
        ----------
        method : 'rank'（Rank 标准化，[0,1]）| 'zscore'（均值 0 方差 1）| None（跳过）
        """
        if method is None:
            return self

        from factor_framework.operators import cs_rank, cs_zscore

        if method == "rank":
            cs_fn = cs_rank
        elif method == "zscore":
            cs_fn = cs_zscore
        else:
            raise ValueError(
                f"[TransformPipeline.standardize] method 必须为 'rank' 或 'zscore'，"
                f"收到 {method!r}。"
            )

        def _step(panel: pd.DataFrame) -> pd.DataFrame:
            if self._engine is not None:
                return self._engine.apply_cross_section(panel, cs_fn)
            return panel.rank(axis=1, pct=True, na_option="keep") \
                if method == "rank" else \
                panel.sub(panel.mean(axis=1), axis=0).div(
                    panel.std(axis=1, ddof=1).replace(0, float("nan")), axis=0
                )

        self._steps.append(("standardize", _step))
        return self

    def register_step(self, name: str, fn: _StepFn) -> "TransformPipeline":
        """
        注册自定义变换步骤（任意 panel → panel 函数）。

        Parameters
        ----------
        name : 步骤名称（用于调试输出）
        fn   : (pd.DataFrame) -> pd.DataFrame
        """
        self._steps.append((name, fn))
        return self

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        按顺序执行所有已注册步骤，返回变换后面板。

        Parameters
        ----------
        panel : (日期 × 股票) 因子面板（原始值）

        Returns
        -------
        pd.DataFrame —— 经所有步骤处理后的面板（不修改原始 panel）
        """
        result = panel
        for name, fn in self._steps:
            try:
                result = fn(result)
            except Exception as exc:
                warnings.warn(
                    f"[TransformPipeline] 步骤 '{name}' 执行失败：{exc}，"
                    "已跳过该步骤，继续后续处理。",
                    stacklevel=2,
                )
        return result

    # ── 查看已注册步骤 ────────────────────────────────────────────────────────

    @property
    def step_names(self) -> List[str]:
        """返回已注册步骤的名称列表（有序）。"""
        return [name for name, _ in self._steps]

    def __repr__(self) -> str:
        steps = " → ".join(self.step_names) if self._steps else "(空)"
        return f"TransformPipeline([{steps}])"

    def __len__(self) -> int:
        return len(self._steps)
