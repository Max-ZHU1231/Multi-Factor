"""
advanced_diagnostics.py
=======================
Advanced Diagnostics Pack (Phase 1 scaffold + core implemented modules).

This module provides a standardized output contract for advanced factor
diagnostics. It currently implements:
  - Fama-MacBeth + Newey-West statistics
  - alpha regressions (CAPM / FF3 proxy)
  - monotonicity significance tests
  - turnover boundary effect
  - transaction cost sensitivity scenarios
  - long/short intensity

All remaining diagnostics are emitted as structured placeholder files to keep
the report schema stable while incremental modules are added.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


_OUTPUT_FILES: List[str] = [
    "fm_summary.csv",
    "alpha_models.csv",
    "monotonicity_tests.csv",
    "turnover_boundary.csv",
    "neutralization_comparison.csv",
    "size_bucket_report.csv",
    "cost_scenarios.csv",
    "turnover_attribution.csv",
    "factor_corr_matrix.csv",
    "orthogonal_ic.csv",
    "rolling_oos.csv",
    "param_sensitivity.csv",
    "regime_stability.csv",
    "extreme_value_impact.csv",
    "long_short_intensity.csv",
]


@dataclass
class AdvancedDiagnosticsReport:
    """Container for advanced diagnostics run metadata."""

    factor_name: str
    output_dir: Path
    module_status: Dict[str, str] = field(default_factory=dict)
    timing_rows: List[Dict[str, object]] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)

    def add_timing(self, module: str, elapsed_sec: float) -> None:
        self.timing_rows.append({"module": module, "elapsed_sec": round(elapsed_sec, 4)})

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.timing_rows:
            pd.DataFrame(self.timing_rows).to_csv(self.output_dir / "timing.csv", index=False)
        with open(self.output_dir / "advanced_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2)


def _resolve_nw_lags(n: int, rule: str = "t_pow_0.25") -> int:
    """Resolve Newey-West lag count from a rule string."""
    if n <= 0:
        return 0
    if rule == "t_pow_0.25":
        return max(1, int(n ** 0.25))
    if isinstance(rule, str) and rule.startswith("fixed_"):
        try:
            k = int(rule.split("_", 1)[1])
            return max(1, k)
        except Exception:
            return max(1, int(n ** 0.25))
    return max(1, int(n ** 0.25))


def _series_t_stats(x: pd.Series, nw_lag_rule: str = "t_pow_0.25") -> Dict[str, float]:
    clean = x.dropna().astype(float)
    n = len(clean)
    if n < 2:
        return {
            "mean": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "nw_t_stat": np.nan,
            "nw_p_value": np.nan,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if not np.isnan(t_stat) else np.nan
    # Reuse Newey-West style estimate used in IC path
    lags = _resolve_nw_lags(n, nw_lag_rule)
    e = clean.values - mean
    var_nw = float(np.sum(e ** 2) / n)
    for k in range(1, lags + 1):
        cov_k = float(np.sum(e[k:] * e[:-k]) / n)
        var_nw += 2.0 * (1.0 - k / (lags + 1)) * cov_k
    se_nw = np.sqrt(var_nw / n) if var_nw > 0 else np.nan
    nw_t = mean / se_nw if (se_nw is not None and se_nw > 0) else np.nan
    nw_p = float(2 * (1 - stats.t.cdf(abs(nw_t), df=n - 1))) if not np.isnan(nw_t) else np.nan
    return {"mean": mean, "t_stat": t_stat, "p_value": p_val, "nw_t_stat": nw_t, "nw_p_value": nw_p}


def _build_group_memberships(factor_panel: pd.DataFrame, n_groups: int, direction: int) -> Dict[str, Dict[str, set]]:
    memberships: Dict[str, Dict[str, set]] = {}
    for date in factor_panel.index:
        row = factor_panel.loc[date].dropna()
        if len(row) < n_groups * 2:
            continue
        try:
            labels = pd.qcut(
                row * direction,
                n_groups,
                labels=[f"Q{i+1}" for i in range(n_groups)],
                duplicates="drop",
            )
        except Exception:
            continue
        groups: Dict[str, set] = {}
        for i in range(1, n_groups + 1):
            g = f"Q{i}"
            groups[g] = set(labels[labels == g].index.tolist())
        memberships[str(date)] = groups
    return memberships


def _turnover_between(prev_set: set, curr_set: set) -> float:
    if not prev_set or not curr_set:
        return np.nan
    n_total = len(prev_set | curr_set)
    if n_total == 0:
        return np.nan
    return len(prev_set.symmetric_difference(curr_set)) / (2.0 * n_total)


def _write_placeholder(path: Path, module_name: str, note: str) -> None:
    df = pd.DataFrame([{
        "module": module_name,
        "status": "not_implemented",
        "note": note,
    }])
    df.to_csv(path, index=False)


def _run_fama_macbeth(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    mktcap_panel: Optional[pd.DataFrame] = None,
    industry_map: Optional[pd.Series] = None,
    nw_lag_rule: str = "t_pow_0.25",
) -> pd.DataFrame:
    """
    Fama-MacBeth cross-sectional regression with optional controls.

    Model per period t:
      r_{i,t+1} = a_t + b_t * zfactor_{i,t} + c_t * zlog_mktcap_{i,t}
                  + industry_dummies + e_{i,t}

    We aggregate the lambda_t (= b_t) series with plain t and Newey-West t.
    """
    common_dates = factor_panel.index.intersection(return_panel.index)
    lambda_ts: List[float] = []
    cs_nobs: List[int] = []
    cs_r2: List[float] = []

    use_mktcap = isinstance(mktcap_panel, pd.DataFrame) and (mktcap_panel is not None)
    use_industry = isinstance(industry_map, pd.Series) and (industry_map is not None) and (len(industry_map.dropna()) > 0)

    for dt in common_dates:
        y = return_panel.loc[dt]
        f = factor_panel.loc[dt]
        df = pd.DataFrame({"ret": y, "factor": f}).dropna()
        if df.empty:
            continue

        # factor z-score so lambda is interpretable as 1-sigma return impact
        f_std = float(df["factor"].std(ddof=1))
        if f_std <= 0 or np.isnan(f_std):
            continue
        df["factor_z"] = (df["factor"] - df["factor"].mean()) / f_std

        x_parts: List[pd.DataFrame] = [pd.DataFrame({"factor_z": df["factor_z"]}, index=df.index)]

        if use_mktcap and dt in mktcap_panel.index:
            m = mktcap_panel.loc[dt].reindex(df.index)
            log_m = np.log1p(m.replace([np.inf, -np.inf], np.nan))
            m_df = pd.DataFrame({"log_mktcap": log_m}).dropna()
            df = df.loc[df.index.intersection(m_df.index)]
            if len(df) == 0:
                continue
            m_std = float(m_df.loc[df.index, "log_mktcap"].std(ddof=1))
            if m_std > 0 and not np.isnan(m_std):
                m_df = m_df.loc[df.index]
                m_df["log_mktcap_z"] = (m_df["log_mktcap"] - m_df["log_mktcap"].mean()) / m_std
                x_parts.append(pd.DataFrame({"log_mktcap_z": m_df["log_mktcap_z"]}, index=df.index))

        if use_industry:
            inds = industry_map.reindex(df.index).astype("object")
            dummies = pd.get_dummies(inds, prefix="ind", drop_first=True, dtype=float)
            if not dummies.empty:
                dummies = dummies.loc[df.index]
                x_parts.append(dummies)

        X = pd.concat(x_parts, axis=1).replace([np.inf, -np.inf], np.nan)
        xy = pd.concat([df[["ret"]], X], axis=1).dropna()
        if xy.empty:
            continue
        yv = xy["ret"].values.astype(float)
        Xv = xy.drop(columns=["ret"]).values.astype(float)
        # add intercept
        Xv = np.column_stack([np.ones(len(Xv)), Xv])
        n_obs = Xv.shape[0]
        n_params = Xv.shape[1]
        if n_obs <= n_params + 1:
            continue

        coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        y_hat = Xv @ coef
        resid = yv - y_hat
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((yv - np.mean(yv)) ** 2))
        r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

        lambda_ts.append(float(coef[1]))  # factor_z coefficient
        cs_nobs.append(int(n_obs))
        cs_r2.append(r2)

    lambda_series = pd.Series(lambda_ts, dtype=float)
    agg = _series_t_stats(lambda_series, nw_lag_rule=nw_lag_rule)
    n = int(len(lambda_series.dropna()))
    total_periods = int(len(common_dates))
    nw_lags = _resolve_nw_lags(n, nw_lag_rule) if n > 0 else 0

    controls = []
    if use_mktcap:
        controls.append("log_mktcap_z")
    if use_industry:
        controls.append("industry_dummies")

    out = pd.DataFrame([{
        "fm_lambda_mean": agg["mean"],
        "fm_t": agg["t_stat"],
        "fm_nw_t": agg["nw_t_stat"],
        "fm_nw_p": agg["nw_p_value"],
        "fm_p": agg["p_value"],
        "fm_lambda_std": float(lambda_series.std(ddof=1)) if n > 1 else np.nan,
        "n_periods": n,
        "valid_periods": n,
        "total_periods": total_periods,
        "mean_cs_nobs": float(np.mean(cs_nobs)) if cs_nobs else np.nan,
        "mean_cs_r2": float(np.mean(cs_r2)) if cs_r2 else np.nan,
        "controls_spec": "none" if not controls else "+".join(controls),
        "nw_lags": nw_lags,
        "is_placeholder": False,
    }])
    return out


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance matrix (Newey-West) for OLS coefficients."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    n, k = X.shape
    s = np.zeros((k, k), dtype=float)

    for t in range(n):
        xt = X[t:t + 1].T
        s += (resid[t] ** 2) * (xt @ xt.T)

    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        gamma = np.zeros((k, k), dtype=float)
        for t in range(l, n):
            xt = X[t:t + 1].T
            xlag = X[t - l:t - l + 1].T
            gamma += resid[t] * resid[t - l] * (xt @ xlag.T)
        s += w * (gamma + gamma.T)

    return xtx_inv @ s @ xtx_inv


def _run_alpha_models(
    layer_ret: pd.DataFrame,
    return_panel: pd.DataFrame,
    mktcap_panel: Optional[pd.DataFrame] = None,
    value_panel: Optional[pd.DataFrame] = None,
    nw_lag_rule: str = "t_pow_0.25",
) -> pd.DataFrame:
    """Run CAPM and FF3 time-series alpha regressions on long-short returns."""
    if "LS" not in layer_ret.columns:
        y = pd.Series(dtype=float)
    else:
        y = layer_ret["LS"].astype(float)

    mkt = return_panel.mean(axis=1, skipna=True).rename("mkt") if isinstance(return_panel, pd.DataFrame) else pd.Series(dtype=float, name="mkt")

    # SMB proxy from market-cap split if available; otherwise NaN series.
    smb = pd.Series(index=mkt.index, dtype=float, name="smb")
    if isinstance(mktcap_panel, pd.DataFrame) and not mktcap_panel.empty:
        for dt in mkt.index.intersection(mktcap_panel.index):
            if dt not in return_panel.index:
                continue
            r = return_panel.loc[dt].dropna()
            mc = mktcap_panel.loc[dt].reindex(r.index).dropna()
            common = r.index.intersection(mc.index)
            if len(common) < 10:
                continue
            r = r.loc[common]
            mc = mc.loc[common]
            q = mc.rank(pct=True)
            small = r[q <= 0.3]
            big = r[q >= 0.7]
            if len(small) > 0 and len(big) > 0:
                smb.loc[dt] = float(small.mean() - big.mean())

    # HML (value) should be constructed from independent value exposure.
    # If value data is unavailable, keep HML empty and mark FF2 fallback in model_note.
    hml = pd.Series(index=mkt.index, dtype=float, name="hml")
    if isinstance(value_panel, pd.DataFrame) and not value_panel.empty:
        for dt in mkt.index.intersection(value_panel.index):
            if dt not in return_panel.index:
                continue
            r = return_panel.loc[dt].dropna()
            v = value_panel.loc[dt].reindex(r.index).dropna()
            common = r.index.intersection(v.index)
            if len(common) < 10:
                continue
            r = r.loc[common]
            v = v.loc[common]
            q = v.rank(pct=True)
            high = r[q >= 0.7]
            low = r[q <= 0.3]
            if len(high) > 0 and len(low) > 0:
                hml.loc[dt] = float(high.mean() - low.mean())

    def _fit_model(model: str, factor_names: List[str], factors_df: pd.DataFrame, model_note: str = "") -> Dict[str, object]:
        data = pd.concat([y.rename("y"), factors_df], axis=1).dropna()
        n = len(data)
        p = len(factor_names) + 1  # intercept
        if n <= p + 1:
            return {
                "model": model,
                "alpha": np.nan,
                "alpha_t": np.nan,
                "alpha_nw_t": np.nan,
                "alpha_p": np.nan,
                "alpha_nw_p": np.nan,
                "beta_mkt": np.nan,
                "beta_smb": np.nan,
                "beta_hml": np.nan,
                "r2": np.nan,
                "n_obs": n,
                "nw_lags": _resolve_nw_lags(n, nw_lag_rule) if n > 0 else 0,
                "model_note": model_note,
                "is_placeholder": False,
            }

        yv = data["y"].values.astype(float)
        xv = data[factor_names].values.astype(float)
        X = np.column_stack([np.ones(n), xv])
        coef, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        y_hat = X @ coef
        resid = yv - y_hat

        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((yv - np.mean(yv)) ** 2))
        r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

        dof = n - X.shape[1]
        sigma2 = sse / dof if dof > 0 else np.nan
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv * sigma2)) if not np.isnan(sigma2) else np.full(X.shape[1], np.nan)
        tvals = coef / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=dof))

        nw_lags = _resolve_nw_lags(n, nw_lag_rule)
        nw_cov = _newey_west_cov(X, resid, nw_lags)
        nw_se = np.sqrt(np.diag(nw_cov))
        nw_t = coef / nw_se
        nw_p = 2 * (1 - stats.t.cdf(np.abs(nw_t), df=dof))

        betas = {name: coef[i + 1] for i, name in enumerate(factor_names)}
        return {
            "model": model,
            "alpha": float(coef[0]),
            "alpha_t": float(tvals[0]),
            "alpha_nw_t": float(nw_t[0]),
            "alpha_p": float(pvals[0]),
            "alpha_nw_p": float(nw_p[0]),
            "beta_mkt": float(betas.get("mkt", np.nan)),
            "beta_smb": float(betas.get("smb", np.nan)),
            "beta_hml": float(betas.get("hml", np.nan)),
            "r2": float(r2),
            "n_obs": int(n),
            "nw_lags": int(nw_lags),
            "model_note": model_note,
            "is_placeholder": False,
        }

    capm_df = pd.DataFrame([_fit_model("CAPM", ["mkt"], pd.concat([mkt], axis=1), model_note="CAPM")])
    has_hml = int(hml.dropna().shape[0]) >= 12
    if has_hml:
        ff3_df = pd.DataFrame([_fit_model("FF3", ["mkt", "smb", "hml"], pd.concat([mkt, smb, hml], axis=1), model_note="FF3")])
    else:
        ff3_df = pd.DataFrame([_fit_model("FF3", ["mkt", "smb"], pd.concat([mkt, smb], axis=1), model_note="FF2(no_hml_data)")])
        ff3_df["beta_hml"] = np.nan
    return pd.concat([capm_df, ff3_df], ignore_index=True)


def _run_orthogonal_ic(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    mktcap_panel: Optional[pd.DataFrame] = None,
    nw_lag_rule: str = "t_pow_0.25",
) -> pd.DataFrame:
    """Residual IC after cross-sectional orthogonalization against control factor."""
    # Raw IC time series
    from factor_framework.analytics.ic_analysis import compute_ic

    raw_ic_s = compute_ic(factor_panel, return_panel, method="rank")
    raw_stat = _series_t_stats(raw_ic_s, nw_lag_rule=nw_lag_rule)

    if isinstance(mktcap_panel, pd.DataFrame) and not mktcap_panel.empty:
        control_name = "log_mktcap_z"
        common_dates = factor_panel.index.intersection(mktcap_panel.index)
        resid_panel = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns, dtype=float)
        for dt in common_dates:
            f = factor_panel.loc[dt]
            mc = mktcap_panel.loc[dt].reindex(f.index)
            df = pd.DataFrame({"f": f, "mc": np.log1p(mc)}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 10:
                continue
            mc_std = float(df["mc"].std(ddof=1))
            if mc_std <= 0 or np.isnan(mc_std):
                continue
            x = ((df["mc"] - df["mc"].mean()) / mc_std).values.astype(float)
            y = df["f"].values.astype(float)
            X = np.column_stack([np.ones(len(x)), x])
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - (X @ coef)
            resid_panel.loc[dt, df.index] = resid
    else:
        control_name = "none"
        resid_panel = factor_panel.copy()

    resid_ic_s = compute_ic(resid_panel, return_panel, method="rank")
    resid_stat = _series_t_stats(resid_ic_s, nw_lag_rule=nw_lag_rule)

    n_periods = int(len(raw_ic_s.dropna().index.intersection(resid_ic_s.dropna().index)))
    return pd.DataFrame([{
        "raw_ic": raw_stat["mean"],
        "resid_ic": resid_stat["mean"],
        "delta_ic": (resid_stat["mean"] - raw_stat["mean"]) if (not np.isnan(raw_stat["mean"]) and not np.isnan(resid_stat["mean"])) else np.nan,
        "raw_t": raw_stat["t_stat"],
        "resid_t": resid_stat["t_stat"],
        "raw_nw_t": raw_stat["nw_t_stat"],
        "resid_nw_t": resid_stat["nw_t_stat"],
        "control_factor": control_name,
        "n_periods": n_periods,
        "is_placeholder": False,
    }])


def _run_factor_corr_matrix(
    target_factor_panel: pd.DataFrame,
    peer_factor_panels: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    preferred_peers: Optional[List[str]] = None,
    min_cs_nobs: int = 20,
    corr_method: str = "spearman",
    high_corr_threshold: float = 0.7,
    nw_lag_rule: str = "t_pow_0.25",
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute target-vs-peer monthly cross-sectional Spearman correlations.

    Returns
    -------
    long_df:
        one row per peer factor with robust summary stats
    wide_df:
        1-row wide matrix for mean_cs_corr
    unavailable_peers:
        peers missing in input dictionary
    """
    peer_factor_panels = peer_factor_panels or {}
    preferred_peers = preferred_peers or []

    available = sorted(peer_factor_panels.keys())
    unavailable = [p for p in preferred_peers if p not in peer_factor_panels]

    rows: List[Dict[str, object]] = []
    wide_row: Dict[str, object] = {"target_factor": "target"}
    target_idx = pd.Index(target_factor_panel.index)
    wide_row["as_of"] = str(target_idx.max()) if len(target_idx) else ""
    if len(target_idx):
        wide_row["sample_period"] = f"{target_idx.min()}~{target_idx.max()}"
    else:
        wide_row["sample_period"] = ""

    for peer in available:
        panel = peer_factor_panels.get(peer)
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            continue
        common_dates = target_factor_panel.index.intersection(panel.index)
        monthly_corrs: List[float] = []
        monthly_nobs: List[int] = []
        for dt in common_dates:
            a = target_factor_panel.loc[dt]
            b = panel.loc[dt].reindex(a.index)
            df = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(df) < min_cs_nobs:
                continue
            monthly_nobs.append(int(len(df)))
            corr = df["a"].corr(df["b"], method=corr_method)
            if not np.isnan(corr):
                monthly_corrs.append(float(corr))

        cs = pd.Series(monthly_corrs, dtype=float)
        st = _series_t_stats(cs, nw_lag_rule=nw_lag_rule)
        mean_corr = st["mean"]
        rows.append({
            "peer_factor": peer,
            "mean_cs_corr": mean_corr,
            "median_cs_corr": float(cs.median()) if len(cs) else np.nan,
            "std_cs_corr": float(cs.std(ddof=1)) if len(cs) > 1 else np.nan,
            "t_stat": st["t_stat"],
            "nw_t_stat": st["nw_t_stat"],
            "n_periods": int(len(cs)),
            "mean_cs_nobs": float(np.mean(monthly_nobs)) if monthly_nobs else np.nan,
            "min_cs_nobs": int(np.min(monthly_nobs)) if monthly_nobs else np.nan,
            "high_corr_flag": bool(abs(mean_corr) > high_corr_threshold) if not np.isnan(mean_corr) else False,
            "is_placeholder": False,
        })
        wide_row[peer] = mean_corr

    if rows:
        long_df = pd.DataFrame(rows).sort_values("peer_factor").reset_index(drop=True)
    else:
        long_df = pd.DataFrame([{
            "peer_factor": np.nan,
            "mean_cs_corr": np.nan,
            "median_cs_corr": np.nan,
            "std_cs_corr": np.nan,
            "t_stat": np.nan,
            "nw_t_stat": np.nan,
            "n_periods": 0,
            "mean_cs_nobs": np.nan,
            "min_cs_nobs": np.nan,
            "high_corr_flag": False,
            "is_placeholder": False,
        }])

    wide_df = pd.DataFrame([wide_row])
    return long_df, wide_df, unavailable


def _run_size_bucket_report(
    report,
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Size-bucket performance using point-in-time index membership."""
    from factor_framework.analytics.ic_analysis import compute_ic
    from factor_framework.backtest import layer_backtest, long_short_stats

    pit_df = getattr(report, "index_membership_pit", None)
    if not isinstance(pit_df, pd.DataFrame) or pit_df.empty:
        pit_path = getattr(report, "index_membership_pit_path", None)
        candidate_paths = [pit_path, "universes/index_membership_pit.csv", "universes/index_membership.csv"]
        pit_df = None
        for p in candidate_paths:
            if not p:
                continue
            try:
                tmp = pd.read_csv(p, dtype=str)
                if {"trade_date", "index_code", "ts_code"}.issubset(set(tmp.columns)):
                    pit_df = tmp
                    break
            except Exception:
                continue

    has_pit = isinstance(pit_df, pd.DataFrame) and not pit_df.empty and {"trade_date", "index_code", "ts_code"}.issubset(set(pit_df.columns))
    if has_pit:
        pit_df = pit_df.copy()
        pit_df["trade_date"] = pit_df["trade_date"].astype(str)
        pit_df["index_code"] = pit_df["index_code"].astype(str).str.upper()
        pit_df["ts_code"] = pit_df["ts_code"].astype(str)

    bucket_defs = [
        ("hs300", "000300.SH"),
        ("zz500", "000905.SH"),
        ("zz1000", "000852.SH"),
    ]
    rows: List[Dict[str, object]] = []
    source_map: Dict[str, object] = {"schema": ["trade_date", "index_code", "ts_code"], "buckets": []}
    dates = [str(d) for d in factor_panel.index]
    as_of = max(dates) if dates else ""
    sample_period = f"{min(dates)}~{max(dates)}" if dates else ""

    for bucket_type, index_code in bucket_defs:
        coverage_ts: List[float] = []
        row_returns: List[float] = []
        sub_factor_rows: List[pd.Series] = []
        sub_return_rows: List[pd.Series] = []

        if not has_pit:
            rows.append({
                "bucket_type": bucket_type,
                "ic_mean": np.nan,
                "ls_ann": np.nan,
                "ls_sharpe": np.nan,
                "ls_mdd": np.nan,
                "n_periods": 0,
                "coverage_ratio": 0.0,
                "coverage_ratio_mean": 0.0,
                "coverage_ratio_min": 0.0,
                "membership_source": "missing:pit_membership",
                "point_in_time_flag": False,
                "as_of": as_of,
                "sample_period": sample_period,
                "is_placeholder": False,
            })
            source_map["buckets"].append({
                "bucket_type": bucket_type,
                "index_code": index_code,
                "membership_source": "missing:pit_membership",
                "point_in_time_flag": False,
                "coverage_ratio_mean": 0.0,
                "coverage_ratio_min": 0.0,
            })
            continue

        pit_sub = pit_df.loc[pit_df["index_code"] == index_code, ["trade_date", "ts_code"]]
        if pit_sub.empty:
            rows.append({
                "bucket_type": bucket_type,
                "ic_mean": np.nan,
                "ls_ann": np.nan,
                "ls_sharpe": np.nan,
                "ls_mdd": np.nan,
                "n_periods": 0,
                "coverage_ratio": 0.0,
                "coverage_ratio_mean": 0.0,
                "coverage_ratio_min": 0.0,
                "membership_source": f"missing:index_code:{index_code}",
                "point_in_time_flag": True,
                "as_of": as_of,
                "sample_period": sample_period,
                "is_placeholder": False,
            })
            source_map["buckets"].append({
                "bucket_type": bucket_type,
                "index_code": index_code,
                "membership_source": f"missing:index_code:{index_code}",
                "point_in_time_flag": True,
                "coverage_ratio_mean": 0.0,
                "coverage_ratio_min": 0.0,
            })
            continue

        pit_sub = pit_sub.sort_values("trade_date")
        eff_dates = sorted(pit_sub["trade_date"].unique().tolist())
        universe_cols = set(factor_panel.columns)
        groups = pit_sub.groupby("trade_date")["ts_code"].apply(list).to_dict()

        import bisect
        for dt in dates:
            pos = bisect.bisect_right(eff_dates, dt) - 1
            if pos < 0:
                continue
            eff = eff_dates[pos]
            members = [s for s in groups.get(eff, []) if s in universe_cols]
            base_n = len(groups.get(eff, []))
            if base_n <= 0:
                continue
            coverage_ts.append(float(len(members) / base_n))
            if len(members) < 10 or dt not in factor_panel.index or dt not in return_panel.index:
                continue
            f_row = factor_panel.loc[dt, members].dropna()
            r_row = return_panel.loc[dt, members].reindex(f_row.index).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            sub_factor_rows.append(pd.Series(f_row.loc[common].values, index=common, name=dt))
            sub_return_rows.append(pd.Series(r_row.loc[common].values, index=common, name=dt))

        if sub_factor_rows and sub_return_rows:
            f_sub = pd.DataFrame(sub_factor_rows).sort_index()
            r_sub = pd.DataFrame(sub_return_rows).reindex(f_sub.index)
            ic_s = compute_ic(f_sub, r_sub, method="rank").dropna()
            layer = layer_backtest(f_sub, r_sub, n_groups=5, direction=1)
            ls = long_short_stats(layer, periods_per_year=periods_per_year, rf=0.0)
            row_returns = layer["LS"].dropna().tolist() if "LS" in layer.columns else []
        else:
            ic_s = pd.Series(dtype=float)
            ls = {}
        rows.append({
            "bucket_type": bucket_type,
            "ic_mean": float(ic_s.mean()) if len(ic_s) else np.nan,
            "ls_ann": float(ls.get("ls_annual_return", np.nan)),
            "ls_sharpe": float(ls.get("ls_sharpe", np.nan)),
            "ls_mdd": float(ls.get("ls_max_drawdown", np.nan)),
            "n_periods": int(len(ic_s)),
            "coverage_ratio": float(np.mean(coverage_ts)) if coverage_ts else 0.0,
            "coverage_ratio_mean": float(np.mean(coverage_ts)) if coverage_ts else 0.0,
            "coverage_ratio_min": float(np.min(coverage_ts)) if coverage_ts else 0.0,
            "membership_source": "pit:index_membership",
            "point_in_time_flag": True,
            "as_of": as_of,
            "sample_period": sample_period,
            "is_placeholder": False,
        })
        source_map["buckets"].append({
            "bucket_type": bucket_type,
            "index_code": index_code,
            "membership_source": "pit:index_membership",
            "point_in_time_flag": True,
            "coverage_ratio_mean": float(np.mean(coverage_ts)) if coverage_ts else 0.0,
            "coverage_ratio_min": float(np.min(coverage_ts)) if coverage_ts else 0.0,
            "effective_rebalances": int(len(eff_dates)),
            "n_row_returns": int(len(row_returns)),
        })
    return pd.DataFrame(rows), source_map


def _run_rolling_oos(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    *,
    train_periods: int = 24,
    test_periods: int = 1,
) -> pd.DataFrame:
    """Rolling OOS IC diagnostics on monthly panels."""
    from factor_framework.analytics.ic_analysis import compute_ic
    ic = compute_ic(factor_panel, return_panel, method="rank").dropna()
    if len(ic) < train_periods + test_periods + 2:
        return pd.DataFrame([{
            "oos_ic_mean": np.nan,
            "is_ic_mean": np.nan,
            "oos_is_ratio": np.nan,
            "oos_minus_is": np.nan,
            "oos_over_abs_is": np.nan,
            "degradation": np.nan,
            "oos_t": np.nan,
            "oos_nw_t": np.nan,
            "window_train_len": train_periods,
            "window_test_len": test_periods,
            "step": 1,
            "n_windows": 0,
            "status": "insufficient_windows",
            "status_msg": f"insufficient sample for train={train_periods},test={test_periods},step=1",
            "is_placeholder": False,
        }])
    is_vals: List[float] = []
    oos_vals: List[float] = []
    for i in range(train_periods, len(ic) - test_periods + 1):
        train = ic.iloc[i - train_periods:i]
        test = ic.iloc[i:i + test_periods]
        if len(test) == 0:
            continue
        is_vals.append(float(train.mean()))
        oos_vals.append(float(test.mean()))
    is_mean = float(np.mean(is_vals)) if is_vals else np.nan
    oos_mean = float(np.mean(oos_vals)) if oos_vals else np.nan
    ratio = oos_mean / is_mean if (not np.isnan(oos_mean) and not np.isnan(is_mean) and is_mean != 0) else np.nan
    oos_stat = _series_t_stats(pd.Series(oos_vals, dtype=float))
    return pd.DataFrame([{
        "oos_ic_mean": oos_mean,
        "is_ic_mean": is_mean,
        "oos_is_ratio": ratio,
        "oos_minus_is": (oos_mean - is_mean) if (not np.isnan(oos_mean) and not np.isnan(is_mean)) else np.nan,
        "oos_over_abs_is": (oos_mean / abs(is_mean)) if (not np.isnan(oos_mean) and not np.isnan(is_mean) and is_mean != 0) else np.nan,
        "degradation": (oos_mean - is_mean) if (not np.isnan(oos_mean) and not np.isnan(is_mean)) else np.nan,
        "oos_t": oos_stat["t_stat"],
        "oos_nw_t": oos_stat["nw_t_stat"],
        "window_train_len": train_periods,
        "window_test_len": test_periods,
        "step": 1,
        "n_windows": int(len(oos_vals)),
        "status": "ok",
        "status_msg": "ok",
        "is_placeholder": False,
    }])


def _run_regime_stability(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    layer_ret: pd.DataFrame,
) -> pd.DataFrame:
    """Regime stability split by market return tertiles."""
    from factor_framework.analytics.ic_analysis import compute_ic
    if "LS" not in layer_ret.columns or return_panel is None or return_panel.empty or factor_panel is None or factor_panel.empty:
        return pd.DataFrame([{
            "regime": "unknown",
            "ls_mean": np.nan,
            "ls_sharpe": np.nan,
            "ls_mdd": np.nan,
            "ic_mean": np.nan,
            "regime_method": "market_return_tertile",
            "q1": np.nan,
            "q2": np.nan,
            "n_periods": 0,
            "is_placeholder": False,
        }])
    mkt = return_panel.mean(axis=1, skipna=True).rename("mkt")
    ls = layer_ret["LS"].rename("ls")
    df = pd.concat([mkt, ls], axis=1).dropna()
    if len(df) < 12:
        return pd.DataFrame([{
            "regime": "unknown",
            "ls_mean": np.nan,
            "ls_sharpe": np.nan,
            "ls_mdd": np.nan,
            "ic_mean": np.nan,
            "regime_method": "market_return_tertile",
            "q1": np.nan,
            "q2": np.nan,
            "n_periods": int(len(df)),
            "is_placeholder": False,
        }])
    q1 = df["mkt"].quantile(1 / 3)
    q2 = df["mkt"].quantile(2 / 3)
    rows = []
    for name, mask in [
        ("bear", df["mkt"] <= q1),
        ("sideways", (df["mkt"] > q1) & (df["mkt"] < q2)),
        ("bull", df["mkt"] >= q2),
    ]:
        s = df.loc[mask, "ls"]
        d_idx = df.index[mask]
        ic_s = compute_ic(factor_panel.loc[factor_panel.index.intersection(d_idx)],
                          return_panel.loc[return_panel.index.intersection(d_idx)],
                          method="rank").dropna()
        nav = (1 + s.fillna(0)).cumprod()
        mdd = float((nav / nav.cummax() - 1).min()) if len(nav) else np.nan
        rows.append({
            "regime": name,
            "ls_mean": float(s.mean()) if len(s) else np.nan,
            "ls_sharpe": float(s.mean() / (s.std(ddof=1) + 1e-12)) if len(s) > 1 else np.nan,
            "ls_mdd": mdd,
            "ic_mean": float(ic_s.mean()) if len(ic_s) else np.nan,
            "regime_method": "market_return_tertile",
            "q1": float(q1),
            "q2": float(q2),
            "n_periods": int(len(s)),
            "is_placeholder": False,
        })
    return pd.DataFrame(rows)


def _run_param_sensitivity(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    layer_ret: pd.DataFrame,
    mktcap_panel: Optional[pd.DataFrame] = None,
    industry_map: Optional[pd.Series] = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Parameter grid sensitivity using concrete perturbations."""
    from factor_framework.analytics.ic_analysis import compute_ic
    from factor_framework.backtest import layer_backtest, long_short_stats
    from factor_framework.transform.neutralize import neutralize_regression

    def _eval(panel: pd.DataFrame) -> tuple[float, float, int]:
        ic_s = compute_ic(panel, return_panel, method="rank").dropna()
        layer = layer_backtest(panel, return_panel, n_groups=5, direction=1)
        ls = long_short_stats(layer, periods_per_year=periods_per_year, rf=0.0)
        return (
            float(ic_s.mean()) if len(ic_s) else np.nan,
            float(ls.get("ls_sharpe", np.nan)),
            int(len(ic_s)),
        )

    def _winsor_mad(panel: pd.DataFrame, k: float) -> pd.DataFrame:
        out = panel.copy()
        for dt in out.index:
            row = out.loc[dt]
            med = row.median(skipna=True)
            mad = (row - med).abs().median(skipna=True)
            if mad is None or np.isnan(mad) or mad == 0:
                continue
            lo = med - k * 1.4826 * mad
            hi = med + k * 1.4826 * mad
            out.loc[dt] = row.clip(lower=lo, upper=hi)
        return out

    base_ic, base_sharpe, _ = _eval(factor_panel)
    rows = [{
        "param_name": "base",
        "param_value": "base",
        "ic_mean": base_ic,
        "sharpe": base_sharpe,
        "delta_ic": 0.0,
        "delta_sharpe": 0.0,
        "n_periods": int(len(compute_ic(factor_panel, return_panel, method='rank').dropna())),
        "is_placeholder": False,
    }]

    # lookback window perturbation (temporal smoothing proxy)
    for w in [4, 5, 6]:
        fp = factor_panel.rolling(window=w, min_periods=max(2, w // 2)).mean()
        ic_m, sh, n = _eval(fp)
        rows.append({
            "param_name": "lookback_window",
            "param_value": str(w),
            "ic_mean": ic_m,
            "sharpe": sh,
            "delta_ic": (ic_m - base_ic) if (not np.isnan(ic_m) and not np.isnan(base_ic)) else np.nan,
            "delta_sharpe": (sh - base_sharpe) if (not np.isnan(sh) and not np.isnan(base_sharpe)) else np.nan,
            "n_periods": n,
            "is_placeholder": False,
        })

    # winsor threshold perturbation
    for k in [2.4, 3.0, 3.6]:
        fp = _winsor_mad(factor_panel, k=k)
        ic_m, sh, n = _eval(fp)
        rows.append({
            "param_name": "winsor_k",
            "param_value": str(k),
            "ic_mean": ic_m,
            "sharpe": sh,
            "delta_ic": (ic_m - base_ic) if (not np.isnan(ic_m) and not np.isnan(base_ic)) else np.nan,
            "delta_sharpe": (sh - base_sharpe) if (not np.isnan(sh) and not np.isnan(base_sharpe)) else np.nan,
            "n_periods": n,
            "is_placeholder": False,
        })

    # neutralization scheme perturbation
    if isinstance(mktcap_panel, pd.DataFrame) and not mktcap_panel.empty:
        for scheme in ["none", "cap", "cap_ind"]:
            if scheme == "none":
                fp = factor_panel
            elif scheme == "cap":
                fp = neutralize_regression(factor_panel, mktcap_panel, industry_map=None)
            else:
                fp = neutralize_regression(factor_panel, mktcap_panel, industry_map=industry_map)
            ic_m, sh, n = _eval(fp)
            rows.append({
                "param_name": "neutralize_scheme",
                "param_value": scheme,
                "ic_mean": ic_m,
                "sharpe": sh,
                "delta_ic": (ic_m - base_ic) if (not np.isnan(ic_m) and not np.isnan(base_ic)) else np.nan,
                "delta_sharpe": (sh - base_sharpe) if (not np.isnan(sh) and not np.isnan(base_sharpe)) else np.nan,
                "n_periods": n,
                "is_placeholder": False,
            })
    return pd.DataFrame(rows)


def run_advanced_diagnostics(
    report,
    output_dir: str | Path,
    *,
    n_groups: int = 5,
    direction: int = 1,
    periods_per_year: int = 12,
    cost_scenarios: tuple[float, ...] = (0.001, 0.002, 0.003),
    peer_factors: Optional[List[str]] = None,
    min_cs_nobs: int = 20,
    corr_method: str = "spearman",
    high_corr_threshold: float = 0.7,
    nw_lag_rule: str = "t_pow_0.25",
    enable_wide_output: bool = True,
) -> AdvancedDiagnosticsReport:
    """
    Run advanced diagnostics and write standardized outputs.

    Parameters
    ----------
    report:
        FactorReport-like object with required attributes:
        factor_name, layer_ret, factor_panel, return_panel, ls_stats, turnover.
    output_dir:
        Advanced diagnostics output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adv = AdvancedDiagnosticsReport(
        factor_name=getattr(report, "factor_name", "unknown"),
        output_dir=out,
    )

    implemented = set()
    module_notes: Dict[str, str] = {}
    unavailable_peers: List[str] = []

    # 0) Fama-MacBeth + Newey-West t
    t0 = time.perf_counter()
    fm_df = _run_fama_macbeth(
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        mktcap_panel=getattr(report, "mktcap_panel", None),
        industry_map=getattr(report, "industry_map", None),
        nw_lag_rule=nw_lag_rule,
    )
    fm_df.to_csv(out / "fm_summary.csv", index=False)
    implemented.add("fm_summary.csv")
    adv.add_timing("fama_macbeth", time.perf_counter() - t0)

    # 0.5) Alpha models (CAPM / FF3)
    t0 = time.perf_counter()
    alpha_df = _run_alpha_models(
        layer_ret=getattr(report, "layer_ret", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        mktcap_panel=getattr(report, "mktcap_panel", None),
        value_panel=getattr(report, "value_panel", None),
        nw_lag_rule=nw_lag_rule,
    )
    alpha_df.to_csv(out / "alpha_models.csv", index=False)
    implemented.add("alpha_models.csv")
    adv.add_timing("alpha_models", time.perf_counter() - t0)

    # 0.6) Orthogonalized residual IC
    t0 = time.perf_counter()
    orth_df = _run_orthogonal_ic(
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        mktcap_panel=getattr(report, "mktcap_panel", None),
        nw_lag_rule=nw_lag_rule,
    )
    orth_df.to_csv(out / "orthogonal_ic.csv", index=False)
    implemented.add("orthogonal_ic.csv")
    adv.add_timing("orthogonal_ic", time.perf_counter() - t0)

    # 0.7) Factor correlation matrix (target vs peers)
    t0 = time.perf_counter()
    preferred_peers = list(peer_factors) if peer_factors is not None else [
        "value_pb",
        "momentum_12_1",
        "vol_20d",
        "turnover_rate",
        "size_log_mktcap",
    ]
    corr_df, corr_wide_df, unavailable_peers = _run_factor_corr_matrix(
        target_factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        peer_factor_panels=getattr(report, "peer_factor_panels", None),
        preferred_peers=preferred_peers,
        min_cs_nobs=min_cs_nobs,
        corr_method=corr_method,
        high_corr_threshold=high_corr_threshold,
        nw_lag_rule=nw_lag_rule,
    )
    unavailable_from_pipeline = list(getattr(report, "unavailable_peer_factors", []) or [])
    unavailable_peers = sorted(set(unavailable_peers + unavailable_from_pipeline))

    corr_df.to_csv(out / "factor_corr_matrix.csv", index=False)
    if enable_wide_output:
        corr_wide_df.to_csv(out / "factor_corr_matrix_wide.csv", index=False)
    implemented.add("factor_corr_matrix.csv")
    adv.add_timing("factor_corr_matrix", time.perf_counter() - t0)

    # 1) Monotonicity significance
    t0 = time.perf_counter()
    mono_rows = []
    layer_ret = getattr(report, "layer_ret", pd.DataFrame())
    group_cols = [c for c in layer_ret.columns if c.startswith("Q")]
    for dt, row in layer_ret[group_cols].dropna(how="all").iterrows():
        vals = row.dropna()
        if len(vals) < 3:
            continue
        x = np.arange(1, len(vals) + 1)
        y = vals.values.astype(float)
        sp, sp_p = stats.spearmanr(x, y)
        lr = stats.linregress(x, y)
        mono_rows.append({
            "date": dt,
            "spearman": sp,
            "spearman_p": sp_p,
            "trend_slope": lr.slope,
            "trend_p": lr.pvalue,
        })
    mono_df = pd.DataFrame(mono_rows)
    if mono_df.empty:
        mono_summary = pd.DataFrame([{
            "mono_spearman_mean": np.nan,
            "mono_spearman_p": np.nan,
            "trend_slope": np.nan,
            "trend_p": np.nan,
            "n_periods": 0,
        }])
    else:
        mono_summary = pd.DataFrame([{
            "mono_spearman_mean": float(mono_df["spearman"].mean()),
            "mono_spearman_p": float(stats.ttest_1samp(mono_df["spearman"].dropna(), 0.0, nan_policy="omit").pvalue),
            "trend_slope": float(mono_df["trend_slope"].mean()),
            "trend_p": float(stats.ttest_1samp(mono_df["trend_slope"].dropna(), 0.0, nan_policy="omit").pvalue),
            "n_periods": int(len(mono_df)),
        }])
    mono_summary.to_csv(out / "monotonicity_tests.csv", index=False)
    implemented.add("monotonicity_tests.csv")
    adv.add_timing("monotonicity_tests", time.perf_counter() - t0)

    # 2) Turnover boundary effect
    t0 = time.perf_counter()
    memberships = _build_group_memberships(
        getattr(report, "factor_panel", pd.DataFrame()),
        n_groups=n_groups,
        direction=direction,
    )
    sorted_dates = sorted(memberships.keys())
    top_turnovers: List[float] = []
    bottom_turnovers: List[float] = []
    for i in range(1, len(sorted_dates)):
        prev = memberships[sorted_dates[i - 1]]
        curr = memberships[sorted_dates[i]]
        top_turnovers.append(_turnover_between(prev.get(f"Q{n_groups}", set()), curr.get(f"Q{n_groups}", set())))
        bottom_turnovers.append(_turnover_between(prev.get("Q1", set()), curr.get("Q1", set())))
    top_s = pd.Series(top_turnovers).dropna()
    bottom_s = pd.Series(bottom_turnovers).dropna()
    turnover_gap = float(top_s.mean() - bottom_s.mean()) if len(top_s) and len(bottom_s) else np.nan
    instability = bool(abs(turnover_gap) > 0.15) if not np.isnan(turnover_gap) else False
    pd.DataFrame([{
        "turnover_top": float(top_s.mean()) if len(top_s) else np.nan,
        "turnover_bottom": float(bottom_s.mean()) if len(bottom_s) else np.nan,
        "turnover_gap": turnover_gap,
        "turnover_ratio": float(top_s.mean() / bottom_s.mean()) if len(top_s) and len(bottom_s) and bottom_s.mean() != 0 else np.nan,
        "extreme_group_instability_flag": instability,
    }]).to_csv(out / "turnover_boundary.csv", index=False)
    implemented.add("turnover_boundary.csv")
    adv.add_timing("turnover_boundary", time.perf_counter() - t0)

    # 3) Cost sensitivity scenarios
    t0 = time.perf_counter()
    avg_turn = float(getattr(report, "turnover", {}).get("avg_turnover", np.nan))
    ls_ann = float(getattr(report, "ls_stats", {}).get("ls_annual_return", np.nan))
    ls_sh = float(getattr(report, "ls_stats", {}).get("ls_sharpe", np.nan))
    mdd = float(getattr(report, "ls_stats", {}).get("ls_max_drawdown", np.nan))
    rows = []
    for c in cost_scenarios:
        annual_cost_drag = avg_turn * c * periods_per_year if not np.isnan(avg_turn) else np.nan
        rows.append({
            "cost_per_side": c,
            "ls_ann_gross": ls_ann,
            "ls_ann_net": ls_ann - annual_cost_drag if not np.isnan(ls_ann) and not np.isnan(annual_cost_drag) else np.nan,
            "ls_sharpe_gross": ls_sh,
            "ls_sharpe_net_proxy": (ls_sh * ((ls_ann - annual_cost_drag) / ls_ann)) if (not np.isnan(ls_sh) and not np.isnan(ls_ann) and ls_ann != 0 and not np.isnan(annual_cost_drag)) else np.nan,
            "ls_max_drawdown": mdd,
        })
    pd.DataFrame(rows).to_csv(out / "cost_scenarios.csv", index=False)
    implemented.add("cost_scenarios.csv")
    adv.add_timing("cost_scenarios", time.perf_counter() - t0)

    # 3.5) Size bucket report
    t0 = time.perf_counter()
    size_df, size_source_map = _run_size_bucket_report(
        report=report,
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        periods_per_year=periods_per_year,
    )
    size_df.to_csv(out / "size_bucket_report.csv", index=False)
    implemented.add("size_bucket_report.csv")
    adv.add_timing("size_bucket_report", time.perf_counter() - t0)

    # 3.6) Rolling OOS
    t0 = time.perf_counter()
    oos_df = _run_rolling_oos(
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
    )
    oos_df.to_csv(out / "rolling_oos.csv", index=False)
    implemented.add("rolling_oos.csv")
    adv.add_timing("rolling_oos", time.perf_counter() - t0)

    # 3.7) Regime stability
    t0 = time.perf_counter()
    reg_df = _run_regime_stability(
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        layer_ret=getattr(report, "layer_ret", pd.DataFrame()),
    )
    reg_df.to_csv(out / "regime_stability.csv", index=False)
    implemented.add("regime_stability.csv")
    adv.add_timing("regime_stability", time.perf_counter() - t0)

    # 3.8) Parameter sensitivity
    t0 = time.perf_counter()
    sens_df = _run_param_sensitivity(
        factor_panel=getattr(report, "factor_panel", pd.DataFrame()),
        return_panel=getattr(report, "return_panel", pd.DataFrame()),
        layer_ret=getattr(report, "layer_ret", pd.DataFrame()),
        mktcap_panel=getattr(report, "mktcap_panel", None),
        industry_map=getattr(report, "industry_map", None),
        periods_per_year=periods_per_year,
    )
    sens_df.to_csv(out / "param_sensitivity.csv", index=False)
    implemented.add("param_sensitivity.csv")
    adv.add_timing("param_sensitivity", time.perf_counter() - t0)

    # 4) Long/short intensity
    t0 = time.perf_counter()
    ret_panel = getattr(report, "return_panel", pd.DataFrame())
    bench_ret = ret_panel.mean(axis=1, skipna=True) if isinstance(ret_panel, pd.DataFrame) and not ret_panel.empty else pd.Series(dtype=float)
    top_col = f"Q{n_groups}"
    top_ret = layer_ret[top_col] if top_col in layer_ret.columns else pd.Series(dtype=float)
    bottom_ret = layer_ret["Q1"] if "Q1" in layer_ret.columns else pd.Series(dtype=float)

    long_strength_ts = top_ret.align(bench_ret, join="inner")[0] - top_ret.align(bench_ret, join="inner")[1]
    short_strength_ts = bottom_ret.align(bench_ret, join="inner")[1] - bottom_ret.align(bench_ret, join="inner")[0]

    long_stat = _series_t_stats(long_strength_ts)
    short_stat = _series_t_stats(short_strength_ts)
    pd.DataFrame([
        {
            "leg": "long",
            "benchmark_type": "equal_weight_universe",
            "mean_strength": long_stat["mean"],
            "annualized_strength": long_stat["mean"] * periods_per_year if not np.isnan(long_stat["mean"]) else np.nan,
            "t_stat": long_stat["t_stat"],
            "nw_t_stat": long_stat["nw_t_stat"],
            "p_value": long_stat["p_value"],
            "n_periods": int(long_strength_ts.dropna().shape[0]),
        },
        {
            "leg": "short",
            "benchmark_type": "equal_weight_universe",
            "mean_strength": short_stat["mean"],
            "annualized_strength": short_stat["mean"] * periods_per_year if not np.isnan(short_stat["mean"]) else np.nan,
            "t_stat": short_stat["t_stat"],
            "nw_t_stat": short_stat["nw_t_stat"],
            "p_value": short_stat["p_value"],
            "n_periods": int(short_strength_ts.dropna().shape[0]),
        },
    ]).to_csv(out / "long_short_intensity.csv", index=False)
    implemented.add("long_short_intensity.csv")
    adv.add_timing("long_short_intensity", time.perf_counter() - t0)

    # Placeholders for remaining modules (stable schema contract)
    for fname in _OUTPUT_FILES:
        path = out / fname
        if fname in implemented:
            adv.module_status[fname] = "implemented"
            continue
        note = "Scaffolded output contract. Module computation will be added incrementally."
        _write_placeholder(path, fname.replace(".csv", ""), note)
        module_notes[fname] = note
        adv.module_status[fname] = "placeholder"

    # Mark implemented files explicitly
    for fname in implemented:
        adv.module_status[fname] = "implemented"

    adv.summary = {
        "factor_name": adv.factor_name,
        "is_placeholder": False,
        "has_placeholder_modules": any(v == "placeholder" for v in adv.module_status.values()),
        "implemented_modules": sorted([k for k, v in adv.module_status.items() if v == "implemented"]),
        "placeholder_modules": sorted([k for k, v in adv.module_status.items() if v == "placeholder"]),
        "unavailable_peer_factors": unavailable_peers,
        "universe_membership": size_source_map,
        "total_output_files": len(_OUTPUT_FILES),
        "notes": module_notes,
    }
    adv.save()
    return adv

