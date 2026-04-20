"""
COVID-19 natural experiment analysis.

The 2020-2021 period represents an exogenous shock to healthcare access.
If DDI increases for non-COVID conditions (AMI, stroke, cancer) during this
period, it provides indirect empirical validation of the DDI as a sensor of
diagnostic delay — because the reason for delayed care is independently known.

This module adds:
1. Period-segmented DDI comparison (pre / COVID / post-COVID)
2. Interrupted Time Series (ITS) analysis around March 2020
3. Visualization with COVID period shading
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ICD-10 codes that ARE COVID (to be EXCLUDED from the natural experiment,
# since their severity increase is confounded by the pathology itself)
COVID_ICD_PREFIXES = ["U07", "U08", "U09", "J12", "B34"]


def is_covid_condition(icd_prefixes: list[str]) -> bool:
    """Check if any of the analysis ICD prefixes are COVID-related."""
    return any(
        any(prefix.startswith(covid) for covid in COVID_ICD_PREFIXES)
        for prefix in icd_prefixes
    )


def segment_covid_periods(
    temporal_metrics: pd.DataFrame,
    year_col: str = "ano",
    covid_start: int = 2020,
    covid_end: int = 2021,
) -> pd.DataFrame:
    """Add a period label column: 'pre_covid', 'covid', 'post_covid'."""
    df = temporal_metrics.copy()
    df["period"] = "pre_covid"
    df.loc[df[year_col].between(covid_start, covid_end), "period"] = "covid"
    df.loc[df[year_col] > covid_end, "period"] = "post_covid"
    return df


def compare_covid_periods(
    temporal_metrics: pd.DataFrame,
    config,
    metric: str = "ddi",
) -> dict:
    """Compare DDI across pre-COVID, COVID, and post-COVID periods.
    
    Uses Mann-Whitney U test for pairwise comparisons.
    A significant increase during COVID for non-COVID conditions supports
    the validity of DDI as a diagnostic delay sensor.
    """
    year_col = config.columns.year
    df = segment_covid_periods(temporal_metrics, year_col)
    
    pre = df[df["period"] == "pre_covid"][metric].values
    covid = df[df["period"] == "covid"][metric].values
    post = df[df["period"] == "post_covid"][metric].values
    
    results = {
        "pre_covid_mean": float(np.mean(pre)) if len(pre) > 0 else np.nan,
        "covid_mean": float(np.mean(covid)) if len(covid) > 0 else np.nan,
        "post_covid_mean": float(np.mean(post)) if len(post) > 0 else np.nan,
        "pre_years": len(pre),
        "covid_years": len(covid),
        "post_years": len(post),
    }
    
    # Pre vs COVID comparison
    if len(pre) >= 2 and len(covid) >= 1:
        try:
            stat, p = stats.mannwhitneyu(pre, covid, alternative="less")
            results["pre_vs_covid_pvalue"] = float(p)
            results["pre_vs_covid_increased"] = p < 0.05
        except Exception:
            results["pre_vs_covid_pvalue"] = np.nan
            results["pre_vs_covid_increased"] = False
    
    # Pre vs Post comparison
    if len(pre) >= 2 and len(post) >= 1:
        try:
            stat, p = stats.mannwhitneyu(pre, post, alternative="two-sided")
            results["pre_vs_post_pvalue"] = float(p)
        except Exception:
            results["pre_vs_post_pvalue"] = np.nan
    
    # DDI change during COVID
    if not np.isnan(results["pre_covid_mean"]) and not np.isnan(results["covid_mean"]):
        results["covid_delta_pct"] = float(
            (results["covid_mean"] - results["pre_covid_mean"]) 
            / results["pre_covid_mean"] * 100
        )
    
    return results


def interrupted_time_series(
    temporal_metrics: pd.DataFrame,
    config,
    metric: str = "ddi",
    intervention_year: int = 2020,
) -> dict:
    """Segmented regression (Interrupted Time Series) around COVID intervention.
    
    Model: DDI = β0 + β1*time + β2*intervention + β3*(time × intervention) + ε
    
    β2 = immediate level change at intervention
    β3 = slope change after intervention
    
    Returns:
        dict with regression coefficients and interpretation.
    """
    year_col = config.columns.year
    df = temporal_metrics.dropna(subset=[metric]).copy()
    
    if len(df) < 5:
        return {"error": "insufficient_data_for_ITS"}
    
    df = df.sort_values(year_col).reset_index(drop=True)
    df["time"] = df[year_col] - df[year_col].min()
    df["intervention"] = (df[year_col] >= intervention_year).astype(int)
    df["time_after"] = df["time"] * df["intervention"]
    
    # OLS: DDI ~ time + intervention + time_after
    X = np.column_stack([
        np.ones(len(df)),
        df["time"].values,
        df["intervention"].values,
        df["time_after"].values,
    ])
    y = df[metric].values
    
    try:
        # Using scipy for OLS with p-values
        result = stats.linregress(df["time"].values, y)
        
        # Manual OLS for the full model
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        
        y_pred = X @ beta
        residuals = y - y_pred
        n, k = len(y), 4
        sigma2 = np.sum(residuals**2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(XtX)
        se_beta = np.sqrt(np.diag(var_beta))
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
        
        return {
            "model": "ITS_segmented_regression",
            "intercept": float(beta[0]),
            "pre_slope": float(beta[1]),
            "level_change_at_covid": float(beta[2]),
            "slope_change_after_covid": float(beta[3]),
            "level_change_pvalue": float(p_values[2]),
            "slope_change_pvalue": float(p_values[3]),
            "level_change_significant": float(p_values[2]) < 0.05,
            "slope_change_significant": float(p_values[3]) < 0.05,
            "r_squared": float(1 - np.var(residuals) / np.var(y)),
            "interpretation": _its_interpretation(float(beta[2]), float(beta[3]), 
                                                   float(p_values[2]), float(p_values[3])),
        }
    except np.linalg.LinAlgError:
        return {"error": "singular_matrix"}


def _its_interpretation(level_change: float, slope_change: float, 
                         p_level: float, p_slope: float) -> str:
    parts = []
    if p_level < 0.05:
        direction = "increase" if level_change > 0 else "decrease"
        parts.append(
            f"Significant immediate {direction} in DDI at the COVID intervention point "
            f"(Δ={level_change:+.3f}, p={p_level:.4f})."
        )
    else:
        parts.append(f"No significant immediate level change at COVID (p={p_level:.4f}).")
    
    if p_slope < 0.05:
        direction = "steepening" if slope_change > 0 else "flattening"
        parts.append(
            f"Significant {direction} of DDI trend after COVID "
            f"(Δslope={slope_change:+.4f}/yr, p={p_slope:.4f})."
        )
    
    return " ".join(parts)
