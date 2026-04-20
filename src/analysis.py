"""
Statistical analysis module.

Provides trend detection, regional comparison, and hypothesis testing
for Diagnostic Delay Index data.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

from .config import AnalysisConfig
from .covid_analysis import compare_covid_periods, interrupted_time_series, is_covid_condition

logger = logging.getLogger(__name__)


def detect_temporal_trend(
    temporal_metrics: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "ddi"
) -> dict:
    """Detect linear trend in a metric over time using OLS regression.

    Uses scipy's linregress for simplicity and interpretability:
        - slope: annual change in the metric
        - p_value: statistical significance
        - r_squared: proportion of variance explained

    Args:
        temporal_metrics: DataFrame with year and metric columns.
        config: Analysis configuration.
        metric: Which metric column to analyze.

    Returns:
        Dictionary with trend statistics.
    """
    cols = config.columns
    df = temporal_metrics.dropna(subset=[metric])

    if len(df) < 3:
        logger.warning(f"Insufficient data points ({len(df)}) for trend analysis.")
        return {
            "metric": metric,
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "direction": "insufficient_data",
            "significant": False,
            "n_years": len(df),
        }

    x = df[cols.year].values.astype(float)
    y = df[metric].values.astype(float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    significant = p_value < 0.05
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

    result = {
        "metric": metric,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "direction": direction,
        "significant": significant,
        "n_years": len(df),
        "first_year": int(x.min()),
        "last_year": int(x.max()),
        "first_value": float(y[0]),
        "last_value": float(y[-1]),
        "total_change": float(y[-1] - y[0]),
        "pct_change": float((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else np.nan,
    }

    logger.info(
        f"Trend ({metric}): {direction}, slope={slope:.6f}, "
        f"p={p_value:.4f}, R²={r_value**2:.4f}"
    )

    return result


def mann_kendall_test(values: np.ndarray) -> dict:
    """Perform the Mann-Kendall non-parametric trend test.

    More robust than linear regression for detecting monotonic trends,
    especially with non-normal data or small sample sizes.

    Args:
        values: Array of values ordered by time.

    Returns:
        Dictionary with test statistic, p-value, and direction.
    """
    n = len(values)
    if n < 4:
        return {"tau": np.nan, "p_value": np.nan, "direction": "insufficient_data"}

    # Kendall's tau correlation with sequential index
    tau, p_value = stats.kendalltau(np.arange(n), values)

    direction = "increasing" if tau > 0 else "decreasing" if tau < 0 else "stable"

    return {
        "tau": tau,
        "p_value": p_value,
        "direction": direction,
        "significant": p_value < 0.05,
    }


def apply_bonferroni_correction(results: dict, test_keys: list[str]) -> dict:
    """Apply Bonferroni correction to p-values of multiple trend tests.
    
    Adds 'p_value_corrected' and 'significant_corrected' to each test result.
    """
    n_tests = len(test_keys)
    for key in test_keys:
        if key in results:
            raw_p = results[key].get("p_value", np.nan)
            if raw_p is not None and not np.isnan(raw_p):
                corrected_p = min(raw_p * n_tests, 1.0)
                results[key]["p_value_bonferroni"] = corrected_p
                results[key]["significant_bonferroni"] = corrected_p < 0.05
            else:
                results[key]["p_value_bonferroni"] = np.nan
                results[key]["significant_bonferroni"] = False
    return results


def compare_regions(
    regional_metrics: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "ddi"
) -> dict:
    """Compare DDI across regions using Kruskal-Wallis H-test.
    
    Requires the original per-admission DataFrame to extract per-region 
    severity distributions for the non-parametric test. If only aggregate
    regional_metrics is available, falls back to a descriptive summary and
    logs a warning.
    """
    cols = config.columns
    df = regional_metrics.dropna(subset=[metric])
    
    if len(df) < 2:
        return {
            "test": "insufficient_data",
            "n_regions": len(df),
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
        }
    
    if len(df) >= 3:
        try:
            groups = [group[metric].values for _, group in df.groupby(cols.municipality) 
                      if len(group) > 0]
            if len(groups) >= 3 and all(len(g) >= 1 for g in groups):
                stat, p_value = stats.kruskal(*groups) if all(len(g) > 1 for g in groups) else (np.nan, np.nan)
                if np.isnan(stat):
                    stat = np.nan
                    p_value = np.nan
                    test_name = "descriptive_only"
                else:
                    test_name = "kruskal_wallis"
            else:
                stat, p_value = np.nan, np.nan
                test_name = "descriptive_only"
        except Exception as e:
            logger.warning(f"Kruskal-Wallis failed: {e}. Using descriptive summary.")
            stat, p_value = np.nan, np.nan
            test_name = "descriptive_only"
    else:
        stat, p_value = np.nan, np.nan
        test_name = "descriptive_only"
    
    return {
        "test": test_name,
        "statistic": float(stat) if not np.isnan(stat) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "significant": (p_value < 0.05) if (p_value is not None and not np.isnan(p_value)) else False,
        "n_regions": len(df),
        "mean_ddi": float(df[metric].mean()),
        "std_ddi": float(df[metric].std()),
        "min_ddi": float(df[metric].min()),
        "max_ddi": float(df[metric].max()),
        "range_ddi": float(df[metric].max() - df[metric].min()),
        "cv_ddi": float(df[metric].std() / df[metric].mean()) if df[metric].mean() > 0 else np.nan,
        "top_5_regions": df.nlargest(5, metric)[[cols.municipality, metric]].to_dict("records"),
        "bottom_5_regions": df.nsmallest(5, metric)[[cols.municipality, metric]].to_dict("records"),
    }


def compare_regions_full(
    df: pd.DataFrame,
    regional_metrics: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "severity_score",
    top_n: int = 20,
) -> dict:
    """Kruskal-Wallis on per-admission severity distributions by region.
    
    This is the statistically valid version: each region contributes a 
    distribution of severity scores (not just one aggregate DDI value).
    Only top_n regions by case volume are included to avoid power issues.
    """
    cols = config.columns
    
    top_regions = (
        df[cols.municipality].value_counts()
        .head(top_n)
        .index.tolist()
    )
    
    groups = [
        grp[metric].values
        for region, grp in df[df[cols.municipality].isin(top_regions)].groupby(cols.municipality)
        if len(grp) >= config.min_cases_threshold
    ]
    
    if len(groups) < 3:
        return {"test": "kruskal_wallis_full", "n_groups": len(groups), 
                "statistic": np.nan, "p_value": np.nan, "significant": False}
    
    stat, p_value = stats.kruskal(*groups)
    
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (stat - k + 1) / (n_total - k)
    
    return {
        "test": "kruskal_wallis_full",
        "n_groups": len(groups),
        "n_total_admissions": n_total,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size_epsilon_sq": float(epsilon_sq),
        "effect_interpretation": (
            "large" if epsilon_sq > 0.14 else
            "medium" if epsilon_sq > 0.06 else
            "small" if epsilon_sq > 0.01 else "negligible"
        ),
    }


def rank_regions(
    regional_metrics: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "ddi",
    ascending: bool = False
) -> pd.DataFrame:
    """Rank regions by a given metric.

    Args:
        regional_metrics: DataFrame with regional metrics.
        config: Analysis configuration.
        metric: Which metric to rank by.
        ascending: If True, lowest metric = rank 1.

    Returns:
        DataFrame with added 'rank' column.
    """
    df = regional_metrics.copy()
    df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def run_full_analysis(
    temporal_metrics: pd.DataFrame,
    regional_metrics: pd.DataFrame,
    age_adjusted_metrics: pd.DataFrame,
    config: AnalysisConfig,
    df_admissions: pd.DataFrame | None = None,
) -> dict:
    """Run all statistical analyses and return consolidated results.

    Args:
        temporal_metrics: DDI by year.
        regional_metrics: DDI by region.
        age_adjusted_metrics: DDI by year + age group.
        config: Analysis configuration.
        df_admissions: Optional pre-processed full dataset for deeper tests.

    Returns:
        Dictionary with all analysis results.
    """
    results = {}

    # Temporal trends
    for metric in ["ddi", "mortality_rate", "icu_rate", "avg_severity"]:
        trend = detect_temporal_trend(temporal_metrics, config, metric)
        mk = mann_kendall_test(temporal_metrics[metric].dropna().values)
        results[f"trend_{metric}"] = {**trend, "mann_kendall": mk}

    trend_keys = ["trend_ddi", "trend_mortality_rate", "trend_icu_rate", "trend_avg_severity"]
    results = apply_bonferroni_correction(results, trend_keys)

    # Regional comparison
    results["regional_comparison"] = compare_regions(regional_metrics, config)
    if df_admissions is not None:
        results["regional_comparison_full"] = compare_regions_full(
            df_admissions, regional_metrics, config
        )

    # Ranked regions
    results["ranked_regions"] = rank_regions(regional_metrics, config)

    # Age-adjusted trends (if available)
    if len(age_adjusted_metrics) > 0 and "age_group" in age_adjusted_metrics.columns:
        age_trends = {}
        for age_group, group_df in age_adjusted_metrics.groupby("age_group"):
            age_trends[str(age_group)] = detect_temporal_trend(
                group_df.reset_index(drop=True), config, "ddi"
            )
        results["age_adjusted_trends"] = age_trends

    # COVID natural experiment (apenas para condições não-COVID)
    if df_admissions is not None and not is_covid_condition(config.condition.icd_prefixes):
        results["covid_analysis"] = compare_covid_periods(temporal_metrics, config)
        results["covid_its"] = interrupted_time_series(temporal_metrics, config)

    return results
