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


def compare_regions(
    regional_metrics: pd.DataFrame,
    config: AnalysisConfig,
    metric: str = "ddi"
) -> dict:
    """Compare DDI across regions using Kruskal-Wallis test.

    Non-parametric test for differences between multiple groups.

    Args:
        regional_metrics: DataFrame with regional metrics.
        config: Analysis configuration.
        metric: Which metric to compare.

    Returns:
        Dictionary with test results and summary statistics.
    """
    cols = config.columns

    df = regional_metrics.dropna(subset=[metric])

    if len(df) < 2:
        return {
            "test": "kruskal_wallis",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_regions": len(df),
        }

    return {
        "test": "descriptive",
        "n_regions": len(df),
        "mean_ddi": df[metric].mean(),
        "std_ddi": df[metric].std(),
        "min_ddi": df[metric].min(),
        "max_ddi": df[metric].max(),
        "range_ddi": df[metric].max() - df[metric].min(),
        "top_5_regions": df.nlargest(5, metric)[[cols.municipality, metric]].to_dict("records"),
        "bottom_5_regions": df.nsmallest(5, metric)[[cols.municipality, metric]].to_dict("records"),
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
    config: AnalysisConfig
) -> dict:
    """Run all statistical analyses and return consolidated results.

    Args:
        temporal_metrics: DDI by year.
        regional_metrics: DDI by region.
        age_adjusted_metrics: DDI by year + age group.
        config: Analysis configuration.

    Returns:
        Dictionary with all analysis results.
    """
    results = {}

    # Temporal trends
    for metric in ["ddi", "mortality_rate", "icu_rate", "avg_severity"]:
        trend = detect_temporal_trend(temporal_metrics, config, metric)
        mk = mann_kendall_test(temporal_metrics[metric].dropna().values)
        results[f"trend_{metric}"] = {**trend, "mann_kendall": mk}

    # Regional comparison
    results["regional_comparison"] = compare_regions(regional_metrics, config)

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

    return results
