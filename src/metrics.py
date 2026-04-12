"""
Diagnostic Delay Index (DDI) and related severity metrics.

The DDI is a proxy indicator — it does NOT directly measure diagnostic delay.
Instead, it captures the proportion of cases presenting with high severity at
admission, under the hypothesis that delayed diagnosis leads to worse
condition at presentation.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def compute_group_metrics(
    group: pd.DataFrame,
    config: AnalysisConfig
) -> dict:
    """Compute severity metrics for a group of admissions.

    Metrics:
        - ddi: proportion of high-severity cases (Diagnostic Delay Index)
        - mortality_rate: proportion of deaths
        - icu_rate: proportion of ICU admissions
        - avg_severity: mean severity score
        - median_severity: median severity score
        - avg_los: mean length of stay
        - total_cases: number of cases in the group

    Args:
        group: DataFrame subset (e.g., a single year or region).
        config: Analysis configuration.

    Returns:
        Dictionary of metric values.
    """
    cols = config.columns
    n = len(group)

    if n == 0:
        return {
            "ddi": np.nan,
            "mortality_rate": np.nan,
            "icu_rate": np.nan,
            "avg_severity": np.nan,
            "median_severity": np.nan,
            "avg_los": np.nan,
            "total_cases": 0,
        }

    return {
        "ddi": group["high_severity"].mean(),
        "mortality_rate": group[cols.death].mean(),
        "icu_rate": group[cols.icu].mean(),
        "avg_severity": group["severity_score"].mean(),
        "median_severity": group["severity_score"].median(),
        "avg_los": group[cols.length_of_stay].mean(),
        "total_cases": n,
    }


def compute_temporal_metrics(
    df: pd.DataFrame,
    config: AnalysisConfig
) -> pd.DataFrame:
    """Compute DDI and severity metrics aggregated by year.

    Args:
        df: Feature-engineered DataFrame.
        config: Analysis configuration.

    Returns:
        DataFrame with one row per year and all metrics columns.
    """
    cols = config.columns
    results = []

    for year, group in df.groupby(cols.year):
        if len(group) < config.min_cases_threshold:
            logger.warning(
                f"Year {year}: only {len(group)} cases "
                f"(< threshold {config.min_cases_threshold}). Included but flagged."
            )

        metrics = compute_group_metrics(group, config)
        metrics[cols.year] = int(year)
        metrics["low_sample"] = len(group) < config.min_cases_threshold
        results.append(metrics)

    result_df = pd.DataFrame(results).sort_values(cols.year).reset_index(drop=True)
    logger.info(f"Temporal metrics computed for {len(result_df)} years")
    return result_df


def compute_regional_metrics(
    df: pd.DataFrame,
    config: AnalysisConfig
) -> pd.DataFrame:
    """Compute DDI and severity metrics aggregated by region (municipio).

    Filters out regions with fewer cases than the minimum threshold.

    Args:
        df: Feature-engineered DataFrame.
        config: Analysis configuration.

    Returns:
        DataFrame with one row per region and all metrics columns.
    """
    cols = config.columns
    results = []

    for region, group in df.groupby(cols.municipality):
        if len(group) < config.min_cases_threshold:
            continue

        metrics = compute_group_metrics(group, config)
        metrics[cols.municipality] = region
        results.append(metrics)

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("ddi", ascending=False).reset_index(drop=True)

    logger.info(
        f"Regional metrics computed for {len(result_df)} regions "
        f"(excluded regions with < {config.min_cases_threshold} cases)"
    )
    return result_df


def compute_age_adjusted_metrics(
    df: pd.DataFrame,
    config: AnalysisConfig
) -> pd.DataFrame:
    """Compute DDI metrics stratified by age group and year.

    Useful for detecting whether apparent temporal trends are driven by
    demographic shifts rather than actual changes in diagnostic delay.

    Args:
        df: Feature-engineered DataFrame with 'age_group' column.
        config: Analysis configuration.

    Returns:
        DataFrame with metrics per (year, age_group).
    """
    cols = config.columns

    if "age_group" not in df.columns:
        logger.warning("No age_group column found. Skipping age-adjusted analysis.")
        return pd.DataFrame()

    results = []

    for (year, age_group), group in df.groupby([cols.year, "age_group"]):
        if len(group) < config.min_cases_threshold:
            continue

        metrics = compute_group_metrics(group, config)
        metrics[cols.year] = int(year)
        metrics["age_group"] = age_group
        results.append(metrics)

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values([cols.year, "age_group"]).reset_index(drop=True)

    logger.info(f"Age-adjusted metrics: {len(result_df)} year-age combinations")
    return result_df


def compute_regional_temporal_metrics(
    df: pd.DataFrame,
    config: AnalysisConfig,
    top_n_regions: int = 10
) -> pd.DataFrame:
    """Compute DDI over time for the top N regions by total cases.

    Args:
        df: Feature-engineered DataFrame.
        config: Analysis configuration.
        top_n_regions: Number of top regions to include.

    Returns:
        DataFrame with metrics per (year, region) for top regions.
    """
    cols = config.columns

    # Identify top regions by case volume
    region_counts = df[cols.municipality].value_counts()
    top_regions = region_counts.head(top_n_regions).index.tolist()

    df_top = df[df[cols.municipality].isin(top_regions)]

    results = []
    for (year, region), group in df_top.groupby([cols.year, cols.municipality]):
        if len(group) < config.min_cases_threshold:
            continue
        metrics = compute_group_metrics(group, config)
        metrics[cols.year] = int(year)
        metrics[cols.municipality] = region
        results.append(metrics)

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values([cols.year, cols.municipality]).reset_index(drop=True)

    return result_df
