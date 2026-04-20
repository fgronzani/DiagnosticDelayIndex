"""
Feature engineering module.

Computes the composite Severity Score and age-group features.
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def normalize_length_of_stay(
    series: pd.Series,
    method: str = "minmax"
) -> pd.Series:
    """Normalize length of stay to [0, 1] range.

    Args:
        series: Raw length of stay values.
        method: Normalization method — "minmax" or "zscore".

    Returns:
        Normalized Series clipped to [0, 1].
    """
    if method == "minmax":
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0.5, index=series.index)
        z = (series - mean_val) / std_val
        # Clip z-scores to reasonable range and rescale to [0, 1]
        return ((z.clip(-3, 3) + 3) / 6)

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")


def compute_severity_score(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Compute the composite Severity Score for each admission.

    Formula:
        severity_score = (icu_weight × icu) + (death_weight × obito) + (los_weight × normalized_los)

    Clinical rationale:
        - Death (weight=3): strongest signal of late/severe presentation
        - ICU admission (weight=2): high resource utilization indicates advanced disease
        - Length of stay (weight=1): continuous proxy for disease complexity

    Args:
        df: Preprocessed DataFrame with binary icu/obito columns and numeric los.
        config: Analysis configuration with severity weights.

    Returns:
        DataFrame with added columns: 'los_normalized', 'severity_score'.
    """
    df = df.copy()
    cols = config.columns
    weights = config.severity

    # Normalize length of stay
    df["los_normalized"] = normalize_length_of_stay(
        df[cols.length_of_stay],
        method=weights.los_normalization
    )

    # Compute severity score
    df["severity_score"] = (
        weights.icu_weight * df[cols.icu]
        + weights.death_weight * df[cols.death]
        + weights.los_weight * df["los_normalized"]
    )

    logger.info(
        f"Severity score computed: "
        f"mean={df['severity_score'].mean():.3f}, "
        f"median={df['severity_score'].median():.3f}, "
        f"max={df['severity_score'].max():.3f}"
    )

    return df


def add_age_groups(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Add age group categorization.

    Args:
        df: DataFrame with age column.
        config: Analysis configuration with age bins and labels.

    Returns:
        DataFrame with added 'age_group' column.
    """
    df = df.copy()
    cols = config.columns

    if cols.age not in df.columns:
        logger.warning(f"Age column '{cols.age}' not found. Skipping age groups.")
        return df

    df["age_group"] = pd.cut(
        df[cols.age],
        bins=config.age_bins,
        labels=config.age_labels,
        right=False,
        include_lowest=True
    )

    age_dist = df["age_group"].value_counts().to_dict()
    logger.info(f"Age groups distribution: {age_dist}")

    return df


def classify_high_severity(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Classify cases as high-severity based on threshold.

    Uses either a fixed threshold or a percentile-based threshold.

    Args:
        df: DataFrame with 'severity_score' column.
        config: Analysis configuration with DDI settings.

    Returns:
        DataFrame with added 'high_severity' boolean column.
    """
    df = df.copy()

    if config.ddi.severity_threshold_fixed is not None:
        threshold = config.ddi.severity_threshold_fixed
    else:
        threshold = df["severity_score"].quantile(
            config.ddi.severity_threshold_percentile / 100.0
        )

    df["high_severity"] = (df["severity_score"] >= threshold).astype(int)

    n_high = df["high_severity"].sum()
    logger.info(
        f"High-severity classification: threshold={threshold:.3f}, "
        f"n_high={n_high} ({100 * n_high / len(df):.1f}%)"
    )

    return df


def feature_engineering_pipeline(
    df: pd.DataFrame,
    config: AnalysisConfig
) -> pd.DataFrame:
    """Execute the full feature engineering pipeline.

    Steps:
        1. Compute severity score
        2. Add age groups
        3. Classify high-severity cases

    Args:
        df: Preprocessed DataFrame.
        config: Analysis configuration.

    Returns:
        DataFrame with all engineered features.
    """
    df = compute_severity_score(df, config)
    df = add_age_groups(df, config)
    
    if config.ddi.severity_threshold_fixed is None:
        global_threshold = df["severity_score"].quantile(
            config.ddi.severity_threshold_percentile / 100.0
        )
        config.ddi.severity_threshold_fixed = global_threshold
        logger.info(
            f"Global severity threshold fixed at {global_threshold:.4f} "
            f"(P{config.ddi.severity_threshold_percentile} of full dataset, n={len(df)})"
        )
        
    df = classify_high_severity(df, config)
    return df
