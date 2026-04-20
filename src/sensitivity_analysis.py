"""
Sensitivity analysis for severity score weights.

Tests whether DDI trend conclusions are robust to different weight combinations.
A conclusion is "robust" if it holds for ≥80% of the tested weight combinations.
"""
from __future__ import annotations
import logging
from itertools import product
from copy import deepcopy

import pandas as pd
import numpy as np

from .config import AnalysisConfig, SeverityWeights
from .feature_engineering import feature_engineering_pipeline
from .metrics import compute_temporal_metrics
from .analysis import detect_temporal_trend

logger = logging.getLogger(__name__)


def run_sensitivity_analysis(
    df_preprocessed: pd.DataFrame,
    base_config: AnalysisConfig,
    icu_weights: list[float] = [1.0, 2.0, 3.0],
    death_weights: list[float] = [2.0, 3.0, 4.0],
    los_weights: list[float] = [0.5, 1.0, 2.0],
) -> pd.DataFrame:
    """Run DDI pipeline across all weight combinations.
    
    Returns a DataFrame with one row per weight combination, showing:
    - Weight values
    - DDI trend direction and significance
    - Whether conclusion matches the baseline
    
    Args:
        df_preprocessed: Output of preprocess_pipeline() (before feature engineering).
        base_config: The baseline configuration.
        icu_weights, death_weights, los_weights: Weight values to test.
        
    Returns:
        DataFrame with sensitivity results.
    """
    results = []
    combinations = list(product(icu_weights, death_weights, los_weights))
    logger.info(f"Running sensitivity analysis: {len(combinations)} weight combinations")
    
    for icu_w, death_w, los_w in combinations:
        config = deepcopy(base_config)
        config.severity = SeverityWeights(
            icu_weight=icu_w,
            death_weight=death_w,
            los_weight=los_w,
            los_normalization=base_config.severity.los_normalization,
        )
        config.ddi.severity_threshold_fixed = None  # Force recalculation
        
        try:
            df_fe = feature_engineering_pipeline(df_preprocessed.copy(), config)
            temporal = compute_temporal_metrics(df_fe, config)
            trend = detect_temporal_trend(temporal, config, "ddi")
            
            results.append({
                "icu_weight": icu_w,
                "death_weight": death_w,
                "los_weight": los_w,
                "ddi_direction": trend.get("direction"),
                "ddi_significant": trend.get("significant"),
                "ddi_pvalue": trend.get("p_value"),
                "ddi_slope": trend.get("slope"),
                "r_squared": trend.get("r_squared"),
            })
        except Exception as e:
            logger.warning(f"Weight combo ({icu_w}, {death_w}, {los_w}) failed: {e}")
    
    result_df = pd.DataFrame(results)
    
    # Robustness summary
    if len(result_df) > 0:
        baseline_direction = result_df.iloc[
            result_df.index[
                (result_df.icu_weight == base_config.severity.icu_weight) &
                (result_df.death_weight == base_config.severity.death_weight) &
                (result_df.los_weight == base_config.severity.los_weight)
            ]
        ]["ddi_direction"].values
        
        if len(baseline_direction) > 0:
            baseline_dir = baseline_direction[0]
            n_agree = (result_df["ddi_direction"] == baseline_dir).sum()
            n_significant_agree = (
                (result_df["ddi_direction"] == baseline_dir) & 
                (result_df["ddi_significant"] == True)
            ).sum()
            
            logger.info(
                f"Sensitivity: {n_agree}/{len(result_df)} combinations agree with baseline direction. "
                f"{n_significant_agree}/{len(result_df)} are also significant."
            )
    
    return result_df


def sensitivity_summary(sensitivity_df: pd.DataFrame) -> dict:
    """Summarize robustness of the DDI trend conclusion."""
    total = len(sensitivity_df)
    if total == 0:
        return {}
    
    by_direction = sensitivity_df["ddi_direction"].value_counts(normalize=True).to_dict()
    dominant_direction = max(by_direction, key=by_direction.get)
    robustness_pct = by_direction.get(dominant_direction, 0) * 100
    
    significant_pct = sensitivity_df["ddi_significant"].mean() * 100
    
    return {
        "total_combinations": total,
        "dominant_direction": dominant_direction,
        "direction_robustness_pct": float(robustness_pct),
        "significant_pct": float(significant_pct),
        "conclusion_robust": robustness_pct >= 80,
        "by_direction": {k: float(v * 100) for k, v in by_direction.items()},
    }
