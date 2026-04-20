#!/usr/bin/env python3
"""
Diagnostic Delay Detector — Main Pipeline

Orchestrates the full analysis pipeline:
    1. Data loading and preprocessing
    2. Feature engineering (severity score)
    3. Metrics computation (DDI, temporal, regional)
    4. Statistical analysis (trend detection)
    5. Visualization (publication-quality charts)
    6. Clinical interpretation (plain-language report)

Usage:
    python main.py                                # Run with synthetic data
    python main.py --data path/to/data.csv        # Run with real data
    python main.py --condition I63 --name Stroke   # Different condition
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import AnalysisConfig, ConditionFilter, ColumnConfig, SeverityWeights, DDIConfig
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.metrics import (
    compute_temporal_metrics,
    compute_regional_metrics,
    compute_age_adjusted_metrics,
    compute_regional_temporal_metrics,
)
from src.analysis import run_full_analysis
from src.visualization import generate_all_plots
from src.interpretation import generate_full_report
from src.generate_data import generate_synthetic_data
from src.sensitivity_analysis import run_sensitivity_analysis, sensitivity_summary
from src.metrics import compute_ddi_confidence_intervals


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s │ %(name)-30s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnostic Delay Detector — Detecting Late Presentation Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to input CSV file. If not provided, generates synthetic data.",
    )
    parser.add_argument(
        "--condition", type=str, nargs="+", default=["I21"],
        help="ICD-10 code prefix(es) to filter. Default: I21 (AMI).",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Human-readable condition name. Default: auto-generated from ICD codes.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory for plots and reports. Default: outputs/",
    )
    parser.add_argument(
        "--threshold-percentile", type=float, default=75.0,
        help="Percentile for high-severity classification. Default: 75.0",
    )
    parser.add_argument(
        "--min-cases", type=int, default=30,
        help="Minimum cases per region/year for inclusion. Default: 30",
    )
    parser.add_argument(
        "--synthetic-n", type=int, default=15000,
        help="Number of synthetic records to generate. Default: 15000",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis on severity weights.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


# Condition name lookup
ICD_CONDITION_NAMES = {
    "I21": "Acute Myocardial Infarction (AMI)",
    "I63": "Ischemic Stroke",
    "C34": "Lung Cancer",
    "J18": "Pneumonia",
    "K35": "Acute Appendicitis",
    "C50": "Breast Cancer",
    "C18": "Colon Cancer",
    "I50": "Heart Failure",
}


def main() -> None:
    """Execute the Diagnostic Delay Detector pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("main")
    logger.info("=" * 70)
    logger.info("  DIAGNOSTIC DELAY DETECTOR")
    logger.info("  Detecting Late Presentation Signals in Healthcare Systems")
    logger.info("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────
    condition_name = args.name
    if condition_name is None:
        # Auto-generate from first ICD prefix
        condition_name = ICD_CONDITION_NAMES.get(
            args.condition[0],
            f"ICD {', '.join(args.condition)}"
        )

    config = AnalysisConfig(
        columns=ColumnConfig(),
        severity=SeverityWeights(),
        ddi=DDIConfig(severity_threshold_percentile=args.threshold_percentile),
        condition=ConditionFilter(
            icd_prefixes=args.condition,
            condition_name=condition_name,
        ),
        min_cases_threshold=args.min_cases,
        output_dir=args.output,
    )

    logger.info(f"Condition: {config.condition.condition_name}")
    logger.info(f"ICD prefixes: {config.condition.icd_prefixes}")
    logger.info(f"Severity threshold: P{config.ddi.severity_threshold_percentile}")
    logger.info(f"Output directory: {config.output_dir}")

    # ── Data Loading ─────────────────────────────────────────────────────
    if args.data:
        data_path = args.data
        logger.info(f"Loading real data from: {data_path}")
    else:
        data_path = "data/synthetic_sih_data.csv"
        logger.info("No data file provided. Generating synthetic data...")
        generate_synthetic_data(
            n_records=args.synthetic_n,
            output_path=data_path,
        )

    # ── Pipeline ─────────────────────────────────────────────────────────
    logger.info("─" * 40 + " PREPROCESSING")
    df_preprocessed = preprocess_pipeline(data_path, config)
    logger.info(f"After preprocessing: {len(df_preprocessed)} records")

    if len(df_preprocessed) == 0:
        logger.error("No records after filtering. Check ICD codes and data.")
        sys.exit(1)

    logger.info("─" * 40 + " FEATURE ENGINEERING")
    df = feature_engineering_pipeline(df_preprocessed.copy(), config)

    logger.info("─" * 40 + " METRICS COMPUTATION")
    temporal_metrics = compute_temporal_metrics(df, config)
    ddi_ci = compute_ddi_confidence_intervals(df, config)
    if not ddi_ci.empty:
        temporal_metrics = temporal_metrics.merge(
            ddi_ci[[config.columns.year, 'ddi_ci_lower', 'ddi_ci_upper']], 
            on=config.columns.year, 
            how='left'
        )

    regional_metrics = compute_regional_metrics(df, config)
    age_adjusted_metrics = compute_age_adjusted_metrics(df, config)
    regional_temporal_metrics = compute_regional_temporal_metrics(df, config, top_n_regions=10)

    logger.info("─" * 40 + " STATISTICAL ANALYSIS")
    analysis_results = run_full_analysis(
        temporal_metrics, regional_metrics, age_adjusted_metrics, config, df_admissions=df
    )

    logger.info("─" * 40 + " VISUALIZATION")
    ddi_trend = analysis_results.get("trend_ddi", {})
    plot_paths = generate_all_plots(
        df=df,
        temporal_metrics=temporal_metrics,
        regional_metrics=regional_metrics,
        age_adjusted_metrics=age_adjusted_metrics,
        regional_temporal_metrics=regional_temporal_metrics,
        config=config,
        trend_info=ddi_trend,
    )
    logger.info(f"Generated {len(plot_paths)} plots")

    logger.info("─" * 40 + " CLINICAL INTERPRETATION")
    report = generate_full_report(analysis_results, config)

    # Print report
    print("\n" + report)

    # Save report
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "clinical_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Clinical report saved: {report_path}")

    # Save metrics as CSV
    temporal_metrics.to_csv(output_dir / "temporal_metrics.csv", index=False)
    regional_metrics.to_csv(output_dir / "regional_metrics.csv", index=False)
    if len(age_adjusted_metrics) > 0:
        age_adjusted_metrics.to_csv(output_dir / "age_adjusted_metrics.csv", index=False)

    if args.sensitivity:
        logger.info("─" * 40 + " SENSITIVITY ANALYSIS")
        sens_df = run_sensitivity_analysis(df_preprocessed, config)
        sens_df.to_csv(output_dir / "sensitivity_results.csv", index=False)
        sens_sum = sensitivity_summary(sens_df)
        logger.info(f"Sensitivity completed. Robustness: {sens_sum.get('direction_robustness_pct', 0):.1f}%")

    # Summary statistics
    logger.info("─" * 40 + " SUMMARY")
    logger.info(f"  Total cases analyzed: {len(df)}")
    logger.info(f"  Years covered: {int(df[config.columns.year].min())}–{int(df[config.columns.year].max())}")
    logger.info(f"  Regions with sufficient data: {len(regional_metrics)}")
    
    p_val_display = ddi_trend.get('p_value_bonferroni', ddi_trend.get('p_value', 'N/A'))
    logger.info(f"  DDI trend: {ddi_trend.get('direction', 'N/A')} (p_bonf={p_val_display})")
    
    logger.info(f"  All outputs saved to: {output_dir}/")
    logger.info("=" * 70)
    logger.info("  Analysis complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
