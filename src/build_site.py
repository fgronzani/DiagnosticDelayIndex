"""
Static site generator for the Diagnostic Delay Dashboard.
Builds the website processing multiple conditions automatically.
"""
from __future__ import annotations

import os
import shutil
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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

CONDITIONS = [
    {"id": "I21", "name": "Acute Myocardial Infarction (AMI)"},
    {"id": "I63", "name": "Ischemic Stroke"},
    {"id": "C34", "name": "Lung Cancer"}
]

def run_condition(cond: dict) -> dict:
    """Runs the full analysis pipeline for a single condition and returns data for HTML."""
    print(f"=== Processing {cond['name']} ({cond['id']}) ===")
    
    config = AnalysisConfig(
        columns=ColumnConfig(),
        severity=SeverityWeights(),
        ddi=DDIConfig(severity_threshold_percentile=75.0),
        condition=ConditionFilter(
            icd_prefixes=[cond['id']],
            condition_name=cond['name'],
        ),
        min_cases_threshold=30,
        output_dir="outputs",
    )
    
    data_path = "data/synthetic_sih_data.csv"
    df = preprocess_pipeline(data_path, config)
    df = feature_engineering_pipeline(df, config)
    
    temporal_metrics = compute_temporal_metrics(df, config)
    regional_metrics = compute_regional_metrics(df, config)
    age_adjusted_metrics = compute_age_adjusted_metrics(df, config)
    regional_temporal_metrics = compute_regional_temporal_metrics(df, config, top_n_regions=10)
    
    analysis_results = run_full_analysis(
        temporal_metrics, regional_metrics, age_adjusted_metrics, config
    )
    
    ddi_trend = analysis_results.get("trend_ddi", {})
    generate_all_plots(
        df=df,
        temporal_metrics=temporal_metrics,
        regional_metrics=regional_metrics,
        age_adjusted_metrics=age_adjusted_metrics,
        regional_temporal_metrics=regional_temporal_metrics,
        config=config,
        trend_info=ddi_trend,
    )
    
    # Copy images to docs/assets with prefix
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    for plot in ["ddi_time_series.png", "regional_ddi_bar.png", "regional_temporal_heatmap.png", "severity_distribution.png", "age_adjusted_ddi.png"]:
        src_path = Path(config.output_dir) / plot
        if src_path.exists():
            shutil.copy(src_path, assets_dir / f"{cond['id']}_{plot}")
    
    report_text = generate_full_report(analysis_results, config)
    
    # Format text for HTML
    report_html = report_text.replace("\n", "<br>")
    report_html = report_html.replace("⚠️", "<span style='color:red'>⚠️</span>")
    report_html = report_html.replace("📈", "<span style='color:green'>📈</span>")
    
    # Convert CSVs to HTML
    temporal_html = temporal_metrics.to_html(classes='dataframe', index=False, float_format='%.3f')
    
    return {
        "id": cond['id'],
        "name": cond['name'],
        "report_html": report_html,
        "temporal_table_html": temporal_html
    }

def main():
    print("Starting site build...")
    
    # Ensure assets dir exists
    docs_dir = Path("docs")
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    condition_data = []
    
    for cond in CONDITIONS:
        data = run_condition(cond)
        condition_data.append(data)
        
    print("All models processed. Rendering HTML...")
    
    # Setup Jinja2
    env = Environment(loader=FileSystemLoader("docs"))
    template = env.get_template("template.html")
    
    # Render final internal HTML
    html_out = template.render(conditions=condition_data)
    
    # Write to index.html
    with open(docs_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_out)
        
    print("Site built successfully! Open docs/index.html to view.")

if __name__ == "__main__":
    main()
