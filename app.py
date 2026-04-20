#!/usr/bin/env python3
"""
Diagnostic Delay Detector — Streamlit Dashboard

Interactive web application for exploring Diagnostic Delay Index
analysis results. Allows dynamic configuration and real-time visualization.

Run with:
    streamlit run app.py
"""

import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from src.config import AnalysisConfig, ConditionFilter, DDIConfig
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.metrics import (
    compute_temporal_metrics,
    compute_regional_metrics,
    compute_age_adjusted_metrics,
    compute_regional_temporal_metrics,
)
from src.analysis import run_full_analysis, detect_temporal_trend
from src.interpretation import generate_full_report
from src.generate_data import generate_synthetic_data
from src.covid_analysis import is_covid_condition
from src.sensitivity_analysis import run_sensitivity_analysis, sensitivity_summary
from src.metrics import compute_ddi_confidence_intervals

import plotly.graph_objects as go
import plotly.express as px

def plot_ddi_temporal_plotly(
    temporal_metrics: pd.DataFrame,
    ddi_ci: pd.DataFrame,
    trend_info: dict,
    config,
    show_covid_band: bool = True,
) -> go.Figure:
    """DDI time series with CI band, trend line, and COVID annotation."""
    year_col = config.columns.year
    
    fig = go.Figure()
    
    # CI band
    if ddi_ci is not None and "ddi_ci_lower" in ddi_ci.columns:
        fig.add_trace(go.Scatter(
            x=list(ddi_ci[year_col]) + list(ddi_ci[year_col])[::-1],
            y=list(ddi_ci["ddi_ci_upper"]) + list(ddi_ci["ddi_ci_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(46, 134, 193, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI (bootstrap)",
            hoverinfo="skip",
        ))
    
    # DDI line
    fig.add_trace(go.Scatter(
        x=temporal_metrics[year_col],
        y=temporal_metrics["ddi"],
        mode="lines+markers",
        line=dict(color="#2E86C1", width=2.5),
        marker=dict(size=7, color="#2E86C1"),
        name="DDI",
        hovertemplate="<b>Year:</b> %{x}<br><b>DDI:</b> %{y:.1%}<extra></extra>",
    ))
    
    # OLS trend line
    slope = trend_info.get("slope")
    intercept = trend_info.get("intercept")
    if slope is not None and intercept is not None and not np.isnan(slope):
        x_vals = temporal_metrics[year_col].values
        y_trend = slope * x_vals + intercept
        p_val = trend_info.get("p_value_bonferroni", trend_info.get("p_value", 1.0))
        sig_label = f" (p={p_val:.4f}, {'sig.' if p_val < 0.05 else 'n.s.'})"
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_trend,
            mode="lines",
            line=dict(color="#E74C3C", width=1.5, dash="dash"),
            name=f"OLS trend{sig_label}",
        ))
    
    # COVID shading
    if show_covid_band:
        fig.add_vrect(
            x0=2019.5, x1=2021.5,
            fillcolor="rgba(231, 76, 60, 0.08)",
            layer="below",
            line_width=0,
            annotation_text="COVID-19",
            annotation_position="top left",
            annotation=dict(font_size=11, font_color="#E74C3C"),
        )
    
    fig.update_layout(
        title=dict(text=f"Diagnostic Delay Index — {config.condition.condition_name}", font_size=16),
        xaxis_title="Year",
        yaxis_title="DDI (proportion high-severity)",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    
    return fig

logging.basicConfig(level=logging.WARNING)

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diagnostic Delay Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5D6D7E;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2E86C1;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/?size=80&id=46891&format=png", width=60)
    st.title("⚙️ Configuration")

    st.markdown("---")
    st.subheader("📂 Data Source")

    data_source = st.radio(
        "Select data source:",
        ["🧪 Generate Synthetic Data", "📁 Upload CSV File"],
        index=0,
    )

    if data_source == "🧪 Generate Synthetic Data":
        n_records = st.slider("Number of records", 5000, 50000, 15000, 1000)
        severity_trend = st.slider(
            "Severity trend (annual increase)",
            0.0, 0.05, 0.02, 0.005,
            help="Higher values simulate increasing diagnostic delay",
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.subheader("🎯 Condition")

    CONDITION_OPTIONS = {
        "I21 — Acute Myocardial Infarction": ("I21", "Acute Myocardial Infarction (AMI)"),
        "I63 — Ischemic Stroke": ("I63", "Ischemic Stroke"),
        "C34 — Lung Cancer": ("C34", "Lung Cancer"),
        "J18 — Pneumonia": ("J18", "Pneumonia"),
        "K35 — Acute Appendicitis": ("K35", "Acute Appendicitis"),
    }

    selected_condition = st.selectbox(
        "Target condition:",
        list(CONDITION_OPTIONS.keys()),
    )
    icd_prefix, condition_name = CONDITION_OPTIONS[selected_condition]

    custom_icd = st.text_input(
        "Or enter custom ICD-10 prefix(es):",
        placeholder="e.g., I21,I22",
        help="Comma-separated ICD-10 prefixes",
    )

    if custom_icd.strip():
        icd_prefixes = [p.strip().upper() for p in custom_icd.split(",")]
        condition_name = f"Custom ({', '.join(icd_prefixes)})"
    else:
        icd_prefixes = [icd_prefix]

    st.markdown("---")
    st.subheader("📊 Thresholds")

    threshold_pct = st.slider(
        "High-severity percentile",
        50, 95, 75, 5,
        help="Cases above this percentile are classified as 'high severity'",
    )

    min_cases = st.slider(
        "Minimum cases per group",
        5, 100, 30, 5,
        help="Groups with fewer cases are excluded from analysis",
    )

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🏥 Diagnostic Delay Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Detecting Late Presentation Signals in Healthcare Systems</p>',
    unsafe_allow_html=True,
)

if not run_analysis and "results" not in st.session_state:
    st.info(
        "👈 Configure the analysis parameters in the sidebar and click **Run Analysis** to start.",
        icon="ℹ️",
    )

    with st.expander("📖 About this tool", expanded=True):
        st.markdown("""
        **The Diagnostic Delay Detector** identifies indirect signals of delayed diagnosis
        using hospital-level data. It answers the question:

        > *"Are patients arriving at hospitals in more severe conditions over time?"*

        ### How it works
        1. **Severity Score**: Each admission gets a composite score based on ICU admission (+2),
           death (+3), and normalized length of stay (+1)
        2. **Diagnostic Delay Index (DDI)**: The proportion of cases classified as "high severity"
        3. **Trend Analysis**: Statistical tests detect whether severity is increasing over time
        4. **Regional Comparison**: Identifies geographic disparities in presentation severity

        ### ⚠️ Important Caveats
        - The DDI is a **proxy measure** — it does not directly prove diagnostic delay
        - Changes in coding, capacity, or demographics can affect results
        - This tool generates hypotheses for further clinical investigation
        """)
    st.stop()

if run_analysis:
    # Build config
    config = AnalysisConfig(
        condition=ConditionFilter(
            icd_prefixes=icd_prefixes,
            condition_name=condition_name,
        ),
        ddi=DDIConfig(severity_threshold_percentile=float(threshold_pct)),
        min_cases_threshold=min_cases,
        output_dir="outputs",
    )
    config.ddi.reset_fixed_threshold()

    with st.spinner("Running analysis pipeline..."):
        # Load/generate data
        if data_source == "🧪 Generate Synthetic Data":
            data_path = "data/synthetic_sih_data.csv"
            generate_synthetic_data(
                n_records=n_records,
                severity_trend=severity_trend,
                output_path=data_path,
            )
        else:
            if uploaded_file is None:
                st.error("Please upload a CSV file.")
                st.stop()
            data_path = "data/uploaded_data.csv"
            Path("data").mkdir(exist_ok=True)
            with open(data_path, "wb") as f:
                f.write(uploaded_file.getvalue())

        # Pipeline
        try:
            df_preprocessed = preprocess_pipeline(data_path, config)
            if len(df_preprocessed) == 0:
                st.error(f"No records found for condition {condition_name} with prefixes {icd_prefixes}")
                st.stop()
            
            st.session_state["df_preprocessed"] = df_preprocessed.copy()

            df = feature_engineering_pipeline(df_preprocessed, config)
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
            regional_temporal_metrics = compute_regional_temporal_metrics(df, config)
            analysis_results = run_full_analysis(
                temporal_metrics, regional_metrics, age_adjusted_metrics, config, df_admissions=df
            )
            report = generate_full_report(analysis_results, config)

            # Store in session state
            st.session_state["results"] = {
                "df": df,
                "temporal_metrics": temporal_metrics,
                "regional_metrics": regional_metrics,
                "age_adjusted_metrics": age_adjusted_metrics,
                "regional_temporal_metrics": regional_temporal_metrics,
                "analysis_results": analysis_results,
                "report": report,
                "config": config,
            }
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.stop()

# ── Display Results ──────────────────────────────────────────────────────────

if "results" in st.session_state:
    r = st.session_state["results"]
    df = r["df"]
    temporal_metrics = r["temporal_metrics"]
    regional_metrics = r["regional_metrics"]
    age_adjusted_metrics = r["age_adjusted_metrics"]
    analysis_results = r["analysis_results"]
    report = r["report"]
    config = r["config"]
    cols = config.columns

    # ── Summary Metrics ──────────────────────────────────────────────────
    ddi_trend = analysis_results.get("trend_ddi", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Cases",
            f"{len(df):,}",
            help="Number of admissions analyzed",
        )
    with col2:
        latest_ddi = temporal_metrics["ddi"].iloc[-1] if len(temporal_metrics) > 0 else 0
        st.metric(
            "Latest DDI",
            f"{latest_ddi:.1%}",
            delta=f"{ddi_trend.get('slope', 0)*100:+.2f}%/yr" if ddi_trend.get("slope") else None,
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Mortality Rate",
            f"{df[cols.death].mean():.1%}",
        )
    with col4:
        st.metric(
            "ICU Rate",
            f"{df[cols.icu].mean():.1%}",
        )
    with col5:
        st.metric(
            "Avg Stay (days)",
            f"{df[cols.length_of_stay].mean():.1f}",
        )

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_trends, tab_regional, tab_age, tab_covid, tab_sensitivity, tab_report, tab_data = st.tabs([
        "📈 Temporal Trends",
        "🗺️ Regional Analysis",
        "👥 Age-Adjusted",
        "🦠 COVID Analysis",
        "⚖️ Sensitivity",
        "📋 Clinical Report",
        "🗄️ Raw Data",
    ])

    with tab_trends:
        st.subheader(f"Temporal Trends — {config.condition.condition_name}")

        # Trend significance badge
        if ddi_trend.get("significant"):
            direction = ddi_trend.get("direction", "unknown")
            if direction == "increasing":
                st.error(
                    f"⚠️ **Significant increasing trend** detected "
                    f"(p={ddi_trend.get('p_value', 0):.4f}, R²={ddi_trend.get('r_squared', 0):.3f})"
                )
            else:
                st.success(
                    f"✅ **Significant decreasing trend** detected "
                    f"(p={ddi_trend.get('p_value', 0):.4f})"
                )
        else:
            st.info(f"ℹ️ No statistically significant trend (p={ddi_trend.get('p_value', 0):.4f})")

        # Charts
        st.plotly_chart(
            plot_ddi_temporal_plotly(
                temporal_metrics, temporal_metrics, ddi_trend, config, 
                show_covid_band=True
            ),
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(temporal_metrics, x=cols.year, y=["mortality_rate", "icu_rate"], 
                          title="Mortality & ICU Rates", markers=True)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.line(temporal_metrics, x=cols.year, y="avg_severity", 
                          title="Average Severity Score", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📊 Metrics Table")
        display_cols = [cols.year, "ddi", "mortality_rate", "icu_rate", "avg_severity", "avg_los", "total_cases"]
        display_df = temporal_metrics[display_cols].copy()
        display_df.columns = ["Year", "DDI", "Mortality Rate", "ICU Rate", "Avg Severity", "Avg LOS", "Cases"]

        st.dataframe(
            display_df.style.format({
                "DDI": "{:.1%}",
                "Mortality Rate": "{:.1%}",
                "ICU Rate": "{:.1%}",
                "Avg Severity": "{:.3f}",
                "Avg LOS": "{:.1f}",
                "Cases": "{:,.0f}",
            }),
            use_container_width=True,
        )

    with tab_regional:
        st.subheader(f"Regional Analysis — {config.condition.condition_name}")

        if len(regional_metrics) > 0:
            fig_reg = px.bar(
                regional_metrics.sort_values("ddi", ascending=False).head(20),
                x=cols.municipality, y="ddi",
                title=f"DDI by Region (top {min(20, len(regional_metrics))} regions)",
                color="ddi", color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            st.subheader("📊 Regional Rankings")
            regional_display = regional_metrics.copy()
            regional_display.insert(0, "Rank", range(1, len(regional_display) + 1))
            display_cols_r = ["Rank", cols.municipality, "ddi", "mortality_rate", "icu_rate", "avg_los", "total_cases"]
            st.dataframe(
                regional_display[display_cols_r].style.format({
                    "ddi": "{:.1%}",
                    "mortality_rate": "{:.1%}",
                    "icu_rate": "{:.1%}",
                    "avg_los": "{:.1f}",
                    "total_cases": "{:,.0f}",
                }).background_gradient(subset=["ddi"], cmap="RdYlGn_r"),
                use_container_width=True,
            )
        else:
            st.warning("No regions with sufficient data for analysis.")

    with tab_age:
        st.subheader(f"Age-Adjusted Analysis — {config.condition.condition_name}")

        if len(age_adjusted_metrics) > 0 and "age_group" in age_adjusted_metrics.columns:
            fig_age = px.line(
                age_adjusted_metrics, x=cols.year, y="ddi", color="age_group",
                title="DDI by Age Group over time", markers=True
            )
            st.plotly_chart(fig_age, use_container_width=True)

            # Age-specific trends
            age_trends = analysis_results.get("age_adjusted_trends", {})
            if age_trends:
                st.subheader("Age Group Trend Summary")
                trend_data = []
                for group_name, trend in age_trends.items():
                    trend_data.append({
                        "Age Group": group_name,
                        "Direction": trend.get("direction", "N/A"),
                        "p-value": trend.get("p_value", "N/A"),
                        "Significant": "✅" if trend.get("significant") else "❌",
                    })
                st.dataframe(pd.DataFrame(trend_data), use_container_width=True)
        else:
            st.warning("No age-adjusted data available.")

    with tab_covid:
        st.subheader(f"COVID-19 Natural Experiment — {config.condition.condition_name}")
        if is_covid_condition(config.condition.icd_prefixes):
            st.warning("This analysis is disabled for COVID-related conditions (e.g., Pneumonia), as the shock is not exogenous.")
        else:
            if "covid_its" in analysis_results:
                its = analysis_results["covid_its"]
                comp = analysis_results["covid_analysis"]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Immediate DDI Change (ITS)", 
                              f"{its.get('level_change_at_covid', 0):+.2%}",
                              help="Estimated level change in 2020")
                    st.metric("DDI Trend Slope Change", 
                              f"{its.get('slope_change_after_covid', 0)*100:+.2f}%/yr")
                with c2:
                    st.write("**Interrupted Time Series (ITS) Results**")
                    st.write(f"- Pre-COVID Mean: {comp.get('pre_covid_mean', 0):.1%}")
                    st.write(f"- COVID Period Mean: {comp.get('covid_mean', 0):.1%}")
                    st.write(f"- Post-COVID Mean: {comp.get('post_covid_mean', 0):.1%}")
                
                st.info(f"**Interpretation:** {its.get('interpretation', 'No valid ITS result')}")
            else:
                st.write("Run full analysis to see COVID-19 effect outcomes.")

    with tab_sensitivity:
        st.subheader("⚖️ Sensitivity Analysis (Severity Weights)")
        st.write("Test whether the observed DDI trend is robust to different severity weight assignments.")
        
        if st.checkbox("Run Sensitivity Analysis"):
            with st.spinner("Running pipeline for multiple weight combinations..."):
                sens_df = run_sensitivity_analysis(st.session_state["df_preprocessed"], config)
                sens_summary = sensitivity_summary(sens_df)
                
                st.write(f"**Robustness:** {sens_summary.get('direction_robustness_pct', 0):.1f}% of combinations agree with the baseline trend.")
                if sens_summary.get("conclusion_robust"):
                    st.success("✅ The trend conclusion is robust to weight changes.")
                else:
                    st.warning("⚠️ The trend conclusion is sensitive to weight changes.")
                    
                st.dataframe(sens_df, use_container_width=True)

    with tab_report:
        st.subheader("📋 Clinical Interpretation Report")
        st.text(report)

        st.download_button(
            "📥 Download Report",
            report,
            file_name="diagnostic_delay_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab_data:
        st.subheader("🗄️ Raw Data Preview")
        st.dataframe(df.head(500), use_container_width=True)
        st.caption(f"Showing first 500 of {len(df):,} records")

        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Processed Data",
            csv,
            file_name="processed_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
