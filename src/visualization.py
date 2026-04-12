"""
Visualization module.

Publication-quality charts for Diagnostic Delay Index analysis.
All plots are saved to the configured output directory.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .config import AnalysisConfig

logger = logging.getLogger(__name__)

# ── Style Configuration ──────────────────────────────────────────────────────

PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F39C12",
    "bg": "#FAFBFC",
    "grid": "#E8ECF0",
    "text": "#2C3E50",
}

def _setup_style():
    """Configure matplotlib for clean, professional plots."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.7,
        "grid.linewidth": 0.5,
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })

_setup_style()


def _ensure_output_dir(config: AnalysisConfig) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_ddi_time_series(
    temporal_metrics: pd.DataFrame,
    config: AnalysisConfig,
    trend_info: Optional[dict] = None,
    filename: str = "ddi_time_series.png"
) -> Path:
    """Plot DDI over time with optional trend line.

    Args:
        temporal_metrics: DataFrame with year and DDI columns.
        config: Analysis configuration.
        trend_info: Optional dict with slope/intercept for trend overlay.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    output_dir = _ensure_output_dir(config)
    cols = config.columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Diagnostic Delay Index — {config.condition.condition_name}",
        fontsize=16, fontweight="bold", y=1.02
    )

    years = temporal_metrics[cols.year]

    # DDI over time
    ax = axes[0, 0]
    ax.plot(years, temporal_metrics["ddi"], "o-", color=PALETTE["primary"], linewidth=2.5, markersize=7)
    if trend_info and not np.isnan(trend_info.get("slope", np.nan)):
        trend_y = trend_info["slope"] * years + trend_info["intercept"]
        label = f"Trend (p={trend_info['p_value']:.3f})"
        ax.plot(years, trend_y, "--", color=PALETTE["accent"], linewidth=1.5, alpha=0.8, label=label)
        ax.legend(fontsize=9)
    ax.set_title("Diagnostic Delay Index (DDI)")
    ax.set_xlabel("Year")
    ax.set_ylabel("DDI (proportion high-severity)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Mortality rate
    ax = axes[0, 1]
    ax.plot(years, temporal_metrics["mortality_rate"], "o-", color=PALETTE["accent"], linewidth=2.5, markersize=7)
    ax.set_title("Mortality Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mortality Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ICU rate
    ax = axes[1, 0]
    ax.plot(years, temporal_metrics["icu_rate"], "o-", color=PALETTE["warning"], linewidth=2.5, markersize=7)
    ax.set_title("ICU Admission Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("ICU Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Average severity
    ax = axes[1, 1]
    ax.fill_between(years, temporal_metrics["avg_severity"], alpha=0.3, color=PALETTE["secondary"])
    ax.plot(years, temporal_metrics["avg_severity"], "o-", color=PALETTE["secondary"], linewidth=2.5, markersize=7)
    ax.set_title("Average Severity Score")
    ax.set_xlabel("Year")
    ax.set_ylabel("Severity Score")

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Saved time series plot: {filepath}")
    return filepath


def plot_regional_bar_chart(
    regional_metrics: pd.DataFrame,
    config: AnalysisConfig,
    top_n: int = 20,
    filename: str = "regional_ddi_bar.png"
) -> Path:
    """Plot horizontal bar chart of DDI by region.

    Args:
        regional_metrics: DataFrame with region and DDI columns.
        config: Analysis configuration.
        top_n: Number of top regions to show.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    output_dir = _ensure_output_dir(config)
    cols = config.columns

    df = regional_metrics.nlargest(top_n, "ddi").sort_values("ddi", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))

    # Color gradient based on DDI value
    norm = plt.Normalize(df["ddi"].min(), df["ddi"].max())
    colors = plt.cm.RdYlGn_r(norm(df["ddi"].values))

    bars = ax.barh(
        df[cols.municipality].astype(str),
        df["ddi"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7
    )

    # Add value labels
    for bar, val in zip(bars, df["ddi"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=9,
            color=PALETTE["text"]
        )

    ax.set_title(
        f"DDI by Region — Top {len(df)} · {config.condition.condition_name}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Diagnostic Delay Index (DDI)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Saved regional bar chart: {filepath}")
    return filepath


def plot_severity_distribution(
    df: pd.DataFrame,
    config: AnalysisConfig,
    filename: str = "severity_distribution.png"
) -> Path:
    """Plot severity score distribution with threshold line.

    Args:
        df: Feature-engineered DataFrame.
        config: Analysis configuration.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    output_dir = _ensure_output_dir(config)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(
        df["severity_score"],
        bins=50,
        color=PALETTE["secondary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5
    )

    if config.ddi.severity_threshold_fixed is not None:
        threshold = config.ddi.severity_threshold_fixed
    else:
        threshold = df["severity_score"].quantile(config.ddi.severity_threshold_percentile / 100)

    ax.axvline(threshold, color=PALETTE["accent"], linestyle="--", linewidth=2, label=f"Threshold: {threshold:.2f}")
    ax.legend()
    ax.set_title("Severity Score Distribution")
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Frequency")

    # Box plot by year
    ax = axes[1]
    cols = config.columns
    years_present = sorted(df[cols.year].unique())

    data_by_year = [df[df[cols.year] == y]["severity_score"].values for y in years_present]
    bp = ax.boxplot(
        data_by_year,
        labels=[str(int(y)) for y in years_present],
        patch_artist=True,
        widths=0.6
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["secondary"])
        patch.set_alpha(0.6)

    ax.set_title("Severity Score by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Severity Score")

    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Saved severity distribution: {filepath}")
    return filepath


def plot_heatmap(
    df: pd.DataFrame,
    config: AnalysisConfig,
    filename: str = "regional_temporal_heatmap.png"
) -> Path:
    """Plot heatmap of DDI by region and year.

    Args:
        df: Regional-temporal metrics DataFrame.
        config: Analysis configuration.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    output_dir = _ensure_output_dir(config)
    cols = config.columns

    if len(df) == 0:
        logger.warning("No data for heatmap. Skipping.")
        return output_dir / filename

    pivot = df.pivot_table(
        values="ddi",
        index=cols.municipality,
        columns=cols.year,
        aggfunc="mean"
    )

    if pivot.empty:
        logger.warning("Pivot table empty. Skipping heatmap.")
        return output_dir / filename

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.5)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn_r",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "DDI", "format": mticker.PercentFormatter(xmax=1.0)},
        vmin=0
    )

    ax.set_title(
        f"DDI Heatmap — Region × Year · {config.condition.condition_name}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Region")

    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Saved heatmap: {filepath}")
    return filepath


def plot_age_adjusted_trends(
    age_adjusted_metrics: pd.DataFrame,
    config: AnalysisConfig,
    filename: str = "age_adjusted_ddi.png"
) -> Path:
    """Plot DDI trends stratified by age group.

    Args:
        age_adjusted_metrics: DataFrame with year, age_group, and DDI.
        config: Analysis configuration.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    output_dir = _ensure_output_dir(config)
    cols = config.columns

    if len(age_adjusted_metrics) == 0:
        logger.warning("No age-adjusted data. Skipping.")
        return output_dir / filename

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("husl", n_colors=len(age_adjusted_metrics["age_group"].unique()))

    for i, (age_group, group) in enumerate(age_adjusted_metrics.groupby("age_group")):
        ax.plot(
            group[cols.year],
            group["ddi"],
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=str(age_group)
        )

    ax.set_title(
        f"DDI by Age Group — {config.condition.condition_name}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Diagnostic Delay Index")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(title="Age Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Saved age-adjusted trends: {filepath}")
    return filepath


def generate_all_plots(
    df: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    regional_metrics: pd.DataFrame,
    age_adjusted_metrics: pd.DataFrame,
    regional_temporal_metrics: pd.DataFrame,
    config: AnalysisConfig,
    trend_info: Optional[dict] = None
) -> list[Path]:
    """Generate all visualization plots.

    Args:
        df: Feature-engineered DataFrame.
        temporal_metrics: DDI by year.
        regional_metrics: DDI by region.
        age_adjusted_metrics: DDI by year + age group.
        regional_temporal_metrics: DDI by region × year.
        config: Analysis configuration.
        trend_info: Optional trend line info for DDI.

    Returns:
        List of paths to generated figures.
    """
    paths = []

    paths.append(plot_ddi_time_series(temporal_metrics, config, trend_info))
    paths.append(plot_regional_bar_chart(regional_metrics, config))
    paths.append(plot_severity_distribution(df, config))
    paths.append(plot_heatmap(regional_temporal_metrics, config))
    paths.append(plot_age_adjusted_trends(age_adjusted_metrics, config))

    logger.info(f"Generated {len(paths)} plots")
    return paths
