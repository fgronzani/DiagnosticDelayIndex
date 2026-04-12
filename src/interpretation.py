"""
Clinical interpretation module.

Translates quantitative results into plain-language clinical narratives.
This is a CRITICAL component — raw numbers alone don't inform policy.

IMPORTANT CAVEAT: These interpretations are proxy-based signals, NOT definitive
evidence of diagnostic delay. They should be used to guide further investigation,
not to make direct clinical or policy conclusions.
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def interpret_ddi_trend(trend_info: dict, config: AnalysisConfig) -> str:
    """Generate clinical interpretation of DDI temporal trend.

    Args:
        trend_info: Dictionary with trend statistics.
        config: Analysis configuration.

    Returns:
        Multi-paragraph clinical interpretation.
    """
    condition = config.condition.condition_name
    metric = trend_info.get("metric", "ddi")
    direction = trend_info.get("direction", "unknown")
    p_value = trend_info.get("p_value", 1.0)
    slope = trend_info.get("slope", 0)
    significant = trend_info.get("significant", False)
    first_year = trend_info.get("first_year", "N/A")
    last_year = trend_info.get("last_year", "N/A")
    pct_change = trend_info.get("pct_change", 0)

    lines = []
    lines.append(f"═══ CLINICAL INTERPRETATION — {condition} ═══\n")

    # Main finding
    if direction == "increasing" and significant:
        lines.append(
            f"⚠️  SIGNAL DETECTED: The Diagnostic Delay Index (DDI) shows a "
            f"statistically significant INCREASING trend (p={p_value:.4f}) "
            f"from {first_year} to {last_year}."
        )
        lines.append(
            f"\nThis means a growing proportion of {condition} patients are "
            f"presenting at admission with indicators of advanced disease — "
            f"including higher rates of ICU admission, mortality, and prolonged "
            f"hospitalization."
        )
        lines.append(
            f"\n📋 Possible explanations include:"
            f"\n   1. Delayed access to primary care or diagnostic services"
            f"\n   2. Barriers to seeking care (geographic, financial, cultural)"
            f"\n   3. Reduced availability or effectiveness of screening programs"
            f"\n   4. Changes in referral patterns or hospital admission criteria"
            f"\n   5. Demographic shifts (aging population) increasing baseline severity"
        )
    elif direction == "decreasing" and significant:
        lines.append(
            f"✅ POSITIVE SIGNAL: The DDI shows a statistically significant "
            f"DECREASING trend (p={p_value:.4f}) from {first_year} to {last_year}."
        )
        lines.append(
            f"\nThis suggests that {condition} patients may be arriving at hospitals "
            f"in less severe conditions over time, potentially reflecting improvements "
            f"in early diagnosis, screening programs, or access to care."
        )
    elif not significant:
        lines.append(
            f"ℹ️  NO SIGNIFICANT TREND: The DDI does not show a statistically "
            f"significant temporal trend (p={p_value:.4f}) over the period "
            f"{first_year}–{last_year}."
        )
        lines.append(
            f"\nSeverity patterns for {condition} appear relatively stable, "
            f"though this does not rule out localized or subgroup-level changes."
        )
    else:
        lines.append(f"Trend direction: {direction} (p={p_value:.4f})")

    # Magnitude
    if pct_change is not None and not (isinstance(pct_change, float) and (pct_change != pct_change)):
        lines.append(
            f"\n📊 Total change: {pct_change:+.1f}% over the analysis period."
        )

    # Caveats
    lines.append(
        f"\n⚠️  IMPORTANT LIMITATIONS:"
        f"\n   • The DDI is a PROXY measure. It does not directly prove diagnostic delay."
        f"\n   • Changes in coding practices, hospital capacity, or patient demographics"
        f"\n     may affect these indicators independently of actual diagnostic timing."
        f"\n   • This analysis uses aggregate data without patient-level linkage."
        f"\n   • Ecological fallacy: population-level trends may not apply to individuals."
    )

    return "\n".join(lines)


def interpret_regional_comparison(
    comparison: dict,
    config: AnalysisConfig
) -> str:
    """Generate clinical interpretation of regional DDI variation.

    Args:
        comparison: Dictionary with regional comparison statistics.
        config: Analysis configuration.

    Returns:
        Regional interpretation text.
    """
    condition = config.condition.condition_name
    lines = []

    lines.append(f"\n═══ REGIONAL ANALYSIS — {condition} ═══\n")

    n_regions = comparison.get("n_regions", 0)
    if n_regions == 0:
        lines.append("Insufficient data for regional comparison.")
        return "\n".join(lines)

    mean_ddi = comparison.get("mean_ddi", 0)
    range_ddi = comparison.get("range_ddi", 0)

    lines.append(
        f"Analyzed {n_regions} regions with sufficient case volume "
        f"(≥{config.min_cases_threshold} cases)."
    )
    lines.append(f"\nMean DDI across regions: {mean_ddi:.1%} (range: {range_ddi:.1%})")

    if range_ddi > 0.15:
        lines.append(
            f"\n⚠️  SUBSTANTIAL REGIONAL VARIATION detected. "
            f"The {range_ddi:.1%} spread in DDI across regions suggests "
            f"significant geographic disparities in disease severity at presentation."
        )
        lines.append(
            f"\nRegions with higher DDI may face:"
            f"\n   • Limited access to diagnostic facilities"
            f"\n   • Shortage of primary care providers"
            f"\n   • Geographic barriers to healthcare"
            f"\n   • Socioeconomic factors affecting care-seeking behavior"
        )
    elif range_ddi > 0.05:
        lines.append(
            f"\nModerate regional variation observed. Some disparities in "
            f"presentation severity exist but are within a narrower range."
        )
    else:
        lines.append(
            f"\nRelatively uniform DDI across regions, suggesting similar "
            f"patterns of disease severity at presentation."
        )

    # Top/bottom regions
    top_5 = comparison.get("top_5_regions", [])
    bottom_5 = comparison.get("bottom_5_regions", [])

    if top_5:
        lines.append(f"\n🔴 Highest DDI regions (potential priority areas):")
        for r in top_5:
            region_name = list(r.values())[0]
            ddi_val = list(r.values())[1]
            lines.append(f"   • {region_name}: DDI = {ddi_val:.1%}")

    if bottom_5:
        lines.append(f"\n🟢 Lowest DDI regions:")
        for r in bottom_5:
            region_name = list(r.values())[0]
            ddi_val = list(r.values())[1]
            lines.append(f"   • {region_name}: DDI = {ddi_val:.1%}")

    return "\n".join(lines)


def interpret_component_trends(analysis_results: dict, config: AnalysisConfig) -> str:
    """Interpret individual severity component trends.

    Args:
        analysis_results: Full analysis results dictionary.
        config: Analysis configuration.

    Returns:
        Component-level interpretation text.
    """
    condition = config.condition.condition_name
    lines = []

    lines.append(f"\n═══ COMPONENT ANALYSIS — {condition} ═══\n")

    components = {
        "trend_mortality_rate": ("Mortality Rate", "mortality among hospitalized patients"),
        "trend_icu_rate": ("ICU Admission Rate", "ICU utilization among admissions"),
        "trend_avg_severity": ("Average Severity Score", "composite severity measure"),
    }

    for key, (label, description) in components.items():
        trend = analysis_results.get(key, {})
        direction = trend.get("direction", "unknown")
        p_value = trend.get("p_value", 1.0)
        significant = trend.get("significant", False)

        marker = "📈" if direction == "increasing" else "📉" if direction == "decreasing" else "➡️"
        sig_text = f"(p={p_value:.4f}, {'significant' if significant else 'not significant'})"

        lines.append(f"{marker} {label}: {direction} {sig_text}")
        lines.append(f"   {description}")

    # Combined interpretation
    mort_trend = analysis_results.get("trend_mortality_rate", {})
    icu_trend = analysis_results.get("trend_icu_rate", {})

    if (mort_trend.get("direction") == "increasing" and mort_trend.get("significant")) and \
       (icu_trend.get("direction") == "increasing" and icu_trend.get("significant")):
        lines.append(
            f"\n🚨 CONVERGENT SIGNAL: Both mortality AND ICU rates are increasing "
            f"significantly. This pattern is strongly consistent with later presentation "
            f"of {condition}, though confounding factors must be investigated."
        )
    elif mort_trend.get("direction") == "increasing" and mort_trend.get("significant"):
        lines.append(
            f"\n⚠️  Rising mortality alongside stable ICU rates may suggest "
            f"capacity constraints or changes in patient severity beyond what "
            f"ICU admission captures."
        )

    return "\n".join(lines)


def generate_full_report(
    analysis_results: dict,
    config: AnalysisConfig
) -> str:
    """Generate a complete clinical interpretation report.

    Args:
        analysis_results: Full analysis results from run_full_analysis().
        config: Analysis configuration.

    Returns:
        Complete interpretation report as formatted text.
    """
    sections = []

    # Header
    sections.append("╔" + "═" * 68 + "╗")
    sections.append("║  DIAGNOSTIC DELAY DETECTOR — CLINICAL INTERPRETATION REPORT" + " " * 7 + "║")
    sections.append("╚" + "═" * 68 + "╝")
    sections.append(f"\nCondition: {config.condition.condition_name}")
    sections.append(f"ICD-10 prefixes: {', '.join(config.condition.icd_prefixes)}")

    # DDI trend
    ddi_trend = analysis_results.get("trend_ddi", {})
    sections.append(interpret_ddi_trend(ddi_trend, config))

    # Component trends
    sections.append(interpret_component_trends(analysis_results, config))

    # Regional comparison
    regional = analysis_results.get("regional_comparison", {})
    sections.append(interpret_regional_comparison(regional, config))

    # Age-adjusted insights
    age_trends = analysis_results.get("age_adjusted_trends", {})
    if age_trends:
        sections.append(f"\n═══ AGE-ADJUSTED ANALYSIS ═══\n")
        for age_group, trend in age_trends.items():
            direction = trend.get("direction", "unknown")
            significant = trend.get("significant", False)
            p_value = trend.get("p_value", 1.0)
            marker = "📈" if direction == "increasing" else "📉" if direction == "decreasing" else "➡️"
            sig = "✓" if significant else ""
            sections.append(f"  {marker} {age_group}: {direction} (p={p_value:.4f}) {sig}")

        # Check for age-specific patterns
        increasing_groups = [g for g, t in age_trends.items()
                            if t.get("direction") == "increasing" and t.get("significant")]
        if increasing_groups:
            sections.append(
                f"\n  ⚠️  Age groups with significant increasing DDI: {', '.join(increasing_groups)}"
                f"\n  This may indicate targeted screening or access barriers for these demographics."
            )

    # Final disclaimer
    sections.append(
        f"\n{'─' * 70}"
        f"\n📌 DISCLAIMER: This analysis detects INDIRECT SIGNALS using proxy "
        f"indicators. Results suggest patterns worthy of investigation but do "
        f"NOT constitute proof of diagnostic delay. Clinical audit, patient "
        f"pathway analysis, and expert review are required before policy action."
        f"\n{'─' * 70}"
    )

    report = "\n".join(sections)
    return report
