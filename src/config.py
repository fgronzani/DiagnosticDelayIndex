"""
Configuration module for the Diagnostic Delay Detector.

Centralizes all configurable parameters: column mappings, severity weights,
thresholds, and condition filters. No hardcoded values in other modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnConfig:
    """Maps expected column names to actual dataset column names."""
    diagnosis_code: str = "cid"
    age: str = "idade"
    sex: str = "sexo"
    municipality: str = "municipio"
    year: str = "ano"
    icu: str = "uti"
    death: str = "obito"
    length_of_stay: str = "tempo_internacao"


@dataclass
class SeverityWeights:
    """Weights for severity score computation.

    The severity score is a composite metric:
        score = (icu_weight * icu_flag) + (death_weight * death_flag) + (los_weight * normalized_los)

    Default weights reflect clinical reasoning:
        - Death (+3): strongest indicator of late presentation
        - ICU (+2): significant resource use suggesting advanced disease
        - Length of stay (normalized): continuous severity indicator
    """
    icu_weight: float = 2.0
    death_weight: float = 3.0
    los_weight: float = 1.0
    los_normalization: str = "minmax"  # "minmax" or "zscore"


@dataclass
class DDIConfig:
    """Configuration for the Diagnostic Delay Index computation."""
    severity_threshold_percentile: float = 75.0  # percentile to define "high severity"
    severity_threshold_fixed: Optional[float] = None  # if set, overrides percentile

    def reset_fixed_threshold(self):
        """Reset fixed threshold so it's recalculated on next pipeline run."""
        self.severity_threshold_fixed = None

@dataclass
class ConditionFilter:
    """Defines which ICD-10 conditions to analyze.

    icd_prefixes: list of ICD-10 code prefixes to include (e.g., ["I21"] for AMI)
    condition_name: human-readable name for reports and plots
    """
    icd_prefixes: list[str] = field(default_factory=lambda: ["I21"])
    condition_name: str = "Acute Myocardial Infarction (AMI)"


@dataclass
class AnalysisConfig:
    """Full analysis configuration."""
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    severity: SeverityWeights = field(default_factory=SeverityWeights)
    ddi: DDIConfig = field(default_factory=DDIConfig)
    condition: ConditionFilter = field(default_factory=ConditionFilter)
    age_bins: list[int] = field(default_factory=lambda: [0, 18, 40, 60, 80, 120])
    age_labels: list[str] = field(default_factory=lambda: ["0-17", "18-39", "40-59", "60-79", "80+"])
    min_cases_threshold: int = 30  # minimum cases for a region/year to be included
    output_dir: str = "outputs"
