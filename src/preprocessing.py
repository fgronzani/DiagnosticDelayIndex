"""
Data preprocessing module.

Handles loading, cleaning, type conversion, and filtering of hospital
admission (SIH/AIH) data. Compatible with microdatasus output format.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV data with robust encoding handling.

    Tries UTF-8 first, falls back to latin-1 (common in Brazilian datasets).

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or has no valid rows.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            logger.info(f"Loaded {len(df)} rows from {filepath} (encoding: {encoding})")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode file {filepath} with any supported encoding")


def standardize_columns(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Normalize column names to lowercase and validate required columns exist.

    Args:
        df: Raw DataFrame.
        config: Analysis configuration.

    Returns:
        DataFrame with standardized column names.

    Raises:
        KeyError: If required columns are missing.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    cols = config.columns
    required = [cols.diagnosis_code, cols.year, cols.icu, cols.death, cols.length_of_stay]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_binary_column(series: pd.Series, col_name: str) -> pd.Series:
    """Convert a binary column to int (0/1).

    Handles various representations:
        - Numeric: 0, 1, 0.0, 1.0
        - String: "0", "1", "sim", "não", "yes", "no", "s", "n"
        - Boolean: True, False

    Args:
        series: The column to clean.
        col_name: Column name for logging.

    Returns:
        Integer Series with values 0 or 1.
    """
    s = series.copy()

    # Convert to string for uniform processing
    s = s.astype(str).str.strip().str.lower()

    mapping = {
        "1": 1, "1.0": 1, "sim": 1, "yes": 1, "s": 1, "true": 1,
        "0": 0, "0.0": 0, "não": 0, "nao": 0, "no": 0, "n": 0, "false": 0,
    }

    result = s.map(mapping)
    n_unmapped = result.isna().sum()
    if n_unmapped > 0:
        logger.warning(
            f"Column '{col_name}': {n_unmapped} values could not be mapped to 0/1 "
            f"(unique unmapped: {s[result.isna()].unique()[:5]}). Setting to 0."
        )
        result = result.fillna(0)

    return result.astype(int)


def clean_data(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Clean and type-convert all columns.

    Steps:
        1. Standardize column names
        2. Clean binary columns (uti, obito)
        3. Convert numeric columns (idade, ano, tempo_internacao)
        4. Strip and uppercase ICD codes
        5. Drop rows with critical missing values

    Args:
        df: Raw DataFrame.
        config: Analysis configuration.

    Returns:
        Cleaned DataFrame.
    """
    cols = config.columns
    df = standardize_columns(df, config)

    initial_rows = len(df)

    # Clean binary columns
    df[cols.icu] = clean_binary_column(df[cols.icu], cols.icu)
    df[cols.death] = clean_binary_column(df[cols.death], cols.death)

    # Convert numeric columns
    df[cols.length_of_stay] = pd.to_numeric(df[cols.length_of_stay], errors="coerce")
    df[cols.year] = pd.to_numeric(df[cols.year], errors="coerce")

    if cols.age in df.columns:
        df[cols.age] = pd.to_numeric(df[cols.age], errors="coerce")

    # Clean ICD codes
    df[cols.diagnosis_code] = df[cols.diagnosis_code].astype(str).str.strip().str.upper()

    # Remove invalid length of stay
    df = df[df[cols.length_of_stay] >= 0]
    df = df.dropna(subset=[cols.diagnosis_code, cols.year, cols.length_of_stay])

    final_rows = len(df)
    logger.info(f"Cleaning: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed)")

    return df.reset_index(drop=True)


def filter_condition(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Filter dataset to include only the target clinical condition(s).

    Matches ICD-10 codes by prefix (e.g., "I21" matches "I210", "I211", etc.)

    Args:
        df: Cleaned DataFrame.
        config: Analysis configuration.

    Returns:
        Filtered DataFrame containing only matching conditions.
    """
    cols = config.columns
    prefixes = [p.upper() for p in config.condition.icd_prefixes]

    mask = df[cols.diagnosis_code].apply(
        lambda code: any(code.startswith(prefix) for prefix in prefixes)
    )

    filtered = df[mask].copy()
    logger.info(
        f"Condition filter ({config.condition.condition_name}): "
        f"{len(df)} → {len(filtered)} rows "
        f"(prefixes: {prefixes})"
    )

    if len(filtered) == 0:
        logger.warning(
            f"No cases found for condition '{config.condition.condition_name}'. "
            f"Check ICD prefixes: {prefixes}. "
            f"Sample codes in data: {df[cols.diagnosis_code].unique()[:10]}"
        )

    return filtered.reset_index(drop=True)


def preprocess_pipeline(filepath: str | Path, config: AnalysisConfig) -> pd.DataFrame:
    """Execute the full preprocessing pipeline.

    Load → Clean → Filter by condition.

    Args:
        filepath: Path to the input CSV.
        config: Analysis configuration.

    Returns:
        Preprocessed DataFrame ready for feature engineering.
    """
    df = load_data(filepath)
    df = clean_data(df, config)
    df = filter_condition(df, config)
    return df
