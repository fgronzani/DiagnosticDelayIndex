"""
Synthetic data generator for development and demonstration.

Generates realistic hospital admission data mimicking SIH/DataSUS format
with configurable trends to test the Diagnostic Delay Detector.

NOTE: This is for development/testing ONLY. Real analyses should use
actual DataSUS/SIH data obtained via microdatasus or equivalent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Brazilian Microregions (Health Regiões de Saúde or IBGE Microrregiões)
MUNICIPALITIES = [
    "Micro. São Paulo", "Micro. Guarulhos", "Micro. Osasco", "Micro. Campinas", "Micro. Ribeirão Preto",
    "Micro. Bauru", "Micro. Sorocaba", "Micro. S. J. Rio Preto", "Micro. Presidente Prudente", "Micro. Araçatuba",
    "Micro. Santos", "Micro. São José dos Campos", "Micro. Taubaté", "Micro. Piracicaba", "Micro. Franca",
    "Micro. Araraquara", "Micro. Rio Claro", "Micro. Jundiaí", "Micro. Bragança Paulista", "Micro. Itapeva",
    "Micro. Marília", "Micro. Assis", "Micro. Ourinhos", "Micro. Jaú", "Micro. Lins",
    "Micro. Tupã", "Micro. Adamantina", "Micro. Dracena", "Micro. Andradina", "Micro. Fernandópolis",
    "Micro. Jales", "Micro. Votuporanga", "Micro. Catanduva", "Micro. Barretos", "Micro. São Joaquim da Barra",
    "Micro. Ituverava", "Micro. Batatais", "Micro. Jaboticabal", "Micro. São Carlos", "Micro. Moji Mirim",
    "Micro. Amparo", "Micro. São João da Boa Vista", "Micro. Rio de Janeiro", "Micro. Niterói", "Micro. Campos",
    "Micro. Macaé", "Micro. Cabo Frio", "Micro. Nova Friburgo", "Micro. Petrópolis", "Micro. Teresópolis",
    "Micro. Volta Redonda", "Micro. Resende", "Micro. Angra dos Reis", "Micro. Itaguaí", "Micro. Duque de Caxias",
    "Micro. Nova Iguaçu", "Micro. São Gonçalo", "Micro. Magé", "Micro. Itaboraí", "Micro. Araruama"
]

# ICD-10 codes for conditions of interest
ICD_CODES = {
    "I21": ["I210", "I211", "I212", "I213", "I214", "I219"],  # AMI
    "I63": ["I630", "I631", "I632", "I633", "I634", "I635", "I639"],  # Stroke
    "C34": ["C340", "C341", "C342", "C343", "C349"],  # Lung cancer
    "J18": ["J180", "J181", "J188", "J189"],  # Pneumonia
    "K35": ["K350", "K351", "K353", "K358"],  # Appendicitis
    "other": ["J44", "E11", "N17", "K80", "I50", "J96"],  # Other codes
}


def generate_synthetic_data(
    n_records: int = 150000,
    years: list[int] | None = None,
    severity_trend: float = 0.02,
    regional_variation: float = 0.15,
    seed: int = 42,
    output_path: str = "data/synthetic_sih_data.csv"
) -> pd.DataFrame:
    """Generate synthetic hospital admission data.

    The generator creates data with:
        - Realistic age/sex distributions per condition
        - Configurable temporal trend in severity (simulating increasing delay)
        - Regional variation in baseline severity
        - Correlation between severity components (ICU, death, LOS)

    Args:
        n_records: Total number of records to generate.
        years: List of years. Defaults to 2015-2023.
        severity_trend: Annual increase in severity probability.
            Positive = increasing severity (simulates worsening delay).
        regional_variation: Std dev of regional severity offsets.
        seed: Random seed for reproducibility.
        output_path: Where to save the CSV.

    Returns:
        Generated DataFrame.
    """
    rng = np.random.default_rng(seed)

    if years is None:
        years = list(range(2015, 2024))

    n_years = len(years)

    # Allocate records across years (roughly uniform with some variation)
    year_weights = rng.dirichlet(np.ones(n_years) * 10)
    year_counts = (year_weights * n_records).astype(int)
    year_counts[-1] = n_records - year_counts[:-1].sum()  # Fix rounding

    # Regional severity offsets (some regions have worse baseline)
    regional_offsets = {
        mun: rng.normal(0, regional_variation)
        for mun in MUNICIPALITIES
    }

    records = []

    for year_idx, year in enumerate(years):
        n_year = year_counts[year_idx]

        # Temporal severity modifier (increases over time)
        time_offset = severity_trend * year_idx

        for _ in range(n_year):
            # Select condition (weighted toward target condition)
            if rng.random() < 0.6:
                # Main condition (AMI)
                icd_list = ICD_CODES["I21"]
            elif rng.random() < 0.5:
                # Secondary conditions
                secondary = rng.choice(["I63", "C34", "J18", "K35"])
                icd_list = ICD_CODES[secondary]
            else:
                # Other
                icd_list = ICD_CODES["other"]

            cid = rng.choice(icd_list)

            # Demographics
            if cid.startswith("I21"):
                # AMI: older, male-skewed
                idade = int(rng.normal(65, 14))
                sexo = rng.choice(["M", "F"], p=[0.65, 0.35])
            elif cid.startswith("K35"):
                # Appendicitis: younger
                idade = int(rng.normal(30, 15))
                sexo = rng.choice(["M", "F"], p=[0.55, 0.45])
            else:
                idade = int(rng.normal(55, 18))
                sexo = rng.choice(["M", "F"])

            idade = max(0, min(120, idade))

            # Municipality
            municipio = rng.choice(MUNICIPALITIES)
            reg_offset = regional_offsets[municipio]

            # Base severity probability (increases with age and time)
            base_severity = 0.05 + (idade / 120) * 0.15 + time_offset + reg_offset
            base_severity = np.clip(base_severity, 0.01, 0.95)

            # ICU admission
            icu_prob = base_severity * 0.8
            uti = int(rng.random() < icu_prob)

            # Death (correlated with ICU and severity)
            death_prob = base_severity * 0.3 * (1.5 if uti else 0.5)
            obito = int(rng.random() < death_prob)

            # Length of stay (log-normal, higher for ICU patients)
            base_los = rng.lognormal(mean=1.5 + (0.5 if uti else 0), sigma=0.8)
            if obito:
                base_los *= rng.uniform(0.5, 1.5)  # Deaths can be quick or prolonged
            tempo_internacao = max(1, int(base_los))

            records.append({
                "cid": cid,
                "idade": idade,
                "sexo": sexo,
                "municipio": municipio,
                "ano": year,
                "uti": uti,
                "obito": obito,
                "tempo_internacao": tempo_internacao,
            })

    df = pd.DataFrame(records)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(
        f"Generated {len(df)} synthetic records → {output_path}\n"
        f"  Years: {min(years)}–{max(years)}\n"
        f"  Conditions: {df['cid'].str[:3].nunique()} unique ICD-3 prefixes\n"
        f"  Municipalities: {df['municipio'].nunique()}"
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_synthetic_data()
