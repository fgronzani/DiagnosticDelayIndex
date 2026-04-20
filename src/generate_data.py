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
    output_path: str = "data/synthetic_sih_data.csv",
) -> pd.DataFrame:
    """Generate synthetic SIH/SUS hospital admission data (vectorized, ~0.5s for 150k records)."""
    rng = np.random.default_rng(seed)
    
    if years is None:
        years = list(range(2015, 2024))
    
    n_years = len(years)
    
    # Allocate records across years
    year_weights = rng.dirichlet(np.ones(n_years) * 10)
    year_counts = np.round(year_weights * n_records).astype(int)
    year_counts[-1] = n_records - year_counts[:-1].sum()
    
    # Repeat year values
    year_arr = np.repeat(years, year_counts)
    year_idx_arr = np.repeat(np.arange(n_years), year_counts)
    
    # Regional severity offsets (vectorized)
    region_offsets = rng.normal(0, regional_variation, len(MUNICIPALITIES))
    region_indices = rng.integers(0, len(MUNICIPALITIES), n_records)
    municipio_arr = np.array(MUNICIPALITIES)[region_indices]
    reg_offset_arr = region_offsets[region_indices]
    
    # Condition assignment (vectorized)
    cond_roll = rng.random(n_records)
    main_cond = cond_roll < 0.6
    secondary_cond = (~main_cond) & (cond_roll < 0.8)
    
    # ICD codes (vectorized using choice per condition group)
    cid_arr = np.empty(n_records, dtype=object)
    cid_arr[main_cond] = rng.choice(ICD_CODES["I21"], main_cond.sum())
    
    secondary_cond_types = rng.choice(["I63", "C34", "J18", "K35"], secondary_cond.sum())
    secondary_cids = np.array([
        rng.choice(ICD_CODES[c]) 
        for c in secondary_cond_types
    ])
    cid_arr[secondary_cond] = secondary_cids
    
    other_mask = ~main_cond & ~secondary_cond
    cid_arr[other_mask] = rng.choice(ICD_CODES["other"], other_mask.sum())
    
    # Age (vectorized per condition type)
    idade_arr = np.full(n_records, 55.0)
    idade_arr[main_cond] = rng.normal(65, 14, main_cond.sum())
    k35_mask = np.array([str(c).startswith("K35") for c in cid_arr])
    idade_arr[k35_mask] = rng.normal(30, 15, k35_mask.sum())
    idade_arr = np.clip(np.round(idade_arr).astype(int), 0, 120)
    
    # Sex
    sexo_arr = rng.choice(["M", "F"], n_records, p=[0.55, 0.45])
    
    # Base severity (vectorized)
    time_offset_arr = severity_trend * year_idx_arr
    covid_shock_arr = np.where((year_arr == 2020) | (year_arr == 2021), 0.15, 0.0)
    base_severity = 0.05 + (idade_arr / 120) * 0.15 + time_offset_arr + reg_offset_arr + covid_shock_arr
    base_severity = np.clip(base_severity, 0.01, 0.95)
    
    # ICU
    uti_arr = (rng.random(n_records) < base_severity * 0.8).astype(int)
    
    # Death (correlated with ICU)
    death_prob = base_severity * 0.3 * np.where(uti_arr == 1, 1.5, 0.5)
    obito_arr = (rng.random(n_records) < death_prob).astype(int)
    
    # Length of stay (log-normal)
    los_mean = 1.5 + 0.5 * uti_arr
    base_los = np.exp(rng.normal(los_mean, 0.8, n_records))
    tempo_arr = np.maximum(1, np.round(base_los).astype(int))
    
    df = pd.DataFrame({
        "cid": cid_arr,
        "idade": idade_arr,
        "sexo": sexo_arr,
        "municipio": municipio_arr,
        "ano": year_arr,
        "uti": uti_arr,
        "obito": obito_arr,
        "tempo_internacao": tempo_arr,
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.info(
        f"Generated {len(df)} synthetic records → {output_path}\n"
        f"  Years: {min(years)}–{max(years)}\n"
        f"  Municipalities: {df['municipio'].nunique()}"
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_synthetic_data()
