# Diagnostic Delay Detector

**Detecting Late Presentation Signals in Healthcare Systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Background

Healthcare systems worldwide face a critical challenge: patients arriving at hospitals with advanced, often preventable diseases. When diagnoses are delayed — due to limited access, screening failures, or care-seeking barriers — patients present with higher clinical severity, require more intensive medical treatment, and face worse outcomes.

**How can diagnostic delay be detected when direct patient timeline data is unavailable?**

This project designs a **proxy-based approach**. Instead of tracking individual patient journeys, it analyzes aggregate severity indicators at hospital admission to detect population-level signals of late presentation.

---

## Clinical Rationale

Early diagnosis is a strong predictor of treatment success. Comparing early versus delayed diagnosis scenarios:

| Scenario | Severity at Admission | Clinical Outcome |
|----------|----------------------|------------------|
| Early diagnosis | Lower | Better prognosis, shorter hospitalization |
| Delayed diagnosis | Higher | Increased ICU admission rate, prolonged stay, higher mortality |

By systematically monitoring how severity indicators change **over time** and **across geographic regions**, it is possible to identify:

- **Temporal degradation**: Are patients presenting in worse conditions year over year?
- **Geographic disparities**: Do specific regions exhibit consistently worse presentation severity?
- **Demographic patterns**: Are particular age groups disproportionately affected?

These signals do not provide definitive proof of diagnostic delay individually, but they indicate priority areas for targeted clinical and policy investigation.

---

## Methodology

### Data Source

The current implementation is optimized for hospital admission data from the Brazilian **SIH/SUS** (Sistema de Informações Hospitalares), which can be retrieved via the [`microdatasus`](https://github.com/rfsaldanha/microdatasus) R package. The system is adaptable to any dataset containing equivalent columns.

### Severity Score

Each hospital admission is assigned a composite severity score calculated as follows:

```
severity_score = (2 × ICU_admission) + (3 × death) + (1 × normalized_LOS)
```

| Component | Weight | Clinical Rationale |
|-----------|--------|---------------------|
| Death (óbito) | +3 | Primary indicator of advanced disease upon presentation |
| ICU admission (UTI) | +2 | Indicates significant resource utilization related to disease severity |
| Length of stay (normalized) | +1 | Acts as a continuous proxy for disease complexity |

### Diagnostic Delay Index (DDI)

The Diagnostic Delay Index (DDI) is the core metric of this analysis:

```
DDI = (number of high-severity cases) / (total cases)
```

**High-severity** is categorized as cases falling above the 75th percentile of the severity score distribution (this threshold is configurable).

**Interpretation Guidelines:**
- A DDI of 0.25 indicates 25% of cases present with high severity (baseline under the P75 threshold).
- An increasing DDI over time suggests patients may be arriving in progressively worse conditions.
- A higher DDI in one region compared to another highlights potential healthcare access disparities.

### Statistical Analysis

- **Linear Regression** and **Mann-Kendall tests** are used for temporal trend detection.
- **Age-stratified analysis** is applied to control for demographic confounding variables.
- **Regional ranking** determines outcome comparisons across distinct localities.

---

## Project Structure

```text
diagnostic-delay-detector/
│
├── data/                          # Input data (CSV files)
│   └── synthetic_sih_data.csv     # Generated demonstration data
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration parameters
│   ├── preprocessing.py           # Data loading, cleaning, and filtering
│   ├── feature_engineering.py     # Severity score and feature calculation
│   ├── metrics.py                 # DDI and aggregated metrics logic
│   ├── analysis.py                # Statistical trend computations
│   ├── visualization.py           # Generation of publication-quality charts
│   ├── interpretation.py          # Clinical narrative report synthesis
│   └── generate_data.py           # Synthetic data generation utility
│
├── outputs/                       # Final plotted figures and reports
│
├── docs/                          # GitHub Pages documentation website
│
├── main.py                        # Primary command-line interface
├── app.py                         # Streamlit interactive dashboard
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## Setup and Usage

### Prerequisites

- Python 3.10 or later
- pip package manager

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/diagnostic-delay-detector.git
cd diagnostic-delay-detector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install project dependencies
pip install -r requirements.txt
```

### Command-Line Interface (CLI)

```bash
# Execute with synthetic demonstration data
python main.py

# Execute against a custom dataset
python main.py --data path/to/dataset.csv

# Analyze a different condition (e.g., Ischemic Stroke)
python main.py --condition I63 --name "Ischemic Stroke"

# Configure custom thresholds
python main.py --threshold-percentile 80 --min-cases 50

# Display all available arguments
python main.py --help
```

### Interactive Dashboard

```bash
streamlit run app.py
```

---

## Output Description

### Visual Output

The pipeline automatically generates five publication-quality plots:

1. **DDI Time Series**: A four-panel chart displaying DDI, mortality, ICU rate, and severity over time.
2. **Regional Bar Chart**: The top 20 regions ranked by DDI.
3. **Severity Distribution**: A histogram featuring a threshold marker and yearly box plots.
4. **Regional-Temporal Heatmap**: DDI variation across both regions and years.
5. **Age-Adjusted Trends**: DDI trajectories grouped by patient age categories.

### Text-Based Reports

A clinical interpretation report is generated, covering:

- Primary findings, trend direction, and statistical significance.
- Hypothesized explanations for the observed trends.
- Component-level evaluation of mortality, ICU access, and length of stay.
- Variations across geographic areas and demographic groups.
- Detailed limitations regarding the proxy-based approach.

### Tabular Data Exports

- `temporal_metrics.csv`: DDI and associated metrics grouped by year.
- `regional_metrics.csv`: DDI and associated metrics grouped by region.
- `age_adjusted_metrics.csv`: DDI distributed by year and age group.

---

## Results Interpretation

### Increases in the DDI

A statistically significant increase in the DDI over time implies an increasing proportion of patients are presenting with advanced disease indicators. This may correlate with:

1. Delays in primary care access or diagnostic service availability.
2. Escalating care-seeking barriers (geographic, financial, or cultural limitations).
3. Decrements in screening protocol effectiveness or coverage.
4. Structural shifts in clinical referral pathways or admission criteria.
5. Macro demographic changes, such as population aging.

### Methodological Exclusions (What the DDI Does Not Measure)

- It is not a direct quantitative measurement of delay in diagnostics.
- It cannot isolate which individual patients encountered delays.
- It does not automatically control for every conceivable confounding factor.
- It does not distinguish between a delay in the diagnostic phase versus a delay in treatment administration.

---

## Critical Limitations

These limitations are integral to the methodology and must accompany any conclusions drawn from the tool.

### Methodological Limits

| Limitation | Impact | Mitigation Strategy |
|-----------|--------|------------|
| **Lack of patient-level linkage** | Cannot verify individual clinical pathways | Confine use to population-level surveillance |
| **Severity does not equal delay** | High clinical severity has multifaceted causes | Cross-reference against independent healthcare access indices |
| **Ecological Fallacy** | Group trends do not guarantee individual outcomes | Enforce population-level interpretation scopes |
| **Coding Inconsistencies** | ICD application varies systematically across facilities | Track coding quality variance over time |

### Data Constraints

- **Completeness**: Variable quality and absent fields in systemic administrative documentation.
- **Selection Bias**: Analysis captures only patients reaching hospital admission, excluding pre-admission mortalities.
- **Temporal Stability**: Evolving clinical admission criteria and systemic capacity changes over the study period.
- **Geographic Resolution**: Evaluating at the municipal tier might conceal neighborhood-level variations.

### Analytical Constraints

- **Confounding Variables**: Evolving trends in comorbidities and societal factors might skew severity independently of diagnostic timelines.
- **Arbitrary Thresholds**: The DDI fluctuates based on the selected percentile cutoff.
- **Sample Viability**: Sparsely populated regions yield low case counts, introducing statistical noise.

---

## Configuration Reference

Modifiable parameters available via CLI flags or the Streamlit sidebar menu:

| Argument | Default | Effect |
|-----------|---------|-------------|
| `--condition` | `I21` | ICD-10 prefix filter |
| `--name` | Auto | Readable label for the evaluated condition |
| `--threshold-percentile` | `75` | The severity percentile determining "high-severity" |
| `--min-cases` | `30` | Exclusion margin for sparse groups |
| `--output` | `outputs/` | Target location for output assets |
| `--synthetic-n` | `15000` | Sample size for the generated synthetic dataset |

### Reference Conditions

| ICD-10 Code | Clinical Condition | Relevancy Context |
|--------|-----------|-------------------|
| I21 | Acute Myocardial Infarction | A highly time-sensitive event where delay increases infarction spread. |
| I63 | Ischemic Stroke | Neurological preservation relies on strict timeframes. |
| C34 | Lung Cancer | Detection at advanced stages drastically reduces survival probability. |
| C50 | Breast Cancer | Evaluation of population screening protocol efficacy. |
| K35 | Acute Appendicitis | Delayed intervention directly correlates with perforation risks. |

---

## Generating Real Datasets

### Using microdatasus (R Script)

```r
library(microdatasus)
library(dplyr)

# Download SIH hospitalization data
dados <- fetch_datasus(
  year_start = 2015,
  year_end = 2023,
  uf = "SP",
  information_system = "SIH-RD"
)

# Apply pre-processing
dados <- process_sih(dados)

# Extract and standardize relevant fields
dados_export <- dados %>%
  select(
    cid = DIAG_PRINC,
    idade = IDADE,
    sexo = SEXO,
    municipio = MUNIC_RES,
    ano = ANO_CMPT,
    uti = UTI_MES_TO,
    obito = MORTE,
    tempo_internacao = DIAS_PERM
  )

write.csv(dados_export, "data/sih_sp_2015_2023.csv", row.names = FALSE)
```

Analyze the resulting file via CLI:
```bash
python main.py --data data/sih_sp_2015_2023.csv
```

---

## License

This framework is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Saldanha, R.F., Bastos, R.R., Barcellos, C. (2019). *Microdatasus: pacote para download e pré-processamento de microdados do Departamento de Informática do SUS (DATASUS)*. Cad. Saúde Pública, 35(9).
- World Health Organization. (2022). *Early diagnosis and screening for cancer*.
- Neal, R.D., et al. (2015). *Is increased time to diagnosis and treatment in symptomatic cancer associated with poorer outcomes?* British Journal of Cancer, 112, S92–S107.
