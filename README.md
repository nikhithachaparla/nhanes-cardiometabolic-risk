# Identifying Undiagnosed Cardiometabolic Risk Among U.S. Adults
## NHANES 2021–2023 | HDS 5960-01 Capstone | Saint Louis University

**Author:** Nikhitha Chaparla  
**Advisors:** Dr. Noor Al Hammadi & Dr. Brittany Hollister  
**Program:** Health Data Science, Saint Louis University  
**Semester:** Summer 2026

---

## Project Overview

Millions of U.S. adults have objectively confirmed cardiometabolic risk — elevated HbA1c, blood pressure, or cholesterol — yet lack a formal clinical diagnosis. This capstone project identifies and characterizes this diagnosis gap using nationally representative NHANES 2021–2023 data, examines how undiagnosed cardiometabolic conditions are associated with emergency department visits and hospitalizations, and analyzes how comorbidity combinations — multiple simultaneous undiagnosed conditions — modify these utilization patterns.

This project directly mirrors the population health analytics work performed within clinical informatics platforms such as Epic's Health Maintenance and population health dashboard tools.

---

## Research Questions

1. What is the prevalence of undiagnosed diabetes, hypertension, and hyperlipidemia among U.S. adults, and how does it vary by race/ethnicity, income, and insurance status?
2. Do adults with undiagnosed cardiometabolic conditions demonstrate higher rates of emergency department visits and overnight hospitalizations?
3. Which patient-level factors most strongly predict acute healthcare utilization among high-risk undiagnosed adults?
4. How do comorbidity combinations — multiple simultaneous conditions — modify the relationship between cardiometabolic disease burden and acute healthcare utilization?

---

## Key Findings

### Diagnosis Gap Findings
| Finding | Result |
|---|---|
| Undiagnosed prediabetes prevalence | 23.2% of U.S. adults |
| Undiagnosed hypertension prevalence | 20.4% of U.S. adults |
| Undiagnosed hyperlipidemia prevalence | 17.8% of U.S. adults |
| Undiagnosed diabetes prevalence | 2.5% of U.S. adults |
| Non-Hispanic Black prediabetes gap | 30.0% — highest across all groups |
| Uninsured hypertension gap | 26.4% vs 19.9% among insured |

### Utilization Findings
| Finding | Result |
|---|---|
| High Risk tier ER visit rate | 27.0% vs 20.0% in No Gap group |
| Diabetes + Hyperlipidemia ER rate | 25.6% — highest comorbidity pair |
| Metabolic Syndrome ER rate | 19.9% — nearly identical to non-MS (19.7%) |
| Metabolic Syndrome uninsured rate | 4.2% vs 8.8% non-MS — explains ER paradox |

### Machine Learning Findings
| Finding | Result |
|---|---|
| Best ML model | Random Forest (AUC-ROC = 0.709) |
| Top predictor of ER visits | Gender, usual care site, education |
| Strongest diagnosis gap predictor | Undiagnosed prediabetes (ranked 7th in upgraded model) |
| Comorbidity features AUC improvement | 0.000 — individual lab values already capture comorbidity signal |

---

## Data Source

**National Health and Nutrition Examination Survey (NHANES), August 2021–2023**  
CDC National Center for Health Statistics (NCHS)  
🔗 https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

**Nine component files merged on SEQN (unique respondent ID):**

| File | Contents |
|---|---|
| DEMO_L | Age, sex, race/ethnicity, income, education |
| DIQ_L | Self-reported diabetes diagnosis |
| BPQ_L | Self-reported hypertension and hyperlipidemia diagnoses |
| GHB_L | Lab-measured HbA1c (objective diabetes criterion) |
| BPXO_L | Measured systolic and diastolic blood pressure |
| TCHOL_L | Measured serum total cholesterol |
| BMX_L | Measured BMI and waist circumference |
| HUQ_L | ER visits, hospitalizations, outpatient visits |
| HIQ_L | Health insurance coverage type |

**Final analytic sample:** 5,992 adults aged 20–80 years  
**Access:** All files publicly available at no cost, no credentialing required

> **Note:** Several variable names in the NHANES 2021–2023 cycle differ from prior cycles (HUQ042 replaced HUQ071 for ER visits; HIQ032 series replaced HIQ031 series for insurance types). These were identified, corrected, and documented in the analysis notebooks.

---

## Methods

### Phase 1 — Data Loading, Merging, and Cleaning
- Loaded all 9 NHANES XPT files using `pandas.read_sas()`
- Merged all files on SEQN using SQL LEFT JOIN logic
- Applied inclusion/exclusion criteria: age ≥20, excluded pregnant women, required at least one valid primary lab value
- Handled missing data: median imputation for continuous lab variables, mode imputation for categorical variables
- Recoded NHANES numeric codes to readable labels
- Corrected 2021–2023 cycle variable name changes from prior cycles

### Phase 2 — Feature Engineering and Exploratory Analysis
Engineered four binary diagnosis gap flags using validated clinical thresholds:
- **Undiagnosed Diabetes:** HbA1c ≥6.5% AND no self-reported diagnosis (ADA standard)
- **Undiagnosed Prediabetes:** HbA1c 5.7–6.4% AND no self-reported diagnosis
- **Undiagnosed Hypertension:** Systolic ≥130 OR Diastolic ≥80 mmHg AND no self-reported diagnosis (ACC/AHA 2017)
- **Undiagnosed Hyperlipidemia:** Total cholesterol ≥200 mg/dL AND no self-reported diagnosis

Derived a **Composite Risk Score** (0–4) categorized into risk tiers:
- No Gap (0), Low Risk (1), Moderate Risk (2), High Risk (3–4)

Conducted subgroup prevalence analysis by race/ethnicity, insurance status, and income level.

### Phase 3 — Machine Learning Modeling
- **Target variable:** Emergency department visit in past 12 months (binary)
- **Features:** 20 variables including demographics, lab values, diagnosis gap flags, healthcare access
- **Train/test split:** 80/20 stratified
- **Class imbalance:** Addressed with SMOTE oversampling (balanced 3,848 vs 945 to 3,848 vs 3,848)
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Explainability:** SHAP (SHapley Additive exPlanations) values on best-performing Random Forest

### Phase 3 Extension — Comorbidity Analysis
Extended the analysis to examine how multiple simultaneous conditions interact:

**Comorbidity flags engineered:**
- Total condition burden score (0–4, combining diagnosed and undiagnosed)
- Five comorbidity pair flags: Diabetes+Hypertension, Diabetes+Hyperlipidemia, Hypertension+Hyperlipidemia, Prediabetes+Hypertension, Prediabetes+Hyperlipidemia
- Metabolic Syndrome flag: all three conditions simultaneously (diabetes + hypertension + hyperlipidemia)

**Key comorbidity findings:**
- Condition burden does NOT drive ER visits linearly — adults with zero conditions showed 23.0% ER rate vs 18.7–19.4% for those with multiple conditions, explained by non-chronic acute presentations in the no-condition group
- Diabetes + Hyperlipidemia showed the highest ER visit rate (25.6%) of any comorbidity pair — higher than metabolic syndrome (19.9%)
- Metabolic syndrome patients (10.6% of sample) showed near-identical ER utilization to non-MS adults despite dramatically higher clinical burden (mean HbA1c 7.35 vs 5.59; mean BMI 33.0 vs 29.4), explained by significantly lower uninsured rate (4.2% vs 8.8%) and older age with established care relationships
- Adding comorbidity interaction features to the Random Forest model did not improve AUC (0.709), confirming that individual lab values already capture the comorbidity signal

### Phase 4 — Tableau Dashboards
Four interactive dashboards published to Tableau Public covering diagnosis gap prevalence, healthcare utilization patterns, ML model results, and comorbidity analysis.

---

## Model Performance

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.698 | 0.704 | 0.766 | 0.675 |
| **Random Forest** | **0.709** | **0.749** | **0.739** | **0.791** |
| XGBoost | 0.647 | 0.730 | 0.712 | 0.762 |
| RF + Comorbidity Features | 0.709 | 0.749 | 0.737 | 0.787 |

**Best model: Random Forest** — highest AUC-ROC and F1 score

---

## SHAP Feature Importance — Upgraded Model (Top 10)

| Rank | Feature | Mean \|SHAP Value\| | Interpretation |
|---|---|---|---|
| 1 | Gender | 0.0631 | Females show higher ER utilization |
| 2 | Usual Care Site | 0.0526 | No usual care drives ER dependency |
| 3 | Education | 0.0455 | Lower education increases ER risk |
| 4 | Outpatient Visits | 0.0416 | More visits = sicker patients |
| 5 | Self-Rated Health | 0.0304 | Poor health strongly predicts ER |
| 6 | HbA1c | 0.0298 | Glycemic control is key predictor |
| 7 | Undiagnosed Prediabetes | 0.0290 | Strongest diagnosis gap predictor |
| 8 | Poverty Income Ratio | 0.0289 | Income confounds access and risk |
| 9 | Age | 0.0236 | Older age elevates risk |
| 10 | Insurance Status | 0.0220 | Uninsured = higher ER dependency |

**Key insight:** Social determinants are stronger predictors of ER utilization than clinical diagnosis gap flags alone — fixing diagnosis gaps without addressing structural barriers is insufficient.

---

## Comorbidity Analysis Summary

| Comorbidity Pattern | N | ER Visit Rate | Hosp Rate |
|---|---|---|---|
| No Conditions | 1,089 | 23.0% | 18.8% |
| Diabetes Only | 56 | 16.1% | 10.7% |
| Hypertension Only | 924 | 22.2% | 14.3% |
| Hyperlipidemia Only | 1,024 | 15.3% | 18.1% |
| Diabetes + Hypertension | 192 | 19.3% | 15.1% |
| **Diabetes + Hyperlipidemia** | **129** | **25.6%** | **17.8%** |
| Hypertension + Hyperlipidemia | 1,766 | 18.2% | 13.4% |
| All Three (Metabolic Syndrome) | 638 | 19.9% | 15.7% |

---

## Project Structure

```
Capstone_Project/
├── 01_data/
│   ├── raw/                         ← NHANES .XPT files (download from CDC)
│   └── processed/                   ← Cleaned CSV and chart outputs
├── 01_load_merge.ipynb              ← Phase 1: original load and merge
├── 01_load_merge_v2.ipynb           ← Phase 1: consolidated clean version
├── 02_feature_engineering.ipynb     ← Phase 2: diagnosis gap flags and EDA
├── 03_modeling.ipynb                ← Phase 3: ML modeling and SHAP analysis
├── 03b_comorbidity_analysis.ipynb   ← Phase 3 Extension: comorbidity analysis
├── 03_dashboards/                   ← Tableau workbook files
├── 04_report/                       ← Final capstone report
└── README.md
```

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python (pandas, numpy) | Data loading, cleaning, feature engineering |
| SQL (SQLite/DuckDB) | Multi-file merging on SEQN |
| scikit-learn | Logistic Regression, Random Forest, SMOTE |
| XGBoost | Gradient boosting classification |
| SHAP | Model explainability and feature importance |
| matplotlib / seaborn | EDA and comorbidity charts |
| Tableau Public | Interactive dashboard development and publication |
| Jupyter Notebook | Reproducible analytical workflow |
| GitHub | Version control and project sharing |

---

## Interactive Dashboards

View all four live Tableau dashboards:  
🔗 https://public.tableau.com/views/NHANESCardiometabolicRiskAnalysisNikhithaChaparla/Dashboard3RiskPrediction

| Dashboard | What It Shows |
|---|---|
| Dashboard 1 — Diagnosis Gap | Undiagnosed prevalence by condition and race/ethnicity |
| Dashboard 2 — Healthcare Utilization | ER and hospitalization rates by risk tier and insurance status |
| Dashboard 3 — Risk Prediction | ML model performance, SHAP feature importance, risk tier distribution |
| Dashboard 4 — Comorbidity Analysis | Co-occurrence matrix, burden vs utilization, comorbidity pair outcomes |

---

## How to Reproduce This Analysis

**Step 1 — Download NHANES data:**
Download all 9 component files from https://wwwn.cdc.gov/nchs/nhanes and place in `01_data/raw/`

**Step 2 — Install required libraries:**
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn imbalanced-learn duckdb
```

**Step 3 — Launch Jupyter from project root:**
```bash
cd Documents/Capstone_Project
jupyter notebook
```

**Step 4 — Run notebooks in order:**
1. `01_load_merge_v2.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`
4. `03b_comorbidity_analysis.ipynb`

---

## IRB / Ethical Considerations

This project uses publicly available, fully de-identified secondary data released by the CDC under the NHANES program. No patient contact or PHI access is involved. This project qualifies for IRB exemption under 45 CFR 46.104(d)(4).

---

## References

- Gwira, J. A., Fryar, C. D., & Gu, Q. (2024). Prevalence of total, diagnosed, and undiagnosed diabetes: U.S., August 2021–2023 (NCHS Data Brief No. 516). CDC. https://www.cdc.gov/nchs/products/databriefs/db516.htm
- Saleh, S. N., et al. (2026). Trends in hypertension prevalence and control in the U.S. over 25 years. Journal of Clinical Hypertension. https://doi.org/10.1111/jch.70216
- CDC. (2023). Excess burden of poverty and hypertension by race/ethnicity. Preventing Chronic Disease, 20, E64. https://www.cdc.gov/pcd/issues/2023/23_0065.htm
- NCHS. (2022). ED visits among adults with chronic conditions: U.S., 2017–2019 (NHSR No. 174). CDC. https://cdc.gov/nchs/data/nhsr/nhsr174.pdf
- Grout, R., et al. (2023). Predicting disease onset from EHRs for population health management. Frontiers in Artificial Intelligence, 6. https://doi.org/10.3389/frai.2023.1287541
- Adler, A. (2021). Using machine learning techniques to identify key risk factors for diabetes and undiagnosed diabetes [Preprint]. arXiv. https://arxiv.org/pdf/2105.09379
