# Identifying Undiagnosed Cardiometabolic Risk Among U.S. Adults
## NHANES 2021-2023 | HDS 5960-01 Capstone | Saint Louis University

**Author:** Nikhitha Chaparla
**Advisors:** Dr. Noor Al Hammadi & Dr. Brittany Hollister
**Program:** Health Data Science, Saint Louis University
**Semester:** Summer 2026
**Status:** Manuscript in preparation for journal submission

---

## Project Overview

This project identifies U.S. adults with objectively confirmed cardiometabolic risk who lack a formal clinical diagnosis, and examines how this diagnosis gap drives emergency department visits and hospitalizations. Using nine NHANES 2021-2023 component files, it builds machine learning models to predict avoidable acute care utilization and extends the analysis to examine comorbidity patterns.

---

## Key Findings

| Finding | Result |
|---|---|
| Undiagnosed prediabetes | 23.2% of analytic sample |
| Undiagnosed hypertension | 20.4% of analytic sample |
| Undiagnosed hyperlipidemia | 17.8% of analytic sample |
| Non-Hispanic Black prediabetes gap | 30.0% — highest across all groups |
| High Risk tier ER visit rate | 27.0% vs 20.0% No Gap group |
| Best ML model | Random Forest AUC 0.709 (95% CI 0.674-0.741) |
| Top SHAP predictor | Gender, usual care site, education |
| Highest ER comorbidity pair | Diabetes + Hyperlipidemia (25.6%) |
| Metabolic syndrome ER paradox | 19.9% MS vs 19.7% non-MS despite higher clinical burden |

---

## Data Source

**National Health and Nutrition Examination Survey (NHANES), August 2021-2023**
CDC National Center for Health Statistics
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

Nine component files merged on SEQN: DEMO_L, DIQ_L, BPQ_L, GHB_L, BPXO_L, TCHOL_L, BMX_L, HUQ_L, HIQ_L

**Final analytic sample:** 5,992 adults aged 20-80
**Access:** Publicly available, no credentialing required

---

## Methods

- 4 binary diagnosis gap flags using ADA and ACC/AHA 2017 clinical thresholds
- Composite risk score (0-4) and risk tier classification
- Survey-weighted prevalence estimates using WTMEC2YR
- SMOTE oversampling for class imbalance
- Models: Logistic Regression, Random Forest, XGBoost
- SHAP explainability analysis on best-performing Random Forest
- Bootstrap 95% CI on AUC (1,000 iterations)
- Calibration assessment (Brier score, reliability curve)
- Adjusted prevalence ratio model (modified Poisson, svyglm)
- Comorbidity pair analysis and metabolic syndrome extension

---

## Model Performance

| Model | AUC-ROC | F1 | Recall |
|---|---|---|---|
| Logistic Regression | 0.698 | 0.704 | 0.675 |
| **Random Forest** | **0.709 (95% CI 0.674-0.741)** | **0.749** | **0.791** |
| XGBoost | 0.647 | 0.730 | 0.762 |

---

## Project Structure

Capstone_Project/

├── 01_load_merge_v2.ipynb           <- Phase 1: load, merge, clean

├── 02_feature_engineering.ipynb     <- Phase 2: diagnosis gap flags, EDA

├── 03_modeling.ipynb                <- Phase 3: ML, SHAP, calibration

├── 03b_comorbidity_analysis.ipynb   <- Phase 3 ext: comorbidity analysis

├── nhanes_survey_analysis.R         <- Survey-weighted estimation (R)

├── ml_diagnostics.py                <- Bootstrap AUC and calibration

├── 01_data/processed/               <- Outputs and figures

└── README.md
---

## Tools

Python, SQL, scikit-learn, XGBoost, SHAP, imbalanced-learn, R (survey package), Tableau Public, Jupyter Notebook, GitHub

---

## Interactive Dashboards

Dashboards 1-3 (Diagnosis Gap, Utilization, Risk Prediction):
https://public.tableau.com/views/NHANESCardiometabolicRiskAnalysisNikhithaChaparla

Dashboard 4 (Comorbidity Analysis):
https://public.tableau.com/app/profile/nikhitha.chaparla/viz/NHANESComorbidityAnalysis1/ComorbidityAnalysis

---

## How to Reproduce

1. Download 9 NHANES 2021-2023 XPT files from CDC and place in 01_data/raw/
2. Install: pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn
3. Launch Jupyter from project root: jupyter notebook
4. Run notebooks in order: 01 -> 02 -> 03 -> 03b

---

## IRB

Exempt under 45 CFR 46.104(d)(4) - publicly available de-identified CDC data
