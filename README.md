# RFS Prediction in MIMIC-IV: Analysis Code

**Manuscript:** "Aggregate SOFA Score Imputation Outperforms Component-Level
Disaggregation for Refeeding Syndrome Prediction: A Comparison of Logistic
Regression and Random Forest in Critically Ill Patients"

**Journal:** JAMIA Open (Manuscript ID: JAMIO-2026-0352)

**Authors:** James Green, Scott Parrott, Glenn Brietzke

---

## Overview

This repository contains the analysis code for the JAMIA Open major revision
of manuscript JAMIO-2026-0352. The study compares logistic regression and
random forest models for predicting refeeding syndrome (RFS) in a critically
ill adult cohort derived from MIMIC-IV, with a focus on SOFA score handling
strategies (component-level disaggregation versus aggregate total with
imputation).

**Key findings:**
- Cohort: n=6,843 ICU stays; RFS prevalence 54.2% (3,708 cases)
- 15-variable random forest with KNN SOFA imputation: AUC 0.643 (95% CI 0.629 to 0.656)
- Aggregate SOFA imputation substantially outperforms component-level disaggregation
  (AUC 0.639 to 0.643 vs. 0.570; DeLong Z=9.75, p<0.000001)
- Temporal validation (pre-2020 development, 2020+ held-out): RF AUC 0.655 (95% CI 0.618 to 0.691)

---

## Repository Contents

```
JAMIA_Supplemental_Analyses.py   Primary analysis script
requirements.txt                  Python package dependencies
README.md                         This file
```

---

## Data Availability

The analysis uses MIMIC-IV (v3.1), a publicly available critical care database.
Access requires completion of a credentialing process through PhysioNet:

https://physionet.org/content/mimiciv/

The input data file (`Data_File_02Oct_WITH_CORRECTED_SEVERITY.xlsx`) is a
derived cohort file built from MIMIC-IV. It cannot be shared in this repository
under the MIMIC-IV Data Use Agreement. Researchers with PhysioNet credentialed
access may replicate the cohort by applying the inclusion and exclusion criteria
described in the manuscript Methods section.

---

## Running the Analysis

### Prerequisites

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

### Input File

Place your derived cohort file at the path specified in `DATA_PATH` at the top
of `JAMIA_Supplemental_Analyses.py`, or update that path to match your
environment. The file must contain the following columns:

**Outcome:**
- `overall_rfs_presence_corrected`: binary RFS indicator

**7-variable Brietzke baseline feature set:**
- `age`, `bmi`, `sofa_total`, `first_careunit_medicu_ref`,
  `nutr_initial_icuday`, `diabetes`, `alcohol`

**Additional features for the 15-variable expanded set:**
- `pre_phos`, `pre_pot`, `pre_mag`, `creatinine`, `bilirubin`,
  `malnutrition`, `pancreatitis`, `renal_failure_new`

**SOFA components (for component-level sensitivity analysis):**
- `sofa_resp`, `sofa_coag`, `sofa_liver`, `sofa_cardio`, `sofa_cns`, `sofa_renal`

**Temporal split variable:**
- `admittime` or equivalent admission timestamp (used to split pre-2020 development
  from 2020+ temporal validation cohorts)

### Running

```bash
python JAMIA_Supplemental_Analyses.py
```

Output is written to `JAMIA_Supplemental_Results.csv` in the working directory.

---

## Analysis Structure

The script runs the following analyses in sequence:

1. **2x2 Factorial** (Table 2): LR and RF crossed with 7-variable and 15-variable
   feature sets; 5-fold stratified cross-validation, random seed 42

2. **SOFA Handling Comparison** (Table 4): Five strategies compared within the
   15-variable set: component-level disaggregation, zero imputation, median
   imputation, KNN imputation (k=5), and MICE

3. **ICU LOS Sensitivity** (supplementary): 15-variable set with and without
   ICU length of stay as a feature

4. **XGBoost Sensitivity** (supplementary): GradientBoostingClassifier as a
   third algorithm across the same feature sets

5. **Temporal Validation**: Development cohort (pre-2020, n=6,070) and held-out
   validation cohort (2020 onward, n=773)

6. **DeLong Tests**: Pairwise AUC comparisons using the DeLong et al. (1988)
   nonparametric procedure on pooled out-of-fold predicted probabilities

7. **Bootstrap Confidence Intervals**: 1,000 iterations, random seed 42

---

## Citation

If you use this code, please cite the manuscript (citation to be updated upon
acceptance):

> Green J, Parrott S, Brietzke G. Aggregate SOFA Score Imputation Outperforms
> Component-Level Disaggregation for Refeeding Syndrome Prediction: A
> Comparison of Logistic Regression and Random Forest in Critically Ill
> Patients. JAMIA Open. 2026 (under review).

---

## License

Code is released under the MIT License. The underlying MIMIC-IV data is
governed by its own Data Use Agreement (see PhysioNet link above) and may not
be redistributed.
