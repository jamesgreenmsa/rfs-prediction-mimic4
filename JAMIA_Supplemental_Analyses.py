import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

DATA_PATH = r"Data_File_02Oct_WITH_CORRECTED_SEVERITY.xlsx"
OUTPUT_CSV = "JAMIA_Supplemental_Results.csv"

RANDOM_SEED = 42
N_SPLITS = 5

OUTCOME_COL = "overall_rfs_presence_corrected"

SOFA_COMPONENTS = [
    "sofa_resp",
    "sofa_coag",
    "sofa_liver",
    "sofa_cardio",
    "sofa_cns",
    "sofa_renal"
]

SOFA_TOTAL = "sofa_total"

FEATURES_7VAR = [
    "age",
    "bmi",
    SOFA_TOTAL,
    "first_careunit_medicu_ref",
    "nutr_initial_icuday",
    "diabetes",
    "alcohol"
]

FEATURES_17VAR_NO_ICULOS = [
    "age",
    "bmi",
    SOFA_TOTAL,
    "first_careunit_medicu_ref",
    "nutr_initial_icuday",
    "diabetes",
    "alcohol",
    "pre_phos",
    "pre_pot",
    "pre_mag",
    "creatinine",
    "bilirubin",
    "malnutrition",
    "pancreatitis",
    "renal_failure_new"
]

FEATURES_17VAR_WITH_ICULOS = FEATURES_17VAR_NO_ICULOS + ["icu_los"]

FEATURES_17VAR_COMPONENTS = [
    "age",
    "bmi",
    "first_careunit_medicu_ref",
    "nutr_initial_icuday",
    "diabetes",
    "alcohol",
    "pre_phos",
    "pre_pot",
    "pre_mag",
    "creatinine",
    "bilirubin",
    "malnutrition",
    "pancreatitis",
    "renal_failure_new"
] + SOFA_COMPONENTS

results = []


def delong_test(y_true, pred1, pred2):
    n = len(y_true)
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    pos = y_true == 1
    neg = y_true == 0
    n1 = pos.sum()
    n0 = neg.sum()

    def psi(pred_pos, pred_neg):
        return (pred_pos > pred_neg).astype(float) + 0.5 * (pred_pos == pred_neg).astype(float)

    v10_1 = np.array([np.mean(psi(pred1[pos], pred1[j])) for j in np.where(neg)[0]])
    v01_1 = np.array([np.mean(psi(pred1[pos], pred1[j])) for j in np.where(pos)[0]])
    v10_2 = np.array([np.mean(psi(pred2[pos], pred2[j])) for j in np.where(neg)[0]])
    v01_2 = np.array([np.mean(psi(pred2[pos], pred2[j])) for j in np.where(pos)[0]])

    s10_11 = np.cov(v10_1, v10_1)[0, 1] if n0 > 1 else 0
    s01_11 = np.cov(v01_1, v01_1)[0, 1] if n1 > 1 else 0
    s10_22 = np.cov(v10_2, v10_2)[0, 1] if n0 > 1 else 0
    s01_22 = np.cov(v01_2, v01_2)[0, 1] if n1 > 1 else 0
    s10_12 = np.cov(v10_1, v10_2)[0, 1] if n0 > 1 else 0
    s01_12 = np.cov(v01_1, v01_2)[0, 1] if n1 > 1 else 0

    var = (s10_11 / n0 + s01_11 / n1) + (s10_22 / n0 + s01_22 / n1) - 2 * (s10_12 / n0 + s01_12 / n1)
    if var <= 0:
        return auc1, auc2, np.nan, np.nan
    se = np.sqrt(var)
    z = (auc1 - auc2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if len(scores) < 10:
        return np.nan, np.nan
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def run_cv(X, y, model_fn, imputer_fn=None, n_splits=N_SPLITS, seed=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        if imputer_fn is not None:
            imp = imputer_fn()
            X_train = imp.fit_transform(X_train)
            X_val = imp.transform(X_val)
        model = model_fn()
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    return oof_preds


def lr_fn():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')


def rf_fn():
    return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)


def xgb_fn():
    return GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)


def median_imputer():
    return SimpleImputer(strategy='median')


def zero_imputer():
    return SimpleImputer(strategy='constant', fill_value=0)


def knn_imputer():
    return KNNImputer(n_neighbors=5)


def mice_imputer():
    return IterativeImputer(random_state=RANDOM_SEED, max_iter=10)


def record(analysis, model, feature_set, strategy, auc, ci_lo, ci_hi, brier=None, note=""):
    results.append({
        "analysis": analysis,
        "model": model,
        "feature_set": feature_set,
        "strategy": strategy,
        "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "ci_lo_95": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
        "ci_hi_95": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
        "brier": round(brier, 4) if brier is not None and not np.isnan(brier) else np.nan,
        "note": note
    })
    print(f"  {analysis} | {model} | {feature_set} | {strategy} | AUC={auc:.4f} ({ci_lo:.4f}-{ci_hi:.4f})")


print("Loading data...")
df = pd.read_excel(DATA_PATH)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"  Outcome column: {OUTCOME_COL}")

if OUTCOME_COL not in df.columns:
    print(f"ERROR: outcome column '{OUTCOME_COL}' not found.")
    print(f"  Available columns containing 'rfs': {[c for c in df.columns if 'rfs' in c.lower()]}")
    raise SystemExit(1)

y = df[OUTCOME_COL].values.astype(int)
print(f"  Outcome prevalence: {y.mean():.4f} ({y.sum()}/{len(y)})")

admission_date_col = None
for cand in ["admittime", "admit_date", "admission_date", "intime", "icu_intime", "icustay_intime"]:
    if cand in df.columns:
        admission_date_col = cand
        break
if admission_date_col is None:
    for c in df.columns:
        if "admit" in c.lower() or "intime" in c.lower():
            admission_date_col = c
            break

if admission_date_col:
    print(f"  Admission date column found: {admission_date_col}")
    df[admission_date_col] = pd.to_datetime(df[admission_date_col], errors='coerce')
    cutoff = pd.Timestamp("2020-01-01")
    dev_mask = df[admission_date_col] < cutoff
    val_mask = df[admission_date_col] >= cutoff
    print(f"  Temporal split: dev n={dev_mask.sum()}, val n={val_mask.sum()}")
else:
    print("  WARNING: No admission date column found. Temporal validation will be skipped.")
    dev_mask = None
    val_mask = None


def get_X(feature_list, df=df):
    missing = [f for f in feature_list if f not in df.columns]
    available = [f for f in feature_list if f in df.columns]
    if missing:
        print(f"    WARNING: missing columns: {missing}")
    return df[available].values, available


print("\n--- ANALYSIS 1: 2x2 Factorial (all four cells) ---")

print("  7-var LR (median impute)")
X7, used7 = get_X(FEATURES_7VAR)
preds_7lr = run_cv(X7, y, lr_fn, median_imputer)
auc = roc_auc_score(y, preds_7lr)
ci = bootstrap_ci(y, preds_7lr, roc_auc_score)
brier = brier_score_loss(y, preds_7lr)
record("factorial_2x2", "LR", "7var", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  7-var RF (median impute)")
preds_7rf = run_cv(X7, y, rf_fn, median_imputer)
auc = roc_auc_score(y, preds_7rf)
ci = bootstrap_ci(y, preds_7rf, roc_auc_score)
brier = brier_score_loss(y, preds_7rf)
record("factorial_2x2", "RF", "7var", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  17-var no ICU LOS LR (median impute)")
X17, used17 = get_X(FEATURES_17VAR_NO_ICULOS)
preds_17lr = run_cv(X17, y, lr_fn, median_imputer)
auc = roc_auc_score(y, preds_17lr)
ci = bootstrap_ci(y, preds_17lr, roc_auc_score)
brier = brier_score_loss(y, preds_17lr)
record("factorial_2x2", "LR", "17var_no_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  17-var no ICU LOS RF (median impute)")
preds_17rf = run_cv(X17, y, rf_fn, median_imputer)
auc = roc_auc_score(y, preds_17rf)
ci = bootstrap_ci(y, preds_17rf, roc_auc_score)
brier = brier_score_loss(y, preds_17rf)
record("factorial_2x2", "RF", "17var_no_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)


print("\n--- ANALYSIS 2: ICU LOS sensitivity (with vs without) ---")

if "icu_los" in df.columns:
    print("  17-var WITH ICU LOS LR (median impute)")
    X17los, _ = get_X(FEATURES_17VAR_WITH_ICULOS)
    preds_17lr_los = run_cv(X17los, y, lr_fn, median_imputer)
    auc = roc_auc_score(y, preds_17lr_los)
    ci = bootstrap_ci(y, preds_17lr_los, roc_auc_score)
    brier = brier_score_loss(y, preds_17lr_los)
    record("icu_los_sensitivity", "LR", "17var_with_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

    print("  17-var WITH ICU LOS RF (median impute)")
    preds_17rf_los = run_cv(X17los, y, rf_fn, median_imputer)
    auc = roc_auc_score(y, preds_17rf_los)
    ci = bootstrap_ci(y, preds_17rf_los, roc_auc_score)
    brier = brier_score_loss(y, preds_17rf_los)
    record("icu_los_sensitivity", "RF", "17var_with_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)
else:
    print("  icu_los_days column not found, skipping ICU LOS sensitivity")
    record("icu_los_sensitivity", "LR", "17var_with_iculos", "median_impute_sofa_total", np.nan, np.nan, np.nan, note="column not found")
    record("icu_los_sensitivity", "RF", "17var_with_iculos", "median_impute_sofa_total", np.nan, np.nan, np.nan, note="column not found")


print("\n--- ANALYSIS 3: SOFA strategy comparison (Table 4 check) ---")

sofa_comp_avail = [c for c in SOFA_COMPONENTS if c in df.columns]
print(f"  SOFA components available: {sofa_comp_avail}")

base_features_no_sofa = [
    "age", "bmi", "icu_type_medical", "nutr_initial_icuday",
    "diabetes", "alcohol_use", "pre_nutrition_phosphorus",
    "pre_nutrition_potassium", "pre_nutrition_magnesium",
    "pre_nutrition_sodium", "pre_nutrition_albumin",
    "pre_nutrition_creatinine", "pre_nutrition_bilirubin",
    "malnutrition", "pancreatitis", "aki_present"
]
base_avail = [f for f in base_features_no_sofa if f in df.columns]

if sofa_comp_avail:
    print("  Component-level SOFA LR (zero impute within fold)")
    Xcomp = df[base_avail + sofa_comp_avail].values
    preds_comp_lr = run_cv(Xcomp, y, lr_fn, zero_imputer)
    auc = roc_auc_score(y, preds_comp_lr)
    ci = bootstrap_ci(y, preds_comp_lr, roc_auc_score)
    brier = brier_score_loss(y, preds_comp_lr)
    record("sofa_strategy_table4", "LR", "17var_components", "component_level_zero_impute", auc, ci[0], ci[1], brier)

    print("  Component-level SOFA RF (zero impute within fold)")
    preds_comp_rf = run_cv(Xcomp, y, rf_fn, zero_imputer)
    auc = roc_auc_score(y, preds_comp_rf)
    ci = bootstrap_ci(y, preds_comp_rf, roc_auc_score)
    brier = brier_score_loss(y, preds_comp_rf)
    record("sofa_strategy_table4", "RF", "17var_components", "component_level_zero_impute", auc, ci[0], ci[1], brier)

print("  Zero impute SOFA total LR")
preds_zero_lr = run_cv(X17, y, lr_fn, zero_imputer)
auc = roc_auc_score(y, preds_zero_lr)
ci = bootstrap_ci(y, preds_zero_lr, roc_auc_score)
brier = brier_score_loss(y, preds_zero_lr)
record("sofa_strategy_table4", "LR", "17var_no_iculos", "zero_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  Zero impute SOFA total RF")
preds_zero_rf = run_cv(X17, y, rf_fn, zero_imputer)
auc = roc_auc_score(y, preds_zero_rf)
ci = bootstrap_ci(y, preds_zero_rf, roc_auc_score)
brier = brier_score_loss(y, preds_zero_rf)
record("sofa_strategy_table4", "RF", "17var_no_iculos", "zero_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  Median impute SOFA total LR (already run, recording)")
auc = roc_auc_score(y, preds_17lr)
ci = bootstrap_ci(y, preds_17lr, roc_auc_score)
brier = brier_score_loss(y, preds_17lr)
record("sofa_strategy_table4", "LR", "17var_no_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  Median impute SOFA total RF (already run, recording)")
auc = roc_auc_score(y, preds_17rf)
ci = bootstrap_ci(y, preds_17rf, roc_auc_score)
brier = brier_score_loss(y, preds_17rf)
record("sofa_strategy_table4", "RF", "17var_no_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  KNN impute SOFA total LR")
preds_knn_lr = run_cv(X17, y, lr_fn, knn_imputer)
auc = roc_auc_score(y, preds_knn_lr)
ci = bootstrap_ci(y, preds_knn_lr, roc_auc_score)
brier = brier_score_loss(y, preds_knn_lr)
record("sofa_strategy_table4", "LR", "17var_no_iculos", "knn_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  KNN impute SOFA total RF")
preds_knn_rf = run_cv(X17, y, rf_fn, knn_imputer)
auc = roc_auc_score(y, preds_knn_rf)
ci = bootstrap_ci(y, preds_knn_rf, roc_auc_score)
brier = brier_score_loss(y, preds_knn_rf)
record("sofa_strategy_table4", "RF", "17var_no_iculos", "knn_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  MICE impute SOFA total LR")
preds_mice_lr = run_cv(X17, y, lr_fn, mice_imputer)
auc = roc_auc_score(y, preds_mice_lr)
ci = bootstrap_ci(y, preds_mice_lr, roc_auc_score)
brier = brier_score_loss(y, preds_mice_lr)
record("sofa_strategy_table4", "LR", "17var_no_iculos", "mice_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  MICE impute SOFA total RF")
preds_mice_rf = run_cv(X17, y, rf_fn, mice_imputer)
auc = roc_auc_score(y, preds_mice_rf)
ci = bootstrap_ci(y, preds_mice_rf, roc_auc_score)
brier = brier_score_loss(y, preds_mice_rf)
record("sofa_strategy_table4", "RF", "17var_no_iculos", "mice_impute_sofa_total", auc, ci[0], ci[1], brier)


print("\n--- ANALYSIS 4: XGBoost sensitivity ---")

print("  XGBoost 7-var (median impute)")
preds_xgb_7 = run_cv(X7, y, xgb_fn, median_imputer)
auc = roc_auc_score(y, preds_xgb_7)
ci = bootstrap_ci(y, preds_xgb_7, roc_auc_score)
brier = brier_score_loss(y, preds_xgb_7)
record("xgboost_sensitivity", "XGBoost", "7var", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

print("  XGBoost 17-var no ICU LOS (median impute)")
preds_xgb_17 = run_cv(X17, y, xgb_fn, median_imputer)
auc = roc_auc_score(y, preds_xgb_17)
ci = bootstrap_ci(y, preds_xgb_17, roc_auc_score)
brier = brier_score_loss(y, preds_xgb_17)
record("xgboost_sensitivity", "XGBoost", "17var_no_iculos", "median_impute_sofa_total", auc, ci[0], ci[1], brier)

if sofa_comp_avail:
    print("  XGBoost 17-var components (zero impute)")
    preds_xgb_comp = run_cv(Xcomp, y, xgb_fn, zero_imputer)
    auc = roc_auc_score(y, preds_xgb_comp)
    ci = bootstrap_ci(y, preds_xgb_comp, roc_auc_score)
    brier = brier_score_loss(y, preds_xgb_comp)
    record("xgboost_sensitivity", "XGBoost", "17var_components", "component_level_zero_impute", auc, ci[0], ci[1], brier)


print("\n--- ANALYSIS 5: Temporal validation ---")

if dev_mask is not None and dev_mask.sum() > 100 and val_mask.sum() > 100:
    y_dev = y[dev_mask.values]
    y_val = y[val_mask.values]

    print(f"  Dev: n={dev_mask.sum()}, prevalence={y_dev.mean():.4f}")
    print(f"  Val: n={val_mask.sum()}, prevalence={y_val.mean():.4f}")

    for feat_list, feat_label, imp_fn in [
        (FEATURES_17VAR_NO_ICULOS, "17var_no_iculos", median_imputer),
        (FEATURES_7VAR, "7var", median_imputer)
    ]:
        avail = [f for f in feat_list if f in df.columns]
        X_dev = df.loc[dev_mask, avail].values
        X_val_data = df.loc[val_mask, avail].values

        for model_label, model_fn in [("LR", lr_fn), ("RF", rf_fn)]:
            print(f"  Temporal: {model_label} {feat_label}")
            imp = imp_fn()
            X_dev_imp = imp.fit_transform(X_dev)
            X_val_imp = imp.transform(X_val_data)
            model = model_fn()
            model.fit(X_dev_imp, y_dev)
            val_preds = model.predict_proba(X_val_imp)[:, 1]
            auc = roc_auc_score(y_val, val_preds)
            ci = bootstrap_ci(y_val, val_preds, roc_auc_score)
            brier = brier_score_loss(y_val, val_preds)
            record("temporal_validation", model_label, feat_label, "median_impute_sofa_total", auc, ci[0], ci[1], brier,
                   note=f"dev_n={dev_mask.sum()},val_n={val_mask.sum()}")
else:
    print("  Temporal validation skipped: no admission date column or insufficient cohort sizes")
    for model_label in ["LR", "RF"]:
        record("temporal_validation", model_label, "17var_no_iculos", "median_impute_sofa_total",
               np.nan, np.nan, np.nan, note="no_admission_date_column")


print("\n--- DeLong test: LR vs RF on 17-var no ICU LOS ---")
auc1, auc2, z, p = delong_test(y, preds_17lr, preds_17rf)
print(f"  LR AUC={auc1:.4f}, RF AUC={auc2:.4f}, Z={z:.4f}, p={p:.6f}")
results.append({
    "analysis": "delong_test",
    "model": "LR_vs_RF",
    "feature_set": "17var_no_iculos",
    "strategy": "median_impute",
    "auc": round(auc1, 4),
    "ci_lo_95": round(auc2, 4),
    "ci_hi_95": np.nan,
    "brier": np.nan,
    "note": f"Z={round(z,4) if not np.isnan(z) else 'nan'},p={round(p,6) if not np.isnan(p) else 'nan'}"
})

if sofa_comp_avail:
    print("  DeLong: best component vs best imputation (median)")
    auc1, auc2, z, p = delong_test(y, preds_comp_rf, preds_17rf)
    print(f"  Component RF AUC={auc1:.4f}, Median RF AUC={auc2:.4f}, Z={z:.4f}, p={p:.6f}")
    results.append({
        "analysis": "delong_test",
        "model": "component_RF_vs_median_RF",
        "feature_set": "17var_components_vs_17var_no_iculos",
        "strategy": "component_vs_median",
        "auc": round(auc1, 4),
        "ci_lo_95": round(auc2, 4),
        "ci_hi_95": np.nan,
        "brier": np.nan,
        "note": f"Z={round(z,4) if not np.isnan(z) else 'nan'},p={round(p,6) if not np.isnan(p) else 'nan'}"
    })


results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults written to: {OUTPUT_CSV}")
print(f"Total rows: {len(results_df)}")
print("\nFull results table:")
print(results_df.to_string())
