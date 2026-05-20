"""
04_train_model.py
Train and evaluate three models (Logistic Regression, RandomForest, XGBoost)
using stratified 5-fold cross-validation.
Saves:
  results/metrics/cv_results.csv       — per-fold metrics for each model
  results/metrics/summary_metrics.csv  — mean ± std summary
  results/metrics/best_model.pkl       — best model (by AUC) fitted on full data
  results/metrics/roc_data.pkl         — ROC curve data for visualization
  results/metrics/feature_importance.pkl
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import (roc_auc_score, roc_curve,
                                     confusion_matrix, precision_recall_curve,
                                     average_precision_score, f1_score)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR    = '/data/staff/Jana/Multiomics_Project/data'
METRICS_DIR = '/data/staff/Jana/Multiomics_Project/results/metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(f'{DATA_DIR}/feature_matrix.csv')

with open(f'{DATA_DIR}/feature_cols.txt') as f:
    feature_cols = [l.strip() for l in f.readlines() if l.strip()]

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].values.astype(float)
y = df['label'].values.astype(int)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Positive (LP/P): {y.sum()}  |  Negative: {(y==0).sum()}")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=0.1, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42
        ))
    ]),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / y.sum(),
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0
    ),
}

# ---------------------------------------------------------------------------
# Stratified 5-fold CV
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc         = (tp + tn) / (tp + tn + fp + fn)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    auc         = roc_auc_score(y_true, y_prob)
    auprc       = average_precision_score(y_true, y_prob)
    return {
        'AUC': auc, 'AUPRC': auprc,
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'PPV': ppv, 'NPV': npv,
        'F1': f1, 'Accuracy': acc,
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rows      = []   # validation metrics per fold
train_rows   = []   # training metrics per fold (for overfitting check)
roc_data     = {}   # model_name → list of (fpr, tpr, auc) per fold

for name, model in models.items():
    print(f"\nTraining {name} ...")
    roc_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)

        # Validation metrics
        y_prob_val = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_prob_val)
        metrics.update({'model': name, 'fold': fold_idx + 1, 'split': 'validation'})
        cv_rows.append(metrics)

        # Training metrics (to check for overfitting)
        y_prob_tr = model.predict_proba(X_tr)[:, 1]
        train_metrics = compute_metrics(y_tr, y_prob_tr)
        train_metrics.update({'model': name, 'fold': fold_idx + 1, 'split': 'train'})
        train_rows.append(train_metrics)

        fpr, tpr, _ = roc_curve(y_val, y_prob_val)
        roc_folds.append({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': metrics['AUC']})
        print(f"  Fold {fold_idx+1}: Val AUC={metrics['AUC']:.3f}  "
              f"Train AUC={train_metrics['AUC']:.3f}  "
              f"Gap={train_metrics['AUC']-metrics['AUC']:.3f}")

    roc_data[name] = roc_folds

# Save combined train+val metrics
df_train_val = pd.DataFrame(train_rows + cv_rows)
df_train_val.to_csv(f'{METRICS_DIR}/train_val_metrics.csv', index=False)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

df_cv = pd.DataFrame(cv_rows)
df_cv.to_csv(f'{METRICS_DIR}/cv_results.csv', index=False)

metric_cols = ['AUC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'Accuracy']
summary_rows = []
for name in models:
    sub = df_cv[df_cv['model'] == name][metric_cols]
    row = {'model': name}
    for col in metric_cols:
        row[f'{col}_mean'] = sub[col].mean()
        row[f'{col}_std']  = sub[col].std()
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f'{METRICS_DIR}/summary_metrics.csv', index=False)

print("\n=== Summary (mean ± std over 5 folds) ===")
for _, row in df_summary.iterrows():
    print(f"\n{row['model']}:")
    for col in metric_cols:
        print(f"  {col}: {row[f'{col}_mean']:.3f} ± {row[f'{col}_std']:.3f}")

# ---------------------------------------------------------------------------
# Pick best model by mean AUC and refit on full data
# ---------------------------------------------------------------------------

best_model_name = df_summary.loc[df_summary['AUC_mean'].idxmax(), 'model']
print(f"\nBest model: {best_model_name} (AUC={df_summary.loc[df_summary['AUC_mean'].idxmax(), 'AUC_mean']:.3f})")

best_model = models[best_model_name]
best_model.fit(X, y)

with open(f'{METRICS_DIR}/best_model.pkl', 'wb') as f:
    pickle.dump({'name': best_model_name, 'model': best_model,
                 'feature_cols': feature_cols}, f)

# ---------------------------------------------------------------------------
# Also compute individual feature ROC curves (for visualization)
# ---------------------------------------------------------------------------

single_feature_roc = {}
key_single_features = ['spliceai_max', 'pangolin_max', 'spip_score', 'squirls_max',
                        'loeuf_score', 'tissue_specificity_score', 'aberrant_splicing']

for feat in key_single_features:
    if feat not in feature_cols:
        continue
    fi = feature_cols.index(feat)
    scores = X[:, fi]
    if np.nanstd(scores) == 0:
        continue
    try:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        single_feature_roc[feat] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

fi_data = {}

# Logistic Regression: odds ratios
lr_pipe = models['LogisticRegression']
lr_pipe.fit(X, y)
scaler = lr_pipe.named_steps['scaler']
coefs  = lr_pipe.named_steps['clf'].coef_[0]
odds   = np.exp(coefs)
fi_data['LogisticRegression'] = pd.DataFrame({
    'feature':   feature_cols,
    'coef':      coefs,
    'odds_ratio': odds,
}).sort_values('odds_ratio', ascending=False)

# RandomForest: mean decrease impurity
rf_model = models['RandomForest']
rf_model.fit(X, y)
fi_data['RandomForest'] = pd.DataFrame({
    'feature':    feature_cols,
    'importance': rf_model.feature_importances_,
}).sort_values('importance', ascending=False)

# XGBoost: gain importance
xgb_model = models['XGBoost']
xgb_model.fit(X, y)
fi_data['XGBoost'] = pd.DataFrame({
    'feature':    feature_cols,
    'importance': xgb_model.feature_importances_,
}).sort_values('importance', ascending=False)

# Save all 3 models (each now fit on full data) for the web app ensemble
all_models_dict = {
    'LogisticRegression': {'model': lr_pipe,   'feature_cols': feature_cols},
    'RandomForest':       {'model': rf_model,  'feature_cols': feature_cols},
    'XGBoost':            {'model': xgb_model, 'feature_cols': feature_cols},
}
with open(f'{METRICS_DIR}/all_models.pkl', 'wb') as f:
    pickle.dump(all_models_dict, f)

# Save imputation medians for missing-feature handling in predictor.py
impute_stats = {col: float(np.nanmedian(X[:, i])) for i, col in enumerate(feature_cols)}
with open(f'{METRICS_DIR}/imputation_stats.pkl', 'wb') as f:
    pickle.dump(impute_stats, f)

# Save
with open(f'{METRICS_DIR}/roc_data.pkl', 'wb') as f:
    pickle.dump({'cv_roc': roc_data, 'single_feature_roc': single_feature_roc}, f)

with open(f'{METRICS_DIR}/feature_importance.pkl', 'wb') as f:
    pickle.dump(fi_data, f)

# Best model predictions on full data (for violin plot)
y_prob_all = best_model.predict_proba(X)[:, 1]
df_scores = df[['variant_id', 'gene', 'dna_variant', 'label']].copy()
df_scores['integrated_score'] = y_prob_all
df_scores.to_csv(f'{METRICS_DIR}/integrated_scores.csv', index=False)

print(f"\nSaved all outputs to {METRICS_DIR}/")
print("04_train_model.py complete.")
