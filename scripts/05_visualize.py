"""
05_visualize.py
Generate all figures for the VUS reclassification prototype model.
Outputs saved to results/plots/:
  roc_curves.png           — ROC curves: 3 models + key single features
  violin_plot.png          — Integrated score by reclassification outcome
  feature_importance_lr.png  — LR coefficients (odds ratios)
  feature_importance_rf.png  — RF + XGBoost importance
  correlation_heatmap.png  — Pairwise feature correlation
  confusion_matrix.png     — Confusion matrix for best model
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

DATA_DIR    = '/data/staff/Jana/Multiomics_Project/data'
METRICS_DIR = '/data/staff/Jana/Multiomics_Project/results/metrics'
PLOTS_DIR   = '/data/staff/Jana/Multiomics_Project/results/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    'LogisticRegression': '#2196F3',
    'RandomForest':        '#4CAF50',
    'XGBoost':             '#FF5722',
}
SINGLE_COLORS = {
    'spliceai_max':             '#9C27B0',
    'pangolin_max':             '#E91E63',
    'spip_score':               '#FF9800',
    'squirls_max':              '#795548',
    'loeuf_score':              '#607D8B',
    'tissue_specificity_score': '#009688',
    'aberrant_splicing':        '#FFC107',
}

# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

df       = pd.read_csv(f'{DATA_DIR}/feature_matrix.csv')
df_score = pd.read_csv(f'{METRICS_DIR}/integrated_scores.csv')
df_cv    = pd.read_csv(f'{METRICS_DIR}/cv_results.csv')
df_summ  = pd.read_csv(f'{METRICS_DIR}/summary_metrics.csv')

with open(f'{METRICS_DIR}/roc_data.pkl', 'rb') as f:
    roc_bundle = pickle.load(f)
cv_roc            = roc_bundle['cv_roc']
single_feature_roc = roc_bundle['single_feature_roc']

with open(f'{METRICS_DIR}/feature_importance.pkl', 'rb') as f:
    fi_data = pickle.load(f)

with open(f'{METRICS_DIR}/best_model.pkl', 'rb') as f:
    best_bundle = pickle.load(f)
best_model_name = best_bundle['name']
best_model      = best_bundle['model']
feature_cols    = best_bundle['feature_cols']

# ---------------------------------------------------------------------------
# 1. ROC Curves
# ---------------------------------------------------------------------------

def mean_roc(folds):
    """Interpolate per-fold ROC curves to a common FPR grid and average."""
    base_fpr = np.linspace(0, 1, 200)
    tprs = []
    for fold in folds:
        fpr = np.array(fold['fpr'])
        tpr = np.array(fold['tpr'])
        tprs.append(np.interp(base_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs,  axis=0)
    return base_fpr, mean_tpr, std_tpr

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')

# Model curves (mean ± std band)
for model_name, folds in cv_roc.items():
    auc_mean = np.mean([f['auc'] for f in folds])
    auc_std  = np.std( [f['auc'] for f in folds])
    fpr_g, tpr_g, tpr_std = mean_roc(folds)
    color = PALETTE[model_name]
    ax.plot(fpr_g, tpr_g, color=color, lw=2,
            label=f'{model_name} (AUC={auc_mean:.3f}±{auc_std:.3f})')
    ax.fill_between(fpr_g, tpr_g - tpr_std, tpr_g + tpr_std,
                    color=color, alpha=0.12)

# Single-feature curves
for feat, data in single_feature_roc.items():
    label = feat.replace('_', ' ').title()
    ax.plot(data['fpr'], data['tpr'],
            color=SINGLE_COLORS.get(feat, 'grey'),
            lw=1.3, linestyle='--',
            label=f'{label} (AUC={data["auc"]:.3f})')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — VUS Reclassification Model\n(5-fold CV, mean ± std)', fontsize=13)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/roc_curves.png', dpi=150)
plt.close(fig)
print("Saved: roc_curves.png")

# ---------------------------------------------------------------------------
# 2. Violin / Box Plot — integrated score by outcome
# ---------------------------------------------------------------------------

df_score['Outcome'] = df_score['label'].map({1: 'Reclassified (LP/P)', 0: 'Stayed VUS / LB'})

fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(data=df_score, x='Outcome', y='integrated_score',
               palette={'Reclassified (LP/P)': '#4CAF50', 'Stayed VUS / LB': '#F44336'},
               inner='box', cut=0, ax=ax)
sns.stripplot(data=df_score, x='Outcome', y='integrated_score',
              color='black', alpha=0.35, size=3, jitter=True, ax=ax)

ax.set_ylabel('Integrated Probability Score', fontsize=12)
ax.set_xlabel('')
ax.set_title(f'Integrated Score Distribution by Reclassification Outcome\n({best_model_name})', fontsize=12)
ax.axhline(0.5, ls='--', color='grey', lw=1, alpha=0.7, label='Threshold = 0.5')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/violin_plot.png', dpi=150)
plt.close(fig)
print("Saved: violin_plot.png")

# ---------------------------------------------------------------------------
# 3a. Feature Importance — Logistic Regression (Odds Ratios)
# ---------------------------------------------------------------------------

df_lr = fi_data['LogisticRegression'].head(20).copy()

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#2196F3' if o >= 1 else '#F44336' for o in df_lr['odds_ratio']]
bars = ax.barh(df_lr['feature'][::-1], df_lr['odds_ratio'][::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.5)
ax.axvline(1.0, color='black', lw=1, ls='--', alpha=0.7)
ax.set_xlabel('Odds Ratio', fontsize=12)
ax.set_title('Logistic Regression — Top 20 Features (Odds Ratios)', fontsize=12)
patches = [mpatches.Patch(color='#2196F3', label='OR > 1 (risk increasing)'),
           mpatches.Patch(color='#F44336', label='OR < 1 (protective)')]
ax.legend(handles=patches, fontsize=10)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/feature_importance_lr.png', dpi=150)
plt.close(fig)
print("Saved: feature_importance_lr.png")

# ---------------------------------------------------------------------------
# 3b. Feature Importance — RF vs XGBoost (side by side)
# ---------------------------------------------------------------------------

top_n = 15
df_rf  = fi_data['RandomForest'].head(top_n).set_index('feature')['importance']
df_xgb = fi_data['XGBoost'].head(top_n).set_index('feature')['importance']
top_feats = list(dict.fromkeys(df_rf.index.tolist() + df_xgb.index.tolist()))[:top_n]

rf_vals  = [fi_data['RandomForest'].set_index('feature')['importance'].get(f, 0) for f in top_feats]
xgb_vals = [fi_data['XGBoost'].set_index('feature')['importance'].get(f, 0) for f in top_feats]

x = np.arange(len(top_feats))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, rf_vals,  width, label='RandomForest', color='#4CAF50', alpha=0.85)
ax.bar(x + width/2, xgb_vals, width, label='XGBoost',      color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', '\n') for f in top_feats], fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_title(f'Feature Importance — RandomForest vs XGBoost (Top {top_n})', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/feature_importance_rf_xgb.png', dpi=150)
plt.close(fig)
print("Saved: feature_importance_rf_xgb.png")

# ---------------------------------------------------------------------------
# 4. Correlation Heatmap (core features only to keep readable)
# ---------------------------------------------------------------------------

core_features = [
    'aberrant_splicing', 'wt_transcript',
    'pct_aberrant_sanger', 'pct_aberrant_agarose',
    'is_frameshift_ptc', 'splice_distance',
    'spliceai_max', 'pangolin_max', 'spip_score', 'squirls_max',
    'loeuf_score', 'tissue_specificity_score', 'gnomad_af',
    'alamut_ese_code', 'exon_proportion',
]
core_features = [c for c in core_features if c in df.columns]

df_corr = df[core_features].corr()

nice_labels = {
    'aberrant_splicing':        'Aberrant Splicing',
    'wt_transcript':            'WT Transcript',
    'pct_aberrant_sanger':      '% Aberrant (Sanger)',
    'pct_aberrant_agarose':     '% Aberrant (Agarose)',
    'is_frameshift_ptc':        'Frameshift/PTC',
    'splice_distance':          'Splice Distance',
    'spliceai_max':             'SpliceAI Max',
    'pangolin_max':             'Pangolin Max',
    'spip_score':               'SPiP Score',
    'squirls_max':              'SQUIRLS Max',
    'loeuf_score':              'LOEUF Score',
    'tissue_specificity_score': 'Tissue Specificity',
    'gnomad_af':                'gnomAD AF',
    'alamut_ese_code':          'Alamut ESE',
    'exon_proportion':          'Exon Proportion',
}
df_corr.index   = [nice_labels.get(c, c) for c in df_corr.index]
df_corr.columns = [nice_labels.get(c, c) for c in df_corr.columns]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, annot_kws={'size': 7},
            linewidths=0.4, linecolor='white',
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Heatmap', fontsize=13, pad=14)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/correlation_heatmap.png', dpi=150)
plt.close(fig)
print("Saved: correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 5. Confusion Matrix — best model (threshold 0.5, full data, indicative)
# ---------------------------------------------------------------------------

X_full = df[feature_cols].values.astype(float)
y_full = df['label'].values.astype(int)
y_prob = best_model.predict_proba(X_full)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_full, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed VUS/LB', 'Reclassified LP/P'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix — {best_model_name}\n(full dataset, threshold = 0.5)', fontsize=12)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/confusion_matrix.png', dpi=150)
plt.close(fig)
print("Saved: confusion_matrix.png")

# ---------------------------------------------------------------------------
# 6. Summary metrics bar chart
# ---------------------------------------------------------------------------

metric_cols = ['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metric_cols))
width = 0.25
offsets = [-width, 0, width]

for i, model_name in enumerate(['LogisticRegression', 'RandomForest', 'XGBoost']):
    row = df_summ[df_summ['model'] == model_name].iloc[0]
    means = [row[f'{m}_mean'] for m in metric_cols]
    stds  = [row[f'{m}_std']  for m in metric_cols]
    ax.bar(x + offsets[i], means, width, yerr=stds,
           label=model_name, color=list(PALETTE.values())[i],
           alpha=0.85, capsize=4, error_kw={'elinewidth': 1.5})

ax.set_xticks(x)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim([0, 1.12])
ax.axhline(0.8, ls='--', color='grey', lw=1, alpha=0.5)
ax.set_title('Model Performance — 5-Fold Cross-Validation (mean ± std)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOTS_DIR}/summary_metrics.png', dpi=150)
plt.close(fig)
print("Saved: summary_metrics.png")

print(f"\nAll figures saved to {PLOTS_DIR}/")
print("05_visualize.py complete.")
