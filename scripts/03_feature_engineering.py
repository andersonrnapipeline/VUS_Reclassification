"""
03_feature_engineering.py
Merge S1 experimental, S2 in-silico, and external API features into a single
feature matrix ready for model training.
Output: data/feature_matrix.csv
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import LabelEncoder

DATA_DIR = '/data/staff/Jana/Multiomics_Project/data'

# ---------------------------------------------------------------------------
# Load parsed data
# ---------------------------------------------------------------------------

df_s1  = pd.read_csv(f'{DATA_DIR}/s1_vus.csv')
df_s2  = pd.read_csv(f'{DATA_DIR}/s2_insilico.csv')
df_ext = pd.read_csv(f'{DATA_DIR}/external_features.csv')

print(f"S1 (VUS): {len(df_s1)} rows")
print(f"S2 (in silico): {len(df_s2)} rows")
print(f"External features: {len(df_ext)} rows")

# ---------------------------------------------------------------------------
# Join S1 + S2 on cnomen_key
# ---------------------------------------------------------------------------
# cnomen_key in S1 is extracted from DNA Variant; in S2 it is the raw cNomen.
# We attempt exact match, then fall back to fuzzy match on accession only.

def accession_only(key):
    """Strip the c. notation, keep only NM_XXXX.Y part."""
    if isinstance(key, str):
        return key.split(':')[0]
    return None

df_s1['acc_key'] = df_s1['cnomen_key'].apply(accession_only)
df_s2['acc_key'] = df_s2['cnomen_key'].apply(accession_only)

# Exact join first
df = df_s1.merge(df_s2, on='cnomen_key', how='left', suffixes=('', '_s2'))

# For rows that didn't match (NaN spliceai_max), try accession-only match
unmatched = df['spliceai_max'].isna()
if unmatched.sum() > 0:
    # Drop duplicates to avoid Series assignment error
    s2_acc = df_s2.drop_duplicates(subset='acc_key').set_index('acc_key')
    extra_cols = ['variant_type', 'spliceai_max', 'pangolin_max', 'spip_score', 'squirls_max']
    for idx in df[unmatched].index:
        acc = df.at[idx, 'acc_key']
        if acc in s2_acc.index:
            for col in extra_cols:
                df.at[idx, col] = s2_acc.at[acc, col]

n_matched = df['spliceai_max'].notna().sum()
print(f"S1–S2 match: {n_matched}/{len(df)} variants have in silico scores")

# ---------------------------------------------------------------------------
# Join external features
# ---------------------------------------------------------------------------

df = df.merge(
    df_ext[['variant_id', 'loeuf', 'loeuf_score', 'tpm_blood', 'tpm_fibro',
            'tissue_specificity_score', 'gnomad_af', 'vep_consequence', 'vep_is_ptc']],
    on='variant_id', how='left'
)

# ---------------------------------------------------------------------------
# Encode categorical features
# ---------------------------------------------------------------------------

# Effect category → one-hot
effect_cats = [
    'No effect', 'Exon skipping',
    'Use of alternative SS, inclusion of intron',
    'Use of alternative SS, exclusion of exon',
    'Multiple events', 'Pseudoexon', 'U12-U2 switch',
]
for cat in effect_cats:
    col = 'eff_' + re.sub(r'[^a-z0-9]', '_', cat.lower())[:30]
    df[col] = (df['effect_category'] == cat).astype(int)

# VulExMap category → one-hot
vulexmap_cats = df['vulexmap_category'].dropna().unique().tolist()
for cat in vulexmap_cats:
    col = 'vex_' + re.sub(r'[^a-z0-9]', '_', str(cat).lower())[:30]
    df[col] = (df['vulexmap_category'] == cat).astype(int)

# Variant type from S2 → one-hot
vt_cats = df['variant_type'].dropna().unique().tolist()
for cat in vt_cats:
    col = 'vtype_' + re.sub(r'[^a-z0-9]', '_', str(cat).lower())[:30]
    df[col] = (df['variant_type'] == cat).astype(int)

# Tissue code already numeric (1=blood, 2=fibroblast, 0=other)
# Alamut ESE already encoded as 0/1/2

# ---------------------------------------------------------------------------
# Final feature selection
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Experimental RNA features (S1)
    'aberrant_splicing',
    'wt_transcript',
    'pct_aberrant_sanger',
    'pct_aberrant_agarose',
    'is_frameshift_ptc',
    'splice_distance',
    'exon_number',
    'exon_proportion',
    'tissue_code',
    'alamut_ese_code',
    # In silico scores (S2)
    'spliceai_max',
    'pangolin_max',
    'spip_score',
    'squirls_max',
    # External annotations
    'loeuf',
    'loeuf_score',
    'tpm_blood',
    'tpm_fibro',
    'tissue_specificity_score',
    'gnomad_af',
    'vep_is_ptc',
]

# Add one-hot columns
ohe_cols = [c for c in df.columns if c.startswith(('eff_', 'vex_', 'vtype_'))]
FEATURE_COLS += ohe_cols

# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

df_model = df[['variant_id', 'gene', 'dna_variant', 'label'] + FEATURE_COLS].copy()

# ---------------------------------------------------------------------------
# Imputation: median for numeric, 0 for binary indicators
# ---------------------------------------------------------------------------

binary_cols = [c for c in FEATURE_COLS if df_model[c].dropna().isin([0, 1]).all()]
numeric_cols = [c for c in FEATURE_COLS if c not in binary_cols]

for col in numeric_cols:
    med = df_model[col].median()
    df_model[col] = df_model[col].fillna(med)

for col in binary_cols:
    df_model[col] = df_model[col].fillna(0)

print(f"\nFeature matrix shape: {df_model.shape}")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Label distribution:\n{df_model['label'].value_counts()}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df_model.to_csv(f'{DATA_DIR}/feature_matrix.csv', index=False)

# Also save feature list for reference
with open(f'{DATA_DIR}/feature_cols.txt', 'w') as f:
    for col in FEATURE_COLS:
        f.write(col + '\n')

print(f"\nSaved: {DATA_DIR}/feature_matrix.csv")
print("03_feature_engineering.py complete.")
