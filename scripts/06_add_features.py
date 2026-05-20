"""
06_add_features.py
Add NMD prediction and rule-based ACMG evidence scores to the feature matrix,
then merge in full VEP consequences from local VEP output if available.

New features added:
  NMD features     — nmd_sensitive, nmd_score
  ACMG evidence    — acmg_pvs1, acmg_ps3, acmg_pm2, acmg_pp3,
                     acmg_bp4, acmg_bp7, acmg_total_score
  Specificity aids — no_rna_effect, high_pct_aberrant,
                     is_canonical_ss, tools_agree_benign, tools_agree_pathogenic
  VEP (if available) — vep_consequence_local (from local VEP TSV)

Overwrites data/feature_matrix.csv and updates data/feature_cols.txt
"""

import os
import re
import numpy as np
import pandas as pd

DATA_DIR    = '/data/staff/Jana/Multiomics_Project/data'
VEP_OUT     = f'{DATA_DIR}/vep_output.txt'   # local VEP TSV if it exists

df    = pd.read_csv(f'{DATA_DIR}/feature_matrix.csv')
df_s1 = pd.read_csv(f'{DATA_DIR}/s1_vus.csv')

# Bring raw columns back for feature computation
raw_cols = ['variant_id', 'effect_category', 'effect_reading_frame',
            'exon', 'exon_number', 'total_exons', 'exon_proportion',
            'pct_aberrant_sanger', 'pct_aberrant_agarose',
            'aberrant_splicing', 'wt_transcript', 'splice_distance',
            'reclassified', 'vulexmap_category', 'alamut_predictions']
raw = df_s1[[c for c in raw_cols if c in df_s1.columns]].copy()
df = df.merge(raw, on='variant_id', how='left', suffixes=('', '_raw'))

# ── 1. NMD Prediction ───────────────────────────────────────────────────────
# Rule: PTC >50 nt upstream of last exon-exon junction triggers NMD.
# Approximation: frameshift/stop_gained AND not in last exon (exon < total-1).
# We use exon_proportion as proxy (< 0.85 = not near last exon).

def nmd_sensitive(row):
    """1 if variant likely triggers NMD, 0 otherwise."""
    ptc = row.get('is_frameshift_ptc', 0)
    if ptc != 1:
        return 0
    # If exon info available, check position
    ep = row.get('exon_proportion', np.nan)
    if not np.isnan(ep):
        # NMD escaped if PTC is in last exon or within 50nt of last junction
        # Conservative: last exon = exon_proportion > 0.90
        return 1 if ep < 0.90 else 0
    # Fallback: check effect_reading_frame text
    ef = str(row.get('effect_reading_frame', ''))
    if re.search(r'frameshift|nonsense|stop|PTC', ef, re.I):
        return 1
    return 0

def nmd_score(row):
    """
    Graded NMD probability (0-1):
    - frameshift/PTC in early exons → high
    - frameshift/PTC in penultimate exon → medium
    - last exon / no PTC → 0
    """
    if row.get('is_frameshift_ptc', 0) != 1:
        return 0.0
    ep = row.get('exon_proportion', np.nan)
    if np.isnan(ep):
        return 0.5   # unknown position — moderate probability
    if ep < 0.50:
        return 1.0   # early exon, definitely NMD
    elif ep < 0.75:
        return 0.85
    elif ep < 0.90:
        return 0.60
    else:
        return 0.10  # last exon — NMD escape likely

df['nmd_sensitive'] = df.apply(nmd_sensitive, axis=1)
df['nmd_score']     = df.apply(nmd_score,     axis=1)

# ── 2. ACMG Evidence Codes (rule-based) ────────────────────────────────────
# Weighted sum reflecting ACMG/AMP 2015 framework for splice variants.
# Weights: PVS1=8, PS=4, PM=2, PP=1, BP=-1, BS=-4

# PVS1: Null variant in LoF-intolerant gene
# Trigger: frameshift/PTC AND LOEUF_score > 0.5 (gene sensitive to LoF)
def acmg_pvs1(row):
    ptc = row.get('is_frameshift_ptc', 0)
    ls  = row.get('loeuf_score', np.nan)
    if ptc == 1:
        if not np.isnan(ls) and ls > 0.6:
            return 1   # Full PVS1
        return 0
    # Splice ±1,2 (canonical) with aberrant splicing — PVS1 moderate
    sd = row.get('splice_distance', 0)
    ab = row.get('aberrant_splicing', 0)
    if sd <= 2 and ab == 1 and not np.isnan(ls) and ls > 0.5:
        return 1
    return 0

# PS3: Functional RNA evidence of aberrant splicing
def acmg_ps3(row):
    ab  = row.get('aberrant_splicing', 0)
    pct = row.get('pct_aberrant_sanger', np.nan)   # values are 0–1
    if ab == 1:
        if not np.isnan(pct) and pct >= 0.5:
            return 1   # Strong — majority of transcripts affected
        return 0
    return 0

# PM2: Absent/very rare in gnomAD (< 0.1%)
def acmg_pm2(row):
    af = row.get('gnomad_af', np.nan)
    # All are VUS so likely rare; if AF is 0 or NaN treat as absent
    if np.isnan(af) or af < 0.001:
        return 1
    return 0

# PP3: Multiple in silico tools predict pathogenic
def acmg_pp3(row):
    sa  = row.get('spliceai_max', np.nan)
    pan = row.get('pangolin_max', np.nan)
    sp  = row.get('spip_score',   np.nan)
    votes = 0
    if not np.isnan(sa)  and sa  > 0.5: votes += 1
    if not np.isnan(pan) and pan > 0.5: votes += 1
    if not np.isnan(sp)  and sp  > 0.9: votes += 1
    return 1 if votes >= 2 else 0

# BP4: Multiple in silico tools predict benign
def acmg_bp4(row):
    sa  = row.get('spliceai_max', np.nan)
    pan = row.get('pangolin_max', np.nan)
    sp  = row.get('spip_score',   np.nan)
    sq  = row.get('squirls_max',  np.nan)
    votes = 0
    if not np.isnan(sa)  and sa  < 0.1: votes += 1
    if not np.isnan(pan) and pan < 0.1: votes += 1
    if not np.isnan(sp)  and sp  < 0.3: votes += 1
    if not np.isnan(sq)  and sq  < 0.1: votes += 1
    return 1 if votes >= 3 else 0

# BP7: No RNA effect observed AND WT transcript maintained
def acmg_bp7(row):
    ab  = row.get('aberrant_splicing', 0)
    wt  = row.get('wt_transcript', 0)
    cat = str(row.get('effect_category', ''))
    if ab == 0 and wt == 1 and 'No effect' in cat:
        return 1
    return 0

df['acmg_pvs1'] = df.apply(acmg_pvs1, axis=1)
df['acmg_ps3']  = df.apply(acmg_ps3,  axis=1)
df['acmg_pm2']  = df.apply(acmg_pm2,  axis=1)
df['acmg_pp3']  = df.apply(acmg_pp3,  axis=1)
df['acmg_bp4']  = df.apply(acmg_bp4,  axis=1)
df['acmg_bp7']  = df.apply(acmg_bp7,  axis=1)

# Weighted ACMG total score (positive = pathogenic evidence, negative = benign)
df['acmg_total_score'] = (
    df['acmg_pvs1'] * 8 +
    df['acmg_ps3']  * 4 +
    df['acmg_pm2']  * 2 +
    df['acmg_pp3']  * 1 +
    df['acmg_bp4']  * -1 +
    df['acmg_bp7']  * -2
)

# ── 3. Specificity-boosting features ───────────────────────────────────────

# Strong benign signal: no aberrant splicing, WT transcript present
df['no_rna_effect'] = ((df['aberrant_splicing'] == 0) & (df['wt_transcript'] == 1)).astype(int)

# Strong pathogenic signal: majority of transcripts aberrant
pct_s = pd.to_numeric(df.get('pct_aberrant_sanger', pd.Series(dtype=float)), errors='coerce')
pct_a = pd.to_numeric(df.get('pct_aberrant_agarose', pd.Series(dtype=float)), errors='coerce')
df['high_pct_aberrant'] = (
    (pct_s.fillna(0) >= 0.5) | (pct_a.fillna(0) >= 0.5)
).astype(int)

# Canonical splice site (±1 or ±2) — high prior for pathogenicity
df['is_canonical_ss'] = (df['splice_distance'].fillna(99) <= 2).astype(int)

# All major in silico tools agree: benign
df['tools_agree_benign'] = (
    (df['spliceai_max'].fillna(1)  < 0.2) &
    (df['pangolin_max'].fillna(1)  < 0.2) &
    (df['spip_score'].fillna(1)    < 0.5) &
    (df['squirls_max'].fillna(1)   < 0.2)
).astype(int)

# All major in silico tools agree: pathogenic
df['tools_agree_pathogenic'] = (
    (df['spliceai_max'].fillna(0)  > 0.5) &
    (df['pangolin_max'].fillna(0)  > 0.5)
).astype(int)

# ── 4. Local VEP output (if available and non-trivial) ──────────────────────
_ptc_vep = {'frameshift_variant', 'stop_gained', 'stop_lost',
             'splice_acceptor_variant', 'splice_donor_variant',
             'start_lost', 'transcript_ablation'}
if os.path.exists(VEP_OUT) and os.path.getsize(VEP_OUT) > 0:
    vep_rows = []
    with open(VEP_OUT) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            # Col 0 = Uploaded_variation (our cnomen_key), col 6 = Consequence
            consequence = parts[6]
            vep_rows.append({
                'cnomen_key':            parts[0],
                'vep_consequence_local': consequence,
                'vep_is_ptc_local':      1 if any(t in consequence for t in _ptc_vep) else 0,
            })
    if vep_rows:
        df_vep = pd.DataFrame(vep_rows).drop_duplicates('cnomen_key')
        df = df.merge(df_vep, on='cnomen_key', how='left')
        mask = df['vep_is_ptc_local'].notna()
        df.loc[mask, 'vep_is_ptc'] = df.loc[mask, 'vep_is_ptc_local']
        print(f"Local VEP coverage: {mask.sum()}/{len(df)}")
    else:
        print("Local VEP file empty — using REST API annotations (164/178).")
else:
    print("No local VEP output — using REST API annotations (164/178).")

# ── 5. Finalise feature list and save ──────────────────────────────────────
NEW_FEATURES = [
    'nmd_sensitive', 'nmd_score',
    'acmg_pvs1', 'acmg_ps3', 'acmg_pm2', 'acmg_pp3',
    'acmg_bp4', 'acmg_bp7', 'acmg_total_score',
    'no_rna_effect', 'high_pct_aberrant',
    'is_canonical_ss', 'tools_agree_benign', 'tools_agree_pathogenic',
]

# Read existing feature list
with open(f'{DATA_DIR}/feature_cols.txt') as f:
    existing = [l.strip() for l in f if l.strip()]

# Avoid duplicates
all_features = existing + [c for c in NEW_FEATURES if c not in existing]

# Drop temp raw columns that were merged in for computation
raw_temp = [c for c in df.columns if c.endswith('_raw')]
df = df.drop(columns=raw_temp, errors='ignore')

# Keep only final feature columns (and meta columns) that exist
meta = ['variant_id', 'gene', 'dna_variant', 'label']
all_features = [c for c in all_features if c in df.columns]
df_out = df[meta + all_features].copy()

# Impute any NaN in new features
for col in NEW_FEATURES:
    if col in df_out.columns:
        df_out[col] = df_out[col].fillna(0)

df_out.to_csv(f'{DATA_DIR}/feature_matrix.csv', index=False)
with open(f'{DATA_DIR}/feature_cols.txt', 'w') as f:
    for c in all_features:
        f.write(c + '\n')

print(f"\nFeature matrix: {df_out.shape[0]} rows × {len(all_features)} features")
print(f"New features added: {len(NEW_FEATURES)}")

print("\nNew feature distributions:")
for col in NEW_FEATURES:
    if col in df_out.columns:
        vals = df_out[col]
        print(f"  {col}: mean={vals.mean():.3f}  range=[{vals.min():.2f}, {vals.max():.2f}]")

print("\n06_add_features.py complete.")
