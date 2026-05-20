"""
01_parse_data.py
Parse Table S1 (experimental RNA data) and Table S2 (in silico scores) from mmc2.xlsx.
Outputs: data/s1_vus.csv, data/s2_insilico.csv, data/variants_for_external.csv
"""

import pandas as pd
import numpy as np
import re
import os
import sys

EXCEL_PATH = '/data/staff/Jana/Multiomics_Project/1-s2.0-S2666247725001241-mmc2.xlsx'
DATA_DIR   = '/data/staff/Jana/Multiomics_Project/data'
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_splice_distance(dna_variant):
    """Return the intronic distance to nearest splice site (0 if exonic)."""
    match = re.search(r'c\.\d+([+-])(\d+)', str(dna_variant))
    return int(match.group(2)) if match else 0

def extract_exon_info(exon_str):
    """Parse 'X/Y' exon string into (exon_num, total_exons, exon_proportion)."""
    match = re.match(r'(\d+)/(\d+)', str(exon_str))
    if match:
        n, t = int(match.group(1)), int(match.group(2))
        return n, t, round(n / t, 4)
    return np.nan, np.nan, np.nan

def extract_cnomen_key(dna_variant):
    """
    Extract 'NM_XXXXXX.Y:c.ZZZ' or 'ENST...:c.ZZZ' from the full DNA Variant string
    for joining with Table S2 cNomen.
    """
    s = str(dna_variant)
    match = re.search(r'((?:NM_|NR_|ENST)\d+\.\d+)(?:\([^)]+\))?:(c\.[^\s,]+)', s)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return np.nan

# ---------------------------------------------------------------------------
# Parse Table S1
# ---------------------------------------------------------------------------

print("Parsing Table S1 ...")
df_s1_raw = pd.read_excel(EXCEL_PATH, sheet_name='Table S1', header=2)

S1_RENAME = {
    '#':                                          'variant_id',
    'DNA Variant':                                'dna_variant',
    'Gene':                                       'gene',
    'Alamut predictions':                         'alamut_predictions',
    'Alamut ESE':                                 'alamut_ese',
    'Experiment Type':                            'experiment_type',
    'Tissue':                                     'tissue',
    'Tested in 2 systems':                        'tested_2_systems',
    'Aberrant splicing observed?':                'aberrant_splicing',
    'WT trancript present?':                      'wt_transcript',
    'Estimated % aberrant splicing (Sanger)':     'pct_aberrant_sanger',
    'Estimated % aberrant splicing (Agarose gel)':'pct_aberrant_agarose',
    'Classification before RNA testing':          'class_before',
    'Classification after RNA testing':           'class_after',
    'Reclassificatied?':                          'reclassified',
    'Effect category':                            'effect_category',
    'Effect on reading frame':                    'effect_reading_frame',
    'EXON':                                       'exon',
    'INTRON':                                     'intron',
    'VulExMap Category':                          'vulexmap_category',
}
df_s1 = df_s1_raw.rename(columns=S1_RENAME)
df_s1 = df_s1[[c for c in S1_RENAME.values() if c in df_s1.columns]]
df_s1 = df_s1.dropna(subset=['variant_id'])
df_s1['variant_id'] = df_s1['variant_id'].astype(int)

# Keep only the 202 numbered variants (rows where # is 1-202)
df_s1 = df_s1[df_s1['variant_id'].between(1, 202)].copy()

# Filter to VUS at baseline (ACMG class 3)
df_vus = df_s1[df_s1['class_before'] == 3].copy()
print(f"  Total variants: {len(df_s1)}")
print(f"  VUS at baseline (class 3): {len(df_vus)}")

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Positive = upgraded from VUS(3) to LP(4) or P(5)
# Negative = stayed at VUS, downgraded, or reclassified to LB/B

def assign_label(row):
    r = str(row['reclassified'])
    if 'Upgraded 3-4' in r or 'Upgraded 3-5' in r:
        return 1
    return 0

df_vus['label'] = df_vus.apply(assign_label, axis=1)
print(f"  Label=1 (LP/P after RNA): {df_vus['label'].sum()}")
print(f"  Label=0 (stayed VUS/LB):  {(df_vus['label']==0).sum()}")

# ---------------------------------------------------------------------------
# Feature extraction from S1
# ---------------------------------------------------------------------------

# Transcript key for S2 join
df_vus['cnomen_key'] = df_vus['dna_variant'].apply(extract_cnomen_key)

# Distance to splice site
df_vus['splice_distance'] = df_vus['dna_variant'].apply(extract_splice_distance)

# Exon proportion
exon_info = df_vus['exon'].apply(lambda x: pd.Series(
    extract_exon_info(x), index=['exon_number', 'total_exons', 'exon_proportion']
))
df_vus = pd.concat([df_vus, exon_info], axis=1)

# Binary Yes/No columns
for col in ['aberrant_splicing', 'wt_transcript', 'tested_2_systems']:
    df_vus[col] = df_vus[col].map({'Yes': 1, 'No': 0}).fillna(np.nan)

# Numeric aberrant splicing percentages
for col in ['pct_aberrant_sanger', 'pct_aberrant_agarose']:
    df_vus[col] = pd.to_numeric(df_vus[col], errors='coerce')

# Binary: frameshift or premature termination codon in effect on reading frame
df_vus['is_frameshift_ptc'] = df_vus['effect_reading_frame'].apply(
    lambda x: 1 if re.search(r'frameshift|nonsense|PTC|stop', str(x), re.IGNORECASE) else 0
)

# Encode tissue (blood=1, fibroblast=2, other=0)
def encode_tissue(t):
    t = str(t).lower()
    if 'blood' in t:
        return 1
    elif 'fibroblast' in t:
        return 2
    return 0

df_vus['tissue_code'] = df_vus['tissue'].apply(encode_tissue)

# Alamut ESE: ESE loss=1, ESE gain=2, other=0
def encode_ese(e):
    e = str(e).lower()
    if 'loss' in e:
        return 1
    elif 'gain' in e:
        return 2
    return 0

df_vus['alamut_ese_code'] = df_vus['alamut_ese'].apply(encode_ese)

df_vus.to_csv(f'{DATA_DIR}/s1_vus.csv', index=False)
print(f"\nSaved: {DATA_DIR}/s1_vus.csv  ({len(df_vus)} rows)")

# ---------------------------------------------------------------------------
# Parse Table S2
# ---------------------------------------------------------------------------

print("\nParsing Table S2 ...")
df_s2_raw = pd.read_excel(EXCEL_PATH, sheet_name='Table S2', header=2)
df_s2_raw.columns = [str(c).strip() for c in df_s2_raw.columns]

df_s2 = df_s2_raw[df_s2_raw['Dataset'] == 'current dataset'].copy()
df_s2 = df_s2.dropna(subset=['cNomen'])
print(f"  S2 current dataset rows: {len(df_s2)}")

# SpliceAI max
for col in ['SpliceAI_AG', 'SpliceAI_AL', 'SpliceAI_DG', 'SpliceAI_DL']:
    df_s2[col] = pd.to_numeric(df_s2[col], errors='coerce')
df_s2['spliceai_max'] = df_s2[['SpliceAI_AG', 'SpliceAI_AL', 'SpliceAI_DG', 'SpliceAI_DL']].max(axis=1)

# Other in silico scores
df_s2['pangolin_max'] = pd.to_numeric(df_s2['Pangolin_maxchange'], errors='coerce')
df_s2['spip_score']   = pd.to_numeric(df_s2['SPiP_SPiPscore'], errors='coerce')
df_s2['squirls_max']  = pd.to_numeric(df_s2['Squirls_max'], errors='coerce')

# Key for joining
df_s2['cnomen_key'] = df_s2['cNomen'].str.strip()

s2_out = df_s2[['cnomen_key', 'VariantType', 'spliceai_max', 'pangolin_max', 'spip_score', 'squirls_max']].copy()
s2_out = s2_out.rename(columns={'VariantType': 'variant_type'})
s2_out.to_csv(f'{DATA_DIR}/s2_insilico.csv', index=False)
print(f"Saved: {DATA_DIR}/s2_insilico.csv  ({len(s2_out)} rows)")

# ---------------------------------------------------------------------------
# Export gene + variant list for external API fetching
# ---------------------------------------------------------------------------

ext = df_vus[['variant_id', 'gene', 'dna_variant', 'cnomen_key']].copy()
ext.to_csv(f'{DATA_DIR}/variants_for_external.csv', index=False)
print(f"Saved: {DATA_DIR}/variants_for_external.csv  ({len(ext)} rows)")
print(f"Unique genes: {df_vus['gene'].nunique()}")
print("\n01_parse_data.py complete.")
