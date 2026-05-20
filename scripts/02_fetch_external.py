"""
02_fetch_external.py
Fetch external annotations via public APIs:
  - gnomAD GraphQL : LOEUF per gene, allele frequency per variant
  - GTEx REST API  : blood + fibroblast TPM per gene
  - Ensembl VEP    : consequence, frameshift/stop-gained flag
All results cached to data/external_cache.json to avoid re-fetching.
Output: data/external_features.csv
"""

import json
import math
import os
import re
import time

import pandas as pd
import numpy as np
import requests

DATA_DIR   = '/data/staff/Jana/Multiomics_Project/data'
CACHE_FILE = f'{DATA_DIR}/external_cache.json'

GNOMAD_API  = 'https://gnomad.broadinstitute.org/api'
GTEX_BASE   = 'https://gtexportal.org/api/v2'
VEP_BASE    = 'https://rest.ensembl.org'

GTEX_TISSUES = {
    'blood':      'Whole_Blood',
    'fibroblast': 'Cells_Cultured_fibroblasts',
}

HEADERS_JSON = {'Content-Type': 'application/json', 'Accept': 'application/json'}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def safe_get(url, params=None, headers=None, timeout=15, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return None

def safe_post(url, payload, headers=None, timeout=15, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return None

# ---------------------------------------------------------------------------
# gnomAD: LOEUF per gene
# ---------------------------------------------------------------------------

GNOMAD_LOEUF_QUERY = """
query GeneConstraint($gene: String!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    gnomad_constraint {
      oe_lof_upper
    }
  }
}
"""

def fetch_loeuf(gene, cache):
    key = f"loeuf:{gene}"
    if key in cache:
        return cache[key]
    payload = {'query': GNOMAD_LOEUF_QUERY, 'variables': {'gene': gene}}
    data = safe_post(GNOMAD_API, payload, headers=HEADERS_JSON)
    loeuf = np.nan
    if data:
        try:
            loeuf = data['data']['gene']['gnomad_constraint']['oe_lof_upper']
        except (TypeError, KeyError):
            pass
    cache[key] = loeuf
    time.sleep(0.3)
    return loeuf

# ---------------------------------------------------------------------------
# gnomAD: allele frequency per variant (via variant ID chrom-pos-ref-alt)
# ---------------------------------------------------------------------------

GNOMAD_AF_QUERY = """
query VariantAF($variantId: String!) {
  variant(variantId: $variantId, dataset: gnomad_r4) {
    genome { af }
    exome { af }
  }
}
"""

def build_gnomad_variant_id(cnomen_key):
    """
    We don't have chrom/pos in S1 directly.
    Return None — AF will be fetched from S2 genomic coords where available.
    """
    return None

def fetch_variant_af(chrom, pos, ref, alt, cache):
    variant_id = f"{chrom}-{pos}-{ref}-{alt}"
    key = f"af:{variant_id}"
    if key in cache:
        return cache[key]
    payload = {'query': GNOMAD_AF_QUERY, 'variables': {'variantId': variant_id}}
    data = safe_post(GNOMAD_API, payload, headers=HEADERS_JSON)
    af = np.nan
    if data and data.get('data', {}).get('variant'):
        v = data['data']['variant']
        afs = []
        for src in ['genome', 'exome']:
            try:
                a = v[src]['af']
                if a is not None:
                    afs.append(a)
            except (TypeError, KeyError):
                pass
        if afs:
            af = max(afs)
    cache[key] = af
    time.sleep(0.3)
    return af

# ---------------------------------------------------------------------------
# GTEx: blood + fibroblast median TPM per gene
# ---------------------------------------------------------------------------

def fetch_gtex_gene_id(gene_symbol, cache):
    key = f"gtex_geneid:{gene_symbol}"
    if key in cache:
        return cache[key]
    url = f"{GTEX_BASE}/reference/gene"
    data = safe_get(url, params={'geneSymbol': gene_symbol, 'gencodeVersion': 'v26',
                                  'genomeBuild': 'GRCh38'}, headers=HEADERS_JSON)
    gencode_id = np.nan
    if data and data.get('data'):
        try:
            gencode_id = data['data'][0]['gencodeId']
        except (IndexError, KeyError):
            pass
    cache[key] = gencode_id
    time.sleep(0.2)
    return gencode_id

def fetch_gtex_tpm(gene_symbol, tissue_key, gencode_id, cache):
    key = f"gtex_tpm:{gene_symbol}:{tissue_key}"
    if key in cache:
        return cache[key]
    if not gencode_id or (isinstance(gencode_id, float) and math.isnan(gencode_id)):
        cache[key] = np.nan
        return np.nan
    url = f"{GTEX_BASE}/expression/geneExpression"
    data = safe_get(url, params={'gencodeId': gencode_id,
                                  'tissueSiteDetailId': GTEX_TISSUES[tissue_key]},
                    headers=HEADERS_JSON)
    tpm = np.nan
    if data and data.get('data'):
        try:
            tpm = data['data'][0]['median']
        except (IndexError, KeyError):
            pass
    cache[key] = tpm
    time.sleep(0.2)
    return tpm

# ---------------------------------------------------------------------------
# Ensembl VEP: consequence + frameshift/PTC flag
# ---------------------------------------------------------------------------

def fetch_vep(cnomen_key, cache):
    """Query Ensembl VEP REST with HGVS notation."""
    key = f"vep:{cnomen_key}"
    if key in cache:
        return cache[key]

    hgvs = requests.utils.quote(str(cnomen_key))
    url = f"{VEP_BASE}/vep/human/hgvs/{hgvs}"
    data = safe_get(url, params={'content-type': 'application/json'},
                    headers={'Accept': 'application/json'})

    consequence = np.nan
    is_ptc = 0

    if data and isinstance(data, list) and len(data) > 0:
        entry = data[0]
        try:
            tc = entry.get('transcript_consequences', [])
            if tc:
                cons_terms = tc[0].get('consequence_terms', [])
                consequence = ','.join(cons_terms)
                ptc_terms = {'frameshift_variant', 'stop_gained', 'stop_lost',
                             'splice_acceptor_variant', 'splice_donor_variant'}
                if ptc_terms & set(cons_terms):
                    is_ptc = 1
        except Exception:
            pass

    result = {'vep_consequence': consequence, 'vep_is_ptc': is_ptc}
    cache[key] = result
    time.sleep(0.3)
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_ext  = pd.read_csv(f'{DATA_DIR}/variants_for_external.csv')
    df_s2   = pd.read_csv(f'{DATA_DIR}/s2_insilico.csv')

    # Load S2 with raw coords for gnomAD AF
    df_s2_raw = pd.read_excel(
        '/data/staff/Jana/Multiomics_Project/1-s2.0-S2666247725001241-mmc2.xlsx',
        sheet_name='Table S2', header=2
    )
    df_s2_raw.columns = [str(c).strip() for c in df_s2_raw.columns]
    df_s2_raw = df_s2_raw[df_s2_raw['Dataset'] == 'current dataset'].dropna(subset=['cNomen'])
    df_s2_raw['cnomen_key'] = df_s2_raw['cNomen'].str.strip()

    # Build a coord lookup: cnomen_key → (chrom, pos, ref, alt)
    coord_map = {}
    for _, row in df_s2_raw.iterrows():
        try:
            coord_map[row['cnomen_key']] = (
                str(int(row['chrom'])), str(int(row['position'])),
                str(row['reference']), str(row['alternative'])
            )
        except Exception:
            pass

    cache = load_cache()
    genes = df_ext['gene'].dropna().unique().tolist()

    print(f"Fetching external data for {len(genes)} genes and {len(df_ext)} variants ...")
    print("This may take several minutes — results cached after first run.\n")

    # --- Gene-level: LOEUF + GTEx ---
    gene_data = {}
    for i, gene in enumerate(genes):
        print(f"  [{i+1}/{len(genes)}] {gene}: gnomAD LOEUF ...", end=' ', flush=True)
        loeuf = fetch_loeuf(gene, cache)

        gencode_id = fetch_gtex_gene_id(gene, cache)
        tpm_blood  = fetch_gtex_tpm(gene, 'blood',      gencode_id, cache)
        tpm_fibro  = fetch_gtex_tpm(gene, 'fibroblast', gencode_id, cache)

        gene_data[gene] = {
            'loeuf':       loeuf,
            'tpm_blood':   tpm_blood,
            'tpm_fibro':   tpm_fibro,
        }
        save_cache(cache)
        print(f"LOEUF={loeuf:.3f}" if isinstance(loeuf, float) and not math.isnan(loeuf) else f"LOEUF=NA", flush=True)

    # --- Variant-level: gnomAD AF + VEP ---
    variant_rows = []
    for i, row in df_ext.iterrows():
        cnomen = row['cnomen_key']
        gene   = row['gene']

        if not isinstance(cnomen, str) or cnomen == 'nan':
            variant_rows.append({'variant_id': row['variant_id'],
                                  'gnomad_af': np.nan,
                                  'vep_consequence': np.nan,
                                  'vep_is_ptc': np.nan})
            continue

        # gnomAD AF via genomic coords
        af = np.nan
        if cnomen in coord_map:
            chrom, pos, ref, alt = coord_map[cnomen]
            af = fetch_variant_af(chrom, pos, ref, alt, cache)

        # VEP consequence
        vep = fetch_vep(cnomen, cache)
        save_cache(cache)

        variant_rows.append({
            'variant_id':      row['variant_id'],
            'gnomad_af':       af,
            'vep_consequence': vep['vep_consequence'],
            'vep_is_ptc':      vep['vep_is_ptc'],
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(df_ext)} variants ...", flush=True)

    # --- Assemble gene-level features ---
    df_gene = pd.DataFrame([
        {'gene': g, **vals} for g, vals in gene_data.items()
    ])

    # Tissue specificity score = max(log2(TPM_blood+1), log2(TPM_fibro+1)), then 0–1 normalise
    df_gene['log_tpm_blood'] = np.log2(pd.to_numeric(df_gene['tpm_blood'], errors='coerce') + 1)
    df_gene['log_tpm_fibro'] = np.log2(pd.to_numeric(df_gene['tpm_fibro'], errors='coerce') + 1)
    df_gene['tissue_score_raw'] = df_gene[['log_tpm_blood', 'log_tpm_fibro']].max(axis=1)
    max_ts = df_gene['tissue_score_raw'].max()
    df_gene['tissue_specificity_score'] = (
        df_gene['tissue_score_raw'] / max_ts if max_ts > 0 else df_gene['tissue_score_raw']
    )

    # LOEUF score = 1 – normalise(LOEUF)
    loeuf_max = df_gene['loeuf'].max()
    loeuf_min = df_gene['loeuf'].min()
    df_gene['loeuf_score'] = 1 - (
        (df_gene['loeuf'] - loeuf_min) / (loeuf_max - loeuf_min)
        if loeuf_max > loeuf_min else 0
    )

    df_gene_out = df_gene[['gene', 'loeuf', 'loeuf_score',
                            'tpm_blood', 'tpm_fibro', 'tissue_specificity_score']]

    # --- Merge variant + gene features ---
    df_var = pd.DataFrame(variant_rows)
    df_ext2 = df_ext.merge(df_gene_out, on='gene', how='left')
    df_ext2 = df_ext2.merge(df_var, on='variant_id', how='left')

    out_path = f'{DATA_DIR}/external_features.csv'
    df_ext2.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_ext2)} rows)")
    print("02_fetch_external.py complete.")

if __name__ == '__main__':
    main()
