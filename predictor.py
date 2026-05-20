"""
predictor.py — VUS Reclassification Predictor
Core class: loads all 3 trained models, accepts raw variant inputs,
auto-fetches external annotations (gnomAD/GTEx/VEP), builds the
56-feature vector, and returns an ensemble Reclassification Score (0–100).
"""

import os, re, json, time
import numpy as np
import pandas as pd
import pickle
import requests
from typing import Optional, Dict, Any, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'results', 'metrics')
DATA_DIR    = os.path.join(BASE_DIR, 'data')

# ── Categorical lookup tables (must match 03_feature_engineering.py) ─────────
EFFECT_CATS = [
    'No effect', 'Exon skipping',
    'Use of alternative SS, inclusion of intron',
    'Use of alternative SS, exclusion of exon',
    'Multiple events', 'Pseudoexon', 'U12-U2 switch',
]
VULEXMAP_CATS_RAW = [
    'intronic', 'splicesite_vulnerable_', 'resilient', 'not_in_database',
    'splicesite_resilient_', 'vulnerable', 'alternative',
    'splicesite_alternative_', 'splicesite_resilient_alternati',
    'splicesite_nasegment_',
]
VARIANT_TYPES_RAW = [
    'canonicalspliceacceptor', 'canonicalsplicedonor',
    'exonicdistant', 'intronicdistant',
]

PTC_CONSEQUENCES = {
    'frameshift_variant', 'stop_gained', 'stop_lost',
    'splice_acceptor_variant', 'splice_donor_variant',
    'start_lost', 'transcript_ablation',
}

# Ensemble AUC weights (from 5-fold CV)
_AUC_WEIGHTS = {
    'RandomForest':       0.947,
    'XGBoost':            0.940,
    'LogisticRegression': 0.935,
}


class VUSPredictor:
    """Load trained models and predict LP/P probability for a splice variant."""

    def __init__(self):
        self._load_models()
        self._load_imputation_stats()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        all_path  = os.path.join(MODELS_DIR, 'all_models.pkl')
        best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        if os.path.exists(all_path):
            with open(all_path, 'rb') as f:
                self.models = pickle.load(f)
        else:
            with open(best_path, 'rb') as f:
                best = pickle.load(f)
            self.models = {best['name']: {'model': best['model'],
                                          'feature_cols': best['feature_cols']}}
        self.feature_cols = list(self.models.values())[0]['feature_cols']

    def _load_imputation_stats(self):
        imp_path = os.path.join(MODELS_DIR, 'imputation_stats.pkl')
        if os.path.exists(imp_path):
            with open(imp_path, 'rb') as f:
                self.medians = pickle.load(f)
        else:
            # Fall back: compute from feature matrix
            df = pd.read_csv(os.path.join(DATA_DIR, 'feature_matrix.csv'))
            self.medians = {}
            for col in self.feature_cols:
                if col in df.columns:
                    self.medians[col] = float(df[col].median())
                else:
                    self.medians[col] = 0.0

    # ── External API fetch ────────────────────────────────────────────────────

    def fetch_annotations(self, gene: str, variant_notation: str = '',
                          chrom: str = None, pos: int = None,
                          ref: str = None, alt: str = None) -> Dict:
        """Fetch gnomAD LOEUF+AF, GTEx TPM, VEP consequence for one variant."""
        result = {'fetch_status': {}}

        # gnomAD LOEUF (gene-level constraint)
        try:
            result.update(self._fetch_gnomad_loeuf(gene))
            result['fetch_status']['loeuf'] = 'ok'
        except Exception as e:
            result.update({'loeuf': np.nan, 'loeuf_score': np.nan})
            result['fetch_status']['loeuf'] = str(e)

        # GTEx tissue expression
        try:
            result.update(self._fetch_gtex(gene))
            result['fetch_status']['gtex'] = 'ok'
        except Exception as e:
            result.update({'tpm_blood': np.nan, 'tpm_fibro': np.nan,
                           'tissue_specificity_score': np.nan})
            result['fetch_status']['gtex'] = str(e)

        # VEP consequence
        try:
            result.update(self._fetch_vep(variant_notation, chrom, pos, ref, alt))
            result['fetch_status']['vep'] = 'ok'
        except Exception as e:
            result.update({'vep_is_ptc': 0, 'vep_consequence': 'unknown'})
            result['fetch_status']['vep'] = str(e)

        # gnomAD allele frequency (variant-level)
        try:
            result.update(self._fetch_gnomad_af(chrom, pos, ref, alt))
            result['fetch_status']['gnomad_af'] = 'ok'
        except Exception as e:
            result['gnomad_af'] = np.nan
            result['fetch_status']['gnomad_af'] = str(e)

        return result

    def _fetch_gnomad_loeuf(self, gene: str) -> Dict:
        query = '''query($gs: String!) {
          gene(gene_symbol: $gs, reference_genome: GRCh38) {
            gnomad_constraint { oe_lof_upper }
          }
        }'''
        r = requests.post('https://gnomad.broadinstitute.org/api',
                          json={'query': query, 'variables': {'gs': gene}},
                          timeout=15)
        r.raise_for_status()
        loeuf = r.json()['data']['gene']['gnomad_constraint']['oe_lof_upper']
        score = float(max(0.0, 1.0 - loeuf / 2.0)) if loeuf is not None else np.nan
        return {'loeuf': loeuf, 'loeuf_score': score}

    def _fetch_gtex(self, gene: str) -> Dict:
        # Step 1: resolve gene symbol → Ensembl ID
        url = ('https://gtexportal.org/api/v2/reference/gene'
               f'?geneId={gene}&gencodeVersion=v39&genomeBuild=GRCh38%2Fhg38')
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get('data', [])
        if not data:
            raise ValueError(f'Gene {gene} not found in GTEx v39')
        ensg = data[0]['gencodeId']

        tpm = {}
        for tissue, key in [('Whole_Blood', 'blood'),
                             ('Cells_Cultured_fibroblasts', 'fibro')]:
            url2 = ('https://gtexportal.org/api/v2/expression/medianGeneExpression'
                    f'?datasetId=gtex_v8&gencodeId={ensg}&tissueSiteDetailId={tissue}')
            r2 = requests.get(url2, timeout=15)
            r2.raise_for_status()
            d2 = r2.json().get('data', [])
            tpm[key] = float(np.median([x['median'] for x in d2])) if d2 else np.nan

        blood, fibro = tpm.get('blood', np.nan), tpm.get('fibro', np.nan)
        log_vals = [np.log2(v + 1) for v in [blood, fibro] if not np.isnan(v)]
        tss = float(max(log_vals)) if log_vals else np.nan
        return {'tpm_blood': blood, 'tpm_fibro': fibro,
                'tissue_specificity_score': tss}

    def _fetch_vep(self, variant_notation: str = '',
                   chrom=None, pos=None, ref=None, alt=None) -> Dict:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        # Try HGVS notation first
        if variant_notation:
            r = requests.post('https://rest.ensembl.org/vep/human/hgvs',
                              json={'hgvs_notations': [variant_notation]},
                              headers=headers, timeout=20)
            if r.status_code == 200:
                d = r.json()
                if d:
                    csq = d[0].get('most_severe_consequence', 'unknown')
                    return {'vep_consequence': csq,
                            'vep_is_ptc': 1 if any(t in csq for t in PTC_CONSEQUENCES) else 0}

        # Fallback: genomic coordinates
        if chrom and pos and ref and alt:
            region = f'{chrom} {pos} . {ref} {alt}'
            r2 = requests.post('https://rest.ensembl.org/vep/human/region',
                               json={'variants': [region]},
                               headers=headers, timeout=20)
            if r2.status_code == 200:
                d2 = r2.json()
                if d2:
                    csq = d2[0].get('most_severe_consequence', 'unknown')
                    return {'vep_consequence': csq,
                            'vep_is_ptc': 1 if any(t in csq for t in PTC_CONSEQUENCES) else 0}

        return {'vep_consequence': 'unknown', 'vep_is_ptc': 0}

    def _fetch_gnomad_af(self, chrom=None, pos=None, ref=None, alt=None) -> Dict:
        if not all([chrom, pos, ref, alt]):
            return {'gnomad_af': np.nan}
        var_id = f'{chrom}-{pos}-{ref}-{alt}'
        query = '''query($vid: String!) {
          variant(dataset: gnomad_r3, variantId: $vid) {
            genome { af }
          }
        }'''
        try:
            r = requests.post('https://gnomad.broadinstitute.org/api',
                              json={'query': query, 'variables': {'vid': var_id}},
                              timeout=15)
            af = r.json()['data']['variant']['genome']['af']
            return {'gnomad_af': float(af) if af is not None else np.nan}
        except Exception:
            return {'gnomad_af': np.nan}

    # ── Feature vector construction ───────────────────────────────────────────

    def build_feature_vector(self, user_input: Dict,
                             annotations: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Build the 56-d feature vector.
        Returns (vector, feature_dict) — the dict is useful for debugging.
        Missing values are filled with training-set medians.
        """
        f = {col: self.medians.get(col, 0.0) for col in self.feature_cols}
        raw = {**annotations, **user_input}  # user input takes precedence

        # ── Direct numeric / binary features ──────────────────────────────────
        direct = ['aberrant_splicing', 'wt_transcript', 'is_frameshift_ptc',
                  'splice_distance', 'exon_number', 'exon_proportion',
                  'tissue_code', 'alamut_ese_code',
                  'spliceai_max', 'pangolin_max', 'spip_score', 'squirls_max',
                  'loeuf', 'loeuf_score', 'tpm_blood', 'tpm_fibro',
                  'tissue_specificity_score', 'gnomad_af', 'vep_is_ptc']
        for key in direct:
            val = raw.get(key)
            if val is not None and key in f:
                try:
                    fv = float(val)
                    if not np.isnan(fv):
                        f[key] = fv
                except (TypeError, ValueError):
                    pass

        # Percentage features — normalise to 0–1 if given as 0–100
        for key in ['pct_aberrant_sanger', 'pct_aberrant_agarose']:
            val = raw.get(key)
            if val is not None and key in f:
                try:
                    fv = float(val)
                    if fv > 1.0:
                        fv /= 100.0
                    if not np.isnan(fv):
                        f[key] = fv
                except (TypeError, ValueError):
                    pass

        # ── One-hot: effect_category ───────────────────────────────────────────
        eff_cat = raw.get('effect_category', '')
        for cat in EFFECT_CATS:
            col = 'eff_' + re.sub(r'[^a-z0-9]', '_', cat.lower())[:30]
            if col in f:
                f[col] = 1.0 if eff_cat == cat else 0.0

        # ── One-hot: vulexmap_category ─────────────────────────────────────────
        vex_cat = raw.get('vulexmap_category', '')
        for cat in VULEXMAP_CATS_RAW:
            col = 'vex_' + re.sub(r'[^a-z0-9]', '_', cat.lower())[:30]
            if col in f:
                f[col] = 1.0 if vex_cat == cat else 0.0

        # ── One-hot: variant_type ──────────────────────────────────────────────
        vtype = re.sub(r'[^a-z0-9]', '', raw.get('variant_type', '').lower())
        for cat in VARIANT_TYPES_RAW:
            col = 'vtype_' + re.sub(r'[^a-z0-9]', '_', cat.lower())[:30]
            if col in f:
                f[col] = 1.0 if vtype == cat else 0.0

        # ── NMD prediction ─────────────────────────────────────────────────────
        is_ptc = float(f.get('is_frameshift_ptc', 0))
        ep = f.get('exon_proportion', np.nan)
        if is_ptc == 1:
            if not (isinstance(ep, float) and np.isnan(ep)):
                f['nmd_sensitive'] = 1.0 if ep < 0.90 else 0.0
                if ep < 0.50:
                    f['nmd_score'] = 1.0
                elif ep < 0.75:
                    f['nmd_score'] = 0.85
                elif ep < 0.90:
                    f['nmd_score'] = 0.60
                else:
                    f['nmd_score'] = 0.10
            else:
                f['nmd_sensitive'] = 1.0
                f['nmd_score'] = 0.5
        else:
            f['nmd_sensitive'] = 0.0
            f['nmd_score'] = 0.0

        # ── ACMG evidence codes ────────────────────────────────────────────────
        loeuf_s  = f.get('loeuf_score', np.nan)
        splai    = f.get('spliceai_max', np.nan)
        pang     = f.get('pangolin_max', np.nan)
        spip     = f.get('spip_score', np.nan)
        sqrl     = f.get('squirls_max', np.nan)
        aberrant = float(f.get('aberrant_splicing', 0))
        pct_s    = f.get('pct_aberrant_sanger', np.nan)
        pct_a    = f.get('pct_aberrant_agarose', np.nan)
        sd       = float(f.get('splice_distance', 99))
        gnomad   = f.get('gnomad_af', np.nan)
        wt       = float(f.get('wt_transcript', 0))

        def _valid(v):
            return v is not None and not (isinstance(v, float) and np.isnan(v))

        # PVS1: LoF in constrained gene
        pvs1 = 0
        if is_ptc == 1 and _valid(loeuf_s) and loeuf_s > 0.6:
            pvs1 = 1
        elif sd <= 2 and aberrant == 1 and _valid(loeuf_s) and loeuf_s > 0.5:
            pvs1 = 1
        f['acmg_pvs1'] = pvs1

        # PS3: functional RNA evidence
        f['acmg_ps3'] = 1 if aberrant == 1 and _valid(pct_s) and pct_s >= 0.5 else 0

        # PM2: absent/rare in gnomAD
        f['acmg_pm2'] = 1 if not _valid(gnomad) or gnomad < 0.001 else 0

        # PP3: in silico pathogenic consensus
        votes_p = sum([
            1 if _valid(splai) and splai > 0.5 else 0,
            1 if _valid(pang)  and pang  > 0.5 else 0,
            1 if _valid(spip)  and spip  > 0.9 else 0,
        ])
        f['acmg_pp3'] = 1 if votes_p >= 2 else 0

        # BP4: in silico benign consensus
        votes_b = sum([
            1 if _valid(splai) and splai < 0.1 else 0,
            1 if _valid(pang)  and pang  < 0.1 else 0,
            1 if _valid(spip)  and spip  < 0.3 else 0,
            1 if _valid(sqrl)  and sqrl  < 0.1 else 0,
        ])
        f['acmg_bp4'] = 1 if votes_b >= 3 else 0

        # BP7: no RNA effect, WT maintained
        eff_cat_str = raw.get('effect_category', '')
        f['acmg_bp7'] = 1 if aberrant == 0 and wt == 1 and 'No effect' in eff_cat_str else 0

        f['acmg_total_score'] = (
            f['acmg_pvs1'] * 8 + f['acmg_ps3'] * 4 + f['acmg_pm2'] * 2 +
            f['acmg_pp3'] * 1 + f['acmg_bp4'] * -1 + f['acmg_bp7'] * -2
        )

        # ── Specificity features ───────────────────────────────────────────────
        f['no_rna_effect']          = 1 if aberrant == 0 and wt == 1 else 0
        f['high_pct_aberrant']      = 1 if (
            (_valid(pct_s) and pct_s >= 0.5) or (_valid(pct_a) and pct_a >= 0.5)
        ) else 0
        f['is_canonical_ss']        = 1 if sd <= 2 else 0
        f['tools_agree_benign']     = 1 if (
            _valid(splai) and splai < 0.2 and
            _valid(pang)  and pang  < 0.2 and
            _valid(spip)  and spip  < 0.5 and
            _valid(sqrl)  and sqrl  < 0.2
        ) else 0
        f['tools_agree_pathogenic'] = 1 if (
            _valid(splai) and splai > 0.5 and
            _valid(pang)  and pang  > 0.5
        ) else 0

        vec = np.array([f.get(col, 0.0) for col in self.feature_cols], dtype=float)
        return vec, f

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, feature_vector: np.ndarray) -> Dict:
        """
        Run all 3 models and return ensemble Reclassification Score (0–100).
        """
        probs = {}
        for name, data in self.models.items():
            p = data['model'].predict_proba(feature_vector.reshape(1, -1))[0, 1]
            probs[name] = float(p)

        # Weighted ensemble
        total_w = sum(_AUC_WEIGHTS.get(m, 1.0) for m in probs)
        ens = sum(probs[m] * _AUC_WEIGHTS.get(m, 1.0) for m in probs) / total_w
        score = round(ens * 100, 1)

        if score >= 70:
            cls, tier = 'Likely Pathogenic / Pathogenic', 'LP/P'
            colour = '#d32f2f'
        elif score >= 50:
            cls, tier = 'Uncertain — lean pathogenic', 'VUS+'
            colour = '#f57c00'
        elif score >= 30:
            cls, tier = 'Uncertain — lean benign', 'VUS-'
            colour = '#f9a825'
        else:
            cls, tier = 'Likely Benign', 'LB'
            colour = '#388e3c'

        return {
            'model_probs':            probs,
            'ensemble_prob':          round(ens, 4),
            'reclassification_score': score,
            'classification':         cls,
            'tier':                   tier,
            'colour':                 colour,
        }

    # ── SHAP explanation ──────────────────────────────────────────────────────

    def explain_shap(self, feature_vector: np.ndarray,
                     top_n: int = 15) -> pd.DataFrame:
        """Return top-N SHAP contributors (RandomForest TreeExplainer)."""
        try:
            import shap
            data = self.models.get('RandomForest')
            if data is None:
                return pd.DataFrame()
            explainer = shap.TreeExplainer(data['model'])
            sv = explainer.shap_values(feature_vector.reshape(1, -1))
            # sv is [neg_class, pos_class] for binary RF
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
            return (pd.DataFrame({
                        'feature':    self.feature_cols,
                        'shap_value': vals,
                        'abs_shap':   np.abs(vals),
                    })
                    .sort_values('abs_shap', ascending=False)
                    .head(top_n)
                    .reset_index(drop=True))
        except Exception:
            return pd.DataFrame()

    # ── Convenience: full pipeline ────────────────────────────────────────────

    def run_full_prediction(self, gene: str, variant_notation: str,
                            user_input: Dict,
                            chrom=None, pos=None, ref=None, alt=None) -> Dict:
        """
        End-to-end: fetch annotations → build features → predict → SHAP.
        user_input keys: spliceai_max, pangolin_max, spip_score, squirls_max,
          aberrant_splicing, wt_transcript, pct_aberrant_sanger,
          pct_aberrant_agarose, is_frameshift_ptc, splice_distance,
          exon_number, exon_proportion, tissue_code, effect_category, etc.
        """
        annotations = self.fetch_annotations(gene, variant_notation,
                                             chrom, pos, ref, alt)
        vec, feat_dict = self.build_feature_vector(user_input, annotations)
        result = self.predict(vec)
        shap_df = self.explain_shap(vec)

        return {
            **result,
            'annotations':    annotations,
            'feature_values': feat_dict,
            'shap_top':       shap_df,
        }
