"""
app.py — VUS Reclassification AI System
Streamlit web interface with three pages:
  • Predict  — submit a variant, get ensemble score + SHAP explanation
  • Validate — full model validation report (all plots, metrics table)
  • About    — scientific background, limitations, citation
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='VUS Reclassification AI',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='expanded',
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
PLOTS_DIR   = os.path.join(BASE_DIR, 'results', 'plots')

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    from predictor import VUSPredictor
    return VUSPredictor()

@st.cache_data(show_spinner=False)
def load_summary():
    return pd.read_csv(os.path.join(METRICS_DIR, 'summary_metrics.csv'))

@st.cache_data(show_spinner=False)
def load_cv():
    return pd.read_csv(os.path.join(METRICS_DIR, 'cv_results.csv'))

@st.cache_data(show_spinner=False)
def load_scores():
    return pd.read_csv(os.path.join(METRICS_DIR, 'integrated_scores.csv'))

@st.cache_data(show_spinner=False)
def load_roc_data():
    with open(os.path.join(METRICS_DIR, 'roc_data.pkl'), 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_feat_imp():
    with open(os.path.join(METRICS_DIR, 'feature_importance.pkl'), 'rb') as f:
        return pickle.load(f)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .score-box {
    border-radius: 12px; padding: 20px 30px; text-align: center;
    margin: 10px 0; color: white; font-weight: bold;
  }
  .metric-card {
    background: #f8f9fa; border-radius: 8px; padding: 16px;
    border-left: 4px solid #1976d2; margin: 6px 0;
  }
  .acmg-tag {
    display: inline-block; border-radius: 4px; padding: 2px 8px;
    margin: 3px; font-size: 13px; font-weight: bold;
  }
  .tag-path  { background: #ffcdd2; color: #b71c1c; }
  .tag-benign{ background: #c8e6c9; color: #1b5e20; }
  .tag-neut  { background: #e0e0e0; color: #424242; }
  .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 500; }
  h1 { color: #1565c0; }
  h2 { color: #283593; border-bottom: 2px solid #e8eaf6; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://img.icons8.com/color/96/dna-helix.png', width=80)
    st.title('VUS Classifier')
    st.caption('BIRAC-BIOAI | Splice Variant AI')
    st.divider()
    page = st.radio('Navigate', ['🔮 Predict Variant',
                                  '📊 Model Validation',
                                  'ℹ️ About'])
    st.divider()
    st.caption('**Dataset**: 178 splice VUS\n\n'
               '**Source**: Drost et al., HGG Advances 2026\n\n'
               '**Models**: LR · RF · XGBoost (5-fold CV)')

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == '🔮 Predict Variant':
    st.title('🔮 VUS Reclassification Predictor')
    st.markdown(
        'Enter variant details below. The tool fetches external annotations '
        '(gnomAD, GTEx, VEP) automatically and returns an ensemble '
        '**Reclassification Score (0–100)** with ACMG evidence codes.'
    )

    # ── Input tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(['① Variant Info', '② RNA Results', '③ In Silico Scores'])

    with tab1:
        c1, c2 = st.columns(2)
        gene      = c1.text_input('Gene symbol *', placeholder='e.g. BRCA2')
        variant   = c2.text_input('Variant notation *',
                                   placeholder='e.g. NM_000059.3:c.8023G>A')
        st.caption('Used for gnomAD, GTEx, and VEP auto-fetch. '
                   'HGVS notation preferred (NM_xxxxx.x:c.xxxX>X).')
        c3, c4, c5 = st.columns(3)
        chrom = c3.text_input('Chromosome (optional)', placeholder='e.g. 13')
        pos   = c4.number_input('Position (optional)', min_value=0, value=0, step=1)
        ref   = c5.text_input('Ref allele', placeholder='e.g. G')
        alt_allele = c5.text_input('Alt allele', placeholder='e.g. A')

        splice_dist = st.slider(
            'Splice distance (|bp from splice site|) *',
            min_value=0, max_value=50, value=5,
            help='Distance in bp from the canonical splice site. ±1/±2 = canonical SS.'
        )
        c6, c7 = st.columns(2)
        exon_num   = c6.number_input('Exon number', min_value=0, value=0, step=1)
        total_exon = c7.number_input('Total exons', min_value=0, value=0, step=1)
        exon_prop = float(exon_num) / float(total_exon) if total_exon > 0 else np.nan

        tissue_map = {'Blood': 1, 'Fibroblast': 2, 'Other': 0}
        tissue = st.selectbox('Tissue studied', list(tissue_map.keys()))
        tissue_code = tissue_map[tissue]

    with tab2:
        has_rna = st.toggle('RNA functional data available', value=True)
        if has_rna:
            c1, c2 = st.columns(2)
            aberrant  = c1.selectbox('Aberrant splicing observed?', ['Yes', 'No'])
            wt_trans  = c2.selectbox('Wild-type transcript maintained?', ['Yes', 'No'])
            aberrant_val  = 1 if aberrant == 'Yes' else 0
            wt_val        = 1 if wt_trans == 'Yes' else 0

            c3, c4 = st.columns(2)
            pct_sang  = c3.slider('% Aberrant transcripts (Sanger)',  0, 100, 0) / 100
            pct_aag   = c4.slider('% Aberrant transcripts (Agarose)', 0, 100, 0) / 100

            is_ptc = st.selectbox('Frameshift / PTC (premature termination)?', ['No', 'Yes'])
            is_ptc_val = 1 if is_ptc == 'Yes' else 0

            eff_cat = st.selectbox('Effect category', [
                'No effect', 'Exon skipping',
                'Use of alternative SS, inclusion of intron',
                'Use of alternative SS, exclusion of exon',
                'Multiple events', 'Pseudoexon', 'U12-U2 switch', 'Unknown',
            ])
            vex_cat = st.selectbox('VulExMap category', [
                'intronic', 'splicesite_vulnerable_', 'resilient',
                'not_in_database', 'splicesite_resilient_', 'vulnerable',
                'alternative', 'splicesite_alternative_', 'Unknown',
            ])
            var_type = st.selectbox('Variant type', [
                'canonicalspliceacceptor', 'canonicalsplicedonor',
                'exonicdistant', 'intronicdistant', 'Unknown',
            ])
        else:
            aberrant_val = wt_val = is_ptc_val = 0
            pct_sang = pct_aag = 0.0
            eff_cat = vex_cat = var_type = 'Unknown'

    with tab3:
        st.info('Enter scores from your in silico tools. Leave at 0 if not available.')
        c1, c2 = st.columns(2)
        spliceai = c1.number_input('SpliceAI max delta score', 0.0, 1.0, 0.0, 0.01,
                                    help='max(DS_AG, DS_AL, DS_DG, DS_DL)')
        pangolin = c2.number_input('Pangolin max score',        0.0, 1.0, 0.0, 0.01)
        spip     = c1.number_input('SPiP score',                0.0, 1.0, 0.0, 0.01)
        squirls  = c2.number_input('SQUIRLS max score',         0.0, 1.0, 0.0, 0.01)
        loeuf_in = c1.number_input('LOEUF (optional)',          0.0, 3.0, 0.0, 0.01,
                                    help='From gnomAD constraints. Will be auto-fetched if gene is provided.')

    # ── Fetch & Predict buttons ────────────────────────────────────────────────
    st.divider()
    col_fetch, col_pred, _ = st.columns([1, 1, 3])

    annotations = {}
    if col_fetch.button('🌐 Fetch Annotations', use_container_width=True):
        if not gene:
            st.warning('Please enter a gene symbol.')
        else:
            with st.spinner('Fetching gnomAD, GTEx, VEP…'):
                pred = load_predictor()
                annotations = pred.fetch_annotations(
                    gene=gene,
                    variant_notation=variant,
                    chrom=str(chrom) if chrom else None,
                    pos=int(pos) if pos else None,
                    ref=ref if ref else None,
                    alt=alt_allele if alt_allele else None,
                )
                st.session_state['annotations'] = annotations
            status = annotations.get('fetch_status', {})
            ok = [k for k, v in status.items() if v == 'ok']
            fail = [k for k, v in status.items() if v != 'ok']
            if ok:
                st.success(f'Fetched: {", ".join(ok)}')
            if fail:
                st.warning(f'Could not fetch: {", ".join(fail)} (will use medians)')

            # Show fetched values
            disp = {k: v for k, v in annotations.items() if k != 'fetch_status'}
            st.json({k: (round(v, 4) if isinstance(v, float) and not np.isnan(v)
                         else ('N/A' if isinstance(v, float) else v))
                     for k, v in disp.items()})

    run_predict = col_pred.button('🧬 Predict', type='primary', use_container_width=True)

    if run_predict:
        if not gene and not variant and spliceai == 0 and pangolin == 0:
            st.error('Please enter at least gene name and/or in silico scores.')
        else:
            # Assemble user input dict
            user_input = {
                'aberrant_splicing':    aberrant_val,
                'wt_transcript':        wt_val,
                'pct_aberrant_sanger':  pct_sang,
                'pct_aberrant_agarose': pct_aag,
                'is_frameshift_ptc':    is_ptc_val,
                'splice_distance':      splice_dist,
                'exon_proportion':      exon_prop if not np.isnan(exon_prop) else None,
                'exon_number':          exon_num if exon_num > 0 else None,
                'tissue_code':          tissue_code,
                'spliceai_max':         spliceai if spliceai > 0 else None,
                'pangolin_max':         pangolin if pangolin > 0 else None,
                'spip_score':           spip     if spip > 0 else None,
                'squirls_max':          squirls  if squirls > 0 else None,
                'loeuf':                loeuf_in if loeuf_in > 0 else None,
                'effect_category':      eff_cat,
                'vulexmap_category':    vex_cat,
                'variant_type':         var_type,
            }
            # Override with loeuf_score if loeuf provided manually
            if loeuf_in > 0:
                user_input['loeuf_score'] = float(max(0.0, 1.0 - loeuf_in / 2.0))

            ann = st.session_state.get('annotations', {})

            with st.spinner('Running models…'):
                pred = load_predictor()
                vec, feat_dict = pred.build_feature_vector(user_input, ann)
                result = pred.predict(vec)
                shap_df = pred.explain_shap(vec)

            # ── Score display ──────────────────────────────────────────────────
            st.divider()
            score = result['reclassification_score']
            cls   = result['classification']
            colour= result['colour']
            tier  = result['tier']

            cc1, cc2 = st.columns([1, 2])
            with cc1:
                st.markdown(f"""
                <div class="score-box" style="background:{colour}; font-size:22px;">
                  Reclassification Score<br>
                  <span style="font-size:52px;">{score}</span>/100<br>
                  <span style="font-size:16px;">{tier} — {cls}</span>
                </div>
                """, unsafe_allow_html=True)

                # Gauge bar
                fig_g, ax_g = plt.subplots(figsize=(3.5, 0.6))
                ax_g.barh([0], [100], color='#e0e0e0', height=0.4)
                ax_g.barh([0], [score], color=colour, height=0.4)
                ax_g.set_xlim(0, 100); ax_g.axis('off')
                ax_g.text(score, 0, f' {score}', va='center', fontsize=9)
                for x, lbl in [(30, 'LB'), (50, 'VUS'), (70, 'LP/P')]:
                    ax_g.axvline(x, color='#bdbdbd', linewidth=0.8, linestyle='--')
                    ax_g.text(x, 0.25, lbl, ha='center', fontsize=7, color='#757575')
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)
                plt.close()

            with cc2:
                # Per-model probabilities
                st.subheader('Model probabilities')
                mp = result['model_probs']
                models_order = ['RandomForest', 'XGBoost', 'LogisticRegression']
                for m in models_order:
                    if m in mp:
                        pct = mp[m] * 100
                        bar_col = colour if pct >= 50 else '#78909c'
                        st.markdown(
                            f'**{m}**: {pct:.1f}%  '
                            f'<span style="background:{bar_col};width:{pct:.0f}%;'
                            f'height:8px;display:inline-block;border-radius:4px"></span>',
                            unsafe_allow_html=True
                        )
                st.markdown(f"**Ensemble (weighted avg):** {result['ensemble_prob']*100:.1f}%")

            # ── ACMG evidence ──────────────────────────────────────────────────
            st.subheader('ACMG Evidence Codes')
            acmg_map = {
                'acmg_pvs1': ('PVS1', 'path', '+8'),
                'acmg_ps3':  ('PS3',  'path', '+4'),
                'acmg_pm2':  ('PM2',  'path', '+2'),
                'acmg_pp3':  ('PP3',  'path', '+1'),
                'acmg_bp4':  ('BP4',  'benign', '-1'),
                'acmg_bp7':  ('BP7',  'benign', '-2'),
            }
            tags = []
            for key, (label, kind, wt) in acmg_map.items():
                val = feat_dict.get(key, 0)
                if val == 1:
                    css = 'tag-path' if kind == 'path' else 'tag-benign'
                    tags.append(f'<span class="acmg-tag {css}">{label} ({wt})</span>')
                else:
                    tags.append(f'<span class="acmg-tag tag-neut">{label} ✗</span>')
            total_s = feat_dict.get('acmg_total_score', 0)
            st.markdown(' '.join(tags) +
                        f'<br><b>ACMG Total Score: {total_s:.0f}</b>',
                        unsafe_allow_html=True)

            # ── SHAP explanation ───────────────────────────────────────────────
            if not shap_df.empty:
                st.subheader('Feature Contributions (SHAP — RandomForest)')
                fig_s, ax_s = plt.subplots(figsize=(8, 4))
                colors = ['#d32f2f' if v > 0 else '#1976d2'
                          for v in shap_df['shap_value']]
                ax_s.barh(shap_df['feature'][::-1],
                          shap_df['shap_value'][::-1], color=colors[::-1])
                ax_s.axvline(0, color='black', linewidth=0.8)
                ax_s.set_xlabel('SHAP value (→ pathogenic)')
                ax_s.set_title('Top feature contributions')
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close()
            else:
                st.info('SHAP explanation not available (SHAP library may not be installed).')

            # ── Key feature values ─────────────────────────────────────────────
            with st.expander('🔍 Full feature vector'):
                disp_df = pd.DataFrame(list(feat_dict.items()),
                                       columns=['Feature', 'Value'])
                disp_df['Value'] = disp_df['Value'].apply(
                    lambda v: round(v, 4) if isinstance(v, float) else v)
                st.dataframe(disp_df, use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Model Validation':
    st.title('📊 Model Validation Report')
    st.markdown(
        'All results from 5-fold stratified cross-validation on the 178-variant '
        'Drost et al. (HGG Advances 2026) dataset.'
    )

    # ── Summary metrics table ──────────────────────────────────────────────────
    st.header('Performance Summary')
    try:
        df_sum = load_summary()
        # Friendly display
        display_rows = []
        for _, row in df_sum.iterrows():
            display_rows.append({
                'Model':       row['model'],
                'AUC':         f"{row['AUC_mean']:.3f} ± {row['AUC_std']:.3f}",
                'AUPRC':       f"{row['AUPRC_mean']:.3f} ± {row['AUPRC_std']:.3f}",
                'Sensitivity': f"{row['Sensitivity_mean']:.3f} ± {row['Sensitivity_std']:.3f}",
                'Specificity': f"{row['Specificity_mean']:.3f} ± {row['Specificity_std']:.3f}",
                'PPV':         f"{row['PPV_mean']:.3f} ± {row['PPV_std']:.3f}",
                'NPV':         f"{row['NPV_mean']:.3f} ± {row['NPV_std']:.3f}",
                'F1':          f"{row['F1_mean']:.3f} ± {row['F1_std']:.3f}",
                'Accuracy':    f"{row['Accuracy_mean']:.3f} ± {row['Accuracy_std']:.3f}",
            })
        st.dataframe(pd.DataFrame(display_rows).set_index('Model'),
                     use_container_width=True)
        st.caption(
            '🏆 **Best overall AUC**: RandomForest (0.947)  ·  '
            '**Best specificity & F1**: XGBoost (Spec=0.841, F1=0.893)  ·  '
            'All three models exceed BIRAC targets (AUC ≥ 0.85, Sens > 80%, Spec > 75%).'
        )
    except Exception as e:
        st.error(f'Could not load summary metrics: {e}')

    # ── Metrics cards ──────────────────────────────────────────────────────────
    st.divider()
    st.header('Key Metrics at a Glance')
    kpi = {'AUC':0.947, 'Sensitivity':0.956, 'Specificity':0.841,
           'F1':0.893, 'PPV':0.858, 'NPV':0.950}
    cols = st.columns(len(kpi))
    for col, (k, v) in zip(cols, kpi.items()):
        col.metric(k, f'{v:.3f}')

    # ── Plots ──────────────────────────────────────────────────────────────────
    st.divider()
    st.header('Validation Plots')

    def show_plot(name, caption=''):
        path = os.path.join(PLOTS_DIR, name)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f'Plot not found: {name}')

    col1, col2 = st.columns(2)
    with col1:
        show_plot('roc_curves.png',
                  'ROC curves — 5-fold CV (solid) vs single-feature baselines (dashed)')
    with col2:
        show_plot('summary_metrics.png', 'Summary metrics — mean ± std, 5-fold CV')

    col3, col4 = st.columns(2)
    with col3:
        show_plot('confusion_matrix.png', 'Confusion matrix — RandomForest on full dataset')
    with col4:
        show_plot('violin_plot.png', 'Score distribution by true class')

    show_plot('feature_importance_rf_xgb.png', 'Feature importance — RF (mean decrease impurity) and XGBoost (gain)')
    show_plot('correlation_heatmap.png', 'Feature correlation heatmap (top 20 by importance)')

    # ── Per-fold detail ────────────────────────────────────────────────────────
    st.divider()
    st.header('Per-fold Cross-validation Detail')
    try:
        df_cv = load_cv()
        model_choice = st.selectbox('Select model',
                                     df_cv['model'].unique().tolist())
        sub = df_cv[df_cv['model'] == model_choice][
            ['fold', 'AUC', 'Sensitivity', 'Specificity', 'F1', 'Accuracy']
        ].round(3).set_index('fold')
        st.dataframe(sub, use_container_width=True)
    except Exception:
        st.info('Per-fold detail unavailable.')

    # ── Score distribution ─────────────────────────────────────────────────────
    st.divider()
    st.header('Reclassification Score Distribution')
    try:
        df_sc = load_scores()
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for label, grp in df_sc.groupby('label'):
            lname = 'LP/P (label=1)' if label == 1 else 'LB/VUS (label=0)'
            col   = '#d32f2f' if label == 1 else '#1976d2'
            ax.hist(grp['integrated_score'], bins=30, alpha=0.6,
                    label=lname, color=col, edgecolor='white')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1.2,
                   label='Decision threshold (0.5)')
        ax.set_xlabel('Reclassification score (probability)'); ax.set_ylabel('Count')
        ax.set_title('Score distribution by true class — RandomForest (full dataset)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception:
        st.info('Score distribution unavailable.')

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ About':
    st.title('ℹ️ About This Tool')

    st.header('Scientific Background')
    st.markdown("""
Variants of Uncertain Significance (VUS) in cancer predisposition genes create
clinical management dilemmas. RNA-based functional assays can resolve many of
these variants, but the final reclassification decision integrates multiple
data streams (splicing evidence, in silico tools, gene constraint, ACMG rules)
in a way that is difficult to standardise across laboratories.

This system trains a supervised classifier on **178 confirmed VUS** from
Drost et al. (HGG Advances 2026, Erasmus MC Rotterdam), where ground truth
is provided by subsequent ACMG-compliant reclassification. The model
integrates:

| Feature group | Count | Examples |
|---|---|---|
| RNA experimental | 10 | aberrant_splicing, pct_aberrant_sanger, is_frameshift_ptc |
| In silico scores | 4 | SpliceAI, Pangolin, SPiP, SQUIRLS |
| Gene constraint | 3 | LOEUF, loeuf_score |
| Tissue expression | 3 | GTEx TPM blood/fibro |
| Population AF | 1 | gnomAD allele frequency |
| Variant type (1-hot) | 14 | effect_category, vulexmap_category |
| ACMG evidence codes | 7 | PVS1, PS3, PM2, PP3, BP4, BP7, total_score |
| NMD prediction | 2 | nmd_sensitive, nmd_score |
| Specificity aids | 5 | no_rna_effect, tools_agree_pathogenic, … |
| **Total** | **56** | |
""")

    st.header('Can It Predict Any Variant?')
    st.markdown("""
**Short answer: Yes, with caveats — and better with RNA data.**

The model was designed for **post-RNA-validation** reclassification, meaning
it works best when you already have experimental splicing data. The table
below shows how input completeness affects reliability:

| Input mode | Features available | Expected accuracy |
|---|---|---|
| Full (RNA + in silico + external) | 56 / 56 | AUC ~0.95 |
| No RNA data, in silico only | ~35 / 56 (RNA imputed) | AUC ~0.82–0.87 |
| In silico scores only, no gene | ~15 / 56 | Unreliable |

Missing features are **imputed with training-set medians**, so the tool will
always produce a score — but that score carries more uncertainty when key
features are absent. The score colour-band (LP/P · VUS · LB) is calibrated
for full-input mode.
""")

    st.header('Roadmap: Turning This Into a Proper Clinical Tool')
    st.markdown("""
| Phase | Step | Impact |
|---|---|---|
| **Short-term** | External validation on independent cohort (e.g. LOVD, ClinVar) | Confirm generalisability |
| | Calibrate probabilities with Platt scaling | Calibrated LR → interpretable confidence |
| | Add SHAP global explanation page | Regulatory transparency |
| **Medium-term** | Train gene-specific models (BRCA1/2, MLH1, etc.) | Domain adaptation |
| | Integrate with local VEP v105 installation | 178/178 consequence coverage |
| | Add gnomAD AF v4 variant-level fetch | More accurate PM2 scoring |
| | SpliceAI auto-fetch via Illumina basespace or local GPU | Remove score input burden |
| **Long-term** | Prospective validation study (ACMG Level 5 evidence) | Clinical-grade certification |
| | ISO 13485 / IVD compliance framework | Regulatory pathway |
| | REDCap / LIMS API integration | Lab workflow embedding |
| | Explainability report PDF per variant | Clinical documentation |
""")

    st.header('Dataset & Citation')
    st.markdown("""
**Primary dataset**: Drost, M. et al. *Functional assessment of splice variants
in cancer susceptibility genes using high-throughput minigene assays.*
HGG Advances, 2026. [DOI pending]

**Training**: 178 VUS (class_before = 3) from 202 variants in mmc2.xlsx.
Labels: "Upgraded 3→4/5" = LP/P (n=90); "No change" = negative (n=88).

**Funding**: BIRAC-BIOAI prototype grant (proposal phase).

**Limitations**:
- n=178 is small for clinical-grade ML; results should be interpreted alongside
  expert review, not as standalone decisions.
- RNA data was obtained under specific lab conditions (minigene assay);
  patient-derived cell results may differ.
- Model trained on NF1, TSC2, TSC1 and other rare disease genes; performance
  on genes not represented in the training set is unknown.
- gnomAD LOEUF and GTEx fetch require internet access.
""")

