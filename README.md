# VUS Reclassification AI System

**BIRAC-BIOAI Grant Prototype** | Erasmus MC × [Your Institution]

An integrated ML system that reclassifies Variants of Uncertain Significance (VUS)
in cancer predisposition genes as Likely Pathogenic (LP/P) using RNA splicing evidence,
in silico scores, ACMG evidence codes, and gene-level constraint annotations.

---

## Results (5-Fold Cross-Validation, n=178 VUS)

| Model | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| **RandomForest** ★ | **0.947 ± 0.022** | 0.956 | 0.763 | 0.873 |
| XGBoost | 0.940 ± 0.017 | 0.933 | **0.841** | **0.893** |
| LogisticRegression | 0.935 ± 0.015 | 0.922 | 0.795 | 0.869 |

All three models exceed BIRAC-BIOAI targets (AUC >= 0.85, Sensitivity > 80%, Specificity > 75%).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline (first time only)
```bash
python scripts/01_parse_data.py
python scripts/02_fetch_external.py   # fetches gnomAD, GTEx, VEP
python scripts/03_feature_engineering.py
python scripts/06_add_features.py     # adds ACMG + NMD features
python scripts/04_train_model.py
python scripts/05_visualize.py
python scripts/07_generate_report.py  # generates PDF validation report
```

### 3. Launch the web interface
```bash
streamlit run app.py
```
Open http://localhost:8501

---

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub (see instructions below)
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → `app.py` as main file
4. Click **Deploy**

> The trained model files (`results/metrics/*.pkl`) must be committed to the repo
> for Streamlit Cloud to serve predictions without re-training.

---

## Project Structure

```
.
├── app.py                     # Streamlit web interface
├── predictor.py               # Core prediction class (API fetch + inference)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── scripts/
│   ├── 01_parse_data.py       # Parse Excel (Table S1 + S2)
│   ├── 02_fetch_external.py   # gnomAD / GTEx / VEP API fetch
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py      # LR + RF + XGBoost 5-fold CV
│   ├── 05_visualize.py        # 7 validation plots
│   ├── 06_add_features.py     # ACMG + NMD + specificity features
│   └── 07_generate_report.py  # PDF validation report
├── data/
│   ├── feature_matrix.csv     # 178 x 56 feature matrix
│   ├── feature_cols.txt       # Ordered feature list
│   ├── s1_vus.csv
│   └── s2_insilico.csv
└── results/
    ├── metrics/
    │   ├── all_models.pkl     # All 3 trained models (full data)
    │   ├── best_model.pkl     # RandomForest (best AUC)
    │   ├── imputation_stats.pkl
    │   ├── summary_metrics.csv
    │   └── ...
    ├── plots/
    │   ├── roc_curves.png
    │   └── ...
    └── VUS_Reclassification_Validation_Report.pdf
```

---

## Feature Engineering (56 features)

| Group | Count | Key features |
|---|---|---|
| RNA experimental | 10 | aberrant_splicing, pct_aberrant_sanger, is_frameshift_ptc |
| In silico scores | 4 | SpliceAI, Pangolin, SPiP, SQUIRLS |
| Gene constraint | 2 | LOEUF, loeuf_score |
| Tissue expression | 3 | GTEx TPM blood + fibro |
| Population AF | 1 | gnomAD allele frequency |
| VEP consequence | 1 | vep_is_ptc |
| Variant type (1-hot) | 21 | effect_category, VulExMap, variant_type |
| ACMG evidence codes | 7 | PVS1 (+8), PS3 (+4), PM2 (+2), PP3 (+1), BP4 (-1), BP7 (-2) |
| NMD prediction | 2 | nmd_sensitive, nmd_score |
| Specificity aids | 5 | tools_agree_benign/pathogenic, is_canonical_ss |

---

## Can It Predict Any Variant?

Yes, with caveats:

- **Best accuracy** when you have RNA data + in silico scores + external annotations (~AUC 0.95)
- **Reduced accuracy** without RNA data (features imputed with training medians, ~AUC 0.82-0.87)
- **Auto-fetch** via gnomAD, GTEx, VEP REST APIs when gene/variant notation provided
- **Not recommended** with only gene name and no other data

---

## Dataset

- Source: Drost et al., HGG Advances 2026 (Erasmus MC Rotterdam)
- 178 VUS (class_before = 3) from 202 published variants
- Labels: "Upgraded 3->4/5" = LP/P (n=90); "No change" = negative (n=88)
- Genes: BRCA1/2, MLH1, MSH2/6, PMS2 and others

---

## Citation

If you use this tool, please cite:

```
Drost, M. et al. Functional assessment of splice variants in cancer susceptibility genes
using high-throughput minigene assays. HGG Advances, 2026.
```

---

## Contact

jana100022@gmail.com
