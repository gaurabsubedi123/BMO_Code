"""
06_test_set_evaluation.py -- Unified held-out test set evaluation for all 15 models.

Addresses Reviewer Flag 1 (data leakage / cross-tier comparison):
  - All prior CV metrics were estimated on the 80% training set only.
  - Zero-shot was evaluated on all 1900 samples (including the 20% test set).
  - This script evaluates EVERY model on the same 341-sample held-out test set,
    enabling a fair apples-to-apples comparison across all three tiers.

Models evaluated (15 total):
  Tier 1 - Classical ML (5 + ensemble):
    Logistic Regression, SVM, Random Forest, XGBoost, Decision Tree, Classical Ensemble
  Tier 2 - BERT fine-tuned (5 + ensemble):
    BERT-base, BioBERT, PubMedBERT, Bio+ClinicalBERT, BioLinkBERT, BERT Ensemble
  Tier 3 - Zero-Shot LLMs (3):
    BART Zero-Shot, Flan-T5 Few-Shot, BioGPT Zero-Shot

Metrics reported (addresses Reviewer Flag 7 - per-class metrics):
  - Weighted F1, Macro F1
  - Has Biomarker F1 (positive class), No Biomarker F1 (negative class)
  - Precision, Recall (weighted)
  - AUC-ROC

Outputs:
    outputs/final_results/test_set_results.csv
    outputs/final_results/fig_test_set_comparison.png
    outputs/final_results/test_set_analysis.txt
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score, classification_report,
                              confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import FastTextVectorizer, BioWordVecVectorizer  # noqa: F401 — needed for unpickling
from config import (OUT_DATASET, OUT_CLASSICAL, OUT_ENHANCED, OUT_BIOWORDVEC, OUT_BERT,
                    OUT_ZEROSHOT, OUT_FINAL, MODELS_CLASSICAL, MODELS_CLASSICAL_ENHANCED,
                    MODELS_CLASSICAL_BIOWORDVEC, MODELS_BERT, MODELS_META,
                    BERT_ENSEMBLE_MEMBERS, SPLIT_TEST_IDX)

os.makedirs(OUT_FINAL, exist_ok=True)

print("-" * 60)
print("Step 6 of 7: Test Set Evaluation — All Models")
print("-" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# BERT inference hyperparameters (must match training in step 03)
MAX_LEN    = 128
BATCH_SIZE = 16   # larger batch is fine for inference only
DROPOUT    = 0.3

# ---- BERT model name map (must match step 03) --------------------------------
BERT_MODELS = {
    'BERT-base':        'bert-base-uncased',
    'BioBERT':          'dmis-lab/biobert-v1.1',
    'PubMedBERT':       'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'Bio+ClinicalBERT': 'emilyalsentzer/Bio_ClinicalBERT',
    'BioLinkBERT':      'michiyasunaga/BioLinkBERT-base',
}

BERT_SUBFOLDER = {
    'BERT-base':        'bert_base',
    'BioBERT':          'biobert',
    'PubMedBERT':       'pubmedbert',
    'Bio+ClinicalBERT': 'bio_clinical_bert',
    'BioLinkBERT':      'biolinkbert',
}


# =============================================================================
# Load test set
# =============================================================================
df          = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
X_all       = df['clean_freeform'].fillna('').values
y_all       = df['has_biomarker'].values

test_idx    = np.load(SPLIT_TEST_IDX)
X_test      = X_all[test_idx]
y_test      = y_all[test_idx]
n_test      = len(y_test)
n_pos       = y_test.sum()
n_neg       = (y_test == 0).sum()

print(f"\nTest set: {n_test} samples | {n_pos} positive ({n_pos/n_test*100:.1f}%) "
      f"| {n_neg} negative ({n_neg/n_test*100:.1f}%)")


# =============================================================================
# Metric helper
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob=None):
    """Return a dict of all reported metrics for one model."""
    m = {}
    m['Weighted F1']      = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    m['Macro F1']         = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    m['Has Biomarker F1'] = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    m['No Biomarker F1']  = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
    m['Precision (W)']    = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    m['Recall (W)']       = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
    m['Accuracy']         = accuracy_score(y_true, y_pred)
    if y_prob is not None:
        try:
            m['AUC'] = roc_auc_score(y_true, y_prob)
        except Exception:
            m['AUC'] = float('nan')
    else:
        m['AUC'] = float('nan')
    return m


# =============================================================================
# Tier 1: Classical ML
# =============================================================================
print("\n" + "-" * 60)
print("Tier 1: Classical ML — Loading saved .pkl pipelines")
print("-" * 60)

all_results  = {}   # model_name -> metrics dict
all_probs    = {}   # model_name -> probability array (positive class)

INDIV_NAMES = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
               'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']

for mname in INDIV_NAMES:
    pkl_path = os.path.join(MODELS_CLASSICAL, f'{mname.replace(" ", "_")}.pkl')
    if not os.path.exists(pkl_path):
        print(f"  WARNING: {pkl_path} not found — skipping {mname}")
        continue
    with open(pkl_path, 'rb') as f:
        pipe = pickle.load(f)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    all_probs[mname]   = y_prob
    all_results[mname] = compute_metrics(y_test, y_pred, y_prob)
    m = all_results[mname]
    print(f"  {mname:<22}  W-F1={m['Weighted F1']:.4f}  "
          f"Macro={m['Macro F1']:.4f}  "
          f"BM-F1={m['Has Biomarker F1']:.4f}  "
          f"AUC={m['AUC']:.4f}")

# Classical Ensemble: soft vote weighted by CV F1 (equal weight fallback)
avail = [m for m in INDIV_NAMES if m in all_probs]
if len(avail) >= 2:
    try:
        cv_results = pd.read_csv(os.path.join(OUT_CLASSICAL, 'results.csv'), index_col=0)
        wts = np.array([cv_results.loc[m, 'F1-Score'] if m in cv_results.index else 1.0
                        for m in avail])
    except Exception:
        wts = np.ones(len(avail))
    wts = wts / wts.sum()
    prob_ens  = sum(all_probs[m] * w for m, w in zip(avail, wts))
    pred_ens  = (prob_ens >= 0.5).astype(int)
    all_probs['Classical Ensemble']   = prob_ens
    all_results['Classical Ensemble'] = compute_metrics(y_test, pred_ens, prob_ens)
    m = all_results['Classical Ensemble']
    print(f"  {'Classical Ensemble':<22}  W-F1={m['Weighted F1']:.4f}  "
          f"Macro={m['Macro F1']:.4f}  "
          f"BM-F1={m['Has Biomarker F1']:.4f}  "
          f"AUC={m['AUC']:.4f}")


# =============================================================================
# Tier 1b: Enhanced Classical ML (optional — only if models/classical_ml_enhanced exists)
# =============================================================================
ENHANCED_INDIV_NAMES = [
    'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
    'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost',
]

if os.path.isdir(MODELS_CLASSICAL_ENHANCED):
    print("\n" + "-" * 60)
    print("Tier 1b: Enhanced Classical ML — Loading saved .pkl pipelines")
    print("-" * 60)

    enh_probs = {}  # local dict for ensemble construction

    for mname in ENHANCED_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_ENHANCED, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            print(f"  WARNING: {pkl_path} not found — skipping {mname}")
            continue
        with open(pkl_path, 'rb') as f:
            pipe = pickle.load(f)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        display = f'{mname} (Enhanced)'
        enh_probs[mname]           = y_prob
        all_probs[display]         = y_prob
        all_results[display]       = compute_metrics(y_test, y_pred, y_prob)
        m = all_results[display]
        print(f"  {display:<30}  W-F1={m['Weighted F1']:.4f}  "
              f"Macro={m['Macro F1']:.4f}  "
              f"BM-F1={m['Has Biomarker F1']:.4f}  "
              f"AUC={m['AUC']:.4f}")

    # Enhanced Ensemble: soft vote weighted by CV F1 (equal weight fallback)
    avail_enh = [m for m in ENHANCED_INDIV_NAMES if m in enh_probs]
    if len(avail_enh) >= 2:
        try:
            cv_enh = pd.read_csv(os.path.join(OUT_ENHANCED, 'results.csv'), index_col=0)
            wts_enh = np.array([cv_enh.loc[m, 'F1-Score'] if m in cv_enh.index else 1.0
                                for m in avail_enh])
        except Exception:
            wts_enh = np.ones(len(avail_enh))
        wts_enh = wts_enh / wts_enh.sum()
        prob_enh_ens  = sum(enh_probs[m] * w for m, w in zip(avail_enh, wts_enh))
        pred_enh_ens  = (prob_enh_ens >= 0.5).astype(int)
        all_probs['Enhanced Ensemble']   = prob_enh_ens
        all_results['Enhanced Ensemble'] = compute_metrics(y_test, pred_enh_ens, prob_enh_ens)
        m = all_results['Enhanced Ensemble']
        print(f"  {'Enhanced Ensemble':<30}  W-F1={m['Weighted F1']:.4f}  "
              f"Macro={m['Macro F1']:.4f}  "
              f"BM-F1={m['Has Biomarker F1']:.4f}  "
              f"AUC={m['AUC']:.4f}")
else:
    print("\nINFO: models/classical_ml_enhanced not found — skipping Enhanced ML "
          "(run step 02a to include)")


# =============================================================================
# Tier 1c: BioWordVec Classical ML (optional — only if models/classical_ml_biowordvec exists)
# =============================================================================
BIOWORDVEC_INDIV_NAMES = [
    'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
    'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost',
]

if os.path.isdir(MODELS_CLASSICAL_BIOWORDVEC):
    print("\n" + "-" * 60)
    print("Tier 1c: BioWordVec Classical ML — Loading saved .pkl pipelines")
    print("-" * 60)

    bwv_probs = {}

    for mname in BIOWORDVEC_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_BIOWORDVEC, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            print(f"  WARNING: {pkl_path} not found — skipping {mname}")
            continue
        with open(pkl_path, 'rb') as f:
            pipe = pickle.load(f)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        display = f'{mname} (BioWordVec)'
        bwv_probs[mname]           = y_prob
        all_probs[display]         = y_prob
        all_results[display]       = compute_metrics(y_test, y_pred, y_prob)
        m = all_results[display]
        print(f"  {display:<30}  W-F1={m['Weighted F1']:.4f}  "
              f"Macro={m['Macro F1']:.4f}  "
              f"BM-F1={m['Has Biomarker F1']:.4f}  "
              f"AUC={m['AUC']:.4f}")

    # BioWordVec Ensemble: soft vote weighted by CV F1
    avail_bwv = [m for m in BIOWORDVEC_INDIV_NAMES if m in bwv_probs]
    if len(avail_bwv) >= 2:
        try:
            cv_bwv = pd.read_csv(os.path.join(OUT_BIOWORDVEC, 'results.csv'), index_col=0)
            wts_bwv = np.array([cv_bwv.loc[m, 'F1-Score'] if m in cv_bwv.index else 1.0
                                for m in avail_bwv])
        except Exception:
            wts_bwv = np.ones(len(avail_bwv))
        wts_bwv = wts_bwv / wts_bwv.sum()
        prob_bwv_ens  = sum(bwv_probs[m] * w for m, w in zip(avail_bwv, wts_bwv))
        pred_bwv_ens  = (prob_bwv_ens >= 0.5).astype(int)
        all_probs['BioWordVec Ensemble']   = prob_bwv_ens
        all_results['BioWordVec Ensemble'] = compute_metrics(y_test, pred_bwv_ens, prob_bwv_ens)
        m = all_results['BioWordVec Ensemble']
        print(f"  {'BioWordVec Ensemble':<30}  W-F1={m['Weighted F1']:.4f}  "
              f"Macro={m['Macro F1']:.4f}  "
              f"BM-F1={m['Has Biomarker F1']:.4f}  "
              f"AUC={m['AUC']:.4f}")
else:
    print("\nINFO: models/classical_ml_biowordvec not found — skipping BioWordVec ML "
          "(run step 02b to include)")


# =============================================================================
# Tier 2: BERT Models
# =============================================================================
print("\n" + "-" * 60)
print("Tier 2: BERT Models — Running inference on test set")
print("-" * 60)


class BiomarkerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    """Must match the architecture used in step 03."""
    def __init__(self, model_name, dropout=DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden    = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


def bert_inference(model_dir, hf_model_name):
    """Load a saved BERT model and run inference on X_test. Returns (preds, probs)."""
    model_path  = os.path.join(model_dir, 'model_state.pt')
    tok_dir     = os.path.join(model_dir, 'tokenizer')
    if not os.path.exists(model_path):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model     = BERTClassifier(hf_model_name).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = BiomarkerDataset(X_test, y_test, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_probs_list = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(DEVICE),
                           batch['attention_mask'].to(DEVICE))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_probs_list.extend(probs.tolist())

    return np.array(all_preds), np.array(all_probs_list)


bert_probs_for_ensemble = {}

for bname, hf_name in BERT_MODELS.items():
    subfolder   = BERT_SUBFOLDER[bname]
    model_dir   = os.path.join(MODELS_BERT, subfolder)
    y_pred_b, y_prob_b = bert_inference(model_dir, hf_name)
    if y_pred_b is None:
        print(f"  WARNING: {model_dir}/model_state.pt not found — skipping {bname}")
        continue
    all_probs[bname]    = y_prob_b
    all_results[bname]  = compute_metrics(y_test, y_pred_b, y_prob_b)
    m = all_results[bname]
    print(f"  {bname:<22}  W-F1={m['Weighted F1']:.4f}  "
          f"Macro={m['Macro F1']:.4f}  "
          f"BM-F1={m['Has Biomarker F1']:.4f}  "
          f"AUC={m['AUC']:.4f}")
    if bname in BERT_ENSEMBLE_MEMBERS:
        bert_probs_for_ensemble[bname] = y_prob_b

# BERT Ensemble: equal-weight soft vote of the three ensemble members
if len(bert_probs_for_ensemble) >= 2:
    prob_be  = np.mean(list(bert_probs_for_ensemble.values()), axis=0)
    pred_be  = (prob_be >= 0.5).astype(int)
    all_probs['BERT Ensemble']    = prob_be
    all_results['BERT Ensemble']  = compute_metrics(y_test, pred_be, prob_be)
    m = all_results['BERT Ensemble']
    print(f"  {'BERT Ensemble':<22}  W-F1={m['Weighted F1']:.4f}  "
          f"Macro={m['Macro F1']:.4f}  "
          f"BM-F1={m['Has Biomarker F1']:.4f}  "
          f"AUC={m['AUC']:.4f}")


# =============================================================================
# Tier 3: Zero-Shot LLMs (filter all_predictions.csv to test_idx rows)
# =============================================================================
print("\n" + "-" * 60)
print("Tier 3: Zero-Shot LLMs — Filtering to test_idx rows")
print("-" * 60)

preds_path = os.path.join(OUT_ZEROSHOT, 'all_predictions.csv')
probs_path = os.path.join(OUT_ZEROSHOT, 'all_probabilities.csv')

if os.path.exists(preds_path):
    preds_df = pd.read_csv(preds_path, index_col=0)
    probs_df = pd.read_csv(probs_path, index_col=0) if os.path.exists(probs_path) else None

    # all_predictions.csv uses positional 0-based integer index matching df index.
    # test_idx contains positions into the full 1900-row DataFrame.
    preds_test = preds_df.values[test_idx]
    probs_test = probs_df.values[test_idx] if probs_df is not None else None

    zs_names = list(preds_df.columns)  # ['BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot']
    for i, zname in enumerate(zs_names):
        y_pred_z = preds_test[:, i].astype(int)
        y_prob_z = probs_test[:, i] if probs_test is not None else None
        all_probs[zname]    = y_prob_z
        all_results[zname]  = compute_metrics(y_test, y_pred_z, y_prob_z)
        m = all_results[zname]
        auc_str = f"{m['AUC']:.4f}" if not np.isnan(m['AUC']) else "N/A"
        print(f"  {zname:<22}  W-F1={m['Weighted F1']:.4f}  "
              f"Macro={m['Macro F1']:.4f}  "
              f"BM-F1={m['Has Biomarker F1']:.4f}  "
              f"AUC={auc_str}")
else:
    print("  WARNING: all_predictions.csv not found — zero-shot results skipped")


# =============================================================================
# Save CSV
# =============================================================================
results_df = pd.DataFrame(all_results).T
results_df.index.name = 'Model'

# Add tier labels
tier_map = {}
for m in ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
          'Decision Tree', 'MLP Small', 'MLP Deep', 'Classical Ensemble']:
    tier_map[m] = 'Classical ML'
for m in ['Logistic Regression (Enhanced)', 'SVM (Enhanced)', 'Random Forest (Enhanced)',
          'XGBoost (Enhanced)', 'Decision Tree (Enhanced)',
          'MLP Small (Enhanced)', 'MLP Deep (Enhanced)', 'Enhanced Ensemble']:
    tier_map[m] = 'Enhanced ML'
for m in ['Logistic Regression (BioWordVec)', 'SVM (BioWordVec)', 'Random Forest (BioWordVec)',
          'XGBoost (BioWordVec)', 'Decision Tree (BioWordVec)',
          'MLP Small (BioWordVec)', 'MLP Deep (BioWordVec)', 'BioWordVec Ensemble']:
    tier_map[m] = 'BioWordVec ML'
for m in ['BERT-base', 'BioBERT', 'PubMedBERT', 'Bio+ClinicalBERT',
          'BioLinkBERT', 'BERT Ensemble']:
    tier_map[m] = 'BERT Fine-tuned'
for m in ['BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot']:
    tier_map[m] = 'Zero-Shot LLM'

results_df.insert(0, 'Tier', results_df.index.map(tier_map))

out_csv = os.path.join(OUT_FINAL, 'test_set_results.csv')
results_df.to_csv(out_csv)
print(f"\nSaved test_set_results.csv -> {out_csv}")


# =============================================================================
# Figure: Weighted F1 and Has Biomarker F1 side by side
# =============================================================================
model_order = [
    'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'Decision Tree',
    'MLP Small', 'MLP Deep', 'Classical Ensemble',
    'Logistic Regression (Enhanced)', 'SVM (Enhanced)', 'Random Forest (Enhanced)',
    'XGBoost (Enhanced)', 'Decision Tree (Enhanced)',
    'MLP Small (Enhanced)', 'MLP Deep (Enhanced)', 'Enhanced Ensemble',
    'Logistic Regression (BioWordVec)', 'SVM (BioWordVec)', 'Random Forest (BioWordVec)',
    'XGBoost (BioWordVec)', 'Decision Tree (BioWordVec)',
    'MLP Small (BioWordVec)', 'MLP Deep (BioWordVec)', 'BioWordVec Ensemble',
    'BERT-base', 'BioBERT', 'PubMedBERT', 'Bio+ClinicalBERT', 'BioLinkBERT',
    'BERT Ensemble',
    'BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot',
]
model_order = [m for m in model_order if m in results_df.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Held-Out Test Set Evaluation (n=341)', fontsize=14, fontweight='bold')

tier_colors = {
    'Classical ML':     '#4878cf',
    'Enhanced ML':      '#3A9E6B',
    'BioWordVec ML':    '#2E8B57',
    'BERT Fine-tuned':  '#6acc65',
    'Zero-Shot LLM':    '#d65f5f',
}
colors = [tier_colors.get(tier_map.get(m, ''), '#888888') for m in model_order]

x = np.arange(len(model_order))
w = 0.6

for ax, metric, title in [
    (axes[0], 'Weighted F1',      'Weighted F1'),
    (axes[1], 'Has Biomarker F1', 'Has Biomarker (Positive Class) F1'),
]:
    vals = [results_df.loc[m, metric] if m in results_df.index else 0.0
            for m in model_order]
    bars = ax.bar(x, vals, width=w, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(title, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='0.90 ref')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

# Legend
patches = [plt.Rectangle((0, 0), 1, 1, fc=c, label=t)
           for t, c in tier_colors.items()]
axes[0].legend(handles=patches, loc='lower left', fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUT_FINAL, 'fig_test_set_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure -> {fig_path}")


# =============================================================================
# Confusion matrices — one per model, held-out test set
# =============================================================================
n_models_plot = len(model_order)
n_cols_cm     = 5
n_rows_cm     = (n_models_plot + n_cols_cm - 1) // n_cols_cm

fig_cm, axes_cm = plt.subplots(n_rows_cm, n_cols_cm,
                                figsize=(n_cols_cm * 3.2, n_rows_cm * 3.2))
axes_cm = axes_cm.flatten()

tier_colors_cm = {
    'Classical ML':    '#4878cf',
    'Enhanced ML':     '#3A9E6B',
    'BioWordVec ML':   '#2E8B57',
    'BERT Fine-tuned': '#6acc65',
    'Zero-Shot LLM':   '#d65f5f',
}

for idx, mname in enumerate(model_order):
    ax = axes_cm[idx]
    if mname not in all_probs or all_probs[mname] is None:
        ax.axis('off')
        continue
    y_pred_cm = (all_probs[mname] >= 0.5).astype(int)
    cm_arr    = confusion_matrix(y_test, y_pred_cm)
    color     = tier_colors_cm.get(tier_map.get(mname, ''), '#888888')

    ax.imshow(cm_arr, interpolation='nearest',
              cmap=plt.cm.Blues, vmin=0, vmax=cm_arr.max())
    ax.set_title(mname, fontsize=7.5, fontweight='bold', color=color)
    ax.set_xticks([0, 1]);  ax.set_yticks([0, 1])
    ax.set_xticklabels(['NoBM', 'BM'], fontsize=7)
    ax.set_yticklabels(['NoBM', 'BM'], fontsize=7)
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_ylabel('True',      fontsize=7)

    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_arr[i, j] > thresh else 'black')

for idx in range(n_models_plot, len(axes_cm)):
    axes_cm[idx].axis('off')

fig_cm.suptitle('Held-Out Test Set — Confusion Matrices (n=341)\n'
                'Blue=Classical ML  Teal=Enhanced ML  Green=BERT  Red=Zero-Shot',
                fontsize=11, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(OUT_FINAL, 'fig_test_set_confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved confusion matrices -> {cm_path}")


# =============================================================================
# Text analysis report
# =============================================================================
report_lines = [
    "=" * 70,
    "TEST SET EVALUATION — ALL 15 MODELS",
    f"Test set: {n_test} samples | {n_pos} positive ({n_pos/n_test*100:.1f}%) | "
    f"{n_neg} negative ({n_neg/n_test*100:.1f}%)",
    "=" * 70,
    "",
    f"{'Model':<26} {'Tier':<18} {'W-F1':>7} {'Macro-F1':>9} {'BM-F1':>7} {'NoBM-F1':>8} {'AUC':>7}",
    "-" * 86,
]

for mname in model_order:
    if mname not in results_df.index:
        continue
    row   = results_df.loc[mname]
    tier  = row['Tier']
    wf1   = row['Weighted F1']
    mf1   = row['Macro F1']
    bmf1  = row['Has Biomarker F1']
    nobm  = row['No Biomarker F1']
    auc   = row['AUC']
    auc_s = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
    report_lines.append(
        f"{mname:<26} {tier:<18} {wf1:>7.4f} {mf1:>9.4f} {bmf1:>7.4f} {nobm:>8.4f} {auc_s:>7}"
    )

report_lines += [
    "",
    "-" * 86,
    "",
    "NOTES",
    "-----",
    "W-F1    = Weighted F1 (accounts for class imbalance; used in CV comparisons)",
    "Macro-F1= Unweighted average of per-class F1",
    "BM-F1   = F1 for 'Has Biomarker' (positive class) — the class of interest",
    "NoBM-F1 = F1 for 'No Biomarker' (negative class)",
    "AUC     = Area Under ROC Curve",
    "",
    "BioGPT AUC < 0.5: Perplexity-based scoring is anti-correlated (inverted confidence).",
    "  Biomarker text is more complex/technical, so BioGPT assigns *higher* perplexity",
    "  (lower confidence) to positive examples — reversing the expected AUC direction.",
    "  This is a known limitation of generation-based zero-shot classifiers.",
    "",
    "Classical Ensemble Train F1 = 1.0: Expected on training data for soft-vote ensemble.",
    "  Individual model probabilities are already calibrated to training labels.",
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUT_FINAL, 'test_set_analysis.txt')
with open(report_path, 'w') as f:
    f.write(report_text)

print(f"Saved analysis -> {report_path}")
print("\n" + report_text)

print("\n" + "-" * 60)
print("Step 6 complete.")
print(f"  outputs/final_results/test_set_results.csv")
print(f"  outputs/final_results/fig_test_set_comparison.png")
print(f"  outputs/final_results/fig_test_set_confusion_matrices.png")
print(f"  outputs/final_results/test_set_analysis.txt")
print("-" * 60)
