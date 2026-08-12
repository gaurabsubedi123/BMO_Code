"""
08_full_predictions.py -- Generate per-row predictions for all 1900 samples across all models.

For each sample, records the true label and each model's prediction, then classifies
each prediction as TP / TN / FP / FN.

Strategy (leakage-free):
  - Classical ML & BERT (train set, 1559 rows): use saved OOF predictions (each sample
    was predicted by a model that never trained on it).
  - Classical ML & BERT (test set, 341 rows): reload saved .pkl / model_state.pt and
    run inference (models were trained on train set only, never saw test set).
  - Zero-Shot LLMs (all 1900 rows): use all_predictions.csv directly (no training
    dependency, so no leakage possible).

Outputs:
    outputs/final_results/full_predictions_labeled.csv
        Columns: RowId, NCTId, Field, Outcome Measure, clean_freeform,
                 has_biomarker, Annotation, split, <Model>_pred x15, <Model>_result x15
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
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import FastTextVectorizer, BioWordVecVectorizer  # noqa: F401 — needed for unpickling
from config import (OUT_DATASET, OUT_CLASSICAL, OUT_ENHANCED, OUT_BIOWORDVEC, OUT_BERT,
                    OUT_ZEROSHOT, OUT_FINAL, MODELS_CLASSICAL, MODELS_CLASSICAL_ENHANCED,
                    MODELS_CLASSICAL_BIOWORDVEC, MODELS_BERT,
                    BERT_ENSEMBLE_MEMBERS, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX)

os.makedirs(OUT_FINAL, exist_ok=True)

print("-" * 60)
print("Step 8: Full per-row predictions — all 1900 samples × all models")
print("-" * 60)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN    = 128
BATCH_SIZE = 16
DROPOUT    = 0.3

print(f"Device: {DEVICE}")

# =============================================================================
# Load full dataset and split indices
# =============================================================================
df        = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
train_idx = np.load(SPLIT_TRAIN_IDX)
test_idx  = np.load(SPLIT_TEST_IDX)

X_all = df['clean_freeform'].fillna('').values
y_all = df['has_biomarker'].values

# Build a split label column
split_col = np.array([''] * len(df), dtype=object)
split_col[train_idx] = 'train'
split_col[test_idx]  = 'test'

print(f"\nDataset: {len(df)} rows  |  train={len(train_idx)}  test={len(test_idx)}")

# Master predictions dict: model_name -> array of length 1900 (int, -1 = not yet filled)
all_preds = {}

# =============================================================================
# BERT helpers
# =============================================================================
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


def bert_predict(model_name, model_path, texts, labels):
    """Run inference and return integer predictions."""
    tok_path  = os.path.join(model_path, 'tokenizer')
    tok_src   = tok_path if os.path.isdir(tok_path) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    model     = BERTClassifier(model_name, dropout=DROPOUT).to(DEVICE)
    state     = torch.load(os.path.join(model_path, 'model_state.pt'),
                           map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    dataset    = BiomarkerDataset(texts, labels, tokenizer)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            preds_list.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return np.array(preds_list, dtype=int)


# =============================================================================
# Tier 1: Classical ML
# =============================================================================
print("\n--- Tier 1: Classical ML ---")

INDIV_NAMES = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
               'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']

# Load OOF predictions for train set
oof_classical = pd.read_csv(
    os.path.join(OUT_CLASSICAL, 'oof_predictions.csv'), index_col=0
).reset_index(drop=True)

# Load saved models for test set inference
classical_test_preds = {}
for mname in INDIV_NAMES:
    pkl_path = os.path.join(MODELS_CLASSICAL, f'{mname.replace(" ", "_")}.pkl')
    if not os.path.exists(pkl_path):
        print(f"  WARNING: {pkl_path} not found — skipping {mname}")
        continue
    with open(pkl_path, 'rb') as f:
        pipe = pickle.load(f)
    classical_test_preds[mname] = pipe.predict(X_all[test_idx]).astype(int)
    print(f"  {mname} — test predictions loaded from saved model")

# Combine train (OOF) + test for each classical model
for mname in INDIV_NAMES:
    if mname not in classical_test_preds:
        continue
    arr = np.full(len(df), -1, dtype=int)
    if mname in oof_classical.columns:
        arr[train_idx] = oof_classical[mname].values.astype(int)
    arr[test_idx] = classical_test_preds[mname]
    all_preds[mname] = arr
    print(f"  {mname} — combined (train OOF + test inference) ✓")

# Classical Ensemble: soft vote on test, majority on train OOF
oof_classical_probs_path = os.path.join(OUT_CLASSICAL, 'oof_probabilities.csv')
ens_arr = np.full(len(df), -1, dtype=int)

# Train: average OOF probabilities if available
if os.path.exists(oof_classical_probs_path):
    oof_probs_cl = pd.read_csv(oof_classical_probs_path, index_col=0).reset_index(drop=True)
    avail_cols   = [m for m in INDIV_NAMES if m in oof_probs_cl.columns]
    if avail_cols:
        avg_train = oof_probs_cl[avail_cols].mean(axis=1).values
        ens_arr[train_idx] = (avg_train >= 0.5).astype(int)
else:
    # Fallback: majority vote from OOF predictions
    avail_cols = [m for m in INDIV_NAMES if m in oof_classical.columns]
    if avail_cols:
        votes = oof_classical[avail_cols].values
        ens_arr[train_idx] = (votes.mean(axis=1) >= 0.5).astype(int)

# Test: reload models and soft vote
test_probs_cl = []
for mname in INDIV_NAMES:
    pkl_path = os.path.join(MODELS_CLASSICAL, f'{mname.replace(" ", "_")}.pkl')
    if not os.path.exists(pkl_path):
        continue
    with open(pkl_path, 'rb') as f:
        pipe = pickle.load(f)
    test_probs_cl.append(pipe.predict_proba(X_all[test_idx])[:, 1])

if test_probs_cl:
    avg_test = np.mean(test_probs_cl, axis=0)
    ens_arr[test_idx] = (avg_test >= 0.5).astype(int)

all_preds['Classical Ensemble'] = ens_arr
print("  Classical Ensemble — combined ✓")


# =============================================================================
# Tier 1b: Enhanced Classical ML (optional — only if models/classical_ml_enhanced exists)
# =============================================================================
ENHANCED_INDIV_NAMES = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
                        'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']

if os.path.isdir(MODELS_CLASSICAL_ENHANCED):
    print("\n--- Tier 1b: Enhanced Classical ML ---")

    _enh_oof_path = os.path.join(OUT_ENHANCED, 'oof_predictions.csv')
    if os.path.exists(_enh_oof_path):
        oof_enhanced = pd.read_csv(_enh_oof_path).reset_index(drop=True)
    else:
        oof_enhanced = pd.DataFrame()

    enh_test_preds = {}
    for mname in ENHANCED_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_ENHANCED, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as _f:
            pipe = pickle.load(_f)
        enh_test_preds[mname] = pipe.predict(X_all[test_idx]).astype(int)
        print(f"  {mname} (Enhanced) — test predictions loaded")

    for mname in ENHANCED_INDIV_NAMES:
        if mname not in enh_test_preds:
            continue
        display = f'{mname} (Enhanced)'
        arr = np.full(len(df), -1, dtype=int)
        if not oof_enhanced.empty and mname in oof_enhanced.columns:
            arr[train_idx] = oof_enhanced[mname].values.astype(int)
        arr[test_idx] = enh_test_preds[mname]
        all_preds[display] = arr
        print(f"  {display} — combined (train OOF + test inference) ✓")

    # Enhanced Ensemble: soft vote on test, average OOF probs on train
    _enh_prob_path = os.path.join(OUT_ENHANCED, 'oof_probabilities.csv')
    enh_ens_arr = np.full(len(df), -1, dtype=int)
    if os.path.exists(_enh_prob_path):
        _eof_probs_enh = pd.read_csv(_enh_prob_path).reset_index(drop=True)
        _avail_enh = [m for m in ENHANCED_INDIV_NAMES if m in _eof_probs_enh.columns]
        if _avail_enh:
            avg_tr = _eof_probs_enh[_avail_enh].mean(axis=1).values
            enh_ens_arr[train_idx] = (avg_tr >= 0.5).astype(int)
    enh_test_probs = []
    for mname in ENHANCED_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_ENHANCED, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as _f:
            pipe = pickle.load(_f)
        enh_test_probs.append(pipe.predict_proba(X_all[test_idx])[:, 1])
    if enh_test_probs:
        enh_ens_arr[test_idx] = (np.mean(enh_test_probs, axis=0) >= 0.5).astype(int)
    all_preds['Enhanced Ensemble'] = enh_ens_arr
    print("  Enhanced Ensemble — combined ✓")
else:
    print("\nINFO: models/classical_ml_enhanced not found — skipping Enhanced ML "
          "(run step 02a to include)")


# =============================================================================
# Tier 1c: BioWordVec Classical ML (optional — only if models/classical_ml_biowordvec exists)
# =============================================================================
BIOWORDVEC_INDIV_NAMES = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
                          'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']

if os.path.isdir(MODELS_CLASSICAL_BIOWORDVEC):
    print("\n--- Tier 1c: BioWordVec Classical ML ---")

    _bwv_oof_path = os.path.join(OUT_BIOWORDVEC, 'oof_predictions.csv')
    if os.path.exists(_bwv_oof_path):
        oof_biowordvec = pd.read_csv(_bwv_oof_path).reset_index(drop=True)
    else:
        oof_biowordvec = pd.DataFrame()

    bwv_test_preds = {}
    for mname in BIOWORDVEC_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_BIOWORDVEC, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as _f:
            pipe = pickle.load(_f)
        bwv_test_preds[mname] = pipe.predict(X_all[test_idx])

        display = f'{mname} (BioWordVec)'
        arr = np.full(len(df), -1, dtype=int)
        if not oof_biowordvec.empty and mname in oof_biowordvec.columns:
            arr[train_idx] = oof_biowordvec[mname].values.astype(int)
        arr[test_idx] = bwv_test_preds[mname]
        all_preds[display] = arr
        print(f"  {display} — combined (train OOF + test inference) ✓")

    # BioWordVec Ensemble: soft vote on test, average OOF probs on train
    _bwv_prob_path = os.path.join(OUT_BIOWORDVEC, 'oof_probabilities.csv')
    bwv_ens_arr = np.full(len(df), -1, dtype=int)
    if os.path.exists(_bwv_prob_path):
        _oof_probs_bwv = pd.read_csv(_bwv_prob_path).reset_index(drop=True)
        avail_cols = [m for m in BIOWORDVEC_INDIV_NAMES if m in _oof_probs_bwv.columns]
        if avail_cols:
            avg_tr = _oof_probs_bwv[avail_cols].mean(axis=1).values
            bwv_ens_arr[train_idx] = (avg_tr >= 0.5).astype(int)
    bwv_test_probs = []
    for mname in BIOWORDVEC_INDIV_NAMES:
        pkl_path = os.path.join(MODELS_CLASSICAL_BIOWORDVEC, f'{mname.replace(" ", "_")}.pkl')
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as _f:
            pipe = pickle.load(_f)
        bwv_test_probs.append(pipe.predict_proba(X_all[test_idx])[:, 1])
    if bwv_test_probs:
        bwv_ens_arr[test_idx] = (np.mean(bwv_test_probs, axis=0) >= 0.5).astype(int)
    all_preds['BioWordVec Ensemble'] = bwv_ens_arr
    print("  BioWordVec Ensemble — combined ✓")
else:
    print("\nINFO: models/classical_ml_biowordvec not found — skipping BioWordVec ML "
          "(run step 02b to include)")


# =============================================================================
# Tier 2: BERT Models
# =============================================================================
print("\n--- Tier 2: BERT Models ---")

oof_bert = pd.read_csv(
    os.path.join(OUT_BERT, 'oof_predictions.csv'), index_col=0
).reset_index(drop=True)

bert_test_preds  = {}
bert_test_probs  = {}

for display_name, hf_name in BERT_MODELS.items():
    subfolder  = BERT_SUBFOLDER[display_name]
    model_path = os.path.join(MODELS_BERT, subfolder)
    state_path = os.path.join(model_path, 'model_state.pt')

    if not os.path.exists(state_path):
        print(f"  WARNING: {state_path} not found — skipping {display_name}")
        continue

    print(f"  {display_name} — running test set inference...")
    preds = bert_predict(hf_name, model_path, X_all[test_idx],
                         y_all[test_idx])
    bert_test_preds[display_name] = preds

    arr = np.full(len(df), -1, dtype=int)
    col = display_name
    if col in oof_bert.columns:
        arr[train_idx] = oof_bert[col].values.astype(int)
    arr[test_idx] = preds
    all_preds[display_name] = arr
    print(f"  {display_name} — combined (train OOF + test inference) ✓")

# BERT Ensemble: majority vote across ensemble members
ens_bert_arr = np.full(len(df), -1, dtype=int)

# Train: OOF majority vote
avail_members = [m for m in BERT_ENSEMBLE_MEMBERS if m in oof_bert.columns]
if avail_members:
    votes = oof_bert[avail_members].values.astype(int)
    ens_bert_arr[train_idx] = (votes.mean(axis=1) >= 0.5).astype(int)

# Test: majority vote from test preds
avail_test = [m for m in BERT_ENSEMBLE_MEMBERS if m in bert_test_preds]
if avail_test:
    stacked = np.stack([bert_test_preds[m] for m in avail_test], axis=1)
    ens_bert_arr[test_idx] = (stacked.mean(axis=1) >= 0.5).astype(int)

all_preds['BERT Ensemble'] = ens_bert_arr
print("  BERT Ensemble — combined ✓")


# =============================================================================
# Tier 3: Zero-Shot LLMs (all 1900 rows, no leakage concern)
# =============================================================================
print("\n--- Tier 3: Zero-Shot LLMs ---")

zeroshot_all = pd.read_csv(
    os.path.join(OUT_ZEROSHOT, 'all_predictions.csv'), index_col=0
)

ZEROSHOT_NAMES = ['BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot']

for zname in ZEROSHOT_NAMES:
    if zname not in zeroshot_all.columns:
        print(f"  WARNING: {zname} not found in all_predictions.csv — skipping")
        continue
    all_preds[zname] = zeroshot_all[zname].values.astype(int)
    print(f"  {zname} — loaded (all 1900 rows) ✓")


# =============================================================================
# Build output DataFrame
# =============================================================================
print("\nBuilding output DataFrame...")

keep_cols = ['RowId', 'NCTId', 'Field', 'Outcome Measure', 'clean_freeform', 'has_biomarker', 'Annotation']
out_df = df[[c for c in keep_cols if c in df.columns]].copy()
out_df['split'] = split_col

def classify_result(y_true, y_pred):
    if y_pred == -1:
        return 'N/A'
    if y_true == 1 and y_pred == 1:
        return 'TP'
    elif y_true == 0 and y_pred == 0:
        return 'TN'
    elif y_true == 0 and y_pred == 1:
        return 'FP'
    else:
        return 'FN'

model_order = (
    ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'Decision Tree',
     'MLP Small', 'MLP Deep', 'CatBoost', 'Classical Ensemble'] +
    ['Logistic Regression (Enhanced)', 'SVM (Enhanced)', 'Random Forest (Enhanced)',
     'XGBoost (Enhanced)', 'Decision Tree (Enhanced)',
     'MLP Small (Enhanced)', 'MLP Deep (Enhanced)', 'CatBoost (Enhanced)', 'Enhanced Ensemble'] +
    ['Logistic Regression (BioWordVec)', 'SVM (BioWordVec)', 'Random Forest (BioWordVec)',
     'XGBoost (BioWordVec)', 'Decision Tree (BioWordVec)',
     'MLP Small (BioWordVec)', 'MLP Deep (BioWordVec)', 'CatBoost (BioWordVec)', 'BioWordVec Ensemble'] +
    ['BERT-base', 'BioBERT', 'PubMedBERT', 'Bio+ClinicalBERT', 'BioLinkBERT',
     'BERT Ensemble'] +
    ['BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot']
)

y_true_arr = df['has_biomarker'].values

for mname in model_order:
    if mname not in all_preds:
        continue
    preds = all_preds[mname]
    out_df[f'{mname}_pred'] = preds
    out_df[f'{mname}_result'] = [
        classify_result(y_true_arr[i], preds[i]) for i in range(len(df))
    ]

# =============================================================================
# Save
# =============================================================================
out_path = os.path.join(OUT_FINAL, 'full_predictions_labeled.xlsx')
out_df.to_excel(out_path, index=False)

# ---- Color-code result cells -------------------------------------------------
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

RESULT_COLORS = {
    'TP':  PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid'),  # green
    'TN':  PatternFill(start_color='BDD7EE', end_color='BDD7EE', fill_type='solid'),  # blue
    'FP':  PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid'),  # yellow
    'FN':  PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid'),  # red
}

wb = load_workbook(out_path)
ws = wb.active

# Build mapping: result_col_index -> pred_col_index (1-based)
header = [cell.value for cell in ws[1]]
result_to_pred = {}
for i, h in enumerate(header):
    if h and h.endswith('_result'):
        pred_col = h.replace('_result', '_pred')
        if pred_col in header:
            result_to_pred[i + 1] = header.index(pred_col) + 1

for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for result_idx, pred_idx in result_to_pred.items():
        result_cell = row[result_idx - 1]
        pred_cell   = row[pred_idx - 1]
        fill = RESULT_COLORS.get(result_cell.value)
        if fill:
            result_cell.fill = fill
            pred_cell.fill   = fill

wb.save(out_path)
print(f"\nSaved to {out_path}")
print(f"Shape: {out_df.shape[0]} rows × {out_df.shape[1]} columns")
print("Color coding: TP=green  TN=blue  FP=yellow  FN=red")

# =============================================================================
# Summary: TP/TN/FP/FN counts per model
# =============================================================================
print("\n" + "=" * 70)
print(f"{'Model':<25} {'Split':<6} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}  {'Accuracy':>8}")
print("=" * 70)

for mname in model_order:
    col = f'{mname}_result'
    if col not in out_df.columns:
        continue
    for split in ['train', 'test', 'all']:
        if split == 'all':
            sub = out_df
        else:
            sub = out_df[out_df['split'] == split]
        results = sub[col]
        tp = (results == 'TP').sum()
        tn = (results == 'TN').sum()
        fp = (results == 'FP').sum()
        fn = (results == 'FN').sum()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else float('nan')
        print(f"  {mname:<25} {split:<6} {tp:>5} {tn:>5} {fp:>5} {fn:>5}  {acc:>8.4f}")
    print()

print("\nDone. Open full_predictions_labeled.csv to inspect individual rows.")
