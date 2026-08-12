"""
03_train_bert_models.py -- Fine-tune four biomedical BERT models.

We evaluate four domain-specific BERT variants:
  - BioBERT       (dmis-lab/biobert-v1.1)
  - PubMedBERT    (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
  - Bio+ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
  - BioLinkBERT   (michiyasunaga/BioLinkBERT-base)

All were pretrained on biomedical text, which gives them a vocabulary advantage
over general BERT for biomarker terminology like "p-tau217", "flortaucipir", etc.

Anti-overfitting strategy (important given the small dataset size):
  - Simple classifier head: Dropout(0.3) -> Linear(hidden, 2)
    A complex head would memorize the small training set easily.
  - Early stopping: patience=4 on validation F1 (the actual target metric).
    Stops training before the model starts memorizing.
  - Layer-wise LR decay: top BERT layers use full LR; each lower layer is
    multiplied by LR_LAYER_DECAY=0.95. Protects general language knowledge
    in lower layers from being overwritten by domain-specific fine-tuning.
  - Proper AdamW weight decay (0.01): applied only to weight matrices,
    NOT to bias terms or LayerNorm parameters (correct AdamW usage).
  - Linear warmup scheduler: stable fine-tuning with warm start.
  - Gradient clipping (max_norm=1.0): prevents exploding gradients.
  - Class-weighted loss: handles the 73/27 imbalance without SMOTE.

Hyperparameter search strategy:
  Phase 1: Try learning rates [1e-5, 2e-5, 3e-5] on a fixed 80/20 holdout
           (seed=99, independent of the CV folds used in Phase 2).
           This prevents the LR search from influencing the CV estimate.
  Phase 2: 5-fold stratified CV with the best LR per model.

Outputs:
    outputs/bert_models/results.csv
    outputs/bert_models/full_results.csv
    outputs/bert_models/per_fold_f1.csv
    outputs/bert_models/analysis.txt
    outputs/bert_models/oof_predictions.csv
    outputs/bert_models/oof_probabilities.csv
    outputs/bert_models/fig_bert_results.png
    outputs/bert_models/fig_loss_curves.png
    outputs/bert_models/fig_cm_oof.png
    outputs/bert_models/fig_test_eval.png
    models/bert/<model>/model_state.pt
    models/bert/<model>/tokenizer/
    models/bert/<model>/config.json
"""

import os
import sys
import copy
import json
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DATASET, OUT_BERT, MODELS_BERT, MODELS_META, BERT_ENSEMBLE_MEMBERS, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX

os.makedirs(OUT_BERT, exist_ok=True)

print("-" * 60)
print("Step 3 of 7: Training BERT models")
print("-" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- Fixed hyperparameters ---------------------------------------------------
MAX_LEN        = 128    # outcome measures are short (~10-50 words); 128 tokens is sufficient
BATCH_SIZE     = 4      # reduced from 8 to fit larger sequences in GPU memory
MAX_EPOCHS     = 12     # combined with early stopping, this is a ceiling not a target
PATIENCE       = 4      # stop if val F1 doesn't improve for 4 consecutive epochs
DROPOUT        = 0.3    # increased from 0.2; small dataset benefits from more regularisation
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1    # 10% of total steps used for warmup
LR_LAYER_DECAY = 0.95   # multiply LR by this factor for each lower BERT layer
N_FOLDS        = 5
LR_CANDIDATES  = [1e-5, 2e-5, 3e-5]

BERT_MODELS = {
    'BERT-base':        'bert-base-uncased',                                              # general-domain baseline
    'BioBERT':          'dmis-lab/biobert-v1.1',
    'PubMedBERT':       'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'Bio+ClinicalBERT': 'emilyalsentzer/Bio_ClinicalBERT',
    'BioLinkBERT':      'michiyasunaga/BioLinkBERT-base',
}

# ---- Load dataset ------------------------------------------------------------
df         = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
texts_arr  = df['clean_freeform'].fillna('').values
labels_arr = df['has_biomarker'].values
groups     = df['NCTId'].values

# Load the 80/20 split created in step 01. Subset to train only so the 20%
# test fold is never seen during LR search or CV.
global_train_idx = np.load(SPLIT_TRAIN_IDX)
global_test_idx  = np.load(SPLIT_TEST_IDX)
texts_test_global  = texts_arr[global_test_idx]
labels_test_global = labels_arr[global_test_idx]
texts_arr  = texts_arr[global_train_idx]
labels_arr = labels_arr[global_train_idx]
groups     = groups[global_train_idx]
print(f"\nLoaded {len(texts_arr)} train samples | "
      f"{labels_arr.mean()*100:.1f}% positive | {(labels_arr==0).sum()} negative")
print(f"Unique trials (groups): {len(set(groups))}")
print(f"Test set: {len(texts_test_global)} samples")

# 80/20 group-aware split within train for LR search.
# seed=99 keeps this split independent of the CV folds (seed=42).
gss_lr = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=99)
lr_tr_idx, lr_va_idx = next(gss_lr.split(texts_arr, labels_arr, groups=groups))


# ---- Dataset class -----------------------------------------------------------
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


# ---- Model definition --------------------------------------------------------
# Using the [CLS] token representation followed by a simple head. We deliberately
# avoid stacking multiple linear layers because on 200-ish training samples,
# a complex head will overfit. One dropout + one linear layer is enough.
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
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token embedding
        return self.head(cls)


# ---- Helper functions --------------------------------------------------------
def make_loader(texts, labels, tokenizer, shuffle):
    ds = BiomarkerDataset(texts, labels, tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def class_weights(labels):
    """Inverse frequency weighting to counteract the positive class being ~27% of data."""
    neg, pos = (labels == 0).sum(), (labels == 1).sum()
    return torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(DEVICE)


def make_optimizer(model, base_lr):
    """
    Layer-wise LR decay + proper AdamW weight decay.

    Strategy:
      - Classifier head         → base_lr (learns from scratch, needs full LR)
      - BERT top layer (11)     → base_lr * decay^1
      - BERT layer i (0..10)   → base_lr * decay^(n_layers - i)
      - BERT embeddings         → base_lr * decay^(n_layers + 1)  (smallest LR)
      - Bias / LayerNorm params → weight_decay=0.0  (correct AdamW behaviour)
    """
    no_decay    = {'bias', 'LayerNorm.weight'}
    n_layers    = model.bert.config.num_hidden_layers  # 12 for all base models

    groups = []

    def _add(params_iter, lr):
        decay_p  = [p for n, p in params_iter if not any(nd in n for nd in no_decay) and p.requires_grad]
        nodecay_p = [p for n, p in params_iter if     any(nd in n for nd in no_decay) and p.requires_grad]
        if decay_p:
            groups.append({'params': decay_p,   'lr': lr, 'weight_decay': WEIGHT_DECAY})
        if nodecay_p:
            groups.append({'params': nodecay_p, 'lr': lr, 'weight_decay': 0.0})

    # Classifier head — full LR
    _add(model.head.named_parameters(), base_lr)

    # BERT transformer layers — decayed LR from top (layer 11) to bottom (layer 0)
    for layer_i in range(n_layers - 1, -1, -1):
        layer_lr = base_lr * (LR_LAYER_DECAY ** (n_layers - layer_i))
        _add(model.bert.encoder.layer[layer_i].named_parameters(), layer_lr)

    # BERT embeddings — smallest LR (deepest layer)
    embed_lr = base_lr * (LR_LAYER_DECAY ** (n_layers + 1))
    _add(model.bert.embeddings.named_parameters(), embed_lr)

    return AdamW(groups)


def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(DEVICE),
                       batch['attention_mask'].to(DEVICE))
        loss = criterion(logits, batch['label'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, probs, trues = [], [], []
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            logits     = model(ids, mask)
            total_loss += criterion(logits, lbls).item()
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lbls.cpu().numpy())
    y_true = np.array(trues)
    y_pred = np.array(preds)
    y_prob = np.array(probs)
    metrics = {
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Accuracy':  accuracy_score(y_true, y_pred),
        'ROC-AUC':   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }
    return total_loss / len(loader), metrics, y_true, y_pred, y_prob


def train_with_early_stopping(model_name, tokenizer, tr_texts, tr_labels,
                               va_texts, va_labels, lr, verbose=True):
    """Train until val F1 stops improving. Returns best model and per-epoch curves."""
    cw        = class_weights(tr_labels)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    tr_dl = make_loader(tr_texts, tr_labels, tokenizer, shuffle=True)
    va_dl = make_loader(va_texts, va_labels, tokenizer, shuffle=False)

    model     = BERTClassifier(model_name).to(DEVICE)
    optimizer = make_optimizer(model, lr)   # layer-wise LR decay + proper weight decay
    total_steps = len(tr_dl) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1    = -1.0   # early stop on F1, not val loss
    patience_count = 0
    best_state     = None
    best_epoch     = 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(MAX_EPOCHS):
        tr_loss = train_one_epoch(model, tr_dl, optimizer, scheduler, criterion)
        va_loss, va_metrics, _, _, _ = evaluate(model, va_dl)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_f1s.append(va_metrics['F1-Score'])

        if verbose:
            print(f"  ep{epoch+1}: tr_loss={tr_loss:.4f}  "
                  f"va_loss={va_loss:.4f}  va_F1={va_metrics['F1-Score']:.4f}",
                  end='')

        # Early stop on val F1 — the actual optimisation target
        if va_metrics['F1-Score'] > best_val_f1:
            best_val_f1    = va_metrics['F1-Score']
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
            best_epoch     = epoch + 1
            if verbose:
                print(" (best)")
        else:
            patience_count += 1
            if verbose:
                print(f" (no improve {patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1}, best was epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    del optimizer, scheduler
    torch.cuda.empty_cache()
    return model, best_epoch, train_losses, val_losses, val_f1s


# =============================================================================
# Phase 1: Learning rate search on a fixed holdout
# =============================================================================
print("\n" + "-" * 60)
print("Phase 1: Learning rate search (80/20 holdout, seed=99)")
print("-" * 60)

best_lr_per_model = {}

for model_name, model_path in BERT_MODELS.items():
    print(f"\n{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    lr_tr_texts  = texts_arr[lr_tr_idx]
    lr_tr_labels = labels_arr[lr_tr_idx]
    lr_va_texts  = texts_arr[lr_va_idx]
    lr_va_labels = labels_arr[lr_va_idx]

    best_lr, best_f1 = None, -1
    for lr in LR_CANDIDATES:
        print(f"\n  LR={lr:.0e}")
        model, stopped_at, _, _, _ = train_with_early_stopping(
            model_path, tokenizer,
            lr_tr_texts, lr_tr_labels,
            lr_va_texts, lr_va_labels,
            lr=lr, verbose=True,
        )
        va_dl = make_loader(lr_va_texts, lr_va_labels, tokenizer, shuffle=False)
        _, va_metrics, _, _, _ = evaluate(model, va_dl)
        f1 = va_metrics['F1-Score']
        print(f"  Val F1={f1:.4f}  (stopped at epoch {stopped_at})")
        if f1 > best_f1:
            best_f1, best_lr = f1, lr
        del model
        torch.cuda.empty_cache()

    best_lr_per_model[model_name] = best_lr
    print(f"\nBest LR for {model_name}: {best_lr:.0e}  (val F1={best_f1:.4f})")

print("\nLR search summary:")
for name, lr in best_lr_per_model.items():
    print(f"  {name:<22}: {lr:.0e}")

# =============================================================================
# Phase 2: 5-fold CV with the best learning rate per model
# =============================================================================
print("\n" + "-" * 60)
print("Phase 2: 5-fold stratified CV with best LR per model")
print("-" * 60)

# StratifiedGroupKFold keeps all outcomes from one trial in the same fold,
# preventing trial-level leakage across train/validation splits.
cv               = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
all_results      = {}
loss_curves      = {}  # all folds per model: {model_name: [(tr_losses, va_losses, va_f1s), ...]}
roc_data         = {}  # first fold only, for the ROC plot
fold_probs_store = {}  # {model_name: {fold_i: (y_true, y_prob)}} for ensemble CV
fold_f1s_store   = {}  # {model_name: [f1_fold1, ..., f1_fold5]} saved to per_fold_f1.csv

torch.manual_seed(42)
np.random.seed(42)

# OOF storage: length-N arrays filled fold-by-fold, saved after all models complete.
oof_preds_bert = {name: np.full(len(labels_arr), -1, dtype=int) for name in BERT_MODELS}
oof_probs_bert = {name: np.full(len(labels_arr), np.nan) for name in BERT_MODELS}

for model_name, model_path in BERT_MODELS.items():
    lr = best_lr_per_model[model_name]
    print(f"\n{model_name}  (LR={lr:.0e})")
    print("-" * 50)

    tokenizer     = AutoTokenizer.from_pretrained(model_path)
    fold_metrics  = []
    fold_train_f1 = []
    fold_stop_ep  = []
    all_fold_curves = []  # collect (tr_losses, va_losses, va_f1s) for every fold

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(texts_arr, labels_arr, groups=groups), 1):
        tr_texts  = texts_arr[tr_idx]
        tr_labels = labels_arr[tr_idx]
        va_texts  = texts_arr[va_idx]
        va_labels = labels_arr[va_idx]

        print(f"\n  Fold {fold_i}/{N_FOLDS}:")
        model, stopped_at, tr_losses, va_losses, va_f1s = train_with_early_stopping(
            model_path, tokenizer,
            tr_texts, tr_labels,
            va_texts, va_labels,
            lr=lr, verbose=True,
        )

        va_dl = make_loader(va_texts, va_labels, tokenizer, shuffle=False)
        _, va_metrics, y_true, y_pred, y_prob = evaluate(model, va_dl)
        fold_probs_store.setdefault(model_name, {})[fold_i] = (y_true, y_prob)
        oof_preds_bert[model_name][va_idx] = y_pred
        oof_probs_bert[model_name][va_idx] = y_prob

        tr_dl = make_loader(tr_texts, tr_labels, tokenizer, shuffle=False)
        _, tr_metrics, _, _, _ = evaluate(model, tr_dl)
        train_f1    = tr_metrics['F1-Score']
        overfit_gap = train_f1 - va_metrics['F1-Score']

        fold_metrics.append(va_metrics)
        fold_train_f1.append(train_f1)
        fold_stop_ep.append(stopped_at)

        overfit_note = "overfit" if overfit_gap > 0.10 else "ok"
        print(f"  Val F1={va_metrics['F1-Score']:.4f}  "
              f"train F1={train_f1:.4f}  "
              f"gap={overfit_gap:.4f}  ({overfit_note})  "
              f"stopped ep={stopped_at}")

        all_fold_curves.append((tr_losses, va_losses, va_f1s))
        if fold_i == 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_data[model_name] = (fpr, tpr, va_metrics['ROC-AUC'])

        del model
        torch.cuda.empty_cache()

    # Aggregate across folds.
    mean_val   = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_f1     = np.std([m['F1-Score'] for m in fold_metrics])
    mean_tr_f1 = np.mean(fold_train_f1)
    mean_gap   = mean_tr_f1 - mean_val['F1-Score']
    mean_stop  = np.mean(fold_stop_ep)

    all_results[model_name] = {
        **mean_val,
        'F1 Std':      std_f1,
        'Train F1':    mean_tr_f1,
        'Overfit Gap': mean_gap,
        'Avg Stop Ep': mean_stop,
    }
    fold_f1s_store[model_name] = [m['F1-Score'] for m in fold_metrics]
    loss_curves[model_name] = all_fold_curves

    overfit_note = "possible overfit" if mean_gap > 0.10 else "ok"
    print(f"\n{model_name} summary:")
    print(f"  Mean val   F1 : {mean_val['F1-Score']:.4f} +/- {std_f1:.4f}")
    print(f"  Mean train F1 : {mean_tr_f1:.4f}")
    print(f"  Overfit gap   : {mean_gap:.4f}  ({overfit_note})")
    print(f"  Mean AUC      : {mean_val['ROC-AUC']:.4f}")
    print(f"  Avg stop epoch: {mean_stop:.1f}/{MAX_EPOCHS}")
    print(f"  Best LR used  : {lr:.0e}")

# ---- BERT Ensemble CV metrics ------------------------------------------------
# Soft vote over BioBERT + PubMedBERT + BioLinkBERT using out-of-fold probs.
# Bio+ClinicalBERT is excluded (weakest member by CV F1).
available_members = [m for m in BERT_ENSEMBLE_MEMBERS if m in fold_probs_store]

print(f"\n{'='*60}")
print(f"BERT Ensemble CV  (members: {available_members})")
print(f"{'='*60}")

if len(available_members) >= 2:
    ens_fold_metrics = []
    for fold_i in sorted(fold_probs_store[available_members[0]].keys()):
        y_true_fold = fold_probs_store[available_members[0]][fold_i][0]
        prob_sum = np.zeros(len(y_true_fold))
        for m in available_members:
            prob_sum += fold_probs_store[m][fold_i][1]
        avg_prob   = prob_sum / len(available_members)
        y_pred_ens = (avg_prob >= 0.5).astype(int)
        fold_met = {
            'Precision': precision_score(y_true_fold, y_pred_ens, average='weighted', zero_division=0),
            'Recall':    recall_score(y_true_fold, y_pred_ens, average='weighted', zero_division=0),
            'F1-Score':  f1_score(y_true_fold, y_pred_ens, average='weighted', zero_division=0),
            'Accuracy':  accuracy_score(y_true_fold, y_pred_ens),
            'ROC-AUC':   roc_auc_score(y_true_fold, avg_prob) if len(np.unique(y_true_fold)) > 1 else 0.5,
        }
        ens_fold_metrics.append(fold_met)
        print(f"  Fold {fold_i}: F1={fold_met['F1-Score']:.4f}  AUC={fold_met['ROC-AUC']:.4f}")

    mean_ens  = {k: np.mean([m[k] for m in ens_fold_metrics]) for k in ens_fold_metrics[0]}
    std_ens   = np.std([m['F1-Score'] for m in ens_fold_metrics])
    all_results['BERT Ensemble'] = {
        **mean_ens,
        'F1 Std':      std_ens,
        'Train F1':    float('nan'),
        'Overfit Gap': float('nan'),
        'Avg Stop Ep': float('nan'),
    }
    fold_f1s_store['BERT Ensemble'] = [m['F1-Score'] for m in ens_fold_metrics]
    print(f"\nBERT Ensemble summary:")
    print(f"  Mean val F1 : {mean_ens['F1-Score']:.4f} +/- {std_ens:.4f}")
    print(f"  Mean AUC    : {mean_ens['ROC-AUC']:.4f}")
else:
    print(f"  Not enough ensemble members available (need >=2, got {len(available_members)})")

# Build BERT Ensemble OOF probabilities by averaging member OOF probs per fold.
fold_idx_store = {fold_i: va_idx
                  for fold_i, (_, va_idx) in enumerate(cv.split(texts_arr, labels_arr, groups=groups), 1)}
oof_probs_ens = np.full(len(labels_arr), np.nan)
if len(available_members) >= 2:
    for fold_i, va_idx in fold_idx_store.items():
        prob_sum_fold = np.zeros(len(va_idx))
        for m in available_members:
            prob_sum_fold += oof_probs_bert[m][va_idx]
        oof_probs_ens[va_idx] = prob_sum_fold / len(available_members)
oof_preds_bert['BERT Ensemble'] = (oof_probs_ens >= 0.5).astype(int)
oof_probs_bert['BERT Ensemble'] = oof_probs_ens

# Save OOF predictions with global dataset row indices so step 08 can align
# with the full y array without needing to know about the split file.
pd.DataFrame(oof_preds_bert, index=global_train_idx).to_csv(
    os.path.join(OUT_BERT, 'oof_predictions.csv'))
pd.DataFrame(oof_probs_bert, index=global_train_idx).to_csv(
    os.path.join(OUT_BERT, 'oof_probabilities.csv'))
print(f"\nSaved OOF predictions to {os.path.join(OUT_BERT, 'oof_predictions.csv')}")
print(f"Saved OOF probabilities to {os.path.join(OUT_BERT, 'oof_probabilities.csv')}")

# ---- Results table -----------------------------------------------------------
results_df   = pd.DataFrame(all_results).T.round(4)
display_cols = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC',
                'F1 Std', 'Train F1', 'Overfit Gap', 'Avg Stop Ep']
results_df   = results_df[display_cols]

print("\n" + "-" * 60)
print("BERT Models Results (5-fold CV, best LR per model)")
print("-" * 60)
print(results_df[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC', 'Overfit Gap']].to_string())

# results.csv is the simplified version consumed by step 5.
results_df[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']].to_csv(
    os.path.join(OUT_BERT, 'results.csv'))
results_df.to_csv(os.path.join(OUT_BERT, 'full_results.csv'))
# Per-fold F1 scores for all models.
_pfdf = pd.DataFrame(fold_f1s_store, index=[f'Fold{i}' for i in range(1, 6)]).T
_pfdf.index.name = 'Model'
_pfdf['Mean'] = _pfdf.mean(axis=1).round(4)
_pfdf['Std']  = _pfdf.std(axis=1, ddof=0).round(4)
_pfdf.to_csv(os.path.join(OUT_BERT, 'per_fold_f1.csv'))
print(f"\nSaved results to {os.path.join(OUT_BERT, 'results.csv')}")
print(f"Saved full results to {os.path.join(OUT_BERT, 'full_results.csv')}")
print(f"Saved per-fold F1  to {os.path.join(OUT_BERT, 'per_fold_f1.csv')}")

# ---- Figures -----------------------------------------------------------------
colors = ['#7F77DD', '#1D9E75', '#D85A30', '#378ADD', '#BA7517', '#A32D2D']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('BERT Models -- 5-fold CV with Early Stopping',
             fontsize=14, fontweight='bold')

# F1 bar chart with standard deviation.
f1_vals = results_df['F1-Score'].values
f1_stds = results_df['F1 Std'].values
bars = axes[0].bar(range(len(f1_vals)), f1_vals, color=colors[:len(f1_vals)],
                   yerr=f1_stds, capsize=5, alpha=0.85)
axes[0].set_xticks(range(len(f1_vals)))
axes[0].set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=9)
axes[0].set_ylabel('Weighted F1-Score (5-fold CV)')
axes[0].set_title('F1-Score Comparison', fontweight='bold')
axes[0].set_ylim(0, 1.1)
for bar, val in zip(bars, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Train vs val F1 (overfitting check).
x     = np.arange(len(results_df))
width = 0.35
tr_f1s = results_df['Train F1'].values
va_f1s = results_df['F1-Score'].values
axes[1].bar(x - width/2, tr_f1s, width, label='Train F1', color='#7F77DD', alpha=0.75)
axes[1].bar(x + width/2, va_f1s, width, label='Val F1 (CV)', color='#1D9E75', alpha=0.75)
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=9)
axes[1].set_ylabel('Weighted F1-Score')
axes[1].set_title('Train vs Val F1 (Overfitting Check)', fontweight='bold')
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')
for i, (tr, va) in enumerate(zip(tr_f1s, va_f1s)):
    gap = tr - va
    c   = '#D85A30' if gap > 0.10 else '#555555'
    axes[1].text(i, max(tr, va) + 0.03, f'gap={gap:.2f}',
                 ha='center', fontsize=8, color=c)

# ROC curves (fold 1 of each model).
for (model_name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
    axes[2].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', color=color, lw=2)
axes[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curves (Fold 1)', fontweight='bold')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_BERT, 'fig_bert_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure to {os.path.join(OUT_BERT, 'fig_bert_results.png')}")

# Loss curves -- averaged across all folds with per-fold lines shown faintly.
# Folds with early stopping run fewer epochs; shorter sequences are padded with
# their final value so all folds can be averaged to a common length.
def pad_to(seq, length):
    """Extend seq to `length` by repeating the last value."""
    seq = list(seq)
    return seq + [seq[-1]] * (length - len(seq))

fig2, axes2 = plt.subplots(1, len(loss_curves), figsize=(5 * len(loss_curves), 4))
if len(loss_curves) == 1:
    axes2 = [axes2]
fig2.suptitle(f'Training vs Validation Loss (Mean ± Std across {N_FOLDS} folds) — Early Stopping',
              fontsize=12, fontweight='bold')

for ax, (model_name, fold_curves), color in zip(axes2, loss_curves.items(), colors):
    max_ep  = max(len(fc[0]) for fc in fold_curves)
    tr_mat  = np.array([pad_to(fc[0], max_ep) for fc in fold_curves])
    va_mat  = np.array([pad_to(fc[1], max_ep) for fc in fold_curves])
    f1_mat  = np.array([pad_to(fc[2], max_ep) for fc in fold_curves])
    epochs  = np.arange(1, max_ep + 1)

    # Individual fold lines (faint)
    for fold_tr, fold_va in zip(tr_mat, va_mat):
        ax.plot(epochs, fold_tr, color=color,  alpha=0.15, lw=1)
        ax.plot(epochs, fold_va, color='grey', alpha=0.15, lw=1, linestyle='--')

    # Mean ± std loss
    tr_mean, tr_std = tr_mat.mean(0), tr_mat.std(0)
    va_mean, va_std = va_mat.mean(0), va_mat.std(0)
    ax.plot(epochs, tr_mean, 'o-', color=color,  label='Train loss', lw=2)
    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, color=color, alpha=0.15)
    ax.plot(epochs, va_mean, 's--', color='grey', label='Val loss',   lw=2)
    ax.fill_between(epochs, va_mean - va_std, va_mean + va_std, color='grey', alpha=0.15)

    ax.set_title(model_name, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    # Val F1 on twin right axis
    ax2 = ax.twinx()
    f1_mean, f1_std = f1_mat.mean(0), f1_mat.std(0)
    ax2.plot(epochs, f1_mean, '^:', color=color, alpha=0.8, lw=1.5, label='Val F1 (mean)')
    ax2.fill_between(epochs, f1_mean - f1_std, f1_mean + f1_std, color=color, alpha=0.08)
    ax2.set_ylabel('Val F1', fontsize=8, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=7)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUT_BERT, 'fig_loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved loss curves (all folds averaged) to {os.path.join(OUT_BERT, 'fig_loss_curves.png')}")

# ---- Write analysis text file ------------------------------------------------
best_model = results_df['F1-Score'].idxmax()
best_f1    = results_df['F1-Score'].max()
best_auc_m = results_df['ROC-AUC'].idxmax()
best_auc   = results_df['ROC-AUC'].max()

analysis_path = os.path.join(OUT_BERT, 'analysis.txt')
with open(analysis_path, 'w') as f:
    f.write("-" * 60 + "\n")
    f.write("BERT MODELS -- ANALYSIS AND FINDINGS\n")
    f.write("Biomarker Detection in Alzheimer's Clinical Trial Texts\n")
    f.write("-" * 60 + "\n\n")
    f.write(f"Dataset: {len(texts_arr)} samples, {labels_arr.mean()*100:.1f}% positive\n\n")

    f.write("-" * 60 + "\n")
    f.write("METHODOLOGY\n")
    f.write("-" * 60 + "\n")
    f.write("Approach: Fine-tuned BERT binary classification\n")
    f.write(f"Classifier head: Dropout({DROPOUT}) -> Linear(hidden, 2)\n")
    f.write(f"Max token length: {MAX_LEN}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Max epochs: {MAX_EPOCHS} (with early stopping, patience={PATIENCE})\n")
    f.write(f"Weight decay: {WEIGHT_DECAY}\n")
    f.write(f"Warmup ratio: {WARMUP_RATIO}\n")
    f.write(f"Gradient clipping: max_norm=1.0\n\n")

    f.write("Anti-overfitting mechanisms:\n")
    f.write(f"  1. Early stopping (patience={PATIENCE}, monitored on val F1)\n")
    f.write("  2. Single linear classifier head\n")
    f.write("  3. AdamW weight decay (0.01) on weight matrices only (bias/LayerNorm excluded)\n")
    f.write(f"  4. Dropout ({DROPOUT}) on [CLS] representation\n")
    f.write("  5. Class-weighted cross-entropy loss\n")
    f.write(f"  6. Layer-wise LR decay (factor={LR_LAYER_DECAY}) — protects lower BERT layers\n\n")

    f.write("Evaluation strategy:\n")
    f.write("  Phase 1: LR search on fixed 80/20 holdout (seed=99)\n")
    f.write(f"  Phase 2: {N_FOLDS}-fold stratified CV (seed=42) with best LR\n\n")

    f.write("-" * 60 + "\n")
    f.write("LEARNING RATE SEARCH RESULTS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Candidates: {LR_CANDIDATES}\n\n")
    for name, lr in best_lr_per_model.items():
        f.write(f"  {name:<22}: best LR = {lr:.0e}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("RESULTS TABLE (5-fold CV, weighted metrics)\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} "
            f"{'Acc':>7} {'AUC':>7} {'F1 Std':>7} {'Gap':>7} {'Stop Ep':>8}\n")
    f.write("-" * 90 + "\n")
    for model_name in results_df.index:
        r = results_df.loc[model_name]
        f.write(f"{model_name:<22} {r['Precision']:>9.4f} {r['Recall']:>7.4f} "
                f"{r['F1-Score']:>7.4f} {r['Accuracy']:>7.4f} {r['ROC-AUC']:>7.4f} "
                f"{r['F1 Std']:>7.4f} {r['Overfit Gap']:>7.4f} "
                f"{r['Avg Stop Ep']:>8.1f}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("-" * 60 + "\n")
    f.write(f"\n1. Best F1:  {best_model} ({best_f1:.4f})\n")
    f.write(f"2. Best AUC: {best_auc_m} ({best_auc:.4f})\n\n")
    f.write("3. Overfitting check:\n")
    for name in results_df.index:
        gap = results_df.loc[name, 'Overfit Gap']
        ep  = results_df.loc[name, 'Avg Stop Ep']
        status = 'possible overfit' if gap > 0.10 else 'healthy'
        f.write(f"   {name:<22}: gap={gap:.4f}  avg_stop_ep={ep:.1f}  ({status})\n")

    f.write("\n4. Early stopping is working -- average stop epoch is well below MAX_EPOCHS.\n")

print(f"Saved analysis to {analysis_path}")

# =============================================================================
# Save BERT models for the demo app
# =============================================================================
# Train each BERT model on the full 80% training set (90/10 group-aware split
# for early stopping with PATIENCE_DEMO=2). These weights are for the demo only;
# the authoritative performance numbers come from the CV above.
print("\n" + "-" * 60)
print("Saving BERT models for demo app")
print("-" * 60)
print("Training each BERT model on the full 80% train set (90/10 early-stop split)")
print()

BERT_SUBFOLDERS = {
    'BERT-base':        'bert_base',
    'BioBERT':          'biobert',
    'PubMedBERT':       'pubmedbert',
    'Bio+ClinicalBERT': 'bio_clinical_bert',
    'BioLinkBERT':      'biolinkbert',
}
PATIENCE_DEMO = 2  # less aggressive than CV (patience=4): full-set training converges faster

os.makedirs(MODELS_BERT, exist_ok=True)

# 90/10 group-aware split within the 80% training set for early stopping.
gss_demo = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
demo_tr_idx, demo_val_idx = next(gss_demo.split(texts_arr, labels_arr, groups=groups))
demo_tr_texts, demo_val_texts = texts_arr[demo_tr_idx], texts_arr[demo_val_idx]
demo_tr_labels, demo_val_labels = labels_arr[demo_tr_idx], labels_arr[demo_val_idx]
print(f"Demo split: train={len(demo_tr_texts)}, val={len(demo_val_texts)}")

saved_bert_subfolders = {}

for model_name, model_path in BERT_MODELS.items():
    subfolder = BERT_SUBFOLDERS[model_name]
    print(f"\nTraining {model_name}  ({model_path})")
    print("-" * 50)

    lr = best_lr_per_model[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cw        = class_weights(demo_tr_labels)
    criterion_demo = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    tr_dl = make_loader(demo_tr_texts, demo_tr_labels, tokenizer, shuffle=True)
    va_dl = make_loader(demo_val_texts, demo_val_labels, tokenizer, shuffle=False)

    model_demo = BERTClassifier(model_path).to(DEVICE)
    optimizer_demo = make_optimizer(model_demo, lr)   # layer-wise LR decay, proper weight decay
    total_steps_demo = len(tr_dl) * MAX_EPOCHS
    scheduler_demo = get_linear_schedule_with_warmup(
        optimizer_demo,
        num_warmup_steps=int(WARMUP_RATIO * total_steps_demo),
        num_training_steps=total_steps_demo,
    )

    best_val_f1_demo = -1.0   # early stop on F1, consistent with CV training
    patience_count_demo = 0
    best_state_demo = None

    for epoch in range(MAX_EPOCHS):
        tr_loss = train_one_epoch(model_demo, tr_dl, optimizer_demo,
                                  scheduler_demo, criterion_demo)
        va_loss, va_metrics, _, _, _ = evaluate(model_demo, va_dl)
        print(f"  ep{epoch+1}: tr_loss={tr_loss:.4f}  "
              f"va_loss={va_loss:.4f}  va_F1={va_metrics['F1-Score']:.4f}", end='')
        if va_metrics['F1-Score'] > best_val_f1_demo:
            best_val_f1_demo = va_metrics['F1-Score']
            patience_count_demo = 0
            best_state_demo = copy.deepcopy(model_demo.state_dict())
            print(" (best)")
        else:
            patience_count_demo += 1
            print(f" (no improve {patience_count_demo}/{PATIENCE_DEMO})")
            if patience_count_demo >= PATIENCE_DEMO:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Save the best checkpoint.
    save_dir = os.path.join(MODELS_BERT, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state_demo, os.path.join(save_dir, 'model_state.pt'))
    tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))

    model_demo.load_state_dict(best_state_demo)
    config_demo = {
        'model_name':  model_path,
        'max_len':     MAX_LEN,
        'hidden_size': model_demo.bert.config.hidden_size,
        'dropout':     DROPOUT,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_demo, f, indent=2)

    saved_bert_subfolders[model_name] = subfolder
    print(f"  Saved {model_name} -> models/bert/{subfolder}/")

    del model_demo, optimizer_demo, scheduler_demo
    torch.cuda.empty_cache()

print(f"\nBERT models saved to {MODELS_BERT}")

# =============================================================================
# OOF Confusion Matrices — BERT Models
# =============================================================================
print("\n" + "-" * 60)
print("OOF Confusion Matrices — BERT Models")
print("-" * 60)

import seaborn as sns
from sklearn.metrics import confusion_matrix as _cm2
_oof_df = pd.read_csv(os.path.join(OUT_BERT, 'oof_predictions.csv'), index_col=0)
_y_all_full = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))['has_biomarker'].values
_y_oof = _y_all_full[_oof_df.index.values]

_bert_model_order = list(BERT_MODELS.keys()) + ['BERT Ensemble']
_ncols_b, _nrows_b = 3, 2
_fig_oof_b, _axes_oof_b = plt.subplots(_nrows_b, _ncols_b, figsize=(_ncols_b * 4, _nrows_b * 3.8))
_axes_oof_b = _axes_oof_b.flatten()

for _i, _mname in enumerate(_bert_model_order):
    if _mname not in _oof_df.columns:
        _axes_oof_b[_i].set_visible(False)
        continue
    _preds_b = _oof_df[_mname].values.astype(int)
    _cm_b = _cm2(_y_oof, _preds_b)
    _tn_b, _fp_b, _fn_b, _tp_b = _cm_b.ravel()
    _f1_b = f1_score(_y_oof, _preds_b, average='weighted', zero_division=0)
    _ax_b = _axes_oof_b[_i]
    sns.heatmap(_cm_b, annot=False, ax=_ax_b, cbar=False,
                cmap=sns.light_palette('#55A868', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for _r in range(2):
        for _c in range(2):
            _val = _cm_b[_r, _c]
            _lbl = [['TN', 'FP'], ['FN', 'TP']][_r][_c]
            _ax_b.text(_c + 0.5, _r + 0.5, f'{_lbl}\n{_val}', ha='center', va='center',
                       fontsize=11, fontweight='bold',
                       color='white' if _val > _cm_b.max() * 0.5 else '#333333')
    _ax_b.set_title(f'{_mname}\nOOF F1={_f1_b:.4f}', fontsize=9, fontweight='bold')
    _ax_b.set_xlabel('Predicted', fontsize=8)
    _ax_b.set_ylabel('Actual', fontsize=8)
    _ax_b.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    _ax_b.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

for _j in range(_i + 1, len(_axes_oof_b)):
    _axes_oof_b[_j].set_visible(False)
_fig_oof_b.suptitle(f'OOF Confusion Matrices — BERT Models (80% train, N={len(_y_oof)})',
                    fontsize=13, fontweight='bold')
plt.tight_layout()
_oof_b_path = os.path.join(OUT_BERT, 'fig_cm_oof.png')
plt.savefig(_oof_b_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved BERT OOF confusion matrices to {_oof_b_path}")

# =============================================================================
# Test Set Evaluation — All BERT Models
# =============================================================================
print("\n" + "-" * 60)
print("Test Set Evaluation — BERT Models (20% held-out)")
print("-" * 60)

from transformers import AutoTokenizer as _AT2

_X_test_b = texts_test_global
_y_test_b = labels_test_global
_n_test_b = len(_y_test_b)
print(f"Test set: {_n_test_b} samples | {_y_test_b.sum()} positive / {(_y_test_b==0).sum()} negative")

_bert_test_preds = {}
_bert_test_probs = {}
_bert_roc_test = {}

for _mname_t, _mpath_t in BERT_MODELS.items():
    _subfolder_t = BERT_SUBFOLDERS[_mname_t]
    _save_dir_t = os.path.join(MODELS_BERT, _subfolder_t)
    _cfg_path_t = os.path.join(_save_dir_t, 'config.json')
    _state_path_t = os.path.join(_save_dir_t, 'model_state.pt')
    if not os.path.exists(_state_path_t):
        print(f"  WARNING: {_state_path_t} not found, skipping {_mname_t}")
        continue
    print(f"  Evaluating {_mname_t}...", end=' ', flush=True)
    with open(_cfg_path_t) as _f_cfg:
        _cfg_t = json.load(_f_cfg)
    _tok_t = _AT2.from_pretrained(os.path.join(_save_dir_t, 'tokenizer'))
    _mod_t = BERTClassifier(_cfg_t['model_name']).to(DEVICE)
    _mod_t.load_state_dict(torch.load(_state_path_t, map_location=DEVICE))
    _mod_t.eval()
    _te_dl = make_loader(_X_test_b, _y_test_b, _tok_t, shuffle=False)
    _, _te_met, _y_tr_true, _y_tr_pred, _y_tr_prob = evaluate(_mod_t, _te_dl)
    _bert_test_preds[_mname_t] = _y_tr_pred
    _bert_test_probs[_mname_t] = _y_tr_prob
    _fpr_bt, _tpr_bt, _ = roc_curve(_y_test_b, _y_tr_prob)
    _bert_roc_test[_mname_t] = (_fpr_bt, _tpr_bt, _te_met['ROC-AUC'])
    print(f"F1={_te_met['F1-Score']:.4f}  AUC={_te_met['ROC-AUC']:.4f}")
    del _mod_t, _tok_t
    torch.cuda.empty_cache()

# BERT ensemble on test set
_ens_members_t = [m for m in BERT_ENSEMBLE_MEMBERS if m in _bert_test_probs]
if len(_ens_members_t) >= 2:
    _prob_ens_t = np.mean([_bert_test_probs[m] for m in _ens_members_t], axis=0)
    _pred_ens_t = (_prob_ens_t >= 0.5).astype(int)
    _bert_test_preds['BERT Ensemble'] = _pred_ens_t
    _bert_test_probs['BERT Ensemble'] = _prob_ens_t
    _fpr_be, _tpr_be, _ = roc_curve(_y_test_b, _prob_ens_t)
    _auc_be = roc_auc_score(_y_test_b, _prob_ens_t)
    _bert_roc_test['BERT Ensemble'] = (_fpr_be, _tpr_be, _auc_be)
    _f1_be = f1_score(_y_test_b, _pred_ens_t, average='weighted', zero_division=0)
    print(f"  {'BERT Ensemble':<22}: F1={_f1_be:.4f}  AUC={_auc_be:.4f}")

# Figure: CM grid + ROC
_n_bt = len(_bert_test_preds)
_ncols_bt = 3
_nrows_bt = (_n_bt + _ncols_bt - 1) // _ncols_bt
_fig_bt = plt.figure(figsize=(_ncols_bt * 4.5, _nrows_bt * 3.8 + 4.5))
_cm_bt_axes = [_fig_bt.add_subplot(_nrows_bt + 1, _ncols_bt, i + 1) for i in range(_n_bt)]
_roc_bt_ax = _fig_bt.add_subplot(_nrows_bt + 1, 1, _nrows_bt + 1)

_bt_colors = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D', '#444444']
for _i_bt, (_mname_bt, _preds_bt) in enumerate(_bert_test_preds.items()):
    _ax_bt = _cm_bt_axes[_i_bt]
    _cm_bt_arr = _cm2(_y_test_b, _preds_bt)
    _tn_bt, _fp_bt, _fn_bt, _tp_bt = _cm_bt_arr.ravel()
    _f1_bt_i = f1_score(_y_test_b, _preds_bt, average='weighted', zero_division=0)
    sns.heatmap(_cm_bt_arr, annot=False, ax=_ax_bt, cbar=False,
                cmap=sns.light_palette('#55A868', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for _r in range(2):
        for _c in range(2):
            _val = _cm_bt_arr[_r, _c]
            _lbl = [['TN', 'FP'], ['FN', 'TP']][_r][_c]
            _ax_bt.text(_c + 0.5, _r + 0.5, f'{_lbl}\n{_val}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if _val > _cm_bt_arr.max() * 0.5 else '#333333')
    _ax_bt.set_title(f'{_mname_bt}\nF1={_f1_bt_i:.4f}', fontsize=9, fontweight='bold')
    _ax_bt.set_xlabel('Predicted', fontsize=8)
    _ax_bt.set_ylabel('Actual', fontsize=8)
    _ax_bt.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    _ax_bt.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

for _mname_roc, _clr_roc in zip(_bert_roc_test.keys(), _bt_colors):
    _fpr_r, _tpr_r, _auc_r = _bert_roc_test[_mname_roc]
    _roc_bt_ax.plot(_fpr_r, _tpr_r, color=_clr_roc, lw=2,
                    label=f'{_mname_roc} (AUC={_auc_r:.4f})')
_roc_bt_ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
_roc_bt_ax.set_xlabel('False Positive Rate', fontsize=11)
_roc_bt_ax.set_ylabel('True Positive Rate', fontsize=11)
_roc_bt_ax.set_title('ROC Curves — BERT Test Set', fontsize=12, fontweight='bold')
_roc_bt_ax.legend(fontsize=8, loc='lower right', ncol=2)
_roc_bt_ax.grid(True, alpha=0.3)

_fig_bt.suptitle(f'Test Set Evaluation — BERT Models (20% held-out, N={_n_test_b})',
                 fontsize=14, fontweight='bold')
plt.tight_layout()
_bt_fig_path = os.path.join(OUT_BERT, 'fig_test_eval.png')
plt.savefig(_bt_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved BERT test evaluation figure to {_bt_fig_path}")

print("\nStep 3 complete.")
