"""
02b_classical_ml_biowordvec.py -- Enhanced classical ML with TF-IDF + BioWordVec features,
neural network models, and SHAP explainability.

Mirrors 02a_classical_ml_enhanced.py but substitutes BioWordVec for FastText.
BioWordVec (200-dim) is trained on PubMed abstracts + MIMIC-III clinical notes,
making it far more domain-appropriate than general-domain FastText for
biomarker detection in clinical trial eligibility criteria.

Reference: Zhang Y et al. "BioWordVec, improving biomedical word embeddings with
subword information and MeSH ontology" (Sci Data 2019).
Model download: https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin

  1. FEATURES: TF-IDF (sparse lexical keywords) + pretrained BioWordVec
     (dense biomedical semantic vectors) are concatenated into one wide
     feature matrix. The classifier sees both and learns weights for all columns.

  2. MODELS: Same 8 models as step 02 (LR, SVM, RF, XGBoost, DT, MLP Small, MLP Deep,
     CatBoost), trained on the richer feature matrix instead of TF-IDF alone:
       - MLP Small  : single hidden layer (tuned), ReLU/tanh, Adam, early stopping
       - MLP Deep   : multi-layer (tuned), ReLU/tanh, Adam, early stopping
       - CatBoost   : gradient boosting with auto_class_weights='Balanced'

  3. SHAP: After CV, each model is re-fit on the full training set with its
     consensus best parameters. SHAP values are computed on the held-out test
     set and visualised as:
       - Per-model beeswarm summary plot (top 20 TF-IDF features)
       - Per-model bar chart (top 20 features, TF-IDF vs BioWordVec colour-coded)
       - Cross-model heatmap of the top 30 TF-IDF features by SHAP importance

CV structure: identical to step 02 — 5-fold StratifiedGroupKFold outer,
3-fold StratifiedGroupKFold inner GridSearchCV. Groups = NCTId.

Outputs (mirrors 02a outputs under a different directory):
    outputs/classical_ml_biowordvec/results.csv
    outputs/classical_ml_biowordvec/full_results.csv
    outputs/classical_ml_biowordvec/best_params.csv
    outputs/classical_ml_biowordvec/per_fold_f1.csv
    outputs/classical_ml_biowordvec/analysis.txt
    outputs/classical_ml_biowordvec/oof_predictions.csv
    outputs/classical_ml_biowordvec/oof_probabilities.csv
    outputs/classical_ml_biowordvec/fig_results.png
    outputs/classical_ml_biowordvec/fig_fold_stability.png
    outputs/classical_ml_biowordvec/fig_cm_oof.png
    outputs/classical_ml_biowordvec/fig_test_eval.png
    outputs/classical_ml_biowordvec/shap/shap_beeswarm_<model>.png
    outputs/classical_ml_biowordvec/shap/shap_bar_<model>.png
    outputs/classical_ml_biowordvec/shap/shap_heatmap_cross_model.png
    models/classical_ml_biowordvec/*.pkl
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import issparse, csr_matrix

warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline, FeatureUnion
from matplotlib.patches import Patch
import shap
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

try:
    import torch as _torch
    _USE_GPU = _torch.cuda.is_available()
except ImportError:
    _USE_GPU = False
_XGB_DEVICE   = 'cuda' if _USE_GPU else 'cpu'
_CB_TASK_TYPE = 'GPU'  if _USE_GPU else 'CPU'
print(f"Classical ML device: {'GPU (' + __import__('torch').cuda.get_device_name(0) + ')' if _USE_GPU else 'CPU'}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DATASET, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX, BASE_DIR

OUT_DIR    = os.path.join(BASE_DIR, 'outputs', 'classical_ml_biowordvec')
SHAP_DIR   = os.path.join(OUT_DIR, 'shap')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'classical_ml_biowordvec')
for d in [OUT_DIR, SHAP_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("-" * 60)
print("Step 2b: Enhanced Classical ML (TF-IDF + BioWordVec + SHAP)")
print("-" * 60)

# =============================================================================
# 1.  Load pretrained BioWordVec model
# =============================================================================
# BioWordVec_PubMed_MIMICIII_d200.bin is a 200-dim FastText-format model trained
# on PubMed abstracts (title + abstract) and MIMIC-III clinical notes.
# It handles out-of-vocabulary tokens via character n-gram subwords, which is
# especially valuable for clinical abbreviations and compound biomarker terms.
#
# Download: https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
# Install:  pip install fasttext-wheel   (or pip install fasttext)
#
# Set BWV_MODEL_PATH to the location of your downloaded .bin file.

BWV_MODEL_PATH = os.environ.get(
    'BIOWORDVEC_PATH',
    os.path.join(os.path.expanduser('~'), 'BioWordVec_PubMed_MIMICIII_d200.bin'),
)

print(f"\nLoading BioWordVec from: {BWV_MODEL_PATH}")
try:
    import fasttext as _fasttext
    _bwv    = _fasttext.load_model(BWV_MODEL_PATH)
    BWV_DIM = _bwv.get_dimension()   # 200
    # Publish into shared embeddings module so the transformer's transform()
    # can access the model, and so pickled pipelines unpickle correctly in
    # other contexts (demo/app.py) that import embeddings.
    import embeddings as _emb
    _emb.BWV_MODEL = _bwv
    print(f"BioWordVec loaded — {BWV_DIM}-dim vectors (PubMed + MIMIC-III)")
except Exception as exc:
    print(f"WARNING: BioWordVec unavailable ({exc}).")
    print(f"  Set BIOWORDVEC_PATH or download to {BWV_MODEL_PATH}")
    print(f"  Install: pip install fasttext-wheel")
    _bwv    = None
    BWV_DIM = 0

# Import the shared transformer class (defined in embeddings.py so pickled
# pipelines are stored as embeddings.BioWordVecVectorizer and can be
# unpickled in any context that has src/ on sys.path).
from embeddings import BioWordVecVectorizer  # noqa: E402


# =============================================================================
# 2.  Feature union builder
# =============================================================================


def build_feature_union(tfidf_override_params=None):
    """Return a FeatureUnion of TF-IDF + BioWordVec.

    tfidf_override_params: dict of TfidfVectorizer kwargs to apply on top of
    the defaults (used when re-fitting with consensus best params after CV).
    """
    tfidf_kwargs = dict(sublinear_tf=True, min_df=2)
    if tfidf_override_params:
        tfidf_kwargs.update(tfidf_override_params)

    transformers = [('tfidf', TfidfVectorizer(**tfidf_kwargs))]
    if _bwv is not None:
        transformers.append(('biowordvec', BioWordVecVectorizer(dim=BWV_DIM)))

    return FeatureUnion(transformers)


# =============================================================================
# 3.  Dataset loading
# =============================================================================

dataset_path = os.path.join(OUT_DATASET, 'labeled_dataset.csv')
df           = pd.read_csv(dataset_path)
X            = df['clean_freeform'].fillna('').values
y            = df['has_biomarker'].values
groups       = df['NCTId'].values

global_train_idx = np.load(SPLIT_TRAIN_IDX)
global_test_idx  = np.load(SPLIT_TEST_IDX)

X_cv           = X[global_train_idx]
y_cv           = y[global_train_idx]
groups_cv      = groups[global_train_idx]
X_test_global  = X[global_test_idx]
y_test_global  = y[global_test_idx]

print(f"\nLoaded {len(X):,} samples | {y.mean()*100:.1f}% positive")
print(f"CV set : {len(X_cv):,} samples ({len(set(groups_cv))} trials)")
print(f"Test set: {len(X_test_global):,} samples")

# =============================================================================
# 4.  CV setup and model definitions
# =============================================================================

OUTER_CV  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV  = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
scale_pos = (y_cv == 0).sum() / (y_cv == 1).sum()


def make_models(spw):
    """Instantiate all models. spw = XGBoost scale_pos_weight."""
    return {
        'LR': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs'),
        'SVM': SVC(
            probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            scale_pos_weight=spw, random_state=42,
            eval_metric='logloss', verbosity=0, use_label_encoder=False,
            device=_XGB_DEVICE),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=42),
        # --- Neural networks ------------------------------------------------
        # MLP Small: fast, good regularisation via alpha; early stopping avoids
        # overfitting without needing a fixed epoch count.
        'MLP Small': MLPClassifier(
            hidden_layer_sizes=(256,), activation='relu', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
        # MLP Deep: three layers give the model capacity to learn non-linear
        # combinations of TF-IDF keywords and BioWordVec semantic axes.
        'MLP Deep': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
        # CatBoost: gradient boosting with native sparse support and built-in
        # class balancing via auto_class_weights='Balanced'.
        'CatBoost': CatBoostClassifier(
            auto_class_weights='Balanced', random_seed=42, verbose=0,
            task_type=_CB_TASK_TYPE),
    }


# Hyperparameter grids.
# BioWordVec params are fixed (pretrained) — only TF-IDF params are tuned.
# Pipeline step names: 'features__tfidf__<param>' and 'clf__<param>'.
PARAM_GRIDS = {
    'LR': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__ngram_range':  [(1, 1), (1, 2)],
        'features__tfidf__min_df':       [2, 3],
        'clf__C':                        [0.1, 1.0, 10.0],
    },
    'SVM': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__ngram_range':  [(1, 1), (1, 2)],
        'features__tfidf__min_df':       [2, 3],
        'clf__C':                        [0.1, 1.0, 10.0],
        'clf__kernel':                   ['linear', 'rbf'],
    },
    'Random Forest': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__min_df':       [2, 3],
        'clf__n_estimators':             [100, 200],
        'clf__max_depth':                [10, 20],
        'clf__min_samples_split':        [2, 5],
        'clf__min_samples_leaf':         [1, 2, 4],
        'clf__max_features':             ['sqrt', 'log2'],
    },
    'XGBoost': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__min_df':       [2, 3],
        'clf__n_estimators':             [100, 200],
        'clf__learning_rate':            [0.05, 0.1],
        'clf__max_depth':                [3, 5],
        'clf__min_child_weight':         [1, 3, 5],
        'clf__subsample':                [0.8, 1.0],
        'clf__colsample_bytree':         [0.8, 1.0],
        'clf__reg_alpha':                [0, 1.0, 2.0],
        'clf__reg_lambda':               [1.0, 5.0],
    },
    'Decision Tree': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__ngram_range':  [(1, 1), (1, 2)],
        'features__tfidf__min_df':       [2, 3],
        'clf__max_depth':                [5, 10],
        'clf__min_samples_split':        [2, 5],
        'clf__min_samples_leaf':         [1, 2, 4],
        'clf__ccp_alpha':                [0.0, 0.005, 0.01],
    },
    'MLP Small': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__min_df':       [2, 3],
        'clf__hidden_layer_sizes':       [(128,), (256,), (512,)],
        'clf__activation':               ['relu', 'tanh'],
        'clf__alpha':                    [0.0001, 0.001, 0.01, 0.1],
        'clf__learning_rate_init':       [0.001, 0.0001],
    },
    'MLP Deep': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__min_df':       [2, 3],
        'clf__hidden_layer_sizes':       [(256, 128), (512, 256), (512, 256, 128)],
        'clf__activation':               ['relu', 'tanh'],
        'clf__alpha':                    [0.0001, 0.001, 0.01],
        'clf__learning_rate_init':       [0.001, 0.0001],
    },
    'CatBoost': {
        'features__tfidf__max_features': [3000, 5000],
        'features__tfidf__min_df':       [2, 3],
        'clf__iterations':               [200, 400],
        'clf__learning_rate':            [0.05, 0.1],
        'clf__depth':                    [4, 6, 8],
        'clf__l2_leaf_reg':              [1, 3, 5],
        'clf__min_data_in_leaf':         [1, 5, 10],
    },
}

# =============================================================================
# 5.  Nested cross-validation loop
# =============================================================================

all_results    = {}
best_params    = {}
fold_f1s_store = {}
MODELS         = make_models(scale_pos)

oof_preds_store = {n: np.full(len(y_cv), -1,       dtype=int)   for n in MODELS}
oof_probs_store = {n: np.full(len(y_cv), np.nan,   dtype=float) for n in MODELS}

for model_name, base_clf in MODELS.items():
    print(f"\nTraining {model_name}...")

    fold_test_metrics = []
    fold_train_f1     = []
    fold_best_params  = []

    for fold_i, (tr_idx, te_idx) in enumerate(
            OUTER_CV.split(X_cv, y_cv, groups=groups_cv), 1):

        X_tr, X_te = X_cv[tr_idx], X_cv[te_idx]
        y_tr, y_te = y_cv[tr_idx], y_cv[te_idx]

        pipe = Pipeline([
            ('features', build_feature_union()),
            ('clf',      base_clf),
        ])

        gs = GridSearchCV(
            pipe, PARAM_GRIDS[model_name],
            cv=INNER_CV, scoring='f1_weighted',
            n_jobs=-1, refit=True,
        )
        gs.fit(X_tr, y_tr, groups=groups_cv[tr_idx])

        y_pred = gs.predict(X_te)
        y_prob = gs.predict_proba(X_te)[:, 1]
        oof_preds_store[model_name][te_idx] = y_pred
        oof_probs_store[model_name][te_idx] = y_prob

        fold_f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        fold_test_metrics.append({
            'Precision': precision_score(y_te, y_pred, average='weighted', zero_division=0),
            'Recall':    recall_score(y_te, y_pred, average='weighted', zero_division=0),
            'F1-Score':  fold_f1,
            'Accuracy':  accuracy_score(y_te, y_pred),
            'ROC-AUC':   roc_auc_score(y_te, y_prob),
        })
        y_tr_pred = gs.predict(X_tr)
        fold_train_f1.append(f1_score(y_tr, y_tr_pred, average='weighted', zero_division=0))
        fold_best_params.append(gs.best_params_)

        print(f"  Fold {fold_i}: test F1={fold_f1:.4f}  "
              f"train F1={fold_train_f1[-1]:.4f}  "
              f"gap={fold_train_f1[-1] - fold_f1:.4f}")

    mean_test     = {k: np.mean([f[k] for f in fold_test_metrics]) for k in fold_test_metrics[0]}
    std_f1        = np.std([f['F1-Score'] for f in fold_test_metrics])
    mean_train_f1 = np.mean(fold_train_f1)

    overfit_gap  = mean_train_f1 - mean_test['F1-Score']
    overfit_note = "possible overfit" if overfit_gap > 0.10 else "ok"
    all_results[model_name] = {
        **mean_test,
        'F1 Std':      std_f1,
        'Train F1':    mean_train_f1,
        'Overfit Gap': overfit_gap,
    }
    fold_f1s_store[model_name] = [f['F1-Score'] for f in fold_test_metrics]

    param_keys = fold_best_params[0].keys()
    consensus  = {k: Counter(d[k] for d in fold_best_params).most_common(1)[0][0]
                  for k in param_keys}
    best_params[model_name] = consensus

    print(f"\n  {model_name} summary:")
    print(f"  Mean test  F1 : {mean_test['F1-Score']:.4f} +/- {std_f1:.4f}")
    print(f"  Mean train F1 : {mean_train_f1:.4f}")
    print(f"  Overfit gap   : {overfit_gap:.4f}  ({overfit_note})")
    print(f"  Mean AUC      : {mean_test['ROC-AUC']:.4f}")
    print(f"  Best params   : {consensus}")

# ---- Save CV results ---------------------------------------------------------
from sklearn.metrics import roc_curve, confusion_matrix as _sklearn_cm

results_df = pd.DataFrame(all_results).T.round(4)
results_df.to_csv(os.path.join(OUT_DIR, 'results.csv'))
results_df.to_csv(os.path.join(OUT_DIR, 'full_results.csv'))   # alias used by step 05
pd.DataFrame(best_params).T.to_csv(os.path.join(OUT_DIR, 'best_params.csv'))
pd.DataFrame(oof_preds_store, index=global_train_idx).to_csv(os.path.join(OUT_DIR, 'oof_predictions.csv'))
pd.DataFrame(oof_probs_store, index=global_train_idx).to_csv(os.path.join(OUT_DIR, 'oof_probabilities.csv'))
# Per-fold F1 scores for all models.
_pfdf = pd.DataFrame(fold_f1s_store, index=[f'Fold{i}' for i in range(1, 6)]).T
_pfdf.index.name = 'Model'
_pfdf['Mean'] = _pfdf.mean(axis=1).round(4)
_pfdf['Std']  = _pfdf.std(axis=1, ddof=0).round(4)
_pfdf.to_csv(os.path.join(OUT_DIR, 'per_fold_f1.csv'))
print(f"Saved per-fold F1  to {os.path.join(OUT_DIR, 'per_fold_f1.csv')}")

print("\n\n--- CV Results (TF-IDF + BioWordVec features) ---")
print(results_df[['F1-Score', 'F1 Std', 'ROC-AUC', 'Train F1', 'Overfit Gap']].to_string())

# ---- fig_results.png: ROC curves + F1 bar + train/test overfit ---------------
_model_names_r = list(results_df.index)
_colors_r      = plt.cm.tab10(np.linspace(0, 1, len(_model_names_r)))

fig_r, axes_r = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves using OOF probabilities on the CV set
for i, mname in enumerate(_model_names_r):
    probs = oof_probs_store[mname]
    mask  = ~np.isnan(probs)
    if mask.sum() > 0:
        fpr, tpr, _ = roc_curve(y_cv[mask], probs[mask])
        auc_val     = results_df.loc[mname, 'ROC-AUC']
        axes_r[0].plot(fpr, tpr, color=_colors_r[i], lw=1.5,
                       label=f'{mname} (AUC={auc_val:.3f})')
axes_r[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
axes_r[0].set_xlabel('False Positive Rate')
axes_r[0].set_ylabel('True Positive Rate')
axes_r[0].set_title('ROC Curves (OOF)', fontweight='bold')
axes_r[0].legend(fontsize=7)
axes_r[0].grid(True, alpha=0.3)

# F1 bar chart with std error bars
f1_means = results_df['F1-Score'].values
f1_stds  = results_df['F1 Std'].values
bars = axes_r[1].bar(range(len(f1_means)), f1_means, color=_colors_r,
                     yerr=f1_stds, capsize=5, alpha=0.85)
axes_r[1].set_xticks(range(len(f1_means)))
axes_r[1].set_xticklabels(_model_names_r, rotation=20, ha='right', fontsize=8)
axes_r[1].set_ylabel('Weighted F1-Score (5-fold CV)')
axes_r[1].set_title('F1-Score Comparison', fontweight='bold')
axes_r[1].set_ylim(0, 1.1)
for bar, val in zip(bars, f1_means):
    axes_r[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
axes_r[1].grid(True, alpha=0.3, axis='y')

# Train vs test overfit comparison
x_r    = np.arange(len(_model_names_r))
width  = 0.35
axes_r[2].bar(x_r - width/2, results_df['Train F1'].values,
              width, label='Train F1', color='#7F77DD', alpha=0.75)
axes_r[2].bar(x_r + width/2, results_df['F1-Score'].values,
              width, label='Test F1 (CV)', color='#1D9E75', alpha=0.75)
axes_r[2].set_xticks(x_r)
axes_r[2].set_xticklabels(_model_names_r, rotation=20, ha='right', fontsize=8)
axes_r[2].set_ylabel('Weighted F1-Score')
axes_r[2].set_title('Train vs Test F1 (Overfitting Check)', fontweight='bold')
axes_r[2].set_ylim(0, 1.1)
axes_r[2].legend(fontsize=9)
axes_r[2].grid(True, alpha=0.3, axis='y')
for i, (tr, te) in enumerate(zip(results_df['Train F1'].values, results_df['F1-Score'].values)):
    gap   = tr - te
    color = '#D85A30' if gap > 0.10 else '#555555'
    axes_r[2].text(i, max(tr, te) + 0.03, f'gap={gap:.2f}',
                   ha='center', fontsize=7, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_results.png")

# ---- fig_fold_stability.png: fold-by-fold F1 bars per model ------------------
_n_folds  = 5
_x_base   = np.arange(len(_model_names_r))
_w_fold   = 0.14
_fold_clr = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D']

fig_fold, ax_fold = plt.subplots(figsize=(16, 5))
for fi in range(_n_folds):
    vals = [fold_f1s_store[m][fi] for m in _model_names_r]
    ax_fold.bar(_x_base + fi * _w_fold, vals, _w_fold,
                label=f'Fold {fi+1}', color=_fold_clr[fi], alpha=0.80)
_means_fold = [np.mean(fold_f1s_store[m]) for m in _model_names_r]
ax_fold.plot(_x_base + (_n_folds - 1) / 2 * _w_fold, _means_fold,
             'kD--', markersize=7, lw=1.5, label='Mean CV F1', zorder=5)
ax_fold.set_xticks(_x_base + (_n_folds - 1) / 2 * _w_fold)
ax_fold.set_xticklabels(_model_names_r, rotation=25, ha='right', fontsize=9)
ax_fold.set_ylabel('Weighted F1-Score', fontsize=11)
ax_fold.set_ylim(0, 1.15)
ax_fold.set_title('Fold-by-Fold F1 Scores — BioWordVec Models (5-fold Nested CV)',
                  fontsize=13, fontweight='bold')
ax_fold.legend(fontsize=9, ncol=4)
ax_fold.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_fold_stability.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_fold_stability.png")

# ---- fig_cm_oof.png: OOF confusion matrices ----------------------------------
_ncols_oof = 4
_nrows_oof  = (len(_model_names_r) + _ncols_oof - 1) // _ncols_oof
fig_oof, axes_oof = plt.subplots(_nrows_oof, _ncols_oof,
                                  figsize=(_ncols_oof * 4, _nrows_oof * 3.8))
axes_oof = np.array(axes_oof).flatten()

for i, mname in enumerate(_model_names_r):
    preds_oof = oof_preds_store[mname]
    cm_arr    = _sklearn_cm(y_cv, preds_oof)
    f1_oof    = f1_score(y_cv, preds_oof, average='weighted', zero_division=0)
    ax        = axes_oof[i]
    sns.heatmap(cm_arr, annot=False, ax=ax, cbar=False,
                cmap=sns.light_palette('#4C72B0', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for r in range(2):
        for c in range(2):
            val = cm_arr[r, c]
            lbl = [['TN', 'FP'], ['FN', 'TP']][r][c]
            ax.text(c + 0.5, r + 0.5, f'{lbl}\n{val}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val > cm_arr.max() * 0.5 else '#333333')
    ax.set_title(f'{mname}\nOOF F1={f1_oof:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)
    ax.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    ax.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

for j in range(i + 1, len(axes_oof)):
    axes_oof[j].set_visible(False)
fig_oof.suptitle(f'OOF Confusion Matrices — BioWordVec ML (80% train, N={len(y_cv)})',
                 fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_cm_oof.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_cm_oof.png")

# =============================================================================
# 6.  Save models for demo
# =============================================================================
# Re-fit each model on the full CV training set using consensus best params,
# then pickle to models/classical_ml_biowordvec/<ModelName>.pkl.

print("\n\n--- Saving models for demo ---")

saved_pipes = {}   # reused by SHAP section below — avoids re-fitting twice

for model_name in MODELS:
    bp       = best_params[model_name]
    tfidf_kw = {k.replace('features__tfidf__', ''): v
                for k, v in bp.items() if 'tfidf' in k}
    clf_kw   = {k.replace('clf__', ''): v
                for k, v in bp.items() if k.startswith('clf__')}

    fresh = make_models(scale_pos)[model_name]
    fresh.set_params(**clf_kw)

    pipe = Pipeline([
        ('features', build_feature_union(tfidf_kw)),
        ('clf',      fresh),
    ])
    pipe.fit(X_cv, y_cv)

    safe_name = model_name.replace(' ', '_')
    pkl_path  = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(pipe, fh)

    cv_f1 = all_results[model_name]['F1-Score']
    print(f"  Saved {model_name} -> {safe_name}.pkl  (CV F1={cv_f1:.4f})")
    saved_pipes[model_name] = pipe

print(f"Models saved to {MODELS_DIR}")

# ---- fig_test_eval.png: test set confusion matrices + ROC --------------------
print("\nTest set evaluation...")
_test_results = {}
_roc_test     = {}

for mname, pipe in saved_pipes.items():
    _y_pred_t = pipe.predict(X_test_global)
    _y_prob_t = pipe.predict_proba(X_test_global)[:, 1]
    _test_results[mname] = {'y_pred': _y_pred_t, 'y_prob': _y_prob_t}
    _fpr_t, _tpr_t, _    = roc_curve(y_test_global, _y_prob_t)
    _roc_test[mname]     = (_fpr_t, _tpr_t, roc_auc_score(y_test_global, _y_prob_t))
    _f1_t = f1_score(y_test_global, _y_pred_t, average='weighted', zero_division=0)
    print(f"  {mname:<22}: test F1={_f1_t:.4f}  AUC={_roc_test[mname][2]:.4f}")

_n_test_models = len(_test_results)
_ncols_t = 4
_nrows_t = (_n_test_models + _ncols_t - 1) // _ncols_t

fig_test = plt.figure(figsize=(_ncols_t * 4.5, _nrows_t * 3.8 + 4.5))
_cm_axes = [fig_test.add_subplot(_nrows_t + 1, _ncols_t, i + 1)
            for i in range(_n_test_models)]
_roc_ax  = fig_test.add_subplot(_nrows_t + 1, 1, _nrows_t + 1)

for i, (mname, res) in enumerate(_test_results.items()):
    cm_arr = _sklearn_cm(y_test_global, res['y_pred'])
    f1_t   = f1_score(y_test_global, res['y_pred'], average='weighted', zero_division=0)
    ax     = _cm_axes[i]
    sns.heatmap(cm_arr, annot=False, ax=ax, cbar=False,
                cmap=sns.light_palette('#1D9E75', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for r in range(2):
        for c in range(2):
            val = cm_arr[r, c]
            lbl = [['TN', 'FP'], ['FN', 'TP']][r][c]
            ax.text(c + 0.5, r + 0.5, f'{lbl}\n{val}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val > cm_arr.max() * 0.5 else '#333333')
    ax.set_title(f'{mname}\nTest F1={f1_t:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)
    ax.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    ax.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)
    fpr, tpr, auc_val = _roc_test[mname]
    _roc_ax.plot(fpr, tpr, lw=1.5, label=f'{mname} (AUC={auc_val:.3f})')

_roc_ax.plot([0, 1], [0, 1], 'k--', lw=1)
_roc_ax.set_xlabel('False Positive Rate')
_roc_ax.set_ylabel('True Positive Rate')
_roc_ax.set_title('ROC Curves — Test Set (20% held-out)', fontweight='bold')
_roc_ax.legend(fontsize=8, ncol=2)
_roc_ax.grid(True, alpha=0.3)

fig_test.suptitle(
    f'Test Set Evaluation — BioWordVec ML (20% held-out, N={len(y_test_global)})',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_test_eval.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_test_eval.png")

# ---- analysis.txt: methodology, results table, findings, best params --------
best_model   = results_df['F1-Score'].idxmax()
best_f1      = results_df['F1-Score'].max()
best_auc_m   = results_df['ROC-AUC'].idxmax()
best_auc_val = results_df['ROC-AUC'].max()

analysis_path = os.path.join(OUT_DIR, 'analysis.txt')
with open(analysis_path, 'w') as _af:
    _af.write("-" * 60 + "\n")
    _af.write("ENHANCED CLASSICAL ML -- ANALYSIS AND FINDINGS\n")
    _af.write("Features: TF-IDF + BioWordVec-200 (PubMed + MIMIC-III)\n")
    _af.write("Biomarker Detection in Alzheimer's Clinical Trial Texts\n")
    _af.write("-" * 60 + "\n\n")
    _af.write(f"Dataset: {len(X)} samples, {y.mean()*100:.1f}% positive\n")
    _af.write(f"Class imbalance ratio (neg/pos): {scale_pos:.2f}\n\n")

    _af.write("-" * 60 + "\n")
    _af.write("METHODOLOGY\n")
    _af.write("-" * 60 + "\n")
    _af.write("Feature extraction: TF-IDF (sublinear_tf=True, min_df=2)\n")
    _af.write("  + BioWordVec-200 (pretrained on PubMed abstracts + MIMIC-III)\n")
    _af.write("  Combined via FeatureUnion — model learns weights for all columns.\n")
    _af.write("Models: LR, SVM, Random Forest, XGBoost, Decision Tree,\n")
    _af.write("        MLP Small (arch tuned), MLP Deep (arch tuned), CatBoost\n")
    _af.write("Imbalance: class_weight='balanced' (LR, SVM, RF, DT); scale_pos_weight (XGBoost);\n")
    _af.write("           auto_class_weights='Balanced' (CatBoost); alpha+early_stop (MLP)\n")
    _af.write("CV: 5-fold StratifiedGroupKFold outer + 3-fold inner GridSearchCV\n\n")

    _af.write("-" * 60 + "\n")
    _af.write("RESULTS TABLE (5-fold nested CV, weighted metrics)\n")
    _af.write("-" * 60 + "\n")
    _af.write(f"{'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} "
              f"{'Acc':>7} {'AUC':>7} {'F1 Std':>7} {'Train F1':>9} {'Gap':>6}\n")
    _af.write("-" * 88 + "\n")
    for _mn in results_df.index:
        r    = results_df.loc[_mn]
        note = ' *overfit*' if r['Overfit Gap'] > 0.10 else ''
        _af.write(f"{_mn:<22} {r['Precision']:>9.4f} {r['Recall']:>7.4f} "
                  f"{r['F1-Score']:>7.4f} {r['Accuracy']:>7.4f} {r['ROC-AUC']:>7.4f} "
                  f"{r['F1 Std']:>7.4f} {r['Train F1']:>9.4f} {r['Overfit Gap']:>6.4f}"
                  f"{note}\n")

    _af.write("\n" + "-" * 60 + "\n")
    _af.write("KEY FINDINGS\n")
    _af.write("-" * 60 + "\n")
    _af.write(f"\n1. Best F1:  {best_model} (F1={best_f1:.4f})\n")
    _af.write(f"2. Best AUC: {best_auc_m} (AUC={best_auc_val:.4f})\n\n")
    _af.write("3. Overfitting check:\n")
    for _mn in results_df.index:
        gap    = results_df.loc[_mn, 'Overfit Gap']
        status = 'possible overfit' if gap > 0.10 else 'healthy'
        _af.write(f"   {_mn:<22}: gap={gap:.4f}  ({status})\n")

    _af.write("\n" + "-" * 60 + "\n")
    _af.write("BEST HYPERPARAMETERS (consensus across 5 outer folds)\n")
    _af.write("-" * 60 + "\n")
    for _mn, params in best_params.items():
        _af.write(f"\n{_mn}:\n")
        for k, v in params.items():
            _af.write(f"  {k}: {v}\n")

print(f"Saved analysis.txt")

# =============================================================================
# 7.  SHAP explainability
# =============================================================================
# Uses the already-fitted pipelines from saved_pipes — no re-fitting needed.
# SHAP values are computed on the held-out global test set.
#
# Explainer choice:
#   - LR                  → LinearExplainer  (exact, fast)
#   - RF / XGBoost / DT   → TreeExplainer    (exact, fast)
#   - SVM / MLP Small / MLP Deep → KernelExplainer with k-means background
#                                  (model-agnostic, slower; limited to 150
#                                   test samples for practical speed)

print("\n\n--- SHAP Explainability ---")

TREE_MODELS   = {'Random Forest', 'XGBoost', 'Decision Tree', 'CatBoost'}
LINEAR_MODELS = {'LR'}
KERNEL_MODELS = {'SVM', 'MLP Small', 'MLP Deep'}
KERNEL_SAMPLES = 150     # how many test samples to explain via KernelExplainer
KERNEL_NSAMPLES = 200    # SHAP integration samples per explanation

shap_mean_abs = {}       # {model_name: {feature_name: mean_abs_shap}}
feature_names_cache = None


def extract_feature_names(fitted_pipeline):
    """Return ordered list of feature names from a fitted TF-IDF + BioWordVec pipeline."""
    fu          = fitted_pipeline.named_steps['features']
    tfidf_names = list(fu.transformer_list[0][1].get_feature_names_out())
    if _bwv is not None:
        bwv_names = [f'bwv_dim_{i}' for i in range(BWV_DIM)]
        return tfidf_names + bwv_names
    return tfidf_names


def to_dense(matrix):
    return matrix.toarray() if issparse(matrix) else np.array(matrix)


for model_name in MODELS:
    print(f"\nSHAP -> {model_name} ...")

    # Reuse the already-fitted pipeline from the save step — no re-fitting needed
    final_pipe = saved_pipes[model_name]

    # Cache feature names (same vocab for all models)
    feature_names = extract_feature_names(final_pipe)
    if feature_names_cache is None:
        feature_names_cache = feature_names   # save for cross-model heatmap

    # Transform sets to dense numpy arrays for SHAP
    X_train_dense = to_dense(final_pipe.named_steps['features'].transform(X_cv))
    X_test_dense  = to_dense(final_pipe.named_steps['features'].transform(X_test_global))

    clf_fitted = final_pipe.named_steps['clf']

    try:
        if model_name in LINEAR_MODELS:
            explainer   = shap.LinearExplainer(clf_fitted, X_train_dense)
            shap_vals   = explainer.shap_values(X_test_dense)
            # LinearExplainer returns array for binary; take positive-class values
            sv = shap_vals if shap_vals.ndim == 2 else shap_vals[1]

        elif model_name in TREE_MODELS:
            explainer = shap.TreeExplainer(clf_fitted,
                                           feature_perturbation='tree_path_dependent')
            shap_vals = explainer.shap_values(np.asarray(X_test_dense, dtype=np.float64))
            # TreeExplainer returns list [neg, pos] for binary classifiers
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            if hasattr(sv, 'ndim') and sv.ndim == 3:
                sv = sv[:, :, 1]   # some models return (samples, features, classes)

        else:  # SVM, MLP
            background = shap.kmeans(X_train_dense, 50)
            predict_fn = lambda x: clf_fitted.predict_proba(x)[:, 1]
            explainer  = shap.KernelExplainer(predict_fn, background)
            X_explain  = X_test_dense[:KERNEL_SAMPLES]
            shap_vals  = explainer.shap_values(X_explain, nsamples=KERNEL_NSAMPLES)
            sv         = shap_vals if not isinstance(shap_vals, list) else shap_vals[1]
            X_test_dense = X_test_dense[:KERNEL_SAMPLES]  # align rows for plots

        n_test_rows = sv.shape[0]
        X_display   = X_test_dense[:n_test_rows]

        # --- Separate TF-IDF and BioWordVec column indices
        n_tfidf      = sum(1 for n in feature_names if not n.startswith('bwv_dim_'))
        tfidf_names  = feature_names[:n_tfidf]
        sv_tfidf     = sv[:, :n_tfidf]
        X_tfidf_disp = X_display[:, :n_tfidf]

        # --- Plot 1: Beeswarm summary (TF-IDF features only — interpretable words)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv_tfidf, X_tfidf_disp,
            feature_names=tfidf_names,
            max_display=20, show=False, plot_type='dot',
        )
        plt.title(f'SHAP Summary — {model_name}\n(top 20 TF-IDF features, test set)')
        plt.tight_layout()
        plt.savefig(
            os.path.join(SHAP_DIR, f'shap_beeswarm_{model_name.replace(" ", "_")}.png'),
            dpi=150, bbox_inches='tight')
        plt.close()

        # --- Plot 2: Horizontal bar chart — top 20 features (TF-IDF + BioWordVec combined)
        mean_abs = np.abs(sv).mean(axis=0)
        # Store as plain Python floats so max() comparisons work in section 8.
        shap_mean_abs[model_name] = {k: float(v) for k, v in zip(feature_names, mean_abs)}

        top_idx   = np.argsort(mean_abs)[::-1][:20]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        colors = ['#e74c3c' if not n.startswith('bwv_dim_') else '#27ae60'
                  for n in top_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'Top 20 Features by SHAP Importance — {model_name}')
        ax.legend(handles=[
            Patch(facecolor='#e74c3c', label='TF-IDF (word/ngram)'),
            Patch(facecolor='#27ae60', label='BioWordVec (semantic dim)'),
        ], loc='lower right')
        plt.tight_layout()
        plt.savefig(
            os.path.join(SHAP_DIR, f'shap_bar_{model_name.replace(" ", "_")}.png'),
            dpi=150, bbox_inches='tight')
        plt.close()

        top_words = [n for n in top_names if not n.startswith('bwv_dim_')][:5]
        print(f"  Top TF-IDF features: {top_words}")

    except Exception as exc:
        print(f"  SHAP failed for {model_name}: {exc}")
        print(f"  (Model pkl was already saved before SHAP — safe to continue)")

# =============================================================================
# 8.  Cross-model SHAP heatmap (TF-IDF features only)
# =============================================================================
# Shows which words drive predictions across all models simultaneously.
# Red = high importance, white = low importance.

if shap_mean_abs and feature_names_cache:
  try:
    tfidf_feature_list = [f for f in feature_names_cache if not f.startswith('bwv_dim_')]

    # Pick top 30 TF-IDF features by maximum importance across any model
    max_across_models = {
        f: max(shap_mean_abs[m].get(f, 0) for m in shap_mean_abs)
        for f in tfidf_feature_list
    }
    top30 = sorted(max_across_models, key=max_across_models.get, reverse=True)[:30]

    heatmap_df = pd.DataFrame(
        {m: [shap_mean_abs[m].get(f, 0) for f in top30] for m in shap_mean_abs},
        index=top30,
    )

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd',
        linewidths=0.4, ax=ax, annot_kws={'size': 7},
    )
    ax.set_title('Mean |SHAP value| — Top 30 TF-IDF features across all models',
                 fontsize=12, pad=14)
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('TF-IDF feature (word / n-gram)', fontsize=10)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_DIR, 'shap_heatmap_cross_model.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print("\nCross-model SHAP heatmap saved.")
  except Exception as _hm_exc:
    print(f"  WARNING: Cross-model SHAP heatmap failed: {_hm_exc}")
    print("  (Sections 9 and 10 will still run — rerun 02b to regenerate heatmap)")

# =============================================================================
# 9.  Update metadata.json for demo Model Info tab
# =============================================================================
# Adds a 'biowordvec_ml' section so the demo's Model Info tab shows CV scores
# for BioWordVec-enhanced models. Existing sections are left untouched.

from config import MODELS_META
import json

if os.path.exists(MODELS_META):
    with open(MODELS_META) as _mf:
        _meta = json.load(_mf)
else:
    _meta = {}

_meta['biowordvec_ml'] = {
    'models':    list(results_df.index),
    'cv_f1':     results_df['F1-Score'].round(4).to_dict(),
    'cv_auc':    results_df['ROC-AUC'].round(4).to_dict(),
    'features':  'TF-IDF + BioWordVec-200 (PubMed + MIMIC-III)',
    'note':      'MLP models use early stopping; BioWordVec handles OOV via character subword n-grams',
}

with open(MODELS_META, 'w') as _mf:
    json.dump(_meta, _mf, indent=2)
print(f"Updated metadata.json with biowordvec_ml section.")

# =============================================================================
# 10.  Summary report
# =============================================================================

report_lines = [
    "BioWordVec Classical ML — Summary Report",
    "=" * 60,
    f"Features : TF-IDF (tuned) + BioWordVec-200 (PubMed + MIMIC-III, pretrained)",
    f"Models   : {', '.join(MODELS.keys())}",
    f"CV       : 5-fold outer StratifiedGroupKFold (grouped by NCTId)",
    f"           3-fold inner GridSearchCV (f1_weighted)",
    "",
    "CV Results:",
    results_df[['F1-Score', 'F1 Std', 'ROC-AUC', 'Train F1', 'Overfit Gap']].to_string(),
    "",
    "All outputs saved to:",
    f"  {OUT_DIR}/results.csv",
    f"  {OUT_DIR}/full_results.csv",
    f"  {OUT_DIR}/best_params.csv",
    f"  {OUT_DIR}/analysis.txt",
    f"  {OUT_DIR}/oof_predictions.csv",
    f"  {OUT_DIR}/oof_probabilities.csv",
    f"  {OUT_DIR}/fig_results.png",
    f"  {OUT_DIR}/fig_fold_stability.png",
    f"  {OUT_DIR}/fig_cm_oof.png",
    f"  {OUT_DIR}/fig_test_eval.png",
    f"  {SHAP_DIR}/shap_beeswarm_<model>.png",
    f"  {SHAP_DIR}/shap_bar_<model>.png",
    f"  {SHAP_DIR}/shap_heatmap_cross_model.png",
    f"  {MODELS_DIR}/<model>.pkl",
]

report_path = os.path.join(OUT_DIR, 'summary_report.txt')
with open(report_path, 'w') as _rf:
    _rf.write('\n'.join(report_lines))

print("\n" + "\n".join(report_lines))
print(f"\nStep 2b complete. All outputs in: {OUT_DIR}")
