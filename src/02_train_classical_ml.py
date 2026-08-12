"""
02_train_classical_ml.py -- Nested cross-validation for classical ML models.

Eight models: Logistic Regression, SVM, Random Forest, XGBoost, Decision Tree,
MLP Small, MLP Deep, CatBoost. TF-IDF features. Class imbalance: class_weight='balanced'
(LR, SVM, RF, DT), scale_pos_weight (XGBoost), auto_class_weights='Balanced' (CatBoost).
MLP uses alpha + early stopping.

Nested CV structure:
  - Outer loop: 5-fold StratifiedGroupKFold (grouped by NCTId)
  - Inner loop: 3-fold GridSearchCV per outer fold

After CV: models saved as .pkl for demo; OOF predictions saved for leakage-free
confusion matrices; fold F1 stability chart and 20% held-out test eval produced here.
SHAP explainability runs at the end if shap is installed (pip install shap).

Outputs:
    outputs/classical_ml/results.csv
    outputs/classical_ml/full_results.csv
    outputs/classical_ml/best_params.csv
    outputs/classical_ml/per_fold_f1.csv
    outputs/classical_ml/analysis.txt
    outputs/classical_ml/oof_predictions.csv
    outputs/classical_ml/oof_probabilities.csv
    outputs/classical_ml/fig_results.png
    outputs/classical_ml/fig_cm_oof.png
    outputs/classical_ml/fig_fold_stability.png
    outputs/classical_ml/fig_test_eval.png
    outputs/classical_ml/shap/shap_beeswarm_<model>.png  (if shap installed)
    outputs/classical_ml/shap/shap_bar_<model>.png       (if shap installed)
    outputs/classical_ml/shap/shap_heatmap_cross_model.png (if shap installed)
    models/classical_ml/*.pkl
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

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
# SMOTE removed: TF-IDF vectors are high-dimensional and sparse; interpolating
# between sparse vectors produces meaningless synthetic samples. Class imbalance
# is handled by class_weight='balanced' (LR, SVM, RF, DT) and scale_pos_weight (XGBoost).
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
from config import OUT_DATASET, OUT_CLASSICAL, MODELS_CLASSICAL, MODELS_META, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX

os.makedirs(OUT_CLASSICAL, exist_ok=True)

print("-" * 60)
print("Step 2 of 7: Training classical ML models")
print("-" * 60)

# ---- Load dataset ------------------------------------------------------------
dataset_path = os.path.join(OUT_DATASET, 'labeled_dataset.csv')
df = pd.read_csv(dataset_path)
X      = df['clean_freeform'].fillna('').values
y      = df['has_biomarker'].values
groups = df['NCTId'].values
print(f"\nLoaded {len(X)} samples | {y.mean()*100:.1f}% positive | {(y==0).sum()} negative")

# Load the 80/20 split created once in step 01. All CV and hyperparameter
# tuning uses the 80% training fold only; the 20% test fold is never touched
# here -- it is evaluated on the held-out test set at the bottom of this script.
global_train_idx = np.load(SPLIT_TRAIN_IDX)
global_test_idx  = np.load(SPLIT_TEST_IDX)
X_cv           = X[global_train_idx]
y_cv           = y[global_train_idx]
groups_cv      = groups[global_train_idx]
X_test_global  = X[global_test_idx]
y_test_global  = y[global_test_idx]
print(f"80/20 split loaded -- CV on {len(X_cv)} train samples "
      f"| {y_cv.mean()*100:.1f}% positive | {len(set(groups_cv))} trials"
      f"\nTest set: {len(X_test_global)} samples")

# ---- CV setup ----------------------------------------------------------------
# Both outer and inner CV use StratifiedGroupKFold so trial-level grouping is
# respected at every level of the nested CV. groups_cv[train_idx] is passed to
# gs.fit() so the inner splitter receives the correct trial assignments for
# each outer training fold.
OUTER_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

# scale_pos_weight for XGBoost computed from the training set only.
scale_pos = (y_cv == 0).sum() / (y_cv == 1).sum()


def make_models(scale_pos_weight):
    """Construct base classifiers with imbalance-handling defaults."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs'),
        'SVM': SVC(
            probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric='logloss', verbosity=0, use_label_encoder=False,
            device=_XGB_DEVICE),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=42),
        # MLPClassifier does not support class_weight. Class imbalance (~30/70)
        # is handled via alpha (L2 regularisation) and early stopping.
        'MLP Small': MLPClassifier(
            hidden_layer_sizes=(256,), activation='relu', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
        'MLP Deep': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
            max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
        # CatBoost: gradient boosting with native sparse matrix support and built-in
        # class balancing. l2_leaf_reg and min_data_in_leaf are the main anti-overfit knobs.
        'CatBoost': CatBoostClassifier(
            auto_class_weights='Balanced', random_seed=42, verbose=0,
            task_type=_CB_TASK_TYPE),
    }


# ---- Hyperparameter grids ----------------------------------------------------
# These grids were chosen to cover a useful range without being prohibitively slow.
# The TF-IDF parameters (max_features, ngram_range) are tuned jointly with the
# classifier parameters because the optimal vocabulary size depends on the model.
PARAM_GRIDS = {
    'Logistic Regression': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__ngram_range':  [(1, 1), (1, 2)],
        'tfidf__min_df':       [2, 3],         # higher min_df removes rare memorised tokens
        'clf__C':              [0.01, 0.1, 1.0, 10.0],
    },
    'SVM': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__ngram_range':  [(1, 1), (1, 2)],
        'tfidf__min_df':       [2, 3],
        'clf__C':              [0.1, 1.0, 10.0],
        'clf__kernel':         ['linear', 'rbf'],
    },
    'Random Forest': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__min_df':       [2, 3],
        'clf__n_estimators':   [100, 200],
        'clf__max_depth':      [5, 10, 15, 20],
        'clf__min_samples_split': [2, 5, 10],  # larger values prevent splitting on tiny groups
        'clf__min_samples_leaf':  [1, 2, 4],   # min leaf size directly controls memorisation
        'clf__max_features':   ['sqrt', 'log2'],
    },
    'XGBoost': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__min_df':       [2, 3],
        'clf__n_estimators':   [100, 200],
        'clf__learning_rate':  [0.05, 0.1],
        'clf__max_depth':      [3, 5],
        'clf__min_child_weight': [1, 3, 5],    # min sum of hessian in child; higher = less overfit
        'clf__subsample':      [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
        'clf__reg_alpha':      [0, 1.0, 2.0],  # L1 regularisation on leaf weights
        'clf__reg_lambda':     [1.0, 5.0],     # L2 regularisation (default=1, try stronger)
    },
    'CatBoost': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__min_df':       [2, 3],
        'clf__iterations':     [200, 400],
        'clf__learning_rate':  [0.05, 0.1],
        'clf__depth':          [4, 6, 8],      # equivalent to max_depth; lower = less overfit
        'clf__l2_leaf_reg':    [1, 3, 5],      # L2 regularisation on leaf values
        'clf__min_data_in_leaf': [1, 5, 10],   # min samples per leaf; higher = less overfit
    },
    'Decision Tree': {
        'tfidf__max_features': [2000, 3000, 5000],
        'tfidf__ngram_range':  [(1, 1), (1, 2)],
        'tfidf__min_df':       [2, 3],
        'clf__max_depth':      [3, 5, 7, 10],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf':  [1, 2, 4],
        'clf__criterion':      ['gini', 'entropy'],
        'clf__ccp_alpha':      [0.0, 0.005, 0.01],  # cost-complexity pruning; higher = simpler tree
    },
    'MLP Small': {
        'tfidf__max_features':     [3000, 5000],
        'tfidf__min_df':           [2, 3],
        # Search single-layer widths: narrow vs wide vs very wide
        'clf__hidden_layer_sizes': [(128,), (256,), (512,)],
        'clf__activation':         ['relu', 'tanh'],   # tanh often better for sparse TF-IDF
        # Wider alpha range: TF-IDF is high-dim, heavier L2 can help generalisation
        'clf__alpha':              [0.0001, 0.001, 0.01, 0.1],
        'clf__learning_rate_init': [0.001, 0.0001],
    },
    'MLP Deep': {
        'tfidf__max_features':     [3000, 5000],
        'tfidf__min_df':           [2, 3],
        # Search 2- and 3-layer variants; on 1559 samples very deep nets tend to overfit
        'clf__hidden_layer_sizes': [(256, 128), (512, 256), (512, 256, 128)],
        'clf__activation':         ['relu', 'tanh'],
        'clf__alpha':              [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.0001],
    },
}

# ---- Nested CV loop ----------------------------------------------------------
all_results = {}
best_params = {}

fold_f1s_store = {}
MODELS = make_models(scale_pos)

# OOF storage: length-N arrays filled fold-by-fold for leakage-free
# confusion matrices produced later in this script.
oof_preds_store = {name: np.full(len(y_cv), -1, dtype=int) for name in MODELS}
oof_probs_store = {name: np.full(len(y_cv), np.nan) for name in MODELS}

for model_name, base_clf in MODELS.items():
    print(f"\nTraining {model_name}...")

    fold_test_metrics = []
    fold_train_f1     = []
    fold_best_params  = []

    for fold_i, (train_idx, test_idx) in enumerate(OUTER_CV.split(X_cv, y_cv, groups=groups_cv), 1):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

        # Class imbalance is handled by class_weight='balanced' / scale_pos_weight in each classifier.
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=2)),
            ('clf',   base_clf),
        ])

        gs = GridSearchCV(
            pipe, PARAM_GRIDS[model_name],
            cv=INNER_CV, scoring='f1_weighted',
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train, y_train, groups=groups_cv[train_idx])

        y_pred = gs.predict(X_test)
        y_prob = gs.predict_proba(X_test)[:, 1]
        oof_preds_store[model_name][test_idx] = y_pred
        oof_probs_store[model_name][test_idx] = y_prob

        fold_test_metrics.append({
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'Accuracy':  accuracy_score(y_test, y_pred),
            'ROC-AUC':   roc_auc_score(y_test, y_prob),
        })

        # Train F1 on the same fold's training data, to measure overfitting.
        y_train_pred = gs.predict(X_train)
        fold_train_f1.append(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
        fold_best_params.append(gs.best_params_)

        print(f"  Fold {fold_i}: test F1={fold_test_metrics[-1]['F1-Score']:.4f}  "
              f"train F1={fold_train_f1[-1]:.4f}  "
              f"gap={fold_train_f1[-1] - fold_test_metrics[-1]['F1-Score']:.4f}")

    # Aggregate across folds.
    mean_test     = {k: np.mean([f[k] for f in fold_test_metrics]) for k in fold_test_metrics[0]}
    std_f1        = np.std([f['F1-Score'] for f in fold_test_metrics])
    mean_train_f1 = np.mean(fold_train_f1)
    overfit_gap   = mean_train_f1 - mean_test['F1-Score']

    all_results[model_name] = {
        **mean_test,
        'F1 Std':      std_f1,
        'Train F1':    mean_train_f1,
        'Overfit Gap': overfit_gap,
    }

    # Consensus best params: for each hyperparameter, take the value that appeared
    # most often across the 5 outer folds. This is more robust than a single run.
    param_keys = fold_best_params[0].keys()
    consensus  = {k: Counter(d[k] for d in fold_best_params).most_common(1)[0][0]
                  for k in param_keys}
    best_params[model_name] = consensus
    fold_f1s_store[model_name] = [f['F1-Score'] for f in fold_test_metrics]

    overfit_note = "possible overfit" if overfit_gap > 0.10 else "ok"
    print(f"\n  Mean test  F1 : {mean_test['F1-Score']:.4f} +/- {std_f1:.4f}")
    print(f"  Mean train F1 : {mean_train_f1:.4f}")
    print(f"  Overfit gap   : {overfit_gap:.4f}  ({overfit_note})")
    print(f"  Mean AUC      : {mean_test['ROC-AUC']:.4f}")
    print(f"  Best params   : {consensus}")

# ---- Classical Ensemble: CV-F1-weighted soft vote over all 8 models ----------
# Each fold uses the consensus best params found above. No extra tuning is done
# here -- this is purely an aggregation of what the individual models already learned.
# Models are weighted by their individual CV F1 so stronger models (LR, SVM)
# have proportionally more influence than weaker ones (DT).
print("\nEvaluating Classical Ensemble (CV-F1-weighted soft vote over all 8 models)...")

ensemble_test_metrics = []
ensemble_train_f1     = []
oof_ens_preds = np.full(len(y_cv), -1, dtype=int)
oof_ens_probs = np.full(len(y_cv), np.nan)

for fold_i, (train_idx, test_idx) in enumerate(OUTER_CV.split(X_cv, y_cv, groups=groups_cv), 1):
    X_train, X_test = X_cv[train_idx], X_cv[test_idx]
    y_train, y_test = y_cv[train_idx], y_cv[test_idx]
    sp = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    estimators = []
    for model_name, params in best_params.items():
        clf_p   = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
        tfidf_p = {k.replace('tfidf__', ''): v for k, v in params.items() if k.startswith('tfidf__')}
        tfidf_p.setdefault('sublinear_tf', True)

        if model_name == 'XGBoost':
            clf_p['scale_pos_weight'] = sp
            clf_p.setdefault('eval_metric', 'logloss')
            clf_p.setdefault('verbosity', 0)
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('device', _XGB_DEVICE)
            clf = XGBClassifier(**clf_p)
        elif model_name == 'Logistic Regression':
            clf_p.setdefault('max_iter', 1000)
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('class_weight', 'balanced')
            clf = LogisticRegression(**clf_p)
        elif model_name == 'SVM':
            clf_p.setdefault('probability', True)
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('class_weight', 'balanced')
            clf = SVC(**clf_p)
        elif model_name == 'Random Forest':
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('class_weight', 'balanced')
            clf_p.setdefault('n_jobs', -1)
            clf = RandomForestClassifier(**clf_p)
        elif model_name == 'Decision Tree':
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('class_weight', 'balanced')
            clf = DecisionTreeClassifier(**clf_p)
        elif model_name == 'MLP Small':
            clf_p.setdefault('hidden_layer_sizes', (256,))
            clf_p.setdefault('activation', 'relu')
            clf_p.setdefault('solver', 'adam')
            clf_p.setdefault('max_iter', 500)
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('early_stopping', True)
            clf_p.setdefault('validation_fraction', 0.1)
            clf_p.setdefault('n_iter_no_change', 10)
            clf = MLPClassifier(**clf_p)
        elif model_name == 'MLP Deep':
            clf_p.setdefault('hidden_layer_sizes', (512, 256, 128))
            clf_p.setdefault('activation', 'relu')
            clf_p.setdefault('solver', 'adam')
            clf_p.setdefault('max_iter', 500)
            clf_p.setdefault('random_state', 42)
            clf_p.setdefault('early_stopping', True)
            clf_p.setdefault('validation_fraction', 0.1)
            clf_p.setdefault('n_iter_no_change', 10)
            clf = MLPClassifier(**clf_p)
        else:  # CatBoost
            clf_p.setdefault('auto_class_weights', 'Balanced')
            clf_p.setdefault('random_seed', 42)
            clf_p.setdefault('verbose', 0)
            clf_p.setdefault('task_type', _CB_TASK_TYPE)
            clf = CatBoostClassifier(**clf_p)

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_p)),
            ('clf',   clf),
        ])
        estimators.append((model_name.replace(' ', '_'), pipe))

    # Weight each model by its CV F1 so stronger models have proportionally
    # more influence than weaker ones.
    INDIVIDUAL_MODELS = ['Logistic Regression', 'SVM', 'Random Forest',
                         'XGBoost', 'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']
    weights = [all_results[m]['F1-Score'] for m in INDIVIDUAL_MODELS]
    voter = VotingClassifier(estimators=estimators, voting='soft',
                             weights=weights, n_jobs=-1)
    voter.fit(X_train, y_train)

    y_pred = voter.predict(X_test)
    y_prob = voter.predict_proba(X_test)[:, 1]
    oof_ens_preds[test_idx] = y_pred
    oof_ens_probs[test_idx] = y_prob
    ensemble_test_metrics.append({
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Accuracy':  accuracy_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_prob),
    })
    y_train_pred = voter.predict(X_train)
    ensemble_train_f1.append(
        f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
    print(f"  Fold {fold_i}: test F1={ensemble_test_metrics[-1]['F1-Score']:.4f}  "
          f"train F1={ensemble_train_f1[-1]:.4f}  "
          f"gap={ensemble_train_f1[-1] - ensemble_test_metrics[-1]['F1-Score']:.4f}")

ens_mean  = {k: np.mean([f[k] for f in ensemble_test_metrics]) for k in ensemble_test_metrics[0]}
ens_std   = np.std([f['F1-Score'] for f in ensemble_test_metrics])
ens_gap   = np.mean(ensemble_train_f1) - ens_mean['F1-Score']
all_results['Classical Ensemble'] = {
    **ens_mean, 'F1 Std': ens_std,
    'Train F1': np.mean(ensemble_train_f1), 'Overfit Gap': ens_gap,
}
best_params['Classical Ensemble'] = {'note': 'CV-F1-weighted soft vote over all 8 models with consensus params'}
fold_f1s_store['Classical Ensemble'] = [f['F1-Score'] for f in ensemble_test_metrics]
print(f"\n  Mean test  F1 : {ens_mean['F1-Score']:.4f} +/- {ens_std:.4f}")
print(f"  Mean train F1 : {np.mean(ensemble_train_f1):.4f}")
print(f"  Overfit gap   : {ens_gap:.4f}")
print(f"  Mean AUC      : {ens_mean['ROC-AUC']:.4f}")

# Save OOF predictions for leakage-free confusion matrices (produced later in this script).
oof_preds_store['Classical Ensemble'] = oof_ens_preds
oof_probs_store['Classical Ensemble'] = oof_ens_probs
# Save with global dataset row indices for alignment with the full dataset.
pd.DataFrame(oof_preds_store, index=global_train_idx).to_csv(
    os.path.join(OUT_CLASSICAL, 'oof_predictions.csv'))
pd.DataFrame(oof_probs_store, index=global_train_idx).to_csv(
    os.path.join(OUT_CLASSICAL, 'oof_probabilities.csv'))
print(f"\nSaved OOF predictions to {os.path.join(OUT_CLASSICAL, 'oof_predictions.csv')}")
print(f"Saved OOF probabilities to {os.path.join(OUT_CLASSICAL, 'oof_probabilities.csv')}")

# ---- Results table -----------------------------------------------------------
results_df = pd.DataFrame(all_results).T.round(4)
display_cols = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC',
                'F1 Std', 'Train F1', 'Overfit Gap']
results_df = results_df[display_cols]

print("\n" + "-" * 60)
print("Classical ML Results (5-fold nested CV)")
print("-" * 60)
print(results_df[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC', 'Overfit Gap']].to_string())

# Save results and best params for downstream scripts.
results_df[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']].to_csv(
    os.path.join(OUT_CLASSICAL, 'results.csv'))
# full_results.csv includes Train F1 and Overfit Gap -- read by step 05 for the master results table.
results_df.to_csv(os.path.join(OUT_CLASSICAL, 'full_results.csv'))
pd.DataFrame(best_params).T.to_csv(os.path.join(OUT_CLASSICAL, 'best_params.csv'))
# Per-fold F1 scores for all models (5 folds x n_models).
_pfdf = pd.DataFrame(fold_f1s_store, index=[f'Fold{i}' for i in range(1, 6)]).T
_pfdf.index.name = 'Model'
_pfdf['Mean'] = _pfdf.mean(axis=1).round(4)
_pfdf['Std']  = _pfdf.std(axis=1, ddof=0).round(4)
_pfdf.to_csv(os.path.join(OUT_CLASSICAL, 'per_fold_f1.csv'))
print(f"\nSaved results to {os.path.join(OUT_CLASSICAL, 'results.csv')}")
print(f"Saved full results to {os.path.join(OUT_CLASSICAL, 'full_results.csv')}")
print(f"Saved best params to {os.path.join(OUT_CLASSICAL, 'best_params.csv')}")
print(f"Saved per-fold F1  to {os.path.join(OUT_CLASSICAL, 'per_fold_f1.csv')}")

# ---- Retrain with best params for ROC visualization -------------------------
# Uses an 80/20 split within the training set only -- the held-out 20% test set
# the held-out test set is evaluated at the bottom of this script.
print("\nRetraining with best params for ROC visualization (80/20 within train)...")
gss_roc = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
roc_tr_idx, roc_te_idx = next(gss_roc.split(X_cv, y_cv, groups=groups_cv))
X_tr, X_te, y_tr, y_te = X_cv[roc_tr_idx], X_cv[roc_te_idx], y_cv[roc_tr_idx], y_cv[roc_te_idx]
scale_pos_tr = (y_tr == 0).sum() / (y_tr == 1).sum()

roc_data = {}
for model_name, params in best_params.items():
    clf_params   = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
    tfidf_params = {k.replace('tfidf__', ''): v for k, v in params.items() if k.startswith('tfidf__')}

    if model_name == 'XGBoost':
        clf_params['scale_pos_weight'] = scale_pos_tr
        clf_params.setdefault('eval_metric', 'logloss')
        clf_params.setdefault('verbosity', 0)
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('device', _XGB_DEVICE)
        clf = XGBClassifier(**clf_params)
    elif model_name == 'Logistic Regression':
        clf_params.setdefault('max_iter', 1000)
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('class_weight', 'balanced')
        clf = LogisticRegression(**clf_params)
    elif model_name == 'SVM':
        clf_params.setdefault('probability', True)
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('class_weight', 'balanced')
        clf = SVC(**clf_params)
    elif model_name == 'Random Forest':
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('class_weight', 'balanced')
        clf_params.setdefault('n_jobs', -1)
        clf = RandomForestClassifier(**clf_params)
    elif model_name == 'Decision Tree':
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('class_weight', 'balanced')
        clf = DecisionTreeClassifier(**clf_params)
    elif model_name == 'MLP Small':
        clf_params.setdefault('hidden_layer_sizes', (256,))
        clf_params.setdefault('activation', 'relu')
        clf_params.setdefault('solver', 'adam')
        clf_params.setdefault('max_iter', 500)
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('early_stopping', True)
        clf = MLPClassifier(**clf_params)
    elif model_name == 'MLP Deep':
        clf_params.setdefault('hidden_layer_sizes', (512, 256, 128))
        clf_params.setdefault('activation', 'relu')
        clf_params.setdefault('solver', 'adam')
        clf_params.setdefault('max_iter', 500)
        clf_params.setdefault('random_state', 42)
        clf_params.setdefault('early_stopping', True)
        clf = MLPClassifier(**clf_params)
    elif model_name == 'CatBoost':
        clf_params.setdefault('auto_class_weights', 'Balanced')
        clf_params.setdefault('random_seed', 42)
        clf_params.setdefault('verbose', 0)
        clf_params.setdefault('task_type', _CB_TASK_TYPE)
        clf = CatBoostClassifier(**clf_params)
    else:
        continue  # skip Classical Ensemble note entry

    tfidf_params.setdefault('sublinear_tf', True)
    tfidf_params.setdefault('min_df', 2)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   clf),
    ])
    pipe.fit(X_tr, y_tr)

    y_prob = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    roc_data[model_name] = (fpr, tpr, auc)

# ---- Figures -----------------------------------------------------------------
colors = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D',
          '#2A7DB5', '#8B4EA0', '#3E8C4B', '#C45080']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Classical ML -- Nested CV + Hyperparameter Tuning',
             fontsize=14, fontweight='bold')

# ROC curves (visualization only, not used for final metrics).
for (model_name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
    axes[0].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', color=color, lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves (best params, held-out 20%)', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# F1 bar chart with CV standard deviation as error bars.
f1_means = results_df['F1-Score'].values
f1_stds  = results_df['F1 Std'].values
bars = axes[1].bar(range(len(f1_means)), f1_means, color=colors,
                   yerr=f1_stds, capsize=5, alpha=0.85)
axes[1].set_xticks(range(len(f1_means)))
axes[1].set_xticklabels(results_df.index, rotation=20, ha='right', fontsize=9)
axes[1].set_ylabel('Weighted F1-Score (5-fold CV)')
axes[1].set_title('F1-Score Comparison', fontweight='bold')
axes[1].set_ylim(0, 1.1)
for bar, val in zip(bars, f1_means):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# Train vs test F1 -- the gap tells us how much each model overfits.
x     = np.arange(len(results_df))
width = 0.35
train_f1 = results_df['Train F1'].values
test_f1  = results_df['F1-Score'].values
axes[2].bar(x - width/2, train_f1, width, label='Train F1', color='#7F77DD', alpha=0.75)
axes[2].bar(x + width/2, test_f1,  width, label='Test F1 (CV)', color='#1D9E75', alpha=0.75)
axes[2].set_xticks(x)
axes[2].set_xticklabels(results_df.index, rotation=20, ha='right', fontsize=9)
axes[2].set_ylabel('Weighted F1-Score')
axes[2].set_title('Train vs Test F1 (Overfitting Check)', fontweight='bold')
axes[2].set_ylim(0, 1.1)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')
for i, (tr, te) in enumerate(zip(train_f1, test_f1)):
    gap   = tr - te
    color = '#D85A30' if gap > 0.10 else '#555555'
    axes[2].text(i, max(tr, te) + 0.03, f'gap={gap:.2f}',
                 ha='center', fontsize=7.5, color=color)

plt.tight_layout()
fig_path = os.path.join(OUT_CLASSICAL, 'fig_results.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure to {fig_path}")

# ---- Write analysis text file ------------------------------------------------
best_model = results_df['F1-Score'].idxmax()
best_f1    = results_df['F1-Score'].max()
best_auc_m = results_df['ROC-AUC'].idxmax()
best_auc   = results_df['ROC-AUC'].max()

analysis_path = os.path.join(OUT_CLASSICAL, 'analysis.txt')
with open(analysis_path, 'w') as f:
    f.write("-" * 60 + "\n")
    f.write("CLASSICAL ML -- ANALYSIS AND FINDINGS\n")
    f.write("Biomarker Detection in Alzheimer's Clinical Trial Texts\n")
    f.write("-" * 60 + "\n\n")
    f.write(f"Dataset: {len(X)} samples, {y.mean()*100:.1f}% positive\n")
    f.write(f"Class imbalance ratio (neg/pos): {scale_pos:.2f}\n\n")

    f.write("-" * 60 + "\n")
    f.write("METHODOLOGY\n")
    f.write("-" * 60 + "\n")
    f.write("Feature extraction: TF-IDF with sublinear_tf=True, min_df=2\n")
    f.write("  (tuned: max_features in [3000, 5000], ngram_range in [(1,1), (1,2)])\n")
    f.write("Imbalance handling: class_weight='balanced' (LR, SVM, RF, DT); scale_pos_weight (XGBoost); auto_class_weights='Balanced' (CatBoost)\n")
    f.write("  SMOTE not used -- TF-IDF vectors are high-dimensional/sparse; interpolation is unreliable.\n")
    f.write("Evaluation: 5-fold stratified outer CV + 3-fold inner GridSearchCV\n")
    f.write("  Outer folds provide honest performance estimates.\n")
    f.write("  Inner folds select best hyperparameters per outer fold.\n")
    f.write("  Reported metrics are means over outer test folds only.\n\n")

    f.write("-" * 60 + "\n")
    f.write("RESULTS TABLE (5-fold nested CV, weighted metrics)\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} "
            f"{'Acc':>7} {'AUC':>7} {'F1 Std':>7} {'Train F1':>9} {'Gap':>6}\n")
    f.write("-" * 88 + "\n")
    for model_name in results_df.index:
        r = results_df.loc[model_name]
        gap_note = ' *overfit*' if r['Overfit Gap'] > 0.10 else ''
        f.write(f"{model_name:<22} {r['Precision']:>9.4f} {r['Recall']:>7.4f} "
                f"{r['F1-Score']:>7.4f} {r['Accuracy']:>7.4f} {r['ROC-AUC']:>7.4f} "
                f"{r['F1 Std']:>7.4f} {r['Train F1']:>9.4f} {r['Overfit Gap']:>6.4f}"
                f"{gap_note}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("-" * 60 + "\n")
    f.write(f"\n1. Best F1:  {best_model} (F1={best_f1:.4f})\n")
    f.write(f"2. Best AUC: {best_auc_m} (AUC={best_auc:.4f})\n\n")

    f.write("3. Overfitting check:\n")
    for name in results_df.index:
        gap = results_df.loc[name, 'Overfit Gap']
        status = 'possible overfit' if gap > 0.10 else 'healthy'
        f.write(f"   {name:<22}: gap={gap:.4f}  ({status})\n")

    f.write("\n4. Class imbalance handled via class_weight='balanced' and scale_pos_weight,\n")
    f.write("   not SMOTE -- TF-IDF sparse vectors make SMOTE interpolation unreliable.\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("BEST HYPERPARAMETERS (consensus across 5 outer folds)\n")
    f.write("-" * 60 + "\n")
    for model_name, params in best_params.items():
        f.write(f"\n{model_name}:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("OUTPUT FILES\n")
    f.write("-" * 60 + "\n")
    f.write(f"  {os.path.join(OUT_CLASSICAL, 'results.csv'):<55} -- metrics (used by step 5)\n")
    f.write(f"  {os.path.join(OUT_CLASSICAL, 'best_params.csv'):<55} -- hyperparams (human reference)\n")
    f.write(f"  {os.path.join(OUT_CLASSICAL, 'fig_results.png'):<55} -- ROC, F1 bars, overfit check\n")

print(f"Saved analysis to {analysis_path}")

# =============================================================================
# Save trained models for the demo app
# =============================================================================
# Re-train each model on the full 80% training set using its consensus best
# params and save as a pickle. The demo loads these at startup.
print("\n" + "-" * 60)
print("Saving classical ML models for demo app")
print("-" * 60)

os.makedirs(MODELS_CLASSICAL, exist_ok=True)

def _build_classifier_for_save(model_name, bp, scale_pos_weight):
    """Reconstruct a classifier from consensus best_params dict."""
    clf_p = {k.replace('clf__', ''): v for k, v in bp.items() if k.startswith('clf__')}
    if model_name == 'Logistic Regression':
        clf_p.setdefault('max_iter', 1000)
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('class_weight', 'balanced')
        return LogisticRegression(**clf_p)
    elif model_name == 'SVM':
        clf_p.setdefault('probability', True)
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('class_weight', 'balanced')
        return SVC(**clf_p)
    elif model_name == 'Random Forest':
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('class_weight', 'balanced')
        clf_p.setdefault('n_jobs', -1)
        return RandomForestClassifier(**clf_p)
    elif model_name == 'XGBoost':
        clf_p['scale_pos_weight'] = scale_pos_weight
        clf_p.setdefault('eval_metric', 'logloss')
        clf_p.setdefault('verbosity', 0)
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('device', _XGB_DEVICE)
        return XGBClassifier(**clf_p)
    elif model_name == 'Decision Tree':
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('class_weight', 'balanced')
        return DecisionTreeClassifier(**clf_p)
    elif model_name == 'MLP Small':
        clf_p.setdefault('hidden_layer_sizes', (256,))
        clf_p.setdefault('activation', 'relu')
        clf_p.setdefault('solver', 'adam')
        clf_p.setdefault('max_iter', 500)
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('early_stopping', True)
        clf_p.setdefault('validation_fraction', 0.1)
        clf_p.setdefault('n_iter_no_change', 10)
        return MLPClassifier(**clf_p)
    elif model_name == 'MLP Deep':
        clf_p.setdefault('hidden_layer_sizes', (512, 256, 128))
        clf_p.setdefault('activation', 'relu')
        clf_p.setdefault('solver', 'adam')
        clf_p.setdefault('max_iter', 500)
        clf_p.setdefault('random_state', 42)
        clf_p.setdefault('early_stopping', True)
        clf_p.setdefault('validation_fraction', 0.1)
        clf_p.setdefault('n_iter_no_change', 10)
        return MLPClassifier(**clf_p)
    elif model_name == 'CatBoost':
        clf_p.setdefault('auto_class_weights', 'Balanced')
        clf_p.setdefault('random_seed', 42)
        clf_p.setdefault('verbose', 0)
        clf_p.setdefault('task_type', _CB_TASK_TYPE)
        return CatBoostClassifier(**clf_p)
    else:
        raise ValueError(f"Unknown model: {model_name}")

saved_pipes = {}   # filled below; kept in memory for SHAP section

for model_name, params in best_params.items():
    if model_name == 'Classical Ensemble':
        continue  # ensemble is built dynamically from individual pickles in the demo
    tfidf_p = {k.replace('tfidf__', ''): v for k, v in params.items()
               if k.startswith('tfidf__')}
    tfidf_p.setdefault('sublinear_tf', True)
    clf = _build_classifier_for_save(model_name, params, scale_pos)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_p)),
        ('clf',   clf),
    ])
    pipe.fit(X_cv, y_cv)
    saved_pipes[model_name] = pipe   # keep for SHAP below

    safe_name = model_name.replace(' ', '_')
    save_path = os.path.join(MODELS_CLASSICAL, f'{safe_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(pipe, f)
    cv_f1 = results_df.loc[model_name, 'F1-Score'] if model_name in results_df.index else 0.0
    print(f"  Saved {model_name} -> {safe_name}.pkl  (CV F1={cv_f1:.4f})")

print(f"Classical ML models saved to {MODELS_CLASSICAL}")

# =============================================================================
# SHAP Explainability — Classical ML (TF-IDF features)
# =============================================================================
# Runs only if shap is installed (pip install shap). Skipped gracefully otherwise.
# Outputs beeswarm + bar chart per model, plus a cross-model importance heatmap.
# =============================================================================
try:
    import shap as _shap
    from scipy.sparse import issparse as _issparse
    _shap_available = True
except ImportError:
    _shap_available = False
    print("\nINFO: shap not installed — skipping SHAP explainability (pip install shap)")

if _shap_available and saved_pipes:
    _SHAP_DIR       = os.path.join(OUT_CLASSICAL, 'shap')
    _KERNEL_SAMPLES = 100    # test samples explained via KernelExplainer
    _KERNEL_NSAMPLES = 200   # SHAP integration samples per explanation
    _LINEAR_MODELS  = {'Logistic Regression'}
    _TREE_MODELS    = {'Random Forest', 'XGBoost', 'Decision Tree', 'CatBoost'}
    # SVM and MLP use KernelExplainer (slower but model-agnostic)

    os.makedirs(_SHAP_DIR, exist_ok=True)
    _X_shap_test = X_test_global

    _shap_mean_abs = {}   # {model_name: {feature: importance}} for heatmap

    print("\n" + "-" * 60)
    print("SHAP Explainability — Classical ML")
    print("-" * 60)

    for _sname, _spipe in saved_pipes.items():
        print(f"\nSHAP → {_sname} ...")
        _clf_s       = _spipe.named_steps['clf']
        _feat_names  = list(_spipe.named_steps['tfidf'].get_feature_names_out())
        _X_tr_dense  = _spipe.named_steps['tfidf'].transform(X_cv).toarray().astype(np.float64)
        _X_te_dense  = _spipe.named_steps['tfidf'].transform(_X_shap_test).astype(np.float64)
        if _issparse(_X_te_dense):
            _X_te_dense = _X_te_dense.toarray()

        try:
            if _sname in _LINEAR_MODELS:
                _explainer = _shap.LinearExplainer(_clf_s, _X_tr_dense)
                _sv_raw    = _explainer.shap_values(_X_te_dense)
                _sv = _sv_raw if not isinstance(_sv_raw, list) else _sv_raw[1]
                if hasattr(_sv, 'ndim') and _sv.ndim != 2:
                    _sv = _sv_raw[1]

            elif _sname in _TREE_MODELS:
                _explainer = _shap.TreeExplainer(
                    _clf_s, feature_perturbation='tree_path_dependent')
                _sv_raw = _explainer.shap_values(
                    np.asarray(_X_te_dense, dtype=np.float64))
                _sv = _sv_raw[1] if isinstance(_sv_raw, list) else _sv_raw
                if hasattr(_sv, 'ndim') and _sv.ndim == 3:
                    _sv = _sv[:, :, 1]

            else:  # SVM, MLP Small, MLP Deep — KernelExplainer
                _bg       = _shap.kmeans(_X_tr_dense, 50)
                _pred_fn  = lambda x: _clf_s.predict_proba(x)[:, 1]
                _explainer = _shap.KernelExplainer(_pred_fn, _bg)
                _X_expl   = _X_te_dense[:_KERNEL_SAMPLES]
                _sv_raw   = _explainer.shap_values(_X_expl, nsamples=_KERNEL_NSAMPLES)
                _sv = _sv_raw if not isinstance(_sv_raw, list) else _sv_raw[1]
                _X_te_dense = _X_te_dense[:_KERNEL_SAMPLES]

            # Beeswarm plot
            _n_rows = _sv.shape[0]
            plt.figure(figsize=(10, 8))
            _shap.summary_plot(
                _sv, _X_te_dense[:_n_rows],
                feature_names=_feat_names,
                max_display=20, show=False, plot_type='dot',
            )
            plt.title(f'SHAP Summary — {_sname}\n(top 20 TF-IDF features, test set)')
            plt.tight_layout()
            plt.savefig(os.path.join(_SHAP_DIR,
                f'shap_beeswarm_{_sname.replace(" ", "_")}.png'),
                dpi=150, bbox_inches='tight')
            plt.close()

            # Bar chart
            _mean_abs = np.abs(_sv).mean(axis=0)
            _shap_mean_abs[_sname] = {k: float(v)
                                      for k, v in zip(_feat_names, _mean_abs)}
            _top_idx  = np.argsort(_mean_abs)[::-1][:20]
            _top_names = [_feat_names[i] for i in _top_idx]
            _top_vals  = _mean_abs[_top_idx]

            fig_bar, ax_bar = plt.subplots(figsize=(9, 6))
            ax_bar.barh(_top_names[::-1], _top_vals[::-1], color='#D85A30', alpha=0.85)
            ax_bar.set_xlabel('Mean |SHAP value|', fontsize=11)
            ax_bar.set_title(f'Top 20 TF-IDF Features — {_sname}',
                             fontsize=12, fontweight='bold')
            ax_bar.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_SHAP_DIR,
                f'shap_bar_{_sname.replace(" ", "_")}.png'),
                dpi=150, bbox_inches='tight')
            plt.close()

            _top5 = [_feat_names[i] for i in np.argsort(_mean_abs)[::-1][:5]]
            print(f"  Top 5 features: {_top5}")

        except Exception as _exc:
            print(f"  SHAP failed for {_sname}: {_exc}")

    # Cross-model heatmap
    if _shap_mean_abs:
        try:
            _all_feats = list(next(iter(_shap_mean_abs.values())).keys())
            _max_across = {f: max(_shap_mean_abs[m].get(f, 0.0) for m in _shap_mean_abs)
                           for f in _all_feats}
            _top30 = sorted(_max_across, key=_max_across.get, reverse=True)[:30]
            _hmap_df = pd.DataFrame(
                {m: [_shap_mean_abs[m].get(f, 0.0) for f in _top30]
                 for m in _shap_mean_abs},
                index=_top30,
            )
            fig_hm, ax_hm = plt.subplots(figsize=(13, 10))
            sns.heatmap(_hmap_df, annot=True, fmt='.3f', cmap='YlOrRd',
                        linewidths=0.4, ax=ax_hm, annot_kws={'size': 7})
            ax_hm.set_title(
                'Mean |SHAP value| — Top 30 TF-IDF features across all models',
                fontsize=12, pad=14)
            ax_hm.set_xlabel('Model', fontsize=10)
            ax_hm.set_ylabel('TF-IDF feature (word / n-gram)', fontsize=10)
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(_SHAP_DIR, 'shap_heatmap_cross_model.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("\nCross-model SHAP heatmap saved.")
        except Exception as _hm_exc:
            print(f"  WARNING: Cross-model heatmap failed: {_hm_exc}")

    print(f"\nSHAP outputs saved to {_SHAP_DIR}")

# Update metadata.json so the demo loads the correct model list and CV scores.
# Only the 'classical_ml' key is touched; bert/zeroshot sections are preserved.
import json
_meta = {}
if os.path.exists(MODELS_META):
    with open(MODELS_META) as _mf:
        _meta = json.load(_mf)
_indiv_model_names = ['Logistic Regression', 'SVM', 'Random Forest',
                      'XGBoost', 'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']
_meta['classical_ml'] = {
    'models':         _indiv_model_names,
    'cv_f1':          {m: round(float(results_df.loc[m, 'F1-Score']), 4)
                       for m in _indiv_model_names if m in results_df.index},
    'cv_auc':         {m: round(float(results_df.loc[m, 'ROC-AUC']), 4)
                       for m in _indiv_model_names if m in results_df.index},
    'ensemble_cv_f1': round(float(results_df.loc['Classical Ensemble', 'F1-Score']), 4)
                      if 'Classical Ensemble' in results_df.index else 0.0,
    'ensemble_cv_auc': round(float(results_df.loc['Classical Ensemble', 'ROC-AUC']), 4)
                       if 'Classical Ensemble' in results_df.index else 0.0,
}
with open(MODELS_META, 'w') as _mf:
    json.dump(_meta, _mf, indent=2)
print(f"Updated metadata.json with classical_ml section (8 models).")

# =============================================================================
# OOF Confusion Matrices — Classical ML
# =============================================================================
print("\n" + "-" * 60)
print("OOF Confusion Matrices — Classical ML")
print("-" * 60)

model_order_cm = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
                  'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost', 'Classical Ensemble']
ncols_cm, nrows_cm = 3, 3
fig_oof, axes_oof = plt.subplots(nrows_cm, ncols_cm, figsize=(ncols_cm * 4, nrows_cm * 3.8))
axes_oof = axes_oof.flatten()

from sklearn.metrics import confusion_matrix as _cm2
for i, mname in enumerate(model_order_cm):
    if mname not in oof_preds_store:
        axes_oof[i].set_visible(False)
        continue
    preds_oof = oof_preds_store[mname]
    cm_arr = _cm2(y_cv, preds_oof)
    tn, fp, fn, tp = cm_arr.ravel()
    f1_oof = f1_score(y_cv, preds_oof, average='weighted', zero_division=0)
    ax = axes_oof[i]
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
fig_oof.suptitle(f'OOF Confusion Matrices — Classical ML (80% train, N={len(y_cv)})',
                 fontsize=13, fontweight='bold')
plt.tight_layout()
_oof_cm_path = os.path.join(OUT_CLASSICAL, 'fig_cm_oof.png')
plt.savefig(_oof_cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved OOF confusion matrices to {_oof_cm_path}")

# =============================================================================
# Fold F1 Stability Chart — Classical ML
# =============================================================================
print("\nFold F1 stability chart...")
_model_list_fold = [m for m in model_order_cm if m in fold_f1s_store]
_n_folds_shown = len(fold_f1s_store[_model_list_fold[0]])
_x_base = np.arange(len(_model_list_fold))
_w = 0.12
_fold_colors = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D',
                '#2A7DB5', '#8B4EA0', '#3E8C4B', '#C45080']

fig_fold, ax_fold = plt.subplots(figsize=(16, 5))
for fi in range(_n_folds_shown):
    vals = [fold_f1s_store[m][fi] for m in _model_list_fold]
    ax_fold.bar(_x_base + fi * _w, vals, _w, label=f'Fold {fi+1}',
                color=_fold_colors[fi % len(_fold_colors)], alpha=0.80)
_means = [np.mean(fold_f1s_store[m]) for m in _model_list_fold]
ax_fold.plot(_x_base + (_n_folds_shown - 1) / 2 * _w, _means,
             'kD--', markersize=7, lw=1.5, label='Mean CV F1', zorder=5)
ax_fold.set_xticks(_x_base + (_n_folds_shown - 1) / 2 * _w)
ax_fold.set_xticklabels(_model_list_fold, rotation=25, ha='right', fontsize=10)
ax_fold.set_ylabel('Weighted F1-Score', fontsize=11)
ax_fold.set_ylim(0, 1.15)
ax_fold.set_title('Fold-by-Fold F1 Scores — Classical ML (5-fold Nested CV)',
                  fontsize=13, fontweight='bold')
ax_fold.legend(fontsize=9, ncol=3)
ax_fold.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
_fold_fig_path = os.path.join(OUT_CLASSICAL, 'fig_fold_stability.png')
plt.savefig(_fold_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved fold stability chart to {_fold_fig_path}")

# =============================================================================
# Test Set Evaluation — All Classical ML Models
# =============================================================================
print("\n" + "-" * 60)
print("Test Set Evaluation — Classical ML (20% held-out)")
print("-" * 60)

_X_test = X_test_global
_y_test = y_test_global
_n_test = len(_y_test)
print(f"Test set: {_n_test} samples | {_y_test.sum()} positive / {(_y_test==0).sum()} negative")

_test_results = {}
_roc_test = {}
_indiv_names = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost',
                'Decision Tree', 'MLP Small', 'MLP Deep', 'CatBoost']

for mname in _indiv_names:
    _pkl_path = os.path.join(MODELS_CLASSICAL, f'{mname.replace(" ", "_")}.pkl')
    if not os.path.exists(_pkl_path):
        print(f"  WARNING: {_pkl_path} not found, skipping")
        continue
    with open(_pkl_path, 'rb') as _f:
        _pipe_loaded = pickle.load(_f)
    _y_pred_t = _pipe_loaded.predict(_X_test)
    _y_prob_t = _pipe_loaded.predict_proba(_X_test)[:, 1]
    _test_results[mname] = {'y_pred': _y_pred_t, 'y_prob': _y_prob_t}
    _fpr_t, _tpr_t, _ = roc_curve(_y_test, _y_prob_t)
    _roc_test[mname] = (_fpr_t, _tpr_t, roc_auc_score(_y_test, _y_prob_t))
    _f1_t = f1_score(_y_test, _y_pred_t, average='weighted', zero_division=0)
    print(f"  {mname:<22}: test F1={_f1_t:.4f}  AUC={_roc_test[mname][2]:.4f}")

# Ensemble: soft vote (CV-F1-weighted) over individual pickles
_avail_ens = [m for m in _indiv_names if m in _test_results]
if len(_avail_ens) >= 2:
    _wts = np.array([results_df.loc[m, 'F1-Score'] if m in results_df.index else 1.0
                     for m in _avail_ens])
    _wts = _wts / _wts.sum()
    _prob_ens = sum(_test_results[m]['y_prob'] * w for m, w in zip(_avail_ens, _wts))
    _pred_ens = (_prob_ens >= 0.5).astype(int)
    _test_results['Classical Ensemble'] = {'y_pred': _pred_ens, 'y_prob': _prob_ens}
    _fpr_e, _tpr_e, _ = roc_curve(_y_test, _prob_ens)
    _roc_test['Classical Ensemble'] = (_fpr_e, _tpr_e, roc_auc_score(_y_test, _prob_ens))
    _f1_e = f1_score(_y_test, _pred_ens, average='weighted', zero_division=0)
    print(f"  {'Classical Ensemble':<22}: test F1={_f1_e:.4f}  AUC={_roc_test['Classical Ensemble'][2]:.4f}")

# Figure: CMs (top rows) + ROC curves (bottom)
_n_models_t = len(_test_results)
_ncols_t = 3
_nrows_t = (_n_models_t + _ncols_t - 1) // _ncols_t
_fig_test = plt.figure(figsize=(_ncols_t * 4.5, _nrows_t * 3.8 + 4.5))
_cm_axes = [_fig_test.add_subplot(_nrows_t + 1, _ncols_t, i + 1)
            for i in range(_n_models_t)]
_roc_ax = _fig_test.add_subplot(_nrows_t + 1, 1, _nrows_t + 1)

_roc_colors = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D',
               '#2A7DB5', '#8B4EA0', '#3E8C4B', '#C45080', '#444444']
for i, (mname, res) in enumerate(_test_results.items()):
    ax = _cm_axes[i]
    _cm_t = _cm2(_y_test, res['y_pred'])
    _tn_t, _fp_t, _fn_t, _tp_t = _cm_t.ravel()
    _f1_ti = f1_score(_y_test, res['y_pred'], average='weighted', zero_division=0)
    sns.heatmap(_cm_t, annot=False, ax=ax, cbar=False,
                cmap=sns.light_palette('#2E86AB', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for r in range(2):
        for c in range(2):
            val = _cm_t[r, c]
            lbl = [['TN', 'FP'], ['FN', 'TP']][r][c]
            ax.text(c + 0.5, r + 0.5, f'{lbl}\n{val}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val > _cm_t.max() * 0.5 else '#333333')
    ax.set_title(f'{mname}\nF1={_f1_ti:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)
    ax.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    ax.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

for mname, _clr in zip(_roc_test.keys(), _roc_colors):
    _fpr_t, _tpr_t, _auc_t = _roc_test[mname]
    _roc_ax.plot(_fpr_t, _tpr_t, color=_clr, lw=2,
                 label=f'{mname} (AUC={_auc_t:.4f})')
_roc_ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
_roc_ax.set_xlabel('False Positive Rate', fontsize=11)
_roc_ax.set_ylabel('True Positive Rate', fontsize=11)
_roc_ax.set_title('ROC Curves — Test Set', fontsize=12, fontweight='bold')
_roc_ax.legend(fontsize=8, loc='lower right', ncol=2)
_roc_ax.grid(True, alpha=0.3)

_fig_test.suptitle(f'Test Set Evaluation — Classical ML (20% held-out, N={_n_test})',
                   fontsize=14, fontweight='bold')
plt.tight_layout()
_test_fig_path = os.path.join(OUT_CLASSICAL, 'fig_test_eval.png')
plt.savefig(_test_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved test evaluation figure to {_test_fig_path}")

print("\nStep 2 complete.")
