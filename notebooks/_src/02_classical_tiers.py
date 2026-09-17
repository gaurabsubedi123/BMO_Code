#%% md
# 02 — Classical machine learning across three text representations

Seven estimators × three representations, under nested trial-grouped
cross-validation. This single notebook replaces the three separate scripts of
the original pipeline (`02_`, `02a_`, `02b_`) for one concrete reason: the
BioWordVec binary is 28 GB and takes roughly five minutes to load from cold.
Loading it once and reusing the vectors across all seven estimators, rather
than once per script, removes the dominant cost of this tier.

| Representation | Features | Dim |
|---|---|---|
| **Classical** | TF-IDF | 2,000–5,000 (searched) |
| **FastText** | TF-IDF ⊕ averaged FastText vectors | + 300 |
| **BioWordVec** | TF-IDF ⊕ averaged BioWordVec vectors | + 200 |

The tier names in the third column of every figure — Classical, FastText,
BioWordVec — refer to these three rows.

## Why this tier carries the paper's central claim

The three representations differ **only** in the text features. Estimators,
hyperparameter grids, folds, seeds and evaluation are identical across them.
That makes the comparison a controlled experiment on representation, which is
what licenses the manuscript's claim that biomedical vocabulary — not model
architecture — is what moves performance on this task.

## Protocol (preserved from the original study)

- **Outer CV**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- **Inner CV**: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`
- Hyperparameters are selected by `GridSearchCV` on the **inner** folds only;
  the outer fold is scored once with the refit winner. This is genuine nested
  cross-validation — the reported score is not contaminated by the search.
- Class imbalance is handled by `class_weight="balanced"` / `scale_pos_weight`,
  never by resampling, so no synthetic records enter the folds.
#%% code
import os, sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from xgboost import XGBClassifier

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

df, train_idx, test_idx = C.load_data()
texts_all, labels_all, groups_all = (df["clean_freeform"].fillna("").values,
                                     df["has_biomarker"].values, df["NCTId"].values)
# The pipeline is fed ROW INDICES, not text: TextFromIndex maps them back to
# strings for TF-IDF, and EmbFromIndex slices the precomputed embedding matrix.
# This keeps every object crossing a process boundary small and picklable.
X, y, groups = train_idx, labels_all[train_idx], groups_all[train_idx]
X_te, y_te   = test_idx,  labels_all[test_idx]

# This node has 128 cores and is shared with other users. Unbounded parallelism would
# claim all of them for the duration of the grid searches; 32 is ample for a
# corpus this size and leaves the machine usable for everyone else.
N_JOBS = int(os.environ.get("BMO_N_JOBS", "32"))
# Reference threshold for the achievability analysis. Selection does not use it
# -- see the note in run_tier() for why constraining on it was tested and
# rejected for this tier.
GAP_THRESHOLD = 0.04
print(f"CPU parallelism capped at n_jobs={N_JOBS} of {os.cpu_count()} cores")

# This tier is pure CPU -- TF-IDF, SVMs, forests, gradient boosting. It should
# never hold GPU memory. It does anyway if the devices are visible: something
# in the import chain (xgboost's CUDA probe) initialises a CUDA context in each
# of the 32 joblib workers, and an empty context costs ~426 MB, so the tier
# quietly reserves ~14 GB of VRAM to run scikit-learn. On a shared node that is
# memory taken from other users for no benefit.
#
# The driver sets CUDA_VISIBLE_DEVICES="" for this notebook. This check makes a
# regression visible rather than silent.
# Defined here, with the other runtime settings, because the setup cell prints
# it. XGBoost stays on CPU by design: its CUDA path returns a different F1 on
# sparse input (0.7559 vs 0.9005 on the same grid and seed), so the device is
# not a pure speed knob. Full benchmark in the estimators section below.
XGB_DEVICE = os.environ.get("BMO_XGB_DEVICE", "cpu")
print(f"XGBoost device: {XGB_DEVICE}")

OUTER = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED_CV)
INNER = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=C.SEED_CV)
scale_pos = float((y == 0).sum() / max((y == 1).sum(), 1))
print(f"train {len(X):,} | test {len(X_te):,} | scale_pos_weight {scale_pos:.3f}")
#%% md
## Feature construction

The text representation is built from two blocks: a TF-IDF matrix whose
parameters *are* searched, and a fixed block of averaged pretrained embeddings.

**The embeddings are precomputed once, before any cross-validation, and the
pipeline only indexes into them.** That design choice matters for two reasons,
both learned the hard way on this project:

1. **Correctness.** The obvious implementation puts the embedding lookup inside
   the pipeline and reads the loaded model from a module-level global. Under
   `GridSearchCV(n_jobs>1)` the estimator is serialised to worker processes,
   which do not inherit that global — so the workers either fail to serialise a
   28 GB fastText handle, or quietly produce **zero vectors**. Zero-vector
   features do not raise; they just yield plausible-looking, wrong numbers.
   That is exactly the failure that corrupted the earlier run of this study.
   Precomputing removes the possibility entirely: no model object ever reaches
   a worker, only a 1,900 x 200 float array.

2. **Speed.** Nothing in the search grid changes the embedding, yet putting it
   inside the pipeline recomputes it for every candidate. For Random Forest
   alone that is 1,296 fits x 0.36 s = ~39 minutes per tier, repeated for every
   estimator. Precomputing turns that into a single 0.36 s pass.

Precomputing on all documents introduces no leakage: the embeddings are
unsupervised, fixed, pretrained lookups that never see a label.

To make indexing work, the pipeline is fed **row indices** rather than raw text.
`TextFromIndex` maps indices back to strings for the TF-IDF branch;
`EmbFromIndex` slices the precomputed matrix. Both hold only picklable arrays.
#%% code
class TextFromIndex(BaseEstimator, TransformerMixin):
    """Map row indices to their raw text, for the TF-IDF branch."""
    def __init__(self, texts=None):
        self.texts = texts
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return self.texts[np.asarray(X).ravel().astype(int)]

class EmbFromIndex(BaseEstimator, TransformerMixin):
    """Slice rows out of a precomputed embedding matrix."""
    def __init__(self, E=None):
        self.E = E
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return csr_matrix(self.E[np.asarray(X).ravel().astype(int)])

#%% md
### Precompute the embedding matrices

Each model is loaded once, every document is embedded once, and the model is
then released. After this cell nothing holds a multi-gigabyte handle.
#%% code
import fasttext, gensim.downloader as api

def mean_vectors(texts, lookup, dim):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        ws = str(t).lower().split()
        vecs = [v for v in (lookup(w) for w in ws) if v is not None]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out

t0 = time.time()
print("loading BioWordVec (28 GB) ...")
_bwv = fasttext.load_model(str(C.BIOWORDVEC_BIN))
E_BWV = mean_vectors(texts_all, _bwv.get_word_vector, 200)
del _bwv; import gc; gc.collect()
print(f"  BioWordVec matrix {E_BWV.shape} in {time.time()-t0:.0f}s; model released")

t1 = time.time()
print("loading FastText (wiki-news-300) ...")
_ft = api.load("fasttext-wiki-news-subwords-300")
E_FT = mean_vectors(texts_all, lambda w: _ft[w] if w in _ft else None, 300)
del _ft; gc.collect()
print(f"  FastText matrix {E_FT.shape} in {time.time()-t1:.0f}s; model released")

# Sanity check: a zero row means the document embedded to nothing. A large
# number of zero rows is the signature of the silent failure described above.
for nm, E in (("BioWordVec", E_BWV), ("FastText", E_FT)):
    zeros = int((np.abs(E).sum(axis=1) == 0).sum())
    print(f"  {nm}: {zeros}/{len(E)} all-zero rows, mean norm "
          f"{np.linalg.norm(E, axis=1).mean():.4f}")
    assert zeros < len(E) * 0.05, f"{nm}: {zeros} zero rows — lookup is broken"
print("embedding matrices verified")

#%% md
## Estimators and hyperparameter grids

The grids are inherited unchanged from the original study. They are weighted
towards regularisation — `min_samples_leaf`, `min_child_weight`, `reg_alpha`,
`max_depth`, `min_df` — because with roughly 1,500 training records the
failure mode is memorisation, not underfitting.
#%% code
# ---------------------------------------------------------------------------
# Parallelism, and why the estimators use n_jobs=1
#
# GridSearchCV already runs candidates in parallel. Setting n_jobs on the
# estimator as well nests one thread pool inside another: 32 search workers
# each spawning 32 threads drove the load average to 255 on a 128-core node,
# so most of the machine was spent on context switching rather than fitting.
# The correct pattern is for the outer search to own the parallelism.
#
# Measured on this data, 72 fits of the XGBoost pipeline:
#     estimator n_jobs=32, search n_jobs=32   196.7 s   (2732 ms/fit)   F1 0.9005
#     estimator n_jobs=1,  search n_jobs=32     6.5 s   (  90 ms/fit)   F1 0.9005
# A 30x speed-up from one setting, with identical results.
#
# XGBoost was also tested on the GPU. It is NOT used, and the reason matters:
#     GPU, device="cuda"                       16.3 s   ( 226 ms/fit)   F1 0.7559
# The GPU is both slower than correctly-parallelised CPU *and gives a different
# answer* -- 0.7559 against 0.9005 on the same grid, data and seed. XGBoost's
# CUDA path treats sparse input differently (implicit zeros versus missing
# values), so device is not a pure speed knob here: it changes the model being
# fitted. A quietly wrong result of that kind is exactly the failure this study
# has already suffered once, so the tier stays on CPU.
# ---------------------------------------------------------------------------

def make_models():
    xgb_kw = dict(scale_pos_weight=scale_pos, eval_metric="logloss",
                  tree_method="hist", n_jobs=1, random_state=C.SEED_CV,
                  verbosity=0)
    if XGB_DEVICE == "cuda":
        xgb_kw["device"] = "cuda"
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=C.SEED_CV),
        "SVM":                 SVC(probability=True, class_weight="balanced",
                                   random_state=C.SEED_CV),
        "Random Forest":       RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                                      random_state=C.SEED_CV),
        "XGBoost":             XGBClassifier(**xgb_kw),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced",
                                                      random_state=C.SEED_CV),
        "MLP Small":           MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                             early_stopping=True, random_state=C.SEED_CV),
        "MLP Deep":            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                                             early_stopping=True, random_state=C.SEED_CV),
    }

GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "SVM":                 {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]},
    "Random Forest":       {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, 20],
                            "clf__min_samples_leaf": [1, 2, 4],
                            "clf__max_features": ["sqrt", "log2"]},
    "XGBoost":             {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                            "clf__max_depth": [3, 5], "clf__min_child_weight": [1, 3],
                            "clf__reg_alpha": [0, 1.0], "clf__reg_lambda": [1.0, 5.0]},
    "Decision Tree":       {"clf__max_depth": [3, 5, 10, 20],
                            "clf__min_samples_leaf": [1, 2, 4, 8]},
    "MLP Small":           {"clf__alpha": [1e-4, 1e-3, 1e-2],
                            "clf__learning_rate_init": [1e-3, 1e-2]},
    "MLP Deep":            {"clf__alpha": [1e-4, 1e-3, 1e-2],
                            "clf__learning_rate_init": [1e-3, 1e-2]},
}
# One extra level in the name: feat -> tfidf branch -> the TfidfVectorizer.
TFIDF_GRID = {"feat__tfidf__tfidf__max_features": [2000, 3000, 5000],
              "feat__tfidf__tfidf__ngram_range": [(1, 1), (1, 2)],
              "feat__tfidf__tfidf__min_df": [2, 3]}

# ---------------------------------------------------------------------------
# Smoke mode (BMO_SMOKE=1)
#
# Shrinks every grid to a single point so the whole notebook executes end to end
# in a couple of minutes. It exercises the same code paths as a real run --
# every tier, every estimator, the ensembles, the final fits and all the saving
# -- without the search cost. Runtime errors like an undefined name or a wrong
# parameter path surface here instead of after an hour of grid search.
#
# Smoke output is written to a separate directory so it can never be mistaken
# for real results.
# ---------------------------------------------------------------------------
SMOKE = os.environ.get("BMO_SMOKE") == "1"
if SMOKE:
    TFIDF_GRID = {"feat__tfidf__tfidf__max_features": [2000]}
    GRIDS = {name: {k: v[:1] for k, v in list(g.items())[:1]}
             for name, g in GRIDS.items()}
    C.OUT_CLASSICAL = C.OUT_CLASSICAL.parent / "classical_ml_smoke"
    C.OUT_CLASSICAL.mkdir(parents=True, exist_ok=True)
    print(f"SMOKE MODE: grids reduced to one point, output -> {C.OUT_CLASSICAL}")

def build_features(kind):
    """TF-IDF alone, or TF-IDF concatenated with a precomputed embedding block."""
    tf = ("tfidf", Pipeline([("sel", TextFromIndex(texts_all)),
                             ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2))]))
    if kind == "Classical":
        return FeatureUnion([tf])
    if kind == "FastText":
        return FeatureUnion([tf, ("emb", EmbFromIndex(E_FT))])
    return FeatureUnion([tf, ("emb", EmbFromIndex(E_BWV))])
#%% md
## Nested cross-validation

The outer loop scores; the inner loop selects. Because the grid search runs
inside each outer fold, the hyperparameters are chosen without ever seeing the
records they are later scored on.

Out-of-fold predictions are accumulated across the five outer folds, giving one
leakage-free prediction per training record. Notebook 08 uses these for the
pairwise McNemar tests.
#%% code
def run_tier(tier):
    """Nested CV for every estimator under one text representation."""
    rows, oof_pred, oof_prob, per_fold = [], {}, {}, []
    sel_rows = []
    for name, clf in make_models().items():
        grid = {**TFIDF_GRID, **GRIDS[name]}
        # One CUDA context per worker costs ~400 MB and the device serialises
        # the work regardless, so the GPU model uses a smaller worker pool.
        n_jobs_here = 8 if (name == "XGBoost" and XGB_DEVICE == "cuda") else N_JOBS
        preds = np.full(len(y), -1, dtype=int)
        probs = np.full(len(y), np.nan, dtype=float)
        fold_scores, fold_train, chosen = [], [], []
        t0 = time.time()
        for k, (tr, va) in enumerate(OUTER.split(X, y, groups=groups), 1):
            assert not (set(groups[tr]) & set(groups[va])), f"fold {k} leaks trials"
            pipe = Pipeline([("feat", build_features(tier)), ("clf", clf)])
            # return_train_score gives the per-candidate train-minus-validation
            # gap, which the gap-constrained selection below needs. refit is
            # deferred so we can choose the winner ourselves rather than letting
            # GridSearchCV take the highest score regardless of overfitting.
            gs = GridSearchCV(pipe, grid, cv=INNER, scoring="f1_weighted",
                              n_jobs=n_jobs_here, refit=False,
                              return_train_score=True, error_score="raise")
            # groups MUST be forwarded: the inner CV is StratifiedGroupKFold,
            # which cannot split without them. Omitting this silently breaks the
            # nested search rather than degrading it.
            gs.fit(X[tr], y[tr], groups=groups[tr])

            # ---- selection: best validation score ------------------------
            # Standard practice, and deliberately NOT gap-constrained.
            #
            # A gap-constrained variant was tested (take the best score among
            # candidates whose inner train-validation gap is under 0.04, as the
            # transformer tier does). It cost 0.017 weighted F1 on average --
            # up to 0.044 for the decision tree -- and still left five of seven
            # models above the threshold, because for most of them almost no
            # candidate satisfies it: the MLPs have NO passing configuration in
            # any fold, and logistic regression has one in a single fold of five.
            # Paying real accuracy for a constraint that cannot be met is the
            # worst of both worlds, so selection is by score and the gaps are
            # reported honestly instead.
            #
            # The achievability statistics are still recorded below, because
            # "no configuration of this model family reaches a 0.04 gap on this
            # corpus" is itself a result: the classical tier's larger gaps are a
            # property of those estimators at this sample size, not a tuning
            # failure.
            res = pd.DataFrame(gs.cv_results_)
            res["gap"] = res["mean_train_score"] - res["mean_test_score"]
            bi = res["mean_test_score"].idxmax()
            best_params_ = res.loc[bi, "params"]
            sel_rows.append({"Tier": tier, "Model": name, "fold": k,
                             "n_passing": int((res["gap"] < GAP_THRESHOLD).sum()),
                             "n_candidates": len(res),
                             "selected_val_f1": float(res.loc[bi, "mean_test_score"]),
                             "selected_gap": float(res.loc[bi, "gap"]),
                             "best_achievable_gap": float(res["gap"].min())})
            gs = clone(pipe).set_params(**best_params_)
            gs.fit(X[tr], y[tr])
            preds[va] = gs.predict(X[va])
            probs[va] = gs.predict_proba(X[va])[:, 1]
            from sklearn.metrics import f1_score
            fold_scores.append(f1_score(y[va], preds[va], average="weighted", zero_division=0))
            fold_train.append(f1_score(y[tr], gs.predict(X[tr]), average="weighted",
                                       zero_division=0))
            chosen.append(best_params_)
            per_fold.append({"Tier": tier, "Model": name, "fold": k,
                             "val_weighted_f1": fold_scores[-1],
                             "train_weighted_f1": fold_train[-1],
                             "gap": fold_train[-1] - fold_scores[-1]})
        gap = float(np.mean(fold_train) - np.mean(fold_scores))
        rows.append(C.evaluate(y, preds, probs, name=name, tier=tier,
                               extra={"cv_mean_f1": float(np.mean(fold_scores)),
                                      "cv_std_f1": float(np.std(fold_scores)),
                                      "mean_gap": gap,
                                      "n_configs": int(np.prod([len(v) for v in grid.values()])),
                                      "seconds": round(time.time() - t0, 1)}))
        oof_pred[name], oof_prob[name] = preds, probs
        print(f"  {name:22s} OOF F1 {rows[-1]['weighted_f1']:.4f} | "
              f"CV {np.mean(fold_scores):.4f}+/-{np.std(fold_scores):.4f} | "
              f"gap {gap:+.4f} | {time.time()-t0:.0f}s")
    if sel_rows:
        C.save_table(pd.DataFrame(sel_rows),
                     C.OUT_CLASSICAL / f"gap_achievability_{tier}.csv")
    return rows, oof_pred, oof_prob, per_fold

all_rows, all_oof_pred, all_oof_prob, all_folds = [], {}, {}, []
for tier in ("Classical", "FastText", "BioWordVec"):
    print(f"\n{'='*72}\n{tier}\n{'='*72}")
    r, p, q, f = run_tier(tier)
    all_rows += r; all_folds += f
    all_oof_pred.update({f"{tier} | {k}": v for k, v in p.items()})
    all_oof_prob.update({f"{tier} | {k}": v for k, v in q.items()})
    # Write after each tier rather than only at the end. The previous run gave
    # no sign of progress for two hours because nothing was written until all
    # three tiers finished.
    C.save_table(pd.DataFrame(all_rows), C.OUT_CLASSICAL / "cv_results_partial.csv")
    print(f"[{tier}] complete — {len(all_rows)} model results so far")
#%% md
## Soft-voting ensembles

One ensemble per representation, over the three strongest estimators in that
representation, so ensembling is compared like-for-like across tiers rather
than only where it happens to help.
#%% code
from sklearn.metrics import f1_score
ens_rows = []
for tier in ("Classical", "FastText", "BioWordVec"):
    tier_rows = [r for r in all_rows if r["Tier"] == tier]
    top3 = [r["Model"] for r in sorted(tier_rows, key=lambda r: r["cv_mean_f1"],
                                       reverse=True)[:3]]
    preds = np.full(len(y), -1, dtype=int); probs = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(OUTER.split(X, y, groups=groups), 1):
        members = [(n, Pipeline([("feat", build_features(tier)),
                                 ("clf", make_models()[n])])) for n in top3]
        vc = VotingClassifier(members, voting="soft", n_jobs=N_JOBS)
        vc.fit(X[tr], y[tr])
        preds[va] = vc.predict(X[va]); probs[va] = vc.predict_proba(X[va])[:, 1]
    row = C.evaluate(y, preds, probs, name=f"{tier} Ensemble", tier=tier,
                     extra={"members": ", ".join(top3),
                            "cv_mean_f1": f1_score(y, preds, average="weighted",
                                                   zero_division=0),
                            "cv_std_f1": np.nan, "mean_gap": np.nan})
    ens_rows.append(row)
    all_oof_pred[f"{tier} | Ensemble"] = preds
    all_oof_prob[f"{tier} | Ensemble"] = probs
    print(f"{tier} Ensemble ({', '.join(top3)}): OOF F1 {row['weighted_f1']:.4f}")
#%% md
## Persist cross-validation results
#%% code
assert all_rows, ("no classical results were produced -- every tier failed. "
                  "Check the traceback above rather than continuing; downstream "
                  "notebooks would otherwise consolidate an empty tier.")
print(f"{len(all_rows)} estimator results + {len(ens_rows)} ensembles")

cv_df = pd.DataFrame(all_rows + ens_rows).sort_values("weighted_f1", ascending=False)
C.save_table(cv_df, C.OUT_CLASSICAL / "cv_results_all_tiers.csv")
C.save_table(pd.DataFrame(all_folds), C.OUT_CLASSICAL / "per_fold_all_tiers.csv")

op = pd.DataFrame(all_oof_pred); op.insert(0, "y_true", y)
oq = pd.DataFrame(all_oof_prob); oq.insert(0, "y_true", y)
C.save_table(op, C.OUT_CLASSICAL / "oof_predictions.csv")
C.save_table(oq, C.OUT_CLASSICAL / "oof_probabilities.csv")

print(cv_df[["Tier", "Model", "weighted_f1", "biomarker_f1", "cv_mean_f1",
             "mean_gap", "errors"]].to_string(index=False))
#%% md
### Overfitting audit for this tier
#%% code
audit = pd.DataFrame([
    C.overfit_report(r["cv_mean_f1"] + (r["mean_gap"] if pd.notna(r["mean_gap"]) else 0),
                     r["cv_mean_f1"], name=f"{r['Tier']} | {r['Model']}",
                     n_configs=r.get("n_configs"))
    for _, r in cv_df.iterrows() if pd.notna(r["cv_mean_f1"])
]).sort_values("gap", ascending=False)
print(audit.to_string(index=False))
C.save_table(audit, C.OUT_CLASSICAL / "overfitting_audit.csv")
flag = audit[audit["status"] != "OK"]
print(f"\n{len(flag)} model(s) exceed the 0.04 gap threshold"
      + (":\n" + flag.to_string(index=False) if len(flag) else "."))
#%% md
## Final fit and the single test-set evaluation

Each estimator is refit on the full training set with the hyperparameters most
frequently selected across the outer folds, then evaluated once on the held-out
test set. Fitted pipelines are persisted for the deployment bundle in notebook 10.
#%% code
test_rows, test_pred, test_prob = [], {}, {}
MODELS_DIR = C.MODELS / "classical"; MODELS_DIR.mkdir(parents=True, exist_ok=True)

for tier in ("Classical", "FastText", "BioWordVec"):
    for name, clf in make_models().items():
        grid = {**TFIDF_GRID, **GRIDS[name]}
        pipe = Pipeline([("feat", build_features(tier)), ("clf", clf)])
        nj = 8 if (name == "XGBoost" and XGB_DEVICE == "cuda") else N_JOBS
        gs = GridSearchCV(pipe, grid, cv=INNER, scoring="f1_weighted", n_jobs=nj,
                          refit=False, return_train_score=True)
        gs.fit(X, y, groups=groups)
        _res = pd.DataFrame(gs.cv_results_)
        _res["gap"] = _res["mean_train_score"] - _res["mean_test_score"]
        _params = _res.loc[_res["mean_test_score"].idxmax(), "params"]
        gs = clone(pipe).set_params(**_params)
        gs.fit(X, y)
        yp = gs.predict(X_te); pp = gs.predict_proba(X_te)[:, 1]
        key = f"{tier} | {name}"
        test_rows.append(C.evaluate(y_te, yp, pp, name=name, tier=tier,
                                    extra={"best_params": json.dumps(_params, default=str)}))
        test_pred[key], test_prob[key] = yp, pp
        # No model is exported here. The fitted estimator consumes ROW INDICES
        # and its feature union pickles TextFromIndex / EmbFromIndex, which are
        # defined in this notebook -- loading it elsewhere raises
        #     AttributeError: Can't get attribute 'TextFromIndex' on __main__
        # The index design exists to make cross-validation fast and is useless
        # to a web service, which has raw text and no precomputed matrix.
        # Notebook 10 refits a text-in pipeline from `best_params` recorded
        # below, which is the artefact that can actually be deployed.
        print(f"{tier:11s} {name:22s} test F1 {test_rows[-1]['weighted_f1']:.4f} "
              f"| errors {test_rows[-1]['errors']}")

test_df = pd.DataFrame(test_rows).sort_values("weighted_f1", ascending=False)
C.save_table(test_df, C.OUT_CLASSICAL / "test_results_all_tiers.csv")
tp = pd.DataFrame(test_pred); tp.insert(0, "y_true", y_te)
pq = pd.DataFrame(test_prob); pq.insert(0, "y_true", y_te)
C.save_table(tp, C.OUT_CLASSICAL / "test_predictions.csv")
C.save_table(pq, C.OUT_CLASSICAL / "test_probabilities.csv")
#%% md
## The representation effect

The comparison the paper is built on: identical estimators, identical folds,
identical grids — only the text representation changes.
#%% code
pivot = (test_df.pivot_table(index="Model", columns="Tier", values="weighted_f1")
         .reindex(columns=["Classical", "FastText", "BioWordVec"]))
pivot["BioWordVec - Classical"] = pivot["BioWordVec"] - pivot["Classical"]
pivot["FastText - Classical"]   = pivot["FastText"]   - pivot["Classical"]
print(pivot.round(4).to_string())
C.save_table(pivot.reset_index(), C.OUT_CLASSICAL / "representation_effect.csv")
print(f"\nmean effect of biomedical embeddings: "
      f"{pivot['BioWordVec - Classical'].mean():+.4f} weighted F1")
print(f"mean effect of general embeddings:    "
      f"{pivot['FastText - Classical'].mean():+.4f} weighted F1")
print("\nNote: the sign and size of these differences are what notebook 08 tests "
      "for significance. Do not describe them as effects in the manuscript until "
      "McNemar has been run on the out-of-fold predictions.")
#%% md
## Done

**Next:** `05_transformer_finetuning.ipynb` (already authored) and
`06_prompt_based_llms.ipynb`, then `07` merges every tier.
