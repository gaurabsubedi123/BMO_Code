"""
Reference baselines for the BMO re-run: the training-free AT(N) keyword rule and
the majority-class floor.

Both are deterministic functions of the prepared corpus, so no training is
involved. The keyword rule reuses `keyword_count`, written by
src/01_prepare_dataset.py from the AT(N) term list defined there; a record is
flagged positive when it contains at least one term. The majority-class floor
predicts the majority label of the training partition for every record.

Bootstrap intervals replicate 07_consolidated_evaluation exactly: percentile
method, 2000 resamples, seed 20260828, resampling test records.

Writes $BMO_RUN_DIR/tables/baselines.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, accuracy_score)

RUN = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
N_BOOT, RNG_SEED = 2000, 20260828

ds = pd.read_csv(RUN / "outputs/dataset/labeled_dataset.csv")
train_idx = np.load(RUN / "outputs/dataset/train_idx.npy")
test_idx = np.load(RUN / "outputs/dataset/test_idx.npy")

te = ds.loc[test_idx]
y = te["has_biomarker"].to_numpy().astype(int)
print(f"test partition: {len(y)} records, {y.sum()} positive ({y.mean()*100:.2f}%)")

majority_label = int(ds.loc[train_idx, "has_biomarker"].mode().iloc[0])
print(f"majority class on train: {majority_label}")

preds = {
    "Keyword rule (AT(N) terms)": (te["keyword_count"].to_numpy() > 0).astype(int),
    "Majority class": np.full(len(y), majority_label, dtype=int),
}


def boot_ci(y_true, y_pred, n=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx = np.arange(len(y_true))
    w, b = [], []
    for _ in range(n):
        s_ = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s_])) < 2:
            continue
        w.append(f1_score(y_true[s_], y_pred[s_], average="weighted", zero_division=0))
        b.append(f1_score(y_true[s_], y_pred[s_], pos_label=1, zero_division=0))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(w) + q(b)


rows = []
for name, yhat in preds.items():
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    wlo, whi, blo, bhi = boot_ci(y, yhat)
    rows.append(dict(
        System=name, Tier="Baseline",
        weighted_f1=round(f1_score(y, yhat, average="weighted", zero_division=0), 6),
        biomarker_f1=round(f1_score(y, yhat, pos_label=1, zero_division=0), 6),
        precision=round(precision_score(y, yhat, pos_label=1, zero_division=0), 6),
        recall=round(recall_score(y, yhat, pos_label=1, zero_division=0), 6),
        accuracy=round(accuracy_score(y, yhat), 6),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        errors=int(fp + fn), n=int(len(y)),
        wf1_lo=round(wlo, 6), wf1_hi=round(whi, 6),
        bf1_lo=round(blo, 6), bf1_hi=round(bhi, 6),
        Note="training-free; deterministic given the corpus"))
    print(f"  {name:28} wF1={rows[-1]['weighted_f1']:.4f} "
          f"[{wlo:.4f}, {whi:.4f}]  errors={rows[-1]['errors']} "
          f"(FP {fp} / FN {fn})")

out = pd.DataFrame(rows)
dest = RUN / "tables/baselines.csv"
out.to_csv(dest, index=False)
print(f"\nwrote {dest}")
