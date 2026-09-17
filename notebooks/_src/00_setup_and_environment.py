#%% md
# 00 — Setup, environment, and data provenance

**Biomarker or Not (BMO): classifying biomarker outcome measures in Alzheimer's disease clinical trials**

Entry point for the full re-run. This notebook:

1. Records the exact software and hardware environment.
2. Fixes every random seed used anywhere in the pipeline.
3. Checksums the input data and the fixed train/test split, so every later
   notebook can assert it is working from the same bytes.
4. Writes `bmo_common.py` — the shared module later notebooks import for paths,
   metrics, diagnostics, and the figure style.

`bmo_common.py` is *authored here*, in the cell below, and written out as a
file. Nothing is hidden in an unversioned helper: all modelling code lives in
the notebook that owns it, and the only shared code is what genuinely needs one
definition (paths, the metric function, the plot style).

## Experimental protocol

The most important property of this pipeline is that **the test set is touched
exactly once per model, at the very end.** Prompt selection, learning-rate
search, epoch selection and threshold tuning all happen inside the training
portion only.

| Stage | Data used | Seed | Purpose |
|---|---|---|---|
| Global split | all 1,900 records, grouped by `NCTId` | 42 | 80/20 train/test, no trial on both sides |
| Hyperparameter search | inner 80/20 holdout of *train only* | 99 | pick LR, batch size, epochs |
| Cross-validation | 5-fold, group-aware, *train only* | 42 | the honest estimate we report |
| Final fit | full train | 42 | the model evaluated on test |
| Test evaluation | held-out 20% | — | reported once, never used for any decision |

Grouping by `NCTId` matters: one trial contributes many outcome measures with
near-identical phrasing. A record-level split would leak that phrasing across
the train/test boundary and inflate every number in the paper.

**Why the tuning seed differs from the CV seed.** If hyperparameters were
chosen on the same folds used to report cross-validated performance, the
reported CV score would be optimistically biased — it would partly measure how
well we searched, not how well the model generalises. Seed 99 carves an inner
holdout that is disjoint from the CV folds, so the CV number stays honest.
#%% code
import os, sys, json, platform, subprocess, hashlib
from pathlib import Path

RUN_DIR = Path(os.environ.get(
    "BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
print("Run directory:", RUN_DIR)
assert RUN_DIR.exists(), f"Run directory missing: {RUN_DIR}"

# All large artefacts live on the research NFS mount, never on the node's
# local disk: model weights, the HF cache and the BioWordVec binary are all
# shared resources and the node's root filesystem is not sized for them.
print("HF cache      :", os.environ.get("HF_HOME", "(unset — see env.sh)"))
print("BioWordVec bin:", os.environ.get("BIOWORDVEC_BIN", "(unset — see env.sh)"))
#%% md
## 1. Environment capture

Written verbatim to `outputs/provenance/environment.json`. The original
training environment for this project was lost, which left the earlier saved
pipelines unloadable and cost significant time to diagnose. Capturing this now
prevents a repeat.
#%% code
import torch, transformers, sklearn, numpy as np, pandas as pd, scipy, xgboost

def gpu_table():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip()
        return [l.strip() for l in out.split("\n")]
    except Exception as e:
        return [f"nvidia-smi unavailable: {e}"]

ENV = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "n_gpu": torch.cuda.device_count(),
    "gpus": gpu_table(), "transformers": transformers.__version__,
    "scikit_learn": sklearn.__version__, "numpy": np.__version__,
    "pandas": pd.__version__, "scipy": scipy.__version__, "xgboost": xgboost.__version__,
}
for k, v in ENV.items():
    print(f"{k:16s} {v}")
#%% md
## 2. Global seeds

Every stochastic component is seeded from these constants, imported by every
downstream notebook rather than redefined.
#%% code
SEED_SPLIT, SEED_CV, SEED_TUNE = 42, 42, 99
N_FOLDS, N_BOOTSTRAP = 5, 1000

def set_all_seeds(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return seed

set_all_seeds(SEED_SPLIT)
print(f"seeds: split={SEED_SPLIT} cv={SEED_CV} tune={SEED_TUNE} | folds={N_FOLDS}")
#%% md
### On determinism — a caveat that belongs in the manuscript

We seed everything, but deliberately do **not** set
`torch.use_deterministic_algorithms(True)`: fully deterministic cuDNN kernels
would slow fine-tuning substantially, and non-determinism at the fourth decimal
place changes no conclusion here.

What this means concretely: **re-running the transformer tier reproduces metrics
to roughly ±0.005 weighted F1, not bit-for-bit.** An earlier run of this project
was retrained without recording that fact, and reconciling the two sets of
numbers afterwards was expensive. Every figure in this run therefore ships with
the manifest written by notebook 09, and the manuscript should state the
tolerance rather than implying exact reproducibility.
#%% md
## 3. Data provenance and checksums

`labeled_dataset.csv` and the two index files are the inputs to everything
downstream. Hashing them here — and re-asserting the hashes in each later
notebook — makes it impossible to silently pool results computed on different
dataset versions. That is precisely the failure mode that produced conflicting
numbers across earlier runs of this project.
#%% code
DATA_DIR = RUN_DIR / "outputs" / "dataset"

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

INPUTS = ["labeled_dataset.csv", "train_idx.npy", "test_idx.npy"]
CHECKSUMS = {n: sha256(DATA_DIR / n) for n in INPUTS}
for n, d in CHECKSUMS.items():
    print(f"{d}  {n}")
#%% code
df = pd.read_csv(DATA_DIR / "labeled_dataset.csv")
train_idx = np.load(DATA_DIR / "train_idx.npy")
test_idx  = np.load(DATA_DIR / "test_idx.npy")

print(f"records             {len(df):,}")
print(f"trials (NCTId)      {df['NCTId'].nunique():,}")
print(f"positive rate       {df['has_biomarker'].mean()*100:.2f}%")
print(f"train / test        {len(train_idx):,} / {len(test_idx):,}")
print(f"train positive rate {df['has_biomarker'].values[train_idx].mean()*100:.2f}%")
print(f"test  positive rate {df['has_biomarker'].values[test_idx].mean()*100:.2f}%")

overlap = set(df["NCTId"].values[train_idx]) & set(df["NCTId"].values[test_idx])
assert not overlap, f"LEAK: {len(overlap)} trials span the train/test boundary"
assert not (set(train_idx) & set(test_idx)), "LEAK: overlapping record indices"
print("\nsplit integrity     OK — trial-disjoint, no shared records")
#%% md
## 4. Figure design rules

Every figure in this paper must satisfy three constraints simultaneously:

1. **Colour-blind safe.** Palette is Okabe–Ito, which is distinguishable under
   deuteranopia, protanopia and tritanopia.
2. **Legible when printed in black and white.** This is the stricter
   constraint, and it is why colour is never the *only* channel carrying
   information. Bars additionally differ by **hatch pattern**; lines
   additionally differ by **marker shape and dash pattern**. Convert any figure
   here to greyscale and every series remains separable.
3. **Ordered luminance.** The palette is arranged so that tiers also differ in
   greyscale *value*, not just hue — a second layer of redundancy.

Conventions used throughout, chosen to match the manuscript:

- Model groups are named **Classical / FastText / BioWordVec / Transformer /
  Prompt-based** (defined once in each caption).
- Model names are readable short forms (`BioWordVec + MLP Deep`, `PubMedBERT`);
  full checkpoint identifiers live in the Methods table, not on axes.
- The main comparison figure is a **horizontal bar chart sorted by score**, best
  at top, so 40 long model names read left-to-right without rotation.
- Confusion matrix classes are labelled **No biomarker / Biomarker**.
#%% md
## 5. Write `bmo_common.py`
#%% code
BMO_COMMON = r'''"""
bmo_common.py — shared helpers for the BMO pipeline.

Authored in notebook 00_setup_and_environment.ipynb and written out from there.
Do not edit this file directly; edit the notebook and re-run it.
"""
import os, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef, brier_score_loss)

# ------------------------------------------------------------------ paths
RUN_DIR   = Path(os.environ.get("BMO_RUN_DIR",
                                "/mnt/subedi-research/projects/bmo_rerun_2026"))
DATA_DIR  = RUN_DIR / "outputs" / "dataset"
OUT       = RUN_DIR / "outputs"
MODELS    = RUN_DIR / "models"
FIGS      = RUN_DIR / "figures"
TABLES    = RUN_DIR / "tables"
PROV      = OUT / "provenance"
EXPORT    = RUN_DIR / "export"          # deployable inference bundles
BIOWORDVEC_BIN = Path(os.environ.get(
    "BIOWORDVEC_BIN", "/mnt/subedi-research/data/BioWordVec_PubMed_MIMICIII_d200.bin"))

OUT_CLASSICAL  = OUT / "classical_ml"
OUT_FASTTEXT   = OUT / "classical_ml_enhanced"
OUT_BIOWORDVEC = OUT / "classical_ml_biowordvec"
OUT_BERT       = OUT / "bert_models"
OUT_LLM        = OUT / "prompt_llm"
OUT_FINAL      = OUT / "final_results"
for _d in (OUT, MODELS, FIGS, TABLES, PROV, EXPORT, OUT_CLASSICAL, OUT_FASTTEXT,
           OUT_BIOWORDVEC, OUT_BERT, OUT_LLM, OUT_FINAL):
    _d.mkdir(parents=True, exist_ok=True)

SEED_SPLIT, SEED_CV, SEED_TUNE = 42, 42, 99
N_FOLDS, N_BOOTSTRAP, POS_LABEL = 5, 1000, 1

# Tier vocabulary — short forms, used in every legend and table header.
TIERS = ["Classical", "FastText", "BioWordVec", "Transformer", "Prompt-based"]

# ------------------------------------------------------------------ data
def load_data():
    """Return (df, train_idx, test_idx); asserts split integrity every call."""
    df = pd.read_csv(DATA_DIR / "labeled_dataset.csv")
    tr = np.load(DATA_DIR / "train_idx.npy")
    te = np.load(DATA_DIR / "test_idx.npy")
    assert not (set(df["NCTId"].values[tr]) & set(df["NCTId"].values[te])), \
        "train/test split is not trial-disjoint"
    return df, tr, te

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def assert_inputs(expected: dict):
    """Fail loudly if a notebook is running on different input bytes."""
    for name, want in expected.items():
        got = sha256(DATA_DIR / name)
        assert got == want, f"{name} checksum drift: expected {want[:12]}, got {got[:12]}"
    return True

# ------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred, y_prob=None, name="", tier="", extra=None):
    """One metric dict, identical shape for every model in every tier.

    weighted_f1  headline metric; respects the 73/27 class imbalance.
    biomarker_f1 F1 on the positive class alone — what the triage use case
                 actually cares about, and what separates models once weighted
                 F1 saturates near the ceiling.
    mcc          robust to imbalance; with systems separated by 1-2 errors,
                 F1 alone is a blunt instrument.
    brier        calibration. A model that is right but wildly overconfident is
                 worse for a human-in-the-loop triage tool than one that is
                 right and appropriately uncertain.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "Model": name, "Tier": tier,
        "accuracy":     accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "weighted_f1":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "biomarker_f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "precision":    precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall":       recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "specificity":  tn / (tn + fp) if (tn + fp) else 0.0,
        "mcc":          matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "errors": int(fp + fn), "n": int(len(y_true)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            row["auc"]   = roc_auc_score(y_true, y_prob)
            row["auprc"] = average_precision_score(y_true, y_prob)
            row["brier"] = brier_score_loss(y_true, np.clip(y_prob, 0, 1))
        except ValueError:
            row["auc"] = row["auprc"] = row["brier"] = np.nan
    if extra:
        row.update(extra)
    return row

def bootstrap_ci(y_true, y_pred, metric="weighted_f1", n=N_BOOTSTRAP, seed=SEED_CV):
    """Percentile bootstrap CI, for systems that have no CV folds."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx, vals = np.arange(len(y_true)), []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y_true[b])) < 2:
            continue
        avg = "weighted" if metric == "weighted_f1" else None
        vals.append(f1_score(y_true[b], y_pred[b], average=avg,
                             pos_label=POS_LABEL, zero_division=0)
                    if avg else f1_score(y_true[b], y_pred[b],
                                         pos_label=POS_LABEL, zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def overfit_report(train_score, val_score, name="", n_configs=None):
    """Train-minus-validation gap — the paper's overfitting check, made checkable.

    Also carries the tuning budget (`n_configs`): how many hyperparameter
    configurations were evaluated to select this model. Reporting the search
    budget alongside the gap is what makes "we tuned carefully" a verifiable
    claim rather than an assertion.
    """
    gap = float(train_score) - float(val_score)
    status = "OK" if gap < 0.04 else ("WATCH" if gap < 0.10 else "OVERFIT")
    return {"Model": name, "train_f1": float(train_score), "val_f1": float(val_score),
            "gap": gap, "status": status, "n_configs_tried": n_configs}

# ------------------------------------------------------------------ figure style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4, "grid.color": "#999999",
    "legend.frameon": False, "figure.constrained_layout.use": False,
})

# Typography: figure text uses plain ASCII punctuation only. Em and en dashes
# render inconsistently across PDF viewers and journal typesetting, and a title
# like "Model - test set" is clearer than one built from dashes. Dashes belong
# in prose, not in axis labels.

# Okabe-Ito, ordered so greyscale luminance also separates the tiers.
# Colour is never the sole channel: bars add HATCH, lines add MARKER + DASH.
TIER_STYLE = {
    "Classical":    {"color": "#0072B2", "hatch": "",    "marker": "o", "ls": "-"},
    "FastText":     {"color": "#56B4E9", "hatch": "/",    "marker": "s", "ls": "--"},
    "BioWordVec":   {"color": "#009E73", "hatch": "\\",   "marker": "^", "ls": "-."},
    "Transformer":  {"color": "#D55E00", "hatch": "x",    "marker": "D", "ls": ":"},
    "Prompt-based": {"color": "#CC79A7", "hatch": ".",    "marker": "v", "ls": (0, (3, 1, 1, 1))},
}
def tier_style(tier):
    return TIER_STYLE.get(tier, {"color": "#666666", "hatch": "++",
                                 "marker": "P", "ls": "-"})

def tier_legend(ax, tiers=None, **kw):
    """Legend whose swatches carry both colour and hatch, so it survives B&W."""
    tiers = tiers or list(TIER_STYLE)
    handles = [Patch(facecolor=TIER_STYLE[t]["color"], hatch=TIER_STYLE[t]["hatch"],
                     edgecolor="black", linewidth=0.5, label=t)
               for t in tiers if t in TIER_STYLE]
    ax.legend(handles=handles, **kw)

def plot_ranked_bars(rows, metric, title, xlabel, path, annotate=True,
                     figsize=None, xlim=None, ci=None):
    """Main comparison figure: horizontal bars, best at top.

    rows: dicts with Model, Tier, <metric>, and optionally the CI bounds named
    by `ci` as (lo_key, hi_key).

    Two design decisions worth recording. The x-axis starts near the lowest
    value rather than at zero: with forty systems clustered between 0.95 and
    1.00, a zero-based axis renders them as forty identical bars and hides the
    only thing the figure exists to show. And hatch density is deliberately low
    (single-stroke patterns) -- dense crosshatch repeated across long bars reads
    as visual noise rather than as an accessibility aid, while a single stroke
    still separates the tiers in greyscale.
    """
    rows = sorted(rows, key=lambda r: r[metric], reverse=True)
    names = [r["Model"] for r in rows]
    vals  = [r[metric] for r in rows]
    tiers = [r.get("Tier", "") for r in rows]
    h = figsize[1] if figsize else max(2.6, 0.17 * len(rows) + 1.1)
    fig, ax = plt.subplots(figsize=(figsize[0] if figsize else 6.6, h))
    ypos = np.arange(len(rows))[::-1]

    lo = xlim[0] if xlim else max(0.0, min(vals) - 0.02)
    hi = xlim[1] if xlim else min(1.0, max(vals) + 0.005)

    err = None
    if ci:
        lo_k, hi_k = ci
        if all(lo_k in r and r[lo_k] == r[lo_k] for r in rows):
            err = np.array([[r[metric] - r[lo_k] for r in rows],
                            [r[hi_k] - r[metric] for r in rows]])
            lo = min(lo, min(r[lo_k] for r in rows) - 0.005)

    for k, (y, v, t) in enumerate(zip(ypos, vals, tiers)):
        st = tier_style(t)
        ax.barh(y, v - lo, left=lo, height=0.72, color=st["color"],
                hatch=st["hatch"], edgecolor="black", linewidth=0.4)
    if err is not None:
        ax.errorbar(vals, ypos, xerr=err, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.6, alpha=0.75)

    ax.set_yticks(ypos, names)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    ax.set_xlim(lo, hi)
    if annotate:
        for y, v in zip(ypos, vals):
            ax.text(hi + (hi - lo) * 0.012, y, f"{v:.3f}", va="center", fontsize=6.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    seen = [t for t in TIERS if t in set(tiers)]
    # Legend below the axes: on a sorted horizontal bar chart the interior is
    # occupied everywhere, so an in-axes legend lands on top of data.
    tier_legend(ax, seen, loc="upper center",
                bbox_to_anchor=(0.5, -0.55 / h), ncol=min(len(seen), 5))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"figure -> {path}")

def plot_ranked_dots(rows, metric, title, xlabel, path, ci=None,
                     figsize=None, xlim=None, reference=None):
    """Dot-and-interval plot: one point per system with its confidence interval.

    Preferred over bars for this comparison. Thirty of the forty systems fall
    between 0.95 and 1.00, so bars are nearly full-width regardless of scaling
    and their fills carry no information; worse, a hatch pattern repeated across
    a long bar reads as noise. A point with an interval encodes exactly what the
    figure is about -- the estimate and its uncertainty -- and the convention is
    familiar from forest plots in the clinical literature.

    Greyscale safety comes from marker SHAPE rather than fill: circle, square,
    triangle, diamond and inverted triangle remain distinguishable when printed
    in black and white, without any hatching.
    """
    rows = sorted(rows, key=lambda r: r[metric], reverse=True)
    names = [r["Model"] for r in rows]
    vals  = np.array([r[metric] for r in rows], dtype=float)
    tiers = [r.get("Tier", "") for r in rows]
    h = figsize[1] if figsize else max(2.6, 0.155 * len(rows) + 1.1)
    fig, ax = plt.subplots(figsize=(figsize[0] if figsize else 6.4, h))
    ypos = np.arange(len(rows))[::-1]

    err = None
    if ci and all(ci[0] in r and r[ci[0]] == r[ci[0]] for r in rows):
        err = np.array([vals - np.array([r[ci[0]] for r in rows]),
                        np.array([r[ci[1]] for r in rows]) - vals])

    if reference is not None:
        ax.axvspan(reference[0], reference[1], color="0.88", zorder=0, lw=0)
        ax.axvline(reference[2], color="0.45", ls="--", lw=0.7, zorder=1)

    for y, v, t, k in zip(ypos, vals, tiers, range(len(rows))):
        st = tier_style(t)
        if err is not None:
            ax.plot([v - err[0][k], v + err[1][k]], [y, y],
                    color=st["color"], lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot(v, y, marker=st["marker"], ms=4.2, color=st["color"],
                markeredgecolor="black", markeredgewidth=0.45, zorder=3)

    ax.set_yticks(ypos, names)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    seen = [t for t in TIERS if t in set(tiers)]
    handles = [plt.Line2D([], [], marker=TIER_STYLE[t]["marker"], ls="",
                          color=TIER_STYLE[t]["color"], ms=4.2,
                          markeredgecolor="black", markeredgewidth=0.45, label=t)
               for t in seen]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.5 / h), ncol=min(len(seen), 5))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"figure -> {path}")

def draw_confusion(ax, y_true, y_pred, title=None,
                   labels=("No biomarker", "Biomarker"), cmap="Blues"):
    """Draw a confusion matrix onto an existing axis.

    A sequential blue ramp rather than greyscale. Greyscale was the original
    choice for print safety, but on a well-performing classifier the diagonal
    fractions approach 1.0 and the cells render as solid black, which is both
    ugly and less legible than the numbers printed on them. A light sequential
    ramp keeps the numbers readable, still reproduces sensibly in monochrome,
    and matches the convention readers expect.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1.35)   # headroom keeps 100% off-black
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]*100:.1f}%", ha="center",
                    va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.62 else "black")
    ax.set_xticks([0, 1], labels, fontsize=7)
    ax.set_yticks([0, 1], labels, fontsize=7, rotation=90, va="center")
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left", fontsize=7.5)
    return cm

def plot_confusion(y_true, y_pred, title, path,
                   labels=("No biomarker", "Biomarker"), figsize=(2.9, 2.6)):
    """Standalone confusion matrix: counts plus row-normalised percentages."""
    fig, ax = plt.subplots(figsize=figsize)
    cm = draw_confusion(ax, y_true, y_pred, labels=labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Annotated")
    ax.set_title(title, loc="left")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return cm

def plot_curves(history, title, path):
    """Per-epoch loss and F1, with the early-stopping epoch marked.

    This is the figure that demonstrates absence of overfitting: if validation
    F1 turns over while training loss keeps falling, it is visible here.
    Train and validation are distinguished by marker and dash pattern, not
    colour alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], marker="o", ms=3.2, ls="-",
                 color="#0072B2", label="train")
    if history.get("val_loss"):
        axes[0].plot(ep, history["val_loss"], marker="s", ms=3.2, ls="--",
                     color="#D55E00", label="validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(); axes[0].set_title("Loss", loc="left")
    axes[1].plot(ep, history["train_f1"], marker="o", ms=3.2, ls="-",
                 color="#0072B2", label="train")
    axes[1].plot(ep, history["val_f1"], marker="s", ms=3.2, ls="--",
                 color="#D55E00", label="validation")
    if history.get("best_epoch"):
        axes[1].axvline(history["best_epoch"], ls=":", lw=1, c="black")
        axes[1].annotate("selected epoch", (history["best_epoch"], min(history["val_f1"])),
                         fontsize=6.5, rotation=90, va="bottom", ha="right")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Weighted F1")
    axes[1].legend(); axes[1].set_title("Weighted F1", loc="left")
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)

def save_table(df, path, index=False):
    df.to_csv(path, index=index)
    print(f"table  -> {path}  ({len(df)} rows)")
    return df
'''

common_path = RUN_DIR / "bmo_common.py"
common_path.write_text(BMO_COMMON, encoding="utf-8")
print(f"wrote {common_path}  ({len(BMO_COMMON.splitlines())} lines)")
#%% code
# Import it back and smoke-test the metric helper.
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C
import importlib; importlib.reload(C)

demo = C.evaluate([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.9, 0.8],
                  name="smoke-test", tier="Classical")
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in demo.items()}, indent=2))
#%% md
### Verify the figure style is B&W-safe

The check below renders the tier palette, then converts it to greyscale using
the ITU-R BT.601 luma weights a printer effectively applies. If two tiers
collapse to the same grey value, the hatch pattern is what keeps them apart —
so we print both the luma values and the hatch assignment for the record.
#%% code
def luma(hex_color):
    r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b

print(f"{'Tier':14s} {'colour':9s} {'grey':>6s}  hatch  marker")
for t in C.TIERS:
    s = C.TIER_STYLE[t]
    print(f"{t:14s} {s['color']:9s} {luma(s['color']):6.3f}  "
          f"{s['hatch'] or '(solid)':7s} {s['marker']}")
gaps = sorted(luma(C.TIER_STYLE[t]["color"]) for t in C.TIERS)
print(f"\nsmallest greyscale separation between adjacent tiers: "
      f"{min(b - a for a, b in zip(gaps, gaps[1:])):.3f}")
print("hatch patterns are unique per tier:",
      len({C.TIER_STYLE[t]['hatch'] for t in C.TIERS}) == len(C.TIERS))
#%% md
## 6. Run manifest

Everything a future reader needs about *this* run in one file: environment,
seeds, input checksums, dataset shape. Notebook 09 appends output checksums
once the run completes.
#%% code
MANIFEST = {
    "run_id": "bmo_rerun_2026", "run_dir": str(RUN_DIR),
    "environment": ENV,
    "seeds": {"split": SEED_SPLIT, "cv": SEED_CV, "tune": SEED_TUNE,
              "n_folds": N_FOLDS, "n_bootstrap": N_BOOTSTRAP},
    "input_checksums_sha256": CHECKSUMS,
    "dataset": {
        "n_records": int(len(df)), "n_trials": int(df["NCTId"].nunique()),
        "positive_rate": float(df["has_biomarker"].mean()),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "train_positive_rate": float(df["has_biomarker"].values[train_idx].mean()),
        "test_positive_rate": float(df["has_biomarker"].values[test_idx].mean()),
    },
    "determinism_note": ("Seeds fixed; cuDNN left non-deterministic for speed. "
                         "Transformer metrics reproduce to approximately "
                         "+/-0.005 weighted F1, not bit-for-bit."),
    "storage_note": ("All weights, caches and the BioWordVec binary live on the "
                     "research NFS mount (/mnt/subedi-research), not on the "
                     "compute node's local disk."),
}
(C.PROV / "environment.json").write_text(json.dumps(ENV, indent=2))
(C.PROV / "run_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
print(json.dumps(MANIFEST["dataset"], indent=2))
print("\nmanifest ->", C.PROV / "run_manifest.json")
#%% md
## Done

Downstream notebooks import `bmo_common` and call `C.assert_inputs(...)` with
the checksums printed in section 3. If that assertion fails, the run is mixing
data versions and results must not be pooled.

**Next:** `01_dataset_preparation.ipynb` — corpus construction, label
derivation, and the descriptive statistics the Methods section needs.
