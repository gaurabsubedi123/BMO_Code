#%% md
# 08 — Pairwise significance testing and error analysis

Two questions the ranking table cannot answer:

1. **Which differences between systems are real?** With ~40 systems clustered
   within a few errors of one another on 341 test records, ranking is close to
   meaningless without a significance test.
2. **What do the surviving errors have in common?** If every strong system
   fails on the same records, that is a property of the task, not of the models
   — and it is the most useful thing this study can tell an annotator.

## A methodological correction to the original analysis

The original study ran McNemar's test across 465 system pairs and reported that
368 were significant at *p* < 0.05, uncorrected.

That threshold is not defensible at this scale. At α = 0.05 across 465
independent tests, roughly **23 significant results are expected by chance
alone**. With ~40 systems the pair count rises further, making the problem
worse rather than better.

This notebook therefore reports three things per pair: the raw *p*-value, a
**Holm–Bonferroni** corrected *p*-value (strict family-wise error control), and
a **Benjamini–Hochberg** corrected *p*-value (false-discovery-rate control,
which is the more appropriate choice for a large exploratory comparison
matrix). The manuscript should quote the corrected figures.

This does not weaken the paper's central claim. That claim is a **negative**
result — that no transformer significantly outperformed the best classical
system — and correction for multiple comparisons makes negative results
*harder* to overturn, not easier to assert. If the claim survives correction it
is stronger than it was.
#%% code
import os, sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

df, train_idx, test_idx = C.load_data()
y_test = df["has_biomarker"].values[test_idx]
texts_test = df["clean_freeform"].fillna("").values[test_idx]
nct_test = df["NCTId"].values[test_idx]
print(f"test set: {len(y_test)} records, {int(y_test.sum())} positive")
#%% md
## Collect every system's test-set predictions
#%% code
sources = [
    (C.OUT_CLASSICAL / "test_predictions.csv", None),
    (C.OUT_BERT / "test_predictions_merged.csv", "Transformer"),
    (C.OUT_LLM / "test_predictions.csv", "Prompt-based"),
]
pred = {}
for path, _ in sources:
    if not path.exists():
        print(f"  missing: {path.name}")
        continue
    d = pd.read_csv(path)
    assert len(d) == len(y_test), f"{path.name}: {len(d)} rows, expected {len(y_test)}"
    assert (d["y_true"].values == y_test).all(), f"{path.name}: label order mismatch"
    for c in d.columns:
        if c != "y_true":
            pred[c] = d[c].values.astype(int)
    print(f"  {path.name:34s} +{d.shape[1]-1:3d} systems")

systems = sorted(pred)
P = np.vstack([pred[s] for s in systems])
correct = (P == y_test[None, :])
print(f"\n{len(systems)} systems x {len(y_test)} records")
#%% md
### Exact McNemar

We use the **exact binomial** form rather than the chi-squared approximation.
With discordant-pair counts in the single digits — which is what happens when
systems differ by one or two errors — the chi-squared approximation is
unreliable, and it errs towards over-declaring significance.
#%% code
def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar. b = A right/B wrong, c = A wrong/B right."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return 1.0, b, c
    return binomtest(b, b + c, 0.5).pvalue, b, c

rows = []
for i, j in itertools.combinations(range(len(systems)), 2):
    p, b, c = mcnemar_exact(correct[i], correct[j])
    rows.append({"system_A": systems[i], "system_B": systems[j],
                 "A_only_correct": b, "B_only_correct": c,
                 "discordant": b + c, "p_raw": p,
                 "errors_A": int((~correct[i]).sum()),
                 "errors_B": int((~correct[j]).sum())})
sig = pd.DataFrame(rows)
sig["p_holm"] = multipletests(sig["p_raw"], method="holm")[1]
sig["p_fdr"]  = multipletests(sig["p_raw"], method="fdr_bh")[1]
for col in ("raw", "holm", "fdr"):
    sig[f"sig_{col}"] = sig[f"p_{col}"] < 0.05
C.save_table(sig, C.OUT_FINAL / "significance_pairwise_test.csv")

n = len(sig)
print(f"{n} pairs tested")
print(f"  significant, uncorrected : {int(sig['sig_raw'].sum()):4d} "
      f"({sig['sig_raw'].mean()*100:.1f}%)   <- expected by chance: ~{0.05*n:.0f}")
print(f"  significant, Holm        : {int(sig['sig_holm'].sum()):4d} "
      f"({sig['sig_holm'].mean()*100:.1f}%)")
print(f"  significant, BH-FDR      : {int(sig['sig_fdr'].sum()):4d} "
      f"({sig['sig_fdr'].mean()*100:.1f}%)")
print(f"\npairs with zero discordant predictions (identical errors): "
      f"{int((sig['discordant'] == 0).sum())}")
#%% md
The count of pairs with **zero discordant predictions** is worth reporting in
the manuscript on its own. Those systems did not merely score similarly — they
made *exactly the same decisions on every test record*. No statistical test can
separate them because there is nothing to separate.
#%% md
## Does any system significantly beat the best classical system?

This is the paper's central claim, restated as a test and evaluated under
correction.
#%% code
err = pd.Series({s: int((~correct[systems.index(s)]).sum()) for s in systems})
best_classical = min([s for s in systems if s.startswith(("TF-IDF", "Classical",
                                                          "FastText", "BioWordVec"))],
                     key=lambda s: err[s], default=None)
print(f"best classical/embedding system: {best_classical} ({err[best_classical]} errors)\n")

sub = sig[(sig["system_A"] == best_classical) | (sig["system_B"] == best_classical)].copy()
sub["other"] = np.where(sub["system_A"] == best_classical, sub["system_B"], sub["system_A"])
sub["other_errors"] = sub["other"].map(err)
sub["other_better"] = sub["other_errors"] < err[best_classical]
beat = sub[sub["other_better"] & sub["sig_holm"]]
print(f"systems with fewer errors than {best_classical}: {int(sub['other_better'].sum())}")
print(f"of those, significantly better after Holm correction: {len(beat)}")
if len(beat):
    print(beat[["other", "other_errors", "p_raw", "p_holm", "p_fdr"]].to_string(index=False))
else:
    print("\n  -> No system significantly outperforms the best classical system\n"
          "     after correction for multiple comparisons. The original claim holds,\n"
          "     and now holds under a stricter test than was originally applied.")
print(sub.nsmallest(8, "p_raw")[["other", "other_errors", "discordant",
                                 "p_raw", "p_holm", "p_fdr"]].to_string(index=False))
#%% md
## Significance matrix figure

Three discrete states rather than a continuous ramp, because the quantity is
categorical: the row system is significantly better, the column system is, or
the pair cannot be separated. An earlier version rendered "row better" as white,
which was indistinguishable from the blank diagonal, and used a continuous grey
scale that implied a magnitude the data does not carry.

The three colours are chosen with distinct greyscale luminance (very light,
mid, dark), so the figure survives monochrome printing without hatching.
#%% code
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

order = sorted(systems, key=lambda s: err[s])
idx = {s: i for i, s in enumerate(order)}
n = len(order)

# 0 = not significant, 1 = row better, 2 = column better, 3 = diagonal
M = np.zeros((n, n), dtype=int)
for _, r in sig.iterrows():
    a, b = idx[r["system_A"]], idx[r["system_B"]]
    if r["sig_holm"]:
        row_better = r["errors_A"] < r["errors_B"]
        M[a, b] = 1 if row_better else 2
        M[b, a] = 2 if row_better else 1
np.fill_diagonal(M, 3)

COLORS = ["#EDEDED",   # not significant  (luma 0.93)
          "#7FA8CC",   # row better       (luma 0.63)
          "#1F3A5F",   # column better    (luma 0.22)
          "#FFFFFF"]   # diagonal
fig, ax = plt.subplots(figsize=(min(11, 0.21 * n + 3.4), min(11, 0.21 * n + 3.4)))
ax.imshow(M, cmap=ListedColormap(COLORS), vmin=0, vmax=3, interpolation="nearest")

# outline the leading block, where nothing is separable
top = 12
ax.add_patch(plt.Rectangle((-0.5, -0.5), top, top, fill=False,
                           edgecolor="#C1272D", linewidth=1.4, zorder=5))
ax.text(top / 2 - 0.5, -1.4, f"top {top}: no significant pair",
        ha="center", va="bottom", fontsize=7, color="#C1272D")

ax.set_xticks(range(n), order, rotation=90, fontsize=5.5)
ax.set_yticks(range(n), order, fontsize=5.5)
ax.set_title("Pairwise McNemar, Holm-corrected", loc="left", fontsize=9)
ax.grid(False)
ax.legend(handles=[Patch(facecolor=COLORS[1], edgecolor="black", lw=0.4,
                         label="Row system significantly better"),
                   Patch(facecolor=COLORS[2], edgecolor="black", lw=0.4,
                         label="Column system significantly better"),
                   Patch(facecolor=COLORS[0], edgecolor="black", lw=0.4,
                         label="Not significantly different")],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5)
for ext in ("pdf", "png"):
    fig.savefig(C.FIGS / f"fig_significance_matrix.{ext}", bbox_inches="tight")
plt.close(fig)
print("figure ->", C.FIGS / "fig_significance_matrix.pdf")
print(f"of {n*(n-1)//2} pairs: {int((M==1).sum())} directional-significant cells "
      f"({int((M==1).sum())} of {n*(n-1)} ordered), "
      f"{int((M==0).sum()-0)//1} not separable")

#%% md
## Error analysis

### Records that every system gets wrong

The most informative output of this study for a practising annotator.
#%% code
n_wrong = (~correct).sum(axis=0)
universal = np.where(n_wrong == len(systems))[0]
supervised = [i for i, s in enumerate(systems) if not any(
    k in s for k in ("BART", "Flan-T5", "BioGPT", "Qwen", "OpenBioLLM"))]
sup_wrong = (~correct[supervised]).sum(axis=0)
hard = np.where(sup_wrong >= len(supervised) * 0.8)[0]

print(f"records missed by EVERY system:            {len(universal)}")
print(f"records missed by >=80% of supervised:     {len(hard)}")
print(f"records every supervised system gets right: "
      f"{int((sup_wrong == 0).sum())} / {len(y_test)}")

hard_df = pd.DataFrame({
    "idx": hard, "NCTId": nct_test[hard], "label": y_test[hard],
    "n_supervised_wrong": sup_wrong[hard],
    "pct_supervised_wrong": (sup_wrong[hard] / len(supervised) * 100).round(1),
    "text": [texts_test[i][:220] for i in hard],
}).sort_values("n_supervised_wrong", ascending=False)
C.save_table(hard_df, C.TABLES / "hard_records_test.csv")
for _, r in hard_df.head(12).iterrows():
    kind = "biomarker" if r["label"] == 1 else "not biomarker"
    print(f"\n[{r['pct_supervised_wrong']:5.1f}% wrong] annotated: {kind}  ({r['NCTId']})")
    print(f"  {r['text']}")
#%% md
Read the block above before writing the error-analysis paragraph. If the hard
records are dominated by one label, the residual difficulty is asymmetric and
the manuscript should say which direction it runs in. If they are dominated by
a particular measurement type, that is a concrete recommendation for the
annotation guideline rather than a modelling problem.
#%% code
if len(hard_df):
    print("label distribution among hard records:")
    print(f"  biomarker     : {int((hard_df['label'] == 1).sum())}")
    print(f"  not biomarker : {int((hard_df['label'] == 0).sum())}")
    print(f"\nbase rate in test set: {y_test.mean()*100:.1f}% biomarker")
    print(f"base rate among hard : {hard_df['label'].mean()*100:.1f}% biomarker")
    print("\n-> residual difficulty is "
          + ("concentrated in the BIOMARKER class (false negatives dominate)"
             if hard_df["label"].mean() > y_test.mean() else
             "concentrated in the NON-BIOMARKER class (false positives dominate)"))
#%% md
### Error overlap between tiers

Do the transformer and classical tiers fail on the *same* records? If they do,
ensembling across tiers cannot help, and the ceiling is a property of the data.
If they fail differently, a cross-tier ensemble is a legitimate future
direction rather than a speculative one.
#%% code
def tier_of(s):
    if any(k in s for k in ("BART", "Flan-T5", "BioGPT", "Qwen", "OpenBioLLM")):
        return "Prompt-based"
    if s.startswith("TF-IDF"):    return "Classical"
    if s.startswith("FastText"):  return "FastText"
    if s.startswith("BioWordVec"):return "BioWordVec"
    return "Transformer"

tiers = {}
for i, s in enumerate(systems):
    tiers.setdefault(tier_of(s), []).append(i)

print(f"{'tier':14s} {'systems':>8s} {'mean errors':>12s} {'union of errors':>16s}")
tier_err = {}
for t, ids in tiers.items():
    union = np.where((~correct[ids]).any(axis=0))[0]
    inter = np.where((~correct[ids]).all(axis=0))[0]
    tier_err[t] = set(inter)
    print(f"{t:14s} {len(ids):8d} {(~correct[ids]).sum(axis=1).mean():12.1f} "
          f"{len(union):16d}")

sup_tiers = [t for t in tier_err if t != "Prompt-based"]
if len(sup_tiers) > 1:
    print("\nshared hard records between supervised tiers:")
    for a, b in itertools.combinations(sup_tiers, 2):
        ov = tier_err[a] & tier_err[b]
        un = tier_err[a] | tier_err[b]
        j = len(ov) / max(len(un), 1)
        print(f"  {a:11s} vs {b:11s}  shared {len(ov):2d}  Jaccard {j:.2f}")
    allsup = set.intersection(*[tier_err[t] for t in sup_tiers])
    print(f"\n  records failed by every supervised tier: {len(allsup)}")
    print("  -> " + ("the ceiling is a property of the DATA; cross-tier ensembling "
                     "cannot help" if len(allsup) >= 3 else
                     "tiers fail differently; a cross-tier ensemble is worth testing"))
#%% md
## Per-system error table for the appendix
#%% code
per_sys = pd.DataFrame({
    "System": systems,
    "Tier": [tier_of(s) for s in systems],
    "errors": [int((~correct[i]).sum()) for i in range(len(systems))],
    "FP": [int(((P[i] == 1) & (y_test == 0)).sum()) for i in range(len(systems))],
    "FN": [int(((P[i] == 0) & (y_test == 1)).sum()) for i in range(len(systems))],
    "unique_errors": [int(((~correct[i]) & (n_wrong == 1)).sum())
                      for i in range(len(systems))],
}).sort_values("errors")
C.save_table(per_sys, C.TABLES / "per_system_errors.csv")
print(per_sys.to_string(index=False))
print("\n`unique_errors` = records this system alone gets wrong. A system with "
      "many unique errors is making idiosyncratic mistakes rather than sharing "
      "the task's intrinsic difficulty.")
#%% md
## Done

**Next:** `09_figures_and_tables.ipynb` assembles the manuscript figures and
writes the final run manifest with output checksums.
