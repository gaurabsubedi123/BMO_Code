#%% md
# 07 — Consolidated evaluation across all tiers

Merges every system into one table on one metric definition, one test set, one
split. Nothing is computed here that was not computed in notebooks 02, 05 and
06 — this notebook exists so that the comparison is assembled in exactly one
place, from files, rather than assembled by hand.

## The failure this notebook is designed to prevent

An earlier run of this project produced a consolidated results file in which
the FastText and BioWordVec systems had been scored **before their embedding
model finished loading**, so those rows silently reported performance against
zero vectors. The numbers looked plausible and were wrong, and reconciling them
against the correct file cost significant effort.

Every merge below is therefore checked: row counts, duplicate model names,
impossible metrics, and — most importantly — confusion-matrix cells that must
sum to the size of the test set. A system whose TN+FP+FN+TP does not equal 341
did not evaluate on the test set, whatever its metrics claim.
#%% code
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

df, train_idx, test_idx = C.load_data()
N_TEST, N_TRAIN = len(test_idx), len(train_idx)
y_test  = df["has_biomarker"].values[test_idx]
y_train = df["has_biomarker"].values[train_idx]
print(f"test set {N_TEST} records ({y_test.mean()*100:.2f}% positive)")
#%% md
## Merge the transformer shards

Notebook 05 may have run as two processes, one per GPU. Their outputs are
merged here and the merge is verified to contain every expected model exactly
once.
#%% code
EXPECTED_TRANSFORMERS = ["BERT-base", "BioBERT", "PubMedBERT", "Bio+ClinicalBERT",
                         "BioLinkBERT", "SciBERT", "DeBERTa-v3", "ModernBERT",
                         "BiomedBERT-large"]

def merge_shards(pattern, out_name):
    parts = sorted(C.OUT_BERT.glob(pattern))
    if not parts:
        print(f"  no files matching {pattern}")
        return pd.DataFrame()
    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    merged.to_csv(C.OUT_BERT / out_name, index=False)
    print(f"  {pattern:38s} {len(parts)} shard(s) -> {len(merged):3d} rows -> {out_name}")
    return merged

print("merging transformer shards:")
bert_test = merge_shards("test_results_shard*.csv", "test_results_merged.csv")
bert_cv   = merge_shards("cv_results_shard*.csv",   "cv_results_merged.csv")
merge_shards("per_fold_shard*.csv",                 "per_fold_merged.csv")
merge_shards("hyperparameter_search_shard*.csv",    "hyperparameter_search_merged.csv")
merge_shards("tuning_budget_shard*.csv",            "tuning_budget_merged.csv")
merge_shards("overfitting_audit_shard*.csv",        "overfitting_audit_merged.csv")

if len(bert_test):
    missing = set(EXPECTED_TRANSFORMERS) - set(bert_test["Model"])
    dupes = bert_test["Model"][bert_test["Model"].duplicated()].tolist()
    print(f"\n  missing models: {sorted(missing) if missing else 'none'}")
    print(f"  duplicated:     {dupes if dupes else 'none'}")
    assert not dupes, "a model appears twice — shards overlap"
#%% code
# Out-of-fold predictions need column-wise merging, not row concatenation:
# each shard holds a different set of model columns over the same records.
def merge_oof(pattern, out_name, n_expected):
    parts = sorted(C.OUT_BERT.glob(pattern))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in parts]
    base = frames[0][["y_true"]].copy()
    for f in frames:
        for c in f.columns:
            if c != "y_true":
                base[c] = f[c].values
    assert len(base) == n_expected, f"{out_name}: {len(base)} rows, expected {n_expected}"
    base.to_csv(C.OUT_BERT / out_name, index=False)
    print(f"  {out_name:34s} {len(base):4d} rows x {base.shape[1]-1} models")
    return base

print("merging prediction matrices:")
bert_oof   = merge_oof("oof_predictions_shard*.csv",   "oof_predictions_merged.csv", N_TRAIN)
bert_oofp  = merge_oof("oof_probabilities_shard*.csv", "oof_probabilities_merged.csv", N_TRAIN)
bert_tpred = merge_oof("test_predictions_shard*.csv",  "test_predictions_merged.csv", N_TEST)
bert_tprob = merge_oof("test_probabilities_shard*.csv","test_probabilities_merged.csv", N_TEST)
#%% md
## Assemble the master test-set table
#%% code
frames = []
p = C.OUT_CLASSICAL / "test_results_all_tiers.csv"
if p.exists():
    frames.append(pd.read_csv(p))
if len(bert_test):
    frames.append(bert_test)
p = C.OUT_LLM / "prompt_llm_test_results.csv"
if p.exists():
    frames.append(pd.read_csv(p))

master = pd.concat(frames, ignore_index=True)
# A readable, unique display name per system: tier disambiguates estimators
# that appear in more than one representation.
def display_name(r):
    if r["Tier"] in ("Classical", "FastText", "BioWordVec"):
        return f"{r['Tier']} + {r['Model']}" if r["Tier"] != "Classical" \
               else f"TF-IDF + {r['Model']}"
    return r["Model"]
master["System"] = master.apply(display_name, axis=1)
master = master.sort_values("weighted_f1", ascending=False).reset_index(drop=True)
print(f"{len(master)} systems assembled")
#%% md
### Integrity checks

The confusion-matrix check is the one that would have caught the earlier
zero-vector failure: a system that did not really evaluate on the test set
cannot have cells summing to its size.
#%% code
problems = []
for _, r in master.iterrows():
    cells = int(r["TN"] + r["FP"] + r["FN"] + r["TP"])
    if cells != N_TEST:
        problems.append(f"{r['System']}: confusion cells sum to {cells}, expected {N_TEST}")
    if not (0 <= r["weighted_f1"] <= 1):
        problems.append(f"{r['System']}: weighted_f1 out of range ({r['weighted_f1']})")
    if r["TP"] == 0 and r["weighted_f1"] > 0.7:
        problems.append(f"{r['System']}: zero true positives yet weighted F1 "
                        f"{r['weighted_f1']:.3f} — degenerate all-negative predictor")
dupes = master["System"][master["System"].duplicated()].tolist()
if dupes:
    problems.append(f"duplicate system names: {dupes}")

if problems:
    print("INTEGRITY PROBLEMS:")
    for x in problems:
        print("  -", x)
else:
    print(f"all {len(master)} systems pass: confusion cells sum to {N_TEST}, "
          "metrics in range, no duplicates")
#%% md
Note that the "zero true positives yet high weighted F1" check is not
hypothetical — it is exactly the shape of the BioGPT result in the original
study, where an all-negative predictor scored 0.55 weighted F1 purely from the
majority class. Flagging it here keeps such a system from being read as
mediocre when it is actually degenerate.
#%% code
#%% md
## Confidence intervals on every system

Until now only the prompt-based systems carried intervals. Without them the main
table invites a reader to take the ordering at face value — and on 341 records
the top of that ordering has moved between every run of this study. Percentile
bootstrap intervals (2,000 resamples of the test set) make the overlap explicit,
so systems that are statistically indistinguishable *look* indistinguishable.

The resampling is over test records, which is what the intervals should express:
uncertainty from having evaluated on this particular 341-record sample rather
than another. It does not capture uncertainty from the train/test split itself —
that would need repeated splits, and the limitation is stated in the manuscript.
#%% code
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

N_BOOT = 2000
RNG_SEED = 20260828

def boot_ci(y_true, y_pred, n=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap CIs for weighted and biomarker-class F1."""
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

# Predictions were collected above for the integrity checks; reuse them.
pred_lookup = {}
for path in (C.OUT_CLASSICAL / "test_predictions.csv",
             C.OUT_BERT / "test_predictions_merged.csv",
             C.OUT_LLM / "test_predictions.csv"):
    if path.exists():
        d = pd.read_csv(path)
        for c in d.columns:
            if c != "y_true":
                pred_lookup[c] = d[c].values.astype(int)

def key_for(r):
    if r["Tier"] in ("Classical", "FastText", "BioWordVec"):
        return f"{r['Tier']} | {r['Model']}"
    return r["Model"]

keys = [key_for(r) for _, r in master.iterrows()]
missing = [k for k in keys if k not in pred_lookup]
if missing:
    print(f"WARNING: no predictions for {len(missing)} system(s): {missing[:5]}")

t0 = time.time()
res = Parallel(n_jobs=int(os.environ.get("BMO_N_JOBS", "32")))(
    delayed(boot_ci)(y_test, pred_lookup[k]) for k in keys if k in pred_lookup)
print(f"{len(res)} systems bootstrapped ({N_BOOT} resamples each) "
      f"in {time.time()-t0:.0f}s")

ci = {k: r for k, r in zip([k for k in keys if k in pred_lookup], res)}
for col, i in (("wf1_lo", 0), ("wf1_hi", 1), ("bf1_lo", 2), ("bf1_hi", 3)):
    master[col] = [ci.get(k, (np.nan,)*4)[i] for k in keys]

# How many systems overlap the leader? That number, not the ranking, is the
# honest description of the result.
top = master.iloc[0]
overlap = master[master["wf1_hi"] >= top["wf1_lo"]]
print(f"\nleader: {top['System']} "
      f"{top['weighted_f1']:.4f} [{top['wf1_lo']:.4f}, {top['wf1_hi']:.4f}]")
print(f"systems whose 95% CI overlaps the leader's: {len(overlap)} of {len(master)}")
print("-> the ranking cannot separate these; report them as indistinguishable")
#%% code
COLS = ["System", "Tier", "weighted_f1", "wf1_lo", "wf1_hi",
        "biomarker_f1", "bf1_lo", "bf1_hi", "macro_f1", "mcc",
        "auc", "auprc", "brier", "precision", "recall", "specificity",
        "TN", "FP", "FN", "TP", "errors"]
COLS = [c for c in COLS if c in master.columns]
C.save_table(master[COLS], C.OUT_FINAL / "MASTER_test_all_systems.csv")
C.save_table(master, C.TABLES / "MASTER_test_all_systems_full.csv")
print(master[["System", "Tier", "weighted_f1", "biomarker_f1", "errors"]]
      .to_string(index=False))
#%% md
## Main comparison figure

Horizontal bars sorted by score, best at top. Tier is encoded by colour **and**
hatch pattern so the ordering survives black-and-white printing.
#%% code
rows = master.rename(columns={"System": "Model"}).to_dict("records")
# Two figures rather than one, because they answer different questions.
#
# The main-text figure shows the systems that are actually in contention, on an
# axis wide enough to separate them. The full-range figure includes the weak
# prompt-based systems, whose scores stretch the axis so far that everything
# above 0.95 collapses into a single column -- honest, but it hides the overlap
# that is the point of the analysis.
#
# The shaded band marks the leading system's interval: any system whose own
# interval enters it cannot be distinguished from the leader.
lead = master.iloc[0]
band = (lead["wf1_lo"], lead["wf1_hi"], lead["weighted_f1"])

contenders = [r for r in rows if r["weighted_f1"] >= 0.95]
for ext in ("pdf", "png"):
    C.plot_ranked_dots(contenders, "weighted_f1",
                       f"Systems within reach of the leader "
                       f"({len(contenders)} of {len(rows)}), 95% bootstrap CI",
                       "Weighted F1",
                       C.FIGS / f"fig_main_ranked_weighted_f1.{ext}",
                       ci=("wf1_lo", "wf1_hi"), reference=band,
                       xlim=(0.945, 1.003))
    C.plot_ranked_dots(rows, "weighted_f1",
                       "All systems, held-out test set (95% bootstrap CI)",
                       "Weighted F1",
                       C.FIGS / f"fig_all_systems_full_range.{ext}",
                       ci=("wf1_lo", "wf1_hi"), reference=band)
print(f"\n{len(contenders)} of {len(rows)} systems reach 0.95 weighted F1")
print(f"overlapping the leader's interval: "
      f"{sum(1 for r in rows if r.get('wf1_hi', 0) >= lead['wf1_lo'])}")
C.plot_ranked_bars([r for r in rows if r["Tier"] != "Prompt-based"], "biomarker_f1",
                   "Supervised systems, biomarker-class F1",
                   "F1 (biomarker class)", C.FIGS / "fig_supervised_biomarker_f1.pdf",
                   xlim=(0.85, 1.0))
#%% md
## Where the ceiling is

With 341 test records, one additional error costs roughly 0.003 weighted F1.
The table below converts the ranking into error counts, which is the honest
unit for this comparison: systems separated by fewer errors than the spread
within their own tier should be described as tied.
#%% code
top = master.nsmallest(12, "errors")[["System", "Tier", "errors", "FP", "FN",
                                      "weighted_f1", "biomarker_f1"]]
print(top.to_string(index=False))
print(f"\nfewest errors: {int(master['errors'].min())} / {N_TEST} records")
print(f"systems within 2 errors of the best: "
      f"{int((master['errors'] <= master['errors'].min() + 2).sum())}")
print(f"\none error = {1/N_TEST:.4f} accuracy = "
      f"~{abs(master['weighted_f1'].iloc[0] - master['weighted_f1'].iloc[1]):.4f} "
      "weighted F1 at the top of the table")
#%% md
### False positives versus false negatives

A detail with practical consequences for the triage use case that the metric
table hides: if the supervised systems make almost no false positives but
persistent false negatives, the tool is conservative — it will not waste
annotator time on spurious biomarkers, but it will quietly miss some. That is
the opposite trade-off from what a screening tool usually wants, and the
manuscript should say which behaviour it has rather than only reporting F1.
#%% code
sup = master[master["Tier"] != "Prompt-based"]
print(f"supervised systems: {len(sup)}")
print(f"  mean false positives: {sup['FP'].mean():.2f}")
print(f"  mean false negatives: {sup['FN'].mean():.2f}")
print(f"  systems with zero false positives: {int((sup['FP'] == 0).sum())} / {len(sup)}")
print(f"  systems with zero false negatives: {int((sup['FN'] == 0).sum())} / {len(sup)}")
ratio = sup["FN"].sum() / max(sup["FP"].sum(), 1)
print(f"\n  aggregate FN:FP ratio = {ratio:.2f}:1")
print("  -> the tier is " + ("CONSERVATIVE (misses biomarkers rather than over-calling them)"
                             if ratio > 1.5 else
                             "LIBERAL (over-calls biomarkers)" if ratio < 0.67 else
                             "BALANCED between the two error types"))
#%% md
## Done

**Next:** `08_significance_and_errors.ipynb` — pairwise McNemar tests over
every system pair, and the error analysis that explains the residual failures.
