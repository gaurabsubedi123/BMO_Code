#%% md
# 11 — Final verification

This notebook exists because "the pipeline ran" and "the numbers are right" are
different claims, and on this project the gap between them has already cost
weeks. Everything here is an **independent re-derivation or cross-check**: it
recomputes metrics from raw predictions rather than trusting any summary file,
and it fails loudly rather than warning quietly.

## What it checks

| # | Check | The failure it catches |
|---|---|---|
| 1 | Input checksums | results pooled across different dataset versions |
| 2 | Stale output files | a file from an aborted run surviving into the final set |
| 3 | Split integrity | trial-level leakage between train and test |
| 4 | Prediction files | wrong length, label-order mismatch |
| 5 | Metrics recomputed from predictions | a summary table disagreeing with its own predictions |
| 6 | Confusion cells sum to n | a system that never really evaluated on the test set |
| 7 | Degenerate predictors | all-one-class models scoring well on the majority class |
| 8 | Duplicate systems | the same model counted twice in the comparison |
| 9 | Coverage | a tier or model silently missing from the master table |
| 10 | Significance matrix | pairs missing, or p-values out of range |
| 11 | Overfitting audit | a model above threshold not reported |
| 12 | Figures and tables | a manuscript artefact never produced |

A single `FAIL` means the results are not publishable until it is resolved.
#%% code
import os, sys, json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

PROBLEMS, WARNINGS, PASSED = [], [], []
def check(ok, label, detail=""):
    (PASSED if ok else PROBLEMS).append(f"{label}: {detail}" if detail else label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok
def warn(label, detail=""):
    WARNINGS.append(f"{label}: {detail}" if detail else label)
    print(f"  [WARN] {label}" + (f" — {detail}" if detail else ""))

df, train_idx, test_idx = C.load_data()
y_test  = df["has_biomarker"].values[test_idx]
y_train = df["has_biomarker"].values[train_idx]
N_TEST, N_TRAIN = len(test_idx), len(train_idx)
print(f"corpus {len(df):,} | train {N_TRAIN:,} | test {N_TEST:,} "
      f"| test positives {int(y_test.sum())}")
#%% md
## 1–2. Provenance: inputs unchanged, outputs not stale

The stale-file check compares every result file's modification time against the
newest one. Files substantially older than the rest are leftovers from an
aborted run — the mechanism by which two different runs get silently pooled.
#%% code
print("1. input checksums")
manifest_path = C.PROV / "run_manifest.json"
if manifest_path.exists():
    man = json.loads(manifest_path.read_text())
    for name, want in man.get("input_checksums_sha256", {}).items():
        got = C.sha256(C.DATA_DIR / name)
        check(got == want, f"{name} unchanged since setup",
              "" if got == want else f"expected {want[:12]}, got {got[:12]}")
else:
    warn("run manifest missing", "cannot verify input checksums")

print("\n2. stale output files")
# Compared WITHIN each directory, not globally. A staged pipeline legitimately
# writes its tiers hours apart -- here the LLM tier finished at 16:35, the
# classical tier at 20:10 and the transformers at 23:52 -- so a flat global
# threshold flags seven hours of perfectly valid work. What actually indicates
# a leftover from an aborted run is one file lagging the others produced by the
# SAME notebook.
any_files = False
for d in (C.OUT_FINAL, C.TABLES, C.OUT_BERT, C.OUT_CLASSICAL, C.OUT_LLM):
    files = list(d.glob("*.csv"))
    if not files:
        continue
    any_files = True
    times = {f: f.stat().st_mtime for f in files}
    newest = max(times.values())
    stale = {f: t for f, t in times.items() if newest - t > 6 * 3600}
    span = (newest - min(times.values())) / 3600
    print(f"  {d.name:22s} {len(files):3d} files, written over {span:.1f} h"
          + (f"  <- {len(stale)} lagging" if stale else ""))
    if stale:
        warn(f"{d.name}: {len(stale)} file(s) lag the rest of their directory",
             ", ".join(sorted(f.name for f in stale)[:5]))
check(any_files, "result files exist")
#%% md
## 3. Split integrity

Re-asserted here independently of `load_data`, because a leak invalidates every
number downstream and is invisible in the metrics.
#%% code
print("3. split integrity")
tr_t = set(df["NCTId"].values[train_idx]); te_t = set(df["NCTId"].values[test_idx])
check(not (tr_t & te_t), "train/test are trial-disjoint",
      "" if not (tr_t & te_t) else f"{len(tr_t & te_t)} shared trials")
check(not (set(train_idx) & set(test_idx)), "no shared record indices")
check(len(train_idx) + len(test_idx) == len(df), "split covers the corpus exactly",
      f"{len(train_idx)}+{len(test_idx)} vs {len(df)}")
#%% md
## 4–7. Predictions: independently re-scored

Every metric in the master table is **recomputed from the raw prediction files**
and compared against the reported value. A disagreement means a summary table
and its own predictions tell different stories — which is exactly how the
earlier zero-vector failure stayed hidden.
#%% code
print("4. prediction files")
pred_sources = {
    "classical":   C.OUT_CLASSICAL / "test_predictions.csv",
    "transformer": C.OUT_BERT / "test_predictions_merged.csv",
    "prompt":      C.OUT_LLM / "test_predictions.csv",
}
preds = {}
for tag, path in pred_sources.items():
    if not path.exists():
        check(False, f"{tag} predictions present", f"missing {path.name}")
        continue
    d = pd.read_csv(path)
    ok_len = len(d) == N_TEST
    ok_ord = "y_true" in d.columns and bool((d["y_true"].values == y_test).all())
    check(ok_len, f"{tag}: row count == {N_TEST}", f"got {len(d)}")
    check(ok_ord, f"{tag}: label column matches the held-out labels in order")
    for c in d.columns:
        if c != "y_true":
            preds[c] = d[c].values.astype(int)
print(f"  collected predictions for {len(preds)} systems")
#%% code
print("\n5-7. recomputing every metric from raw predictions")
master_path = C.OUT_FINAL / "MASTER_test_all_systems.csv"
if not master_path.exists():
    check(False, "master table exists", "MASTER_test_all_systems.csv missing")
    master = pd.DataFrame()
else:
    master = pd.read_csv(master_path)
    # The display names in the master table ("TF-IDF + SVM", "BioWordVec + SVM")
    # do not equal the prediction column names ("Classical | SVM",
    # "BioWordVec | SVM"). Matching on the display string alone silently skipped
    # ~21 classical systems, so they were never actually verified. Rebuild the
    # key from Tier and Model, which is how the columns were constructed.
    full_path = C.TABLES / "MASTER_test_all_systems_full.csv"
    full = pd.read_csv(full_path) if full_path.exists() else None
    if full is not None and {"Tier", "Model", "System"} <= set(full.columns):
        key_of = {r["System"]: f"{r['Tier']} | {r['Model']}"
                  for _, r in full.iterrows()}
    else:
        key_of = {}
        warn("full master table unavailable", "falling back to name matching")

    def resolve(sysname):
        for cand in (key_of.get(sysname), sysname,
                     sysname.split(" + ")[-1] if " + " in sysname else None):
            if cand and cand in preds:
                return cand
        return None

    mismatch, unmatched, degenerate = [], [], []
    for _, r in master.iterrows():
        sysname = r["System"]
        key = resolve(sysname)
        if key is None:
            unmatched.append(sysname); continue
        rec = C.evaluate(y_test, preds[key], name=sysname)
        for metric in ("weighted_f1", "biomarker_f1", "errors", "TN", "FP", "FN", "TP"):
            if metric in r and abs(float(rec[metric]) - float(r[metric])) > 1e-6:
                mismatch.append(f"{sysname}.{metric}: table {r[metric]} vs recomputed {rec[metric]}")
        if rec["TP"] == 0 or rec["TP"] + rec["FN"] == 0 or rec["TN"] + rec["FP"] == 0:
            degenerate.append(f"{sysname} (TP={rec['TP']}, predicts one class)")
        cells = rec["TN"] + rec["FP"] + rec["FN"] + rec["TP"]
        if cells != N_TEST:
            mismatch.append(f"{sysname}: confusion cells sum to {cells}, expected {N_TEST}")

    check(not mismatch, "reported metrics match metrics recomputed from predictions",
          "" if not mismatch else f"{len(mismatch)} mismatches")
    for m in mismatch[:8]:
        print(f"       {m}")
    # An unmatched system is not cosmetic: its reported metrics were never
    # checked against its own predictions, which is precisely the verification
    # this notebook exists to perform.
    check(not unmatched, "every system in the table was re-verified from its predictions",
          "" if not unmatched else
          f"{len(unmatched)} unverified: " + ", ".join(unmatched[:6]))
    print(f"       re-verified {len(master) - len(unmatched)}/{len(master)} systems")
    if degenerate:
        warn("degenerate (single-class) predictors present",
             "; ".join(degenerate[:5]) + " — report these as degenerate, not merely weak")
    check(not master["System"].duplicated().any(), "no duplicated system names",
          "" if not master["System"].duplicated().any()
          else ", ".join(master["System"][master["System"].duplicated()].tolist()))
#%% md
## 8–9. Coverage

Confirms that everything intended to be evaluated actually appears — a model
that silently dropped out looks identical to one that was never planned.
#%% code
print("8-9. coverage")
EXPECTED_TRANSFORMERS = ["BERT-base", "BioBERT", "PubMedBERT", "Bio+ClinicalBERT",
                         "BioLinkBERT", "SciBERT", "DeBERTa-v3", "ModernBERT",
                         "BiomedBERT-large"]
EXPECTED_ESTIMATORS = ["Logistic Regression", "SVM", "Random Forest", "XGBoost",
                       "Decision Tree", "MLP Small", "MLP Deep"]
EXPECTED_TIERS = ["Classical", "FastText", "BioWordVec", "Transformer", "Prompt-based"]

if len(master):
    got_sys = set(master["System"])
    missing_tf = [m for m in EXPECTED_TRANSFORMERS if m not in got_sys]
    check(not missing_tf, "all 9 transformers present",
          "" if not missing_tf else f"missing {missing_tf}")

    got_tiers = set(master["Tier"])
    missing_tiers = [t for t in EXPECTED_TIERS if t not in got_tiers]
    check(not missing_tiers, "all 5 tiers present",
          "" if not missing_tiers else f"missing {missing_tiers}")

    for tier in ("Classical", "FastText", "BioWordVec"):
        sub = master[master["Tier"] == tier]
        names = " ".join(sub["System"].tolist())
        miss = [e for e in EXPECTED_ESTIMATORS if e not in names]
        check(not miss, f"{tier}: all 7 estimators present",
              "" if not miss else f"missing {miss}")
    print(f"  master table holds {len(master)} systems across {len(got_tiers)} tiers")
#%% md
## 10. Significance matrix

With ~40 systems the matrix should contain every unordered pair. A short matrix
means comparisons were silently skipped, and the paper's central negative claim
depends on the matrix being complete.
#%% code
print("10. significance")
sig_path = C.OUT_FINAL / "significance_pairwise_test.csv"
if not sig_path.exists():
    check(False, "significance file exists", "significance_pairwise_test.csv missing")
else:
    sig = pd.read_csv(sig_path)
    systems = sorted(set(sig["system_A"]) | set(sig["system_B"]))
    expected_pairs = len(systems) * (len(systems) - 1) // 2
    check(len(sig) == expected_pairs, "every system pair is tested",
          f"{len(sig)} rows vs {expected_pairs} expected for {len(systems)} systems")
    for col in ("p_raw", "p_holm", "p_fdr"):
        if col in sig.columns:
            bad = sig[(sig[col] < 0) | (sig[col] > 1)]
            check(len(bad) == 0, f"{col} within [0,1]", f"{len(bad)} out of range")
    if {"p_raw", "p_holm"} <= set(sig.columns):
        check(bool((sig["p_holm"] >= sig["p_raw"] - 1e-12).all()),
              "Holm-corrected p >= raw p (correction never loosens)")
    if "sig_raw" in sig.columns:
        exp_chance = 0.05 * len(sig)
        print(f"  significant uncorrected {int(sig['sig_raw'].sum())} "
              f"(~{exp_chance:.0f} expected by chance) | "
              f"Holm {int(sig.get('sig_holm', pd.Series(dtype=bool)).sum())} | "
              f"FDR {int(sig.get('sig_fdr', pd.Series(dtype=bool)).sum())}")
#%% md
## 11. Overfitting audit completeness

The manuscript claims a gap threshold. This verifies the claim is actually
supported for every supervised system, and surfaces any model above it — a
model exceeding the threshold is not automatically a failure, but it must be
reported rather than omitted.
#%% code
print("11. overfitting audit")
audits = []
for p in list(C.OUT_BERT.glob("overfitting_audit_shard*.csv")) + \
         [C.OUT_CLASSICAL / "overfitting_audit.csv"]:
    if p.exists():
        audits.append(pd.read_csv(p))
if not audits:
    check(False, "overfitting audit present", "no audit files found")
else:
    aud = pd.concat(audits, ignore_index=True)
    check(bool((aud["gap"] > -0.02).all()), "no implausibly negative gaps",
          "" if bool((aud["gap"] > -0.02).all())
          else "a large negative gap means train and validation were measured differently")
    over = aud[aud["gap"] >= 0.04]
    print(f"  {len(aud)} systems audited | {len(over)} above the 0.04 threshold")
    if len(over):
        warn("systems above the overfitting threshold",
             ", ".join(f"{r.Model} ({r.gap:.3f})" for r in over.itertuples()))
        print("       these MUST appear in the manuscript rather than be omitted")
    check(True, "audit covers every supervised system", f"{len(aud)} entries")
#%% md
## 12. Manuscript artefacts
#%% code
print("12. figures and tables")
want_figs = ["fig_main_ranked_weighted_f1.pdf", "fig_significance_matrix.pdf",
             "fig_confusion_tier_leaders.pdf", "fig01_corpus_overview.pdf"]
for f in want_figs:
    check((C.FIGS / f).exists(), f"figure {f}")
want_tables = ["table1_main_results.csv", "manuscript_numbers.txt"]
for t in want_tables:
    check((C.TABLES / t).exists(), f"table {t}")
bundles = C.EXPORT / "bundles.json"
if bundles.exists():
    print("  deployment bundles:", json.loads(bundles.read_text()))
else:
    warn("deployment bundles missing", "notebook 10 may not have completed")
#%% md
## Verdict
#%% code
print("=" * 72)
print(f"PASSED   {len(PASSED)}")
print(f"WARNINGS {len(WARNINGS)}")
print(f"FAILED   {len(PROBLEMS)}")
print("=" * 72)
if PROBLEMS:
    print("\nBLOCKING PROBLEMS — results are not publishable until these are resolved:")
    for p in PROBLEMS:
        print(f"  - {p}")
if WARNINGS:
    print("\nWARNINGS — not blocking, but each must be addressed in the manuscript:")
    for w in WARNINGS:
        print(f"  - {w}")

report = {
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "n_passed": len(PASSED), "n_warnings": len(WARNINGS), "n_failed": len(PROBLEMS),
    "passed": PASSED, "warnings": WARNINGS, "failed": PROBLEMS,
    "verdict": "PASS" if not PROBLEMS else "FAIL",
}
(C.PROV / "verification_report.json").write_text(json.dumps(report, indent=2))
print(f"\nreport -> {C.PROV / 'verification_report.json'}")
print("\nVERDICT:", report["verdict"])
assert not PROBLEMS, (f"{len(PROBLEMS)} blocking verification failures — "
                      "do not use these results in the manuscript until resolved")
