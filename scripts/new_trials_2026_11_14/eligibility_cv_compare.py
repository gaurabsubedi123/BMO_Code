"""Compare the trial-grouped CV results (eligibility_cv.py) with the outcome-trained models already scored on the
same newer-trial lines (score_trained.py). Paired on identical lines; misses compared with an exact McNemar test
on the biomarker lines, Holm-corrected within each table.

Usage: eligibility_cv_compare.py NEW_TRIALS_DIR CV_DIR
"""
import sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import binomtest, beta


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

new, cv = map(pathlib.Path, sys.argv[1:3])
KEY = ["nct", "section", "line_no"]
MODELS = ["BiomedBERT-large", "SciBERT", "BioLinkBERT"]


def ci(k, n):  # Clopper-Pearson 95%
    return (beta.ppf(.025, k, n - k + 1) if k else 0.0) * 100, (beta.ppf(.975, k + 1, n - k) if k < n else 1.0) * 100


def mcnemar(miss_a, miss_b):
    b, c = int((miss_a & ~miss_b).sum()), int((~miss_a & miss_b).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, .5).pvalue


def table(part, cvfile, basefile, variants):
    d = pd.read_csv(cv / cvfile, dtype={"line_no": str})
    base = pd.read_csv(new / basefile, dtype={"line_no": str})[KEY + MODELS]
    base = base.rename(columns={m: f"{m}|outcome-trained" for m in MODELS})
    d = d.merge(base, on=KEY, how="left", validate="one_to_one")
    assert d.filter(like="|outcome-trained").notna().all().all(), "line join failed"
    y = d.label.values == 1; rows, ref = [], d["BiomedBERT-large|outcome-trained"].values >= .5
    for m in MODELS:
        for v in ["outcome-trained"] + variants:
            p = d[f"{m}|{v}"].values >= .5; miss = y & ~p; fp = ~y & p
            lo, hi = ci(miss.sum(), y.sum())
            trials_pos = d[y].nct.unique(); caught = d[p & y].nct.unique()
            rows.append({"part": part, "model": m, "training": v, "missed": int(miss.sum()), "biomarker_lines": int(y.sum()),
                         "missed_pct": round(100 * miss.mean() / y.mean(), 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
                         "false_pos": int(fp.sum()), "nonbiomarker_lines": int((~y).sum()),
                         "fp_pct": round(100 * fp.sum() / (~y).sum(), 1), "flagged_pct": round(100 * p.mean(), 1),
                         "trials_caught": f"{len(set(caught))}/{len(trials_pos)}",
                         "p_vs_deployed_misses": mcnemar(miss, y & ~ref)})
    t = pd.DataFrame(rows); others = t.index[~((t.model == "BiomedBERT-large") & (t.training == "outcome-trained"))]
    t["p_holm"] = np.nan; t.loc[others, "p_holm"] = holm(t.loc[others, "p_vs_deployed_misses"])
    return t, d


te, de = table("eligibility", "eligibility_cv_scored.csv", "eligibility_trained_scored.csv", ["A", "B"])
to, do = table("outcomes", "outcomes_cv_scored.csv", "outcomes_trained_scored.csv", ["B"])
out = pd.concat([te, to]); out["training"] = out.training.replace(
    {"A": "eligibility only (CV)", "B": "outcomes + eligibility (CV)"})
out.to_csv(cv / "comparison.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns=["p_vs_deployed_misses"]).to_string(index=False))
pf = pd.read_csv(cv / "per_fold.csv")
print("\nVariant B on the paper's 341 held-out outcomes, per fold (deployed model: 0 missed, 2 FP):")
print(pf[pf.variant == "B"][["model", "fold", "heldout_missed", "heldout_fp"]].to_string(index=False))
# what the eligibility-trained model still misses
m = de[(de.label == 1) & (de["BiomedBERT-large|B"] < .5)]
m[["nct", "section", "line_text", "labels"]].to_csv(cv / "biomedbert_B_eligibility_misses.csv", index=False)
print(f"\nBiomedBERT-large (B) eligibility misses written: {len(m)}")
