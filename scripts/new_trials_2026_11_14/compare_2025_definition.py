"""Newer-trial comparison under the paper's (2025 workbook) biomarker definition.

The newer CTO annotations label drug concentration and anti-drug antibody measures as non-biomarkers,
whereas the training data counted them as biomarkers. So that both tests use one definition, lines whose
wording describes such measures (drug_level.py) are set aside, and every system is compared on the rest.

    python compare_2025_definition.py NEW_TRIALS_DIR CLAUDE_REPLIES_JSONL

Writes NEW_TRIALS_DIR/{outcomes,eligibility}_2025def_comparison.csv and prints trial-level counts.
"""
import json, math, sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import beta

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drug_level import drug                       # noqa: E402
LOCAL = ["BiomedBERT-large", "BioLinkBERT", "SciBERT", "BERT-base", "DeBERTa-v3", "BioplusClinicalBERT", "ModernBERT",
         "PubMedBERT", "BioBERT", "TF-IDF + logistic regression"]
CLAUDE = {"claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4-5": "Claude Haiku 4.5",
          "claude-opus-5-5": "Claude Opus 5.5, reasoning", "claude-fable-5-2": "Claude Fable 5.2, reasoning"}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def predictions(d, part, recs):
    y = d.label.values
    P = {s: d[s].values >= 0.5 for s in LOCAL}
    P["Qwen3.8-27B"] = d["Qwen3.8-27B"].values >= 0.5
    P["Qwen3.8-27B, tuned cut-off"] = d["Qwen3.8-27B"].values >= 0.10
    extra = {}
    if part == "outcomes":
        for mid, name in CLAUDE.items():
            got = [recs[f"{mid}__outcomes__{i}"] for i in d.index]
            p = np.array([g["pred"] for g in got])
            P[name] = np.where(p < 0, 1 - y, p).astype(bool)
            ok = p >= 0
            extra[name] = dict(unreadable=int((~ok).sum()), missed_readable=int(((y == 1) & ok & (p == 0)).sum()),
                               biomarker_readable=int(((y == 1) & ok).sum()))
    return P, extra


def main(new_dir, replies):
    new_dir = pathlib.Path(new_dir)
    recs = {}
    for line in open(replies):
        r = json.loads(line)
        if r["result"] == "succeeded" or r["custom_id"] not in recs:
            recs[r["custom_id"]] = r
    for part in ("outcomes", "eligibility"):
        full = pd.read_csv(new_dir / f"{part}_all_scored.csv", dtype={"line_no": str})
        dr = drug(full.model_input)
        d = full[~dr]                                  # keep original index: Claude ids use it
        y = d.label.values
        print(f"\n{part}: {len(full)} lines, {int(dr.sum())} drug-level set aside ({int(full.label[dr].sum())} of them labeled biomarker); "
              f"analysed {len(d)} lines, {int(y.sum())} biomarker, {int((y == 0).sum())} not")
        P, extra = predictions(d, part, recs)
        ref = P["BiomedBERT-large"]; Pn, Nn = int((y == 1).sum()), int((y == 0).sum())
        rows = []
        for s, f in P.items():
            fn = int(((y == 1) & ~f).sum()); fp = int(((y == 0) & f).sum())
            lo = beta.ppf(.025, fn, Pn - fn + 1) if fn else 0.0; hi = beta.ppf(.975, fn + 1, Pn - fn)
            flo = beta.ppf(.025, fp, Nn - fp + 1) if fp else 0.0; fhi = beta.ppf(.975, fp + 1, Nn - fp)
            rows.append(dict(system=s, missed=fn, missed_pct=round(100 * fn / Pn, 1), ci=f"{100*lo:.1f}-{100*hi:.1f}",
                             fp=fp, fp_pct=round(100 * fp / Nn, 1), fp_ci=f"{100*flo:.1f}-{100*fhi:.1f}",
                             flagged=int(f.sum()), flagged_pct=round(100 * f.mean(), 1),
                             p=1.0 if s == "BiomedBERT-large" else mcnemar(int(((y == 1) & ref & ~f).sum()), int(((y == 1) & ~ref & f).sum())),
                             **extra.get(s, {})))
        t = pd.DataFrame(rows); m = t.system != "BiomedBERT-large"
        t.loc[m, "p_holm"] = holm(t.loc[m, "p"].values)
        t = t.sort_values(["missed", "fp"]).reset_index(drop=True)
        t.to_csv(new_dir / f"{part}_2025def_comparison.csv", index=False)
        print(t.drop(columns=["p"]).to_string(index=False))
        if part == "outcomes":
            g = d.assign(f=ref).groupby("nct").agg(bio=("label", "max"), hit=("f", lambda v: 0))
            tb = d[d.label == 1].assign(f=ref[y == 1]).groupby("nct").f.max()
            print(f"trials with a biomarker outcome: {len(tb)}; BiomedBERT-large flagged >=1 in {int(tb.sum())}; missed entirely: {list(tb[~tb].index)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
