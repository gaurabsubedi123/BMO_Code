"""Merge the Claude replies on the newer trials' outcome lines (analysis/claude_new_trials.py) with the
local models' scores and compare all systems against BiomedBERT-large.

    python add_claude.py NEW_TRIALS_DIR CLAUDE_REPLIES_JSONL

Reads NEW_TRIALS_DIR/outcomes_all_scored.csv; writes NEW_TRIALS_DIR/outcomes_all_with_claude_comparison.csv.
A Claude reply that is not a clean yes or no (unreadable, refusal) is scored as wrong, as in the paper.
Misses are compared with an exact McNemar test on the biomarker lines, Holm-corrected across all comparisons.
"""
import json, math, sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import beta

HERE = pathlib.Path(__file__).resolve().parent
ns = {}
exec(open(HERE / "analyze2.py").read().split("c = pd.read_csv")[0]
     .replace("pathlib.Path(__file__).with_name('analyze.py')", repr(str(HERE / "analyze.py"))), ns)
drug = ns["drug"]
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


def main(new_dir, replies):
    new_dir = pathlib.Path(new_dir)
    d = pd.read_csv(new_dir / "outcomes_all_scored.csv", dtype={"line_no": str})
    y = d.label.values; dr = drug(d.model_input).values
    assert len(d) == 2754 and y.sum() == 784
    recs = {}
    for line in open(replies):
        r = json.loads(line)
        if r["result"] == "succeeded" or r["custom_id"] not in recs:
            recs[r["custom_id"]] = r
    preds, extra = {}, {}
    for s in LOCAL:
        preds[s] = d[s].values >= 0.5
    preds["Qwen3.8-27B"] = d["Qwen3.8-27B"].values >= 0.5
    preds["Qwen3.8-27B, tuned cut-off"] = d["Qwen3.8-27B"].values >= 0.10
    for mid, name in CLAUDE.items():
        got = [recs.get(f"{mid}__outcomes__{i}") for i in range(len(d))]
        assert all(g is not None and g["result"] == "succeeded" for g in got), f"{mid}: incomplete run"
        p = np.array([g["pred"] for g in got])
        preds[name] = np.where(p < 0, 1 - y, p).astype(bool)          # unreadable or refused = wrong
        ok = p >= 0
        extra[name] = dict(unreadable=int((~ok).sum()), missed_readable=int(((y == 1) & ok & (p == 0)).sum()),
                           biomarker_readable=int(((y == 1) & ok).sum()))
    ref = preds["BiomedBERT-large"]; P, N, Nx = int((y == 1).sum()), int((y == 0).sum()), int(((y == 0) & ~dr).sum())
    rows = []
    for s, f in preds.items():
        fn = int(((y == 1) & ~f).sum()); lo = beta.ppf(.025, fn, P - fn + 1) if fn else 0.0; hi = beta.ppf(.975, fn + 1, P - fn)
        fpx = int(((y == 0) & f & ~dr).sum())
        rows.append(dict(system=s, kind="Claude" if s in CLAUDE.values() else ("Prompted, open" if s.startswith("Qwen") else "Trained"),
                         missed=fn, missed_pct=round(100 * fn / P, 1), ci=f"{100*lo:.1f}-{100*hi:.1f}",
                         fp=int(((y == 0) & f).sum()), fp_no_drug=fpx, fp_no_drug_pct=round(100 * fpx / Nx, 1),
                         flagged_pct=round(100 * f.mean(), 1),
                         p=1.0 if s == "BiomedBERT-large" else mcnemar(int(((y == 1) & ref & ~f).sum()), int(((y == 1) & ~ref & f).sum())),
                         **extra.get(s, {})))
    t = pd.DataFrame(rows); m = t.system != "BiomedBERT-large"
    t.loc[m, "p_holm"] = holm(t.loc[m, "p"].values)
    t = t.sort_values(["missed", "fp_no_drug"]).reset_index(drop=True)
    t.to_csv(new_dir / "outcomes_all_with_claude_comparison.csv", index=False)
    print(t.drop(columns=["p"]).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
