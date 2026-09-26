"""Test-set metrics and McNemar tests for the added comparisons.

Qwen3.8-27B (default and tuned threshold) and three Claude models were added
after the paper's forty systems were fixed, so they get their own Holm family:
each added system against BioLinkBERT (the best supervised system) and against
Qwen2.5-7B with a tuned threshold (the best of the ten prompted systems). Metrics, bootstrap intervals and the McNemar test
are the paper's own; the script first reproduces published rows to prove it.

    python claude_significance.py
"""
from pathlib import Path

import os

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("BMO_RERUN_DIR", HERE.parent.parent.parent / "rerun_results"))
OUT = RUN / "outputs" / "prompt_llm" / "claude"
# Table 3's intervals come from notebook 07, not bmo_common: 2,000 resamples, this seed.
N_BOOT, RNG_SEED = 2000, 20260828
MODELS = {"claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
          "claude-haiku-4-5": "Claude Haiku 4.5"}


def bootstrap_ci(y_true, y_pred, n=N_BOOT, seed=RNG_SEED):
    """Notebook 07's boot_ci for weighted F1, line for line."""
    rng = np.random.default_rng(seed)
    idx, vals = np.arange(len(y_true)), []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y_true[b])) < 2:
            continue
        vals.append(f1_score(y_true[b], y_pred[b], average="weighted", pos_label=1,
                             zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(b, c):
    """analysis/significance_all_systems.py: exact binomial when min(b, c) < 25."""
    n = b + c
    if n == 0:
        return 1.0
    if min(b, c) < 25:
        return min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5))
    raise ValueError("asymptotic branch not expected at these error counts")


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def row(y, p):
    return {"weighted_f1": f1_score(y, p, average="weighted"),
            "biomarker_f1": f1_score(y, p, pos_label=1),
            "errors": int((p != y).sum()), "missed": int(((y == 1) & (p == 0)).sum()),
            "false_alarms": int(((y == 0) & (p == 1)).sum())}


def main():
    q = pd.read_csv(RUN / "outputs" / "prompt_llm" / "test_predictions.csv")
    bl = pd.read_csv(RUN / "outputs" / "bert_models" / "test_predictions_merged.csv")
    y = q.y_true.values
    assert (bl.y_true.values == y).all()
    ref = {"BioLinkBERT": bl["BioLinkBERT"].values,
           "Qwen2.5-7B-Instruct (tuned)": q["Qwen2.5-7B-Instruct (tuned threshold)"].values}

    # 1. prove the machinery on published Table 3 rows
    published = {"BioLinkBERT": (0.9971, 0.9912, 1.0000, 0.9954, 1),
                 "Qwen2.5-7B-Instruct (tuned)": (0.9763, 0.9610, 0.9912, 0.9619, 8)}
    for name, (wf1, lo, hi, bf1, err) in published.items():
        r, (l, h) = row(y, ref[name]), bootstrap_ci(y, ref[name])
        got = (round(r["weighted_f1"], 4), round(l, 4), round(h, 4),
               round(r["biomarker_f1"], 4), r["errors"])
        status = "matches" if got == (wf1, lo, hi, bf1, err) else "DIFFERS"
        print(f"{name}: {got} vs published {(wf1, lo, hi, bf1, err)} -> {status}")

    # 2. Claude rows; a reply that did not open with Yes or No is scored as wrong
    raw = pd.read_csv(OUT / "raw_replies.csv")
    te = np.load(RUN / "outputs" / "dataset" / "test_idx.npy")
    rows, tests = [], []
    for mid, name in MODELS.items():
        g = raw[(raw.model == mid) & (raw.run == "run1")].set_index("row").pred.reindex(te).values
        g2 = raw[(raw.model == mid) & (raw.run == "run2")].set_index("row").pred.reindex(te).values
        unread = g < 0
        p = np.where(unread, 1 - y, g)
        r = row(y, p)
        lo, hi = bootstrap_ci(y, p)
        rows.append({"System": name, **r, "ci_low": lo, "ci_high": hi,
                     "unreadable": int(unread.sum()),
                     "errors_readable_only": int(((g != y) & ~unread).sum()),
                     "run_agreement": int((g == g2).sum())})
        for rname, rp in ref.items():
            a, b_ = (p == y), (rp == y)
            only_claude_right, only_ref_right = int((a & ~b_).sum()), int((~a & b_).sum())
            tests.append({"Claude": name, "vs": rname, "discordant": only_claude_right + only_ref_right,
                          "only_claude_wrong": only_ref_right, "only_other_wrong": only_claude_right,
                          "p_raw": mcnemar(only_claude_right, only_ref_right)})
    # Qwen3.8-27B: probabilities from qwen38_zeroshot.py, both thresholds
    q38 = pd.read_csv(RUN / "outputs" / "prompt_llm" / "qwen38" / "test_predictions.csv")
    assert (q38.y_true.values == y).all()
    for col in [c for c in q38.columns if c.startswith("Qwen3.8")]:
        p = q38[col].values
        r, (lo, hi) = row(y, p), bootstrap_ci(y, p)
        rows.append({"System": col, **r, "ci_low": lo, "ci_high": hi, "unreadable": 0,
                     "errors_readable_only": r["errors"], "run_agreement": np.nan})
        for rname, rp in ref.items():
            a, b_ = (p == y), (rp == y)
            only_new_right, only_ref_right = int((a & ~b_).sum()), int((~a & b_).sum())
            tests.append({"Claude": col, "vs": rname, "discordant": only_new_right + only_ref_right,
                          "only_claude_wrong": only_ref_right, "only_other_wrong": only_new_right,
                          "p_raw": mcnemar(only_new_right, only_ref_right)})
    t = pd.DataFrame(tests).rename(columns={"Claude": "added_system",
                                            "only_claude_wrong": "only_added_wrong"})
    t["p_holm"] = holm(t.p_raw.values)
    m = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(m.round(4).to_string(index=False))
    print(t.round(4).to_string(index=False))
    m.to_csv(OUT / "added_test_metrics.csv", index=False)
    t.to_csv(OUT / "added_mcnemar_holm.csv", index=False)


if __name__ == "__main__":
    main()
