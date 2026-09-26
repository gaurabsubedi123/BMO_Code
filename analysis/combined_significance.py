"""Test-set metrics and McNemar tests for all 45 systems.

The study's 45 systems are the pipeline's 40 plus Qwen3.8-27B (default and tuned
threshold, from qwen38_zeroshot.py) and three Claude models (from
claude_zeroshot.py). Every pair is compared with the paper's exact McNemar test
and corrected with Holm and Benjamini-Hochberg over all 990 pairs.

The script first proves its machinery: it rebuilds the 40-system family and
checks it against the published file (427 raw, 233 Holm), and it reproduces two
published Table 4 rows with notebook 07's bootstrap, before scoring anything new.

    python combined_significance.py
"""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("BMO_RERUN_DIR", HERE.parent.parent.parent / "rerun_results"))
OUT_DIR = RUN / "outputs"
PUBLISHED_40 = HERE.parent / "results" / "rerun_2026-08-28" / "significance_pairwise_test.csv"
DEST = OUT_DIR / "final_results_45"
DEST.mkdir(parents=True, exist_ok=True)
# Table 4's intervals come from notebook 07, not bmo_common: 2,000 resamples, this seed.
N_BOOT, RNG_SEED = 2000, 20260828
CLAUDE = {"claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
          "claude-haiku-4-5": "Claude Haiku 4.5"}


def mcnemar(b, c):
    """analysis/significance_all_systems.py: exact binomial when min(b, c) < 25."""
    n = b + c
    if n == 0:
        return 1.0
    if min(b, c) < 25:
        return min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5))
    return float(chi2.sf((abs(b - c) - 1.0) ** 2 / n, 1))


def holm(p):
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p):
    p = np.asarray(p)
    order, m, adj, running = np.argsort(p)[::-1], len(p), np.empty(len(p)), 1.0
    for k, i in enumerate(order):
        running = min(running, p[i] * m / (m - k))
        adj[i] = running
    return adj


def bootstrap_ci(y_true, y_pred, n=N_BOOT, seed=RNG_SEED):
    """Notebook 07's boot_ci for weighted F1, line for line."""
    rng = np.random.default_rng(seed)
    idx, vals = np.arange(len(y_true)), []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y_true[b])) < 2:
            continue
        vals.append(f1_score(y_true[b], y_pred[b], average="weighted", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_predictions():
    cl = pd.read_csv(OUT_DIR / "classical_ml" / "test_predictions.csv")
    be = pd.read_csv(OUT_DIR / "bert_models" / "test_predictions_merged.csv")
    pr = pd.read_csv(OUT_DIR / "prompt_llm" / "test_predictions.csv")
    y = cl.y_true.values
    assert (y == be.y_true.values).all() and (y == pr.y_true.values).all()
    pool = {c: d[c].values.astype(int) for d in (cl, be, pr) for c in d.columns if c != "y_true"}
    names40 = sorted(set(pd.read_csv(PUBLISHED_40).system_A) | set(pd.read_csv(PUBLISHED_40).system_B))
    P = {n: pool[n] for n in names40}
    q = pd.read_csv(OUT_DIR / "prompt_llm" / "qwen38" / "test_predictions.csv")
    assert (q.y_true.values == y).all()
    for c in ("Qwen3.8-27B", "Qwen3.8-27B (tuned threshold)"):
        P[c] = q[c].values.astype(int)
    raw = pd.read_csv(OUT_DIR / "prompt_llm" / "claude" / "raw_replies.csv")
    te = np.load(OUT_DIR / "dataset" / "test_idx.npy")
    for mid, name in CLAUDE.items():
        p = raw[(raw.model == mid) & (raw.run == "run1")].set_index("row").pred.reindex(te).values
        assert not np.isnan(p).any()
        P[name] = np.where(p < 0, 1 - y, p).astype(int)   # an unreadable reply is scored as wrong
    return y, P, names40


def family(y, P, names):
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca, cb = P[a] == y, P[b] == y
        x, z = int((ca & ~cb).sum()), int((~ca & cb).sum())
        rows.append(dict(system_A=a, system_B=b, A_only_correct=x, B_only_correct=z,
                         discordant=x + z, p_raw=mcnemar(x, z),
                         errors_A=int((~ca).sum()), errors_B=int((~cb).sum())))
    d = pd.DataFrame(rows)
    d["p_holm"], d["p_fdr"] = holm(d.p_raw.values), bh(d.p_raw.values)
    for k in ("raw", "holm", "fdr"):
        d[f"sig_{k}"] = d[f"p_{k}"] < 0.05
    return d


def main():
    y, P, names40 = load_predictions()

    # 1. the 40-system family must reproduce the published one
    pub = pd.read_csv(PUBLISHED_40)
    d40 = family(y, P, names40)
    both = d40.merge(pub, on=["system_A", "system_B"], suffixes=("", "_pub"))
    assert len(both) == 780 and (both.sig_holm == both.sig_holm_pub).all()
    assert int(d40.sig_raw.sum()) == 427 and int(d40.sig_holm.sum()) == 233
    print("40-system family reproduces the published one: 780 pairs, 427 raw, 233 Holm")

    # 2. the bootstrap must reproduce published Table 4 rows
    for name, want in [("BioLinkBERT", (0.9912, 1.0000)),
                       ("Qwen2.5-7B-Instruct (tuned threshold)", (0.9610, 0.9912))]:
        got = tuple(round(v, 4) for v in bootstrap_ci(y, P[name]))
        assert got == want, (name, got, want)
    print("bootstrap reproduces the published intervals")

    # 3. the 45-system family
    names = sorted(P)
    d = family(y, P, names)
    d.to_csv(DEST / "significance_pairwise_test_45.csv", index=False)
    W = np.array([P[n] != y for n in names])
    alone = W.sum(0) == 1
    err = pd.DataFrame({"System": names, "errors": W.sum(1),
                        "unique_errors": [int((W[i] & alone).sum()) for i in range(len(names))]})
    err.sort_values("errors").to_csv(DEST / "per_system_errors_45.csv", index=False)
    order = err.sort_values(["errors", "System"]).System.tolist()
    k = int((err.errors <= 5).sum())
    top = d[d.system_A.isin(order[:k]) & d.system_B.isin(order[:k])]
    print(f"45 systems, {len(d)} pairs: {int(d.sig_raw.sum())} raw, {int(d.sig_holm.sum())} Holm, "
          f"{int(d.sig_fdr.sum())} BH")
    print(f"leading group (<= 5 errors): {k} systems, {len(top)} pairs, "
          f"{int(top.sig_holm.sum())} significant after Holm")
    print(f"no unique errors: {int((err.unique_errors == 0).sum())} of {len(names)}; "
          f"identical pairs: {int((d.discordant == 0).sum())}")

    # 4. metrics for the five systems new to this family
    rows = []
    for name in ["Qwen3.8-27B (tuned threshold)", "Qwen3.8-27B", *CLAUDE.values()]:
        p = P[name]
        lo, hi = bootstrap_ci(y, p)
        rows.append({"System": name, "weighted_f1": f1_score(y, p, average="weighted"),
                     "biomarker_f1": f1_score(y, p, pos_label=1), "errors": int((p != y).sum()),
                     "missed": int(((y == 1) & (p == 0)).sum()),
                     "false_alarms": int(((y == 0) & (p == 1)).sum()), "ci_low": lo, "ci_high": hi})
    m = pd.DataFrame(rows)
    m.to_csv(OUT_DIR / "prompt_llm" / "claude" / "qwen38_claude_test_metrics.csv", index=False)
    print(m.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
