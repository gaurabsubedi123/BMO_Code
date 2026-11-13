"""Do the systems fail on the same outcome measures?

For every held-out record, count how many systems got it wrong, and break
that down by family (classical, fine-tuned encoder, prompted) and by
biomarker modality. Then look at the strongest systems: are their errors the
same few records or scattered across different ones?

Uses combined_significance.load_predictions(), so the predictions are exactly
the ones behind the published 45-system tables. Reasoning-model predictions are
added when their files exist (claude_reasoning.py, qwen38_thinking.py).

    python error_overlap.py

Writes to $BMO_RERUN_DIR/outputs/error_overlap/:
    per_record_errors.csv   one row per held-out record, which systems missed it
    hard_records.csv        records wrong in at least a quarter of the systems, with text
    leading_overlap.csv     pairwise shared errors among the leading systems
    summary.txt             the numbers quoted in the paper
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_significance as cs                      # noqa: E402


def _load_modality():
    """modality() from recall_by_modality.py, read from its source so the taxonomy is
    identical; importing that module would run its whole analysis."""
    src = (Path(__file__).resolve().parent / "recall_by_modality.py").read_text()
    start = src.index("MRS_UD = ")
    end = src.index('return "Unclassified"') + len('return "Unclassified"')
    ns = {"re": __import__("re")}
    exec(src[start:end], ns)
    return ns["modality"]


modality = _load_modality()

OUT = cs.OUT_DIR / "error_overlap"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTED_KEYS = ("Qwen", "OpenBioLLM", "BART", "Flan", "BioGPT", "Claude")
ENCODER_KEYS = ("BERT", "BioLinkBERT", "PubMedBERT", "BiomedBERT", "ModernBERT", "BlueBERT", "SciBERT",
                "ClinicalBERT", "RoBERTa", "ELECTRA", "DeBERTa")


def family_of(name):
    if any(k in name for k in PROMPTED_KEYS):
        return "prompted"
    if any(k in name for k in ENCODER_KEYS) and "|" not in name:   # classical names are "features | classifier"
        return "encoder"
    return "classical"


def add_reasoning(P, y):
    """Reasoning-model predictions, if those runs have finished."""
    te = np.load(cs.OUT_DIR / "dataset" / "test_idx.npy")
    f = cs.OUT_DIR / "prompt_llm" / "claude_reasoning" / "test_predictions_high.csv"
    if f.exists():
        d = pd.read_csv(f)
        assert (d.row.values == te).all() and (d.y_true.values == y).all()
        for c, name in (("claude-opus-5-5 (high)", "Claude Opus 5.5 (reasoning)"),
                        ("claude-fable-5-2 (high)", "Claude Fable 5.2 (reasoning)")):
            P[name] = d[c].values.astype(int)
    f = cs.OUT_DIR / "prompt_llm" / "qwen38_thinking" / "test_predictions.csv"
    if f.exists():
        d = pd.read_csv(f)
        assert (d.row.values == te).all() and (d.y_true.values == y).all()
        P["Qwen3.8-27B (thinking)"] = d["Qwen3.8-27B (thinking)"].values.astype(int)
    return P


def main():
    y, P, _ = cs.load_predictions()
    assert len(P) == 45, len(P)
    P = add_reasoning(P, y)
    names = sorted(P)
    fam = {n: family_of(n) for n in names}
    E = pd.DataFrame({n: (P[n] != y).astype(int) for n in names})      # 1 = wrong
    errors = E.sum().sort_values()

    df, _, te = (lambda d: (d, None, np.load(cs.OUT_DIR / "dataset" / "test_idx.npy")))(
        pd.read_csv(cs.OUT_DIR / "dataset" / "labeled_dataset.csv"))
    rec = df.iloc[te].reset_index(drop=True)
    rec["modality"] = [modality(a) if yy == 1 else "not a biomarker"
                       for a, yy in zip(rec.Annotation, rec.has_biomarker)]

    per = pd.DataFrame({"row": te, "y_true": y, "modality": rec.modality,
                        "text": rec.clean_freeform,
                        "n_wrong": E.sum(1).values,
                        **{f"n_wrong_{f}": E[[n for n in names if fam[n] == f]].sum(1).values
                           for f in ("classical", "encoder", "prompted")},
                        "systems_wrong": E.apply(lambda r: "; ".join(r.index[r == 1]), axis=1)})
    per.to_csv(OUT / "per_record_errors.csv", index=False)

    n_sys = len(names)
    lines = [f"systems: {n_sys} ({', '.join(f'{f} {sum(v == f for v in fam.values())}' for f in ('classical', 'encoder', 'prompted'))})",
             f"held-out records: {len(y)}; biomarker {int(y.sum())}"]
    never = int((per.n_wrong == 0).sum())
    lines.append(f"records no system got wrong: {never} ({never / len(y):.1%})")
    for k in (1, n_sys // 4, n_sys // 2):
        m = int((per.n_wrong >= k).sum())
        lines.append(f"records wrong in >= {k} systems: {m}")
    hard = per[per.n_wrong >= n_sys / 4].sort_values("n_wrong", ascending=False)
    hard.to_csv(OUT / "hard_records.csv", index=False)
    lines.append(f"hard records (wrong in at least a quarter of systems): {len(hard)}; "
                 f"of them biomarkers {int(hard.y_true.sum())}; by modality: "
                 + ", ".join(f"{k} {v}" for k, v in hard.modality.value_counts().items()))
    share = E.values[per.n_wrong.values >= n_sys / 4].sum() / E.values.sum()
    lines.append(f"share of all errors that fall on these hard records: {share:.1%}")

    # leading systems: at most 5 test errors, as in the paper's leading group
    lead = [n for n in names if errors[n] <= 5]
    lines.append(f"\nleading systems (<= 5 errors): {len(lead)}")
    for n in sorted(lead, key=lambda n: errors[n]):
        idx = np.where(E[n].values == 1)[0]
        lines.append(f"  {n:45s} {errors[n]} errors: "
                     + ", ".join(f"r{te[i]}({'miss' if y[i] else 'FA'},{per.n_wrong[i]}/{n_sys})" for i in idx))
    union = np.where(E[lead].sum(1).values > 0)[0]
    lines.append(f"distinct records behind all leading-system errors: {len(union)} "
                 f"(total leading errors {int(E[lead].values.sum())})")
    shared = [(a, b, int((E[a] & E[b]).sum()), int(errors[a]), int(errors[b]))
              for a, b in itertools.combinations(lead, 2)]
    pd.DataFrame(shared, columns=["A", "B", "shared_errors", "errors_A", "errors_B"]).to_csv(
        OUT / "leading_overlap.csv", index=False)
    top = per.loc[union].sort_values("n_wrong", ascending=False)
    lines.append("records behind leading errors, by how many of the leading systems missed them:")
    for i, r in top.iterrows():
        k = int(E.loc[i, lead].sum())
        lines.append(f"  r{r.row} y={r.y_true} {r.modality:22s} leading {k}/{len(lead)} all {r.n_wrong}/{n_sys}: "
                     f"{str(r.text)[:110]}")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
