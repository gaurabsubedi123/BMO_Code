"""Supplementary tables for the November 2026 rewrite (48 systems, false negatives first).

    python supplement_48.py

Rebuilds every system's held-out predictions with combined_significance.load_predictions()
(the 45 published systems) plus the three reasoning models (error_overlap.add_reasoning), then writes
to $BMO_RERUN_DIR/outputs/supplement_48/:
    all_48_systems_test.csv  false negatives, false positives, flagged, weighted F1, and an exact McNemar
                             test of each system against BiomedBERT-large, Holm-corrected over 47 tests
    hard_records.csv         the held-out outcomes behind the leading systems' errors (<= 5 errors each)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_significance as cs     # noqa: E402
import error_overlap as eo             # noqa: E402

OUT = cs.OUT_DIR / "supplement_48"; OUT.mkdir(parents=True, exist_ok=True)
REF = "BiomedBERT-large"

y, P, _ = cs.load_predictions()
P = eo.add_reasoning(P, y)
assert len(P) == 48, len(P)
rows = []
for n, p in P.items():
    fn, fp = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())
    cr, cn = P[REF] == y, p == y
    b, c = int((cr & ~cn).sum()), int((~cr & cn).sum())
    rows.append(dict(system=n, family=eo.family_of(n), false_negatives=fn, false_positives=fp, flagged=int(p.sum()),
                     errors=fn + fp, weighted_f1=round(f1_score(y, p, average="weighted"), 4),
                     ref_only_correct=b, system_only_correct=c, p_vs_ref=1.0 if n == REF else cs.mcnemar(b, c),
                     # misses only: the same exact test restricted to the 109 biomarker outcomes
                     p_misses_vs_ref=1.0 if n == REF else cs.mcnemar(int((cr & ~cn & (y == 1)).sum()),
                                                                     int((~cr & cn & (y == 1)).sum()))))
d = pd.DataFrame(rows)
m = d.system != REF
d.loc[m, "p_holm_vs_ref"] = cs.holm(d.loc[m, "p_vs_ref"].values)
d.loc[m, "p_holm_misses_vs_ref"] = cs.holm(d.loc[m, "p_misses_vs_ref"].values)
d = d.sort_values(["false_negatives", "false_positives", "system"]).reset_index(drop=True)
r = d[d.system == REF].iloc[0]
assert (r.false_negatives, r.false_positives, r.flagged) == (0, 2, 111)
assert d.query("false_negatives == 0 and false_positives <= 10").system.tolist() == [REF]
d.to_csv(OUT / "all_48_systems_test.csv", index=False)
print(d.to_string(index=False))

text = pd.read_csv(cs.OUT_DIR / "dataset" / "labeled_dataset.csv")
te = np.load(cs.OUT_DIR / "dataset" / "test_idx.npy")
lead = d[d.errors <= 5].system.tolist()
wrong = {i: [n for n in lead if P[n][i] != y[i]] for i in range(len(y))}
h = [dict(row=int(te[i]), label="biomarker" if y[i] else "not a biomarker", leading_wrong=len(w), all_wrong=sum(P[n][i] != y[i] for n in P),
          text=text.clean_freeform.iloc[te[i]]) for i, w in wrong.items() if w]
h = pd.DataFrame(h).sort_values(["leading_wrong", "all_wrong"], ascending=False)
h.to_csv(OUT / "hard_records.csv", index=False)
print(f"\nleading systems {len(lead)}, their errors {int(d[d.errors <= 5].errors.sum())}, on {len(h)} records")
print("significant vs BiomedBERT-large after Holm:", d[d.p_holm_vs_ref < 0.05].system.tolist())
print("misses significantly above BiomedBERT-large after Holm:", d[d.p_holm_misses_vs_ref < 0.05].system.tolist())
print("leading systems significant vs ref:", d[(d.errors <= 5) & (d.p_holm_vs_ref < 0.05)].system.tolist())
