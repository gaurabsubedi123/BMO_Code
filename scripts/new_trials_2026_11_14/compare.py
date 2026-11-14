"""Compare systems on the newer-trial lines: false negatives first, McNemar on biomarker lines vs BiomedBERT-large."""
import sys, re, pandas as pd, numpy as np
from scipy.stats import beta, binomtest
G = str(__import__('pathlib').Path(__file__).resolve().parent) + '/'
src = open(G + 'analyze2.py').read().split('c = pd.read_csv')[0].replace("pathlib.Path(__file__).with_name('analyze.py')", repr(G + 'analyze.py'))
exec(src)
def ci(k, n): return (beta.ppf(.025, k, n-k+1) if k else 0, beta.ppf(.975, k+1, n-k) if k < n else 1)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
for part in sys.argv[1:]:
    d = pd.read_csv(part); y = d.label.values; dr = drug(d.model_input).values
    systems = [c for c in d.columns if c in ("BiomedBERT-large","BioLinkBERT","SciBERT","BERT-base","DeBERTa-v3","BioplusClinicalBERT",
               "ModernBERT","PubMedBERT","BioBERT","TF-IDF + logistic regression") or c.startswith("Qwen3.8")]
    rows = []; ref = d["BiomedBERT-large"].values >= 0.5
    for s in systems:
        for thr in ([0.5, 0.10] if s.startswith("Qwen") else [0.5]):
            f = d[s].values >= thr; name = s + (", tuned cut-off" if thr == 0.10 else "")
            P, N = (y == 1).sum(), (y == 0).sum(); fn = int(((y == 1) & ~f).sum()); fp = int(((y == 0) & f).sum())
            fpx = int(((y == 0) & f & ~dr).sum()); Nx = int(((y == 0) & ~dr).sum())
            b = int(((y == 1) & ref & ~f).sum()); c = int(((y == 1) & ~ref & f).sum())
            lo, hi = ci(fn, P)
            rows.append(dict(system=name, missed=fn, missed_pct=round(100*fn/P, 1), ci=f"{100*lo:.1f}-{100*hi:.1f}", fp=fp,
                             fp_pct=round(100*fp/N, 1), fp_no_drug=fpx, fp_no_drug_pct=round(100*fpx/Nx, 1), flagged_pct=round(100*f.mean(), 1),
                             ref_only_caught=b, sys_only_caught=c, p=1.0 if name == "BiomedBERT-large" else binomtest(b, b+c).pvalue if b+c else 1.0))
    r = pd.DataFrame(rows); m = r.system != "BiomedBERT-large"; r.loc[m, "p_holm"] = holm(r.loc[m, "p"])
    r = r.sort_values(["missed", "fp"]); print(f"\n{part}  (biomarker {P}, non-biomarker {N})"); print(r.drop(columns="p").to_string(index=False))
    r.to_csv(part.replace("_scored.csv", "_comparison.csv"), index=False)
