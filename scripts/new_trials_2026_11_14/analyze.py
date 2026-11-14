import pandas as pd, re, sys, collections
from scipy.stats import beta
def ci(k,n): return (beta.ppf(.025,k,n-k+1) if k else 0.0, beta.ppf(.975,k+1,n-k) if k<n else 1.0)
DRUG = re.compile(r"pharmacokinetic|\bpk\b|\bcmax|\bc ?max\b|\bauc|\btmax|t1/2|half-life|half life|clearance|volume of distribution|"
  r"\bcl/f\b|\bvz|\bvss\b|trough|ctrough|accumulation ratio|elimination|\bada\b|anti-drug|anti-?drug antibod|immunogenicity|"
  r"neutrali[sz]ing antibod|antibodies (to|against) |anti-[a-z0-9]+mab antibod|(plasma|serum|csf|cerebrospinal fluid|blood|urine)( drug)? concentrations? of|"
  r"concentrations? of [a-z]{2,4}-?\d|concentration of (the )?study drug|drug concentration|drug level|exposure of|dose-normali[sz]ed|lambda z|terminal rate|bioavailability|dosimetry|radiation (absorbed )?dose", re.I)
for name in ["outcomes", "eligibility"]:
    d = pd.read_csv(f"{name}_scored.csv"); d["drug"] = d.model_input.str.contains(DRUG)
    def rep(x, tag):
        P, N = (x.label==1).sum(), (x.label==0).sum(); fn=((x.label==1)&(x.pred==0)).sum(); fp=((x.label==0)&(x.pred==1)).sum()
        print(f"  {tag}: n={len(x)} biomarker={P} missed={fn} ({fn/P:.1%}, 95% CI {ci(fn,P)[0]:.1%}-{ci(fn,P)[1]:.1%}) "
              f"false alarms={fp} of {N} ({fp/N:.1%}) flagged={x.pred.sum()} ({x.pred.mean():.0%}) trials={x.nct.nunique()}")
    print(name.upper()); rep(d, "all lines")
    print(f"  drug-level lines: {d.drug.sum()} (labeled YES {int(d[d.drug].label.sum())}, model flagged {int(d[d.drug].pred.sum())})")
    rep(d[~d.drug], "excluding drug-level")
    if name == "outcomes":
        t = d[d.label==1].groupby("nct").apply(lambda g: (g.pred==1).any()); print(f"  trials with >=1 biomarker: {len(t)}, model flagged >=1 in {t.sum()}")
        fa = d[(d.label==0)&(d.pred==1)&~d.drug]; print("  remaining false alarms sample:")
        for s in fa.sample(min(20,len(fa)),random_state=0).model_input: print("   -", s[:140])
