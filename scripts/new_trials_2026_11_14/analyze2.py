import os, pathlib, pandas as pd, re, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
def ci(k,n): return (beta.ppf(.025,k,n-k+1) if k else 0.0, beta.ppf(.975,k+1,n-k) if k<n else 1.0)
exec(open(pathlib.Path(__file__).with_name('analyze.py')).read().split('for name in')[0].split('def ci')[0] + "\n" + "DRUG=" + open(pathlib.Path(__file__).with_name('analyze.py')).read().split("DRUG = ")[1].split("re.I)")[0] + "re.I)")
DRUG2 = re.compile(DRUG.pattern + r"|concentration of [a-z0-9-]+mab|antibodies in serum|anti-?[a-z0-9-]+ antibodies|(csf|serum|blood|plasma) [a-z0-9-]+ concentration|\bcmin\b|\bcav", re.I)
BIO = re.compile(r"amyloid|a[βb]\s?\d|abeta|\btau\b|p-?tau|t-?tau|nfl|neurofilament|apoe|sapp|biomarker|gfap|ykl|trem2|alpha-?synuclein", re.I)
drug = lambda s: s.str.contains(DRUG2) & ~s.str.contains(BIO)
c = pd.read_csv(os.environ.get('BMO_RERUN_DIR', 'rerun_results') + '/outputs/dataset/labeled_dataset.csv')
cd_ = drug(c.clean_freeform.fillna('')); print(f"CORPUS drug-level outcomes: {cd_.sum()}, labeled biomarker {int(c[cd_].has_biomarker.sum())}")
def rep(x, tag):
    P,N=(x.label==1).sum(),(x.label==0).sum(); fn=((x.label==1)&(x.pred==0)).sum(); fp=((x.label==0)&(x.pred==1)).sum()
    lo,hi=ci(fn,P); flo,fhi=ci(fp,N)
    print(f"  {tag}: lines {len(x)}, biomarker {P}, missed {fn} ({fn/P:.1%}; 95% CI {lo:.1%}-{hi:.1%}), caught {P-fn} ({1-fn/P:.1%}), "
          f"false alarms {fp}/{N} ({fp/N:.1%}; {flo:.1%}-{fhi:.1%}), flagged {x.pred.sum()} ({x.pred.mean():.0%})")
for name in ["outcomes","eligibility"]:
    d=pd.read_csv(f"{name}_scored.csv"); d["drug"]=drug(d.model_input); print(name.upper()); rep(d,"all lines")
    print(f"  drug-level: {d.drug.sum()} lines, labeled YES {int(d[d.drug].label.sum())}, flagged {int(d[d.drug].pred.sum())}")
    rep(d[~d.drug],"without drug-level lines")
    if name=="outcomes":
        m=d[(d.label==1)&(d.pred==0)].copy()
        DIG=re.compile(r"cogstate|cantab|nih|c3|computeri|winterlight|actigraph|gait",re.I)
        NAMES=re.compile(r"mri|magnetic|pet\b|amyloid|aria|eeg|electroenceph|volume|hippocamp|imaging|glucose|cholesterol|ldl|hdl",re.I)
        m["kind"]=["digital or computerized test" if DIG.search(l) else "clinical lab (glucose, cholesterol)" if str(l).startswith("N/A_")
                   else "line names no biomarker (label from trial context)" if not NAMES.search(t) else "imaging, ARIA or EEG named in the line"
                   for l,t in zip(m.labels,m.model_input)]
        print(m.kind.value_counts().to_string()); m.to_csv("missed_outcomes_categorized.csv",index=False)
        t=d[d.label==1].groupby("nct").pred.max(); print(f"  trials with a biomarker outcome {len(t)}; model flagged >=1 in {int(t.sum())}")
        fa=d[(d.label==0)&(d.pred==1)&~d.drug]; fa.to_csv("false_alarms_non_drug.csv",index=False)
