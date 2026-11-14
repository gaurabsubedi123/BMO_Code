"""Effect of the deployed tool's digital/EEG review rule (classify_excel.WEAK) on the newer-trial misses.
Usage: python digital_rule.py NEW_TRIALS_DIR"""
import re, pandas as pd, numpy as np
from scipy.stats import beta
WEAK = re.compile(
    r"\b(digital|computeri[sz]ed|computer-based|tablet|ipad|smartphone|app-based|"
    r"wearable|actigraph\w*|accelerometer|continuous performance|reaction time|"
    r"touchscreen|remote assessment|passive monitoring|speech analys\w+|"
    r"keystroke|gait sensor|"
    r"eeg|qeeg|electroencephalo\w*|erp|event-related potential|evoked potential|"
    r"p300|p3b|mismatch negativity|polysomnograph\w*|psg|actimetry|"
    r"meg|magnetoencephalograph\w*|tms|transcranial magnetic stimulation|"
    r"emg|electromyograph\w*|electroretinograph\w*|erg|"
    r"spectral power|resting-state connectivity)\b", re.I)
def ci(k,n): return (beta.ppf(.025,k,n-k+1) if k else 0, beta.ppf(.975,k+1,n-k) if k<n else 1)
def rep(tag, y, f):
    P,N=(y==1).sum(),(y==0).sum(); fn=int(((y==1)&~f).sum()); fp=int(((y==0)&f).sum()); lo,hi=ci(fn,P)
    print(f"  {tag:34s} missed {fn:3d}/{P} ({fn/P:.1%}; {lo:.1%}-{hi:.1%})  FP {fp:4d}/{N} ({fp/N:.1%})  flagged {f.sum():5d} ({f.mean():.1%})")
exec(open(__import__('pathlib').Path(__file__).with_name('analyze2.py')).read().split('c = pd.read_csv')[0].replace("pathlib.Path(__file__).with_name('analyze.py')","str(__import__('pathlib').Path(__file__).with_name('analyze.py'))"))
import sys, pathlib
NEW = pathlib.Path(sys.argv[1])   # folder with outcomes_scored.csv, eligibility_scored.csv, missed_outcomes_categorized.csv
for name, f in [('outcomes', NEW / 'outcomes_scored.csv'), ('eligibility', NEW / 'eligibility_scored.csv')]:
    if f is None:
        continue
    x = pd.read_csv(f); y = x.label.values; m = x.pred.values == 1; r = x.model_input.str.contains(WEAK).values
    print(name.upper(), '| rule matches', r.sum(), 'lines')
    rep('model alone', y, m); rep('model + digital/EEG rule', y, m | r)
    dr = drug(x.model_input).values
    rep('model + rule, drug-level excluded', y[~dr], (m | r)[~dr])
x = pd.read_csv(NEW / 'missed_outcomes_categorized.csv'); x['rule']=x.model_input.str.contains(WEAK)
print('misses caught by rule:', x.rule.sum(), 'of 43'); print(x[~x.rule].labels.value_counts().to_dict())
