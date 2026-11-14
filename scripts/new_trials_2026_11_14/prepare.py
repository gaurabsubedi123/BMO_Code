"""Build model inputs for trials in the CTO 2026-11-14 export that are NOT in the paper corpus.
Outcome lines are matched to their ClinicalTrials.gov (API v2) Outcome Measure + Description, the exact fields the model was
trained on. Eligibility lines are used as written. Usage: prepare.py export.xlsx labeled_dataset.csv outdir"""
import sys, re, json, time, urllib.request, collections
import pandas as pd
xlsx, corpus_csv, out = sys.argv[1:4]
lines = pd.read_excel(xlsx, sheet_name="Biomarker Lines", dtype=str)
corpus = set(pd.read_csv(corpus_csv, dtype=str).NCTId)
new = lines[~lines["NCT ID"].isin(corpus)].copy()
print("lines", len(lines), "| new-trial lines", len(new), "| new trials", new["NCT ID"].nunique())
norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()

def fetch(nct):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct}?fields=OutcomesModule"
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 + 3 * k); err = e
    raise err

TAG = {"PrimaryOutcome": ("primaryOutcomes", "Primary"), "SecondaryOutcome": ("secondaryOutcomes", "Secondary"),
       "OtherOutcome": ("otherOutcomes", "Other")}
api = {}
for i, nct in enumerate(sorted(new.loc[new.Section.isin(TAG), "NCT ID"].unique())):
    api[nct] = fetch(nct).get("protocolSection", {}).get("outcomesModule", {}); time.sleep(0.2)
json.dump(api, open(f"{out}/ctgov_outcomes_raw.json", "w"))

rows, how = [], collections.Counter()
for _, r in new[new.Section.isin(TAG)].iterrows():
    key, tag = TAG[r.Section]
    cands = api.get(r["NCT ID"], {}).get(key, [])
    line = norm(r["Line Text"]); hit = None
    for o in cands:   # export line = "[Tag] measure description timeFrame"
        full = norm(" ".join(x for x in [f"[{tag}]", o.get("measure"), o.get("description"), o.get("timeFrame")] if x))
        if full == line: hit = o; how["exact"] += 1; break
    if hit is None:
        for o in cands:
            if line.startswith(norm(f"[{tag}] {o.get('measure','')}")) and len(norm(o.get("measure",""))) > 5:
                hit = o; how["measure prefix"] += 1; break
    if hit is None: how["unmatched"] += 1
    rows.append(dict(nct=r["NCT ID"], section=r.Section, line_no=r["Line #"], line_text=r["Line Text"],
                     label=int(r["Has Biomarker"] == "YES"), labels=r["Biomarker Labels"], annotator=r["Annotated By"],
                     matched=hit is not None, outcome_measure=(hit or {}).get("measure", ""),
                     outcome_description=(hit or {}).get("description", "")))
oc = pd.DataFrame(rows); print("outcome match:", dict(how))
oc.to_csv(f"{out}/new_outcome_lines.csv", index=False)
el = new[new.Section.str.contains("Criteria")].rename(columns={"NCT ID": "nct", "Section": "section", "Line #": "line_no",
     "Line Text": "line_text", "Biomarker Labels": "labels", "Annotated By": "annotator"})
el["label"] = (el["Has Biomarker"] == "YES").astype(int)
el[["nct", "section", "line_no", "line_text", "label", "labels", "annotator"]].to_csv(f"{out}/new_eligibility_lines.csv", index=False)
print("outcome lines", len(oc), oc.label.sum(), "| eligibility lines", len(el), el.label.sum())
