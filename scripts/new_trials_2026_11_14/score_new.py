"""On grundy: (1) reproduce the deployed BiomedBERT-large on the 341 test records (must be 0 missed / 2 false alarms),
(2) score new-trial outcome lines (input = clean_text(Outcome Measure + ' ' + Outcome Description), as in training) and
eligibility lines (input = clean_text(line text)). Usage: score_new.py classifier_dir run_dir outcomes.csv eligibility.csv outdir"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import classify_excel as ce
torch, tok, model, dev, card = ce.load()
def clean_text(text):                               # verbatim from src/01_prepare_dataset.py
    text = re.sub(r"\['([^']+)'\]", r'\1', str(text))
    text = re.sub(r"[\[\]'\"{}]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()
def probs(texts, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = [re.sub(r"\s+", " ", str(v)).strip().lower() for v in texts[i:i+bs]]
            enc = tok(b, truncation=True, max_length=card["max_length"], padding="max_length", return_tensors="pt").to(dev)
            out += torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1)[:, 1].cpu().tolist()
    return np.array(out)
run = sys.argv[2]
df = pd.read_csv(run + "/outputs/dataset/labeled_dataset.csv"); te = np.load(run + "/outputs/dataset/test_idx.npy")
p = probs(df.clean_freeform.fillna("").values[te]); y = df.has_biomarker.values[te]; yh = (p >= 0.5).astype(int)
fn, fp = int(((y == 1) & (yh == 0)).sum()), int(((y == 0) & (yh == 1)).sum())
print(f"REPRODUCTION on {len(te)} test records: missed {fn}, false alarms {fp}"); assert len(te) == 341 and fn == 0 and fp == 2
print("max_length", card["max_length"], "| device", dev)
for name, path, mk in [("outcomes", sys.argv[3], lambda d: (d.outcome_measure.fillna("").astype(str) + " " + d.outcome_description.fillna("").astype(str)).apply(clean_text)),
                       ("eligibility", sys.argv[4], lambda d: d.line_text.fillna("").astype(str).apply(clean_text))]:
    d = pd.read_csv(path, dtype={"line_no": str}); d["model_input"] = mk(d)
    assert (d.model_input.str.len() > 0).all(), name + ": empty inputs"
    ntok = [len(tok(t)["input_ids"]) for t in d.model_input]; d["truncated"] = np.array(ntok) > card["max_length"]
    d["biomarker_probability"] = probs(d.model_input.tolist()).round(6); d["pred"] = (d.biomarker_probability >= 0.5).astype(int)
    d.to_csv(f"{sys.argv[5]}/{name}_scored.csv", index=False)
    print(name, len(d), "| truncated", int(d.truncated.sum()), "| missed", int(((d.label == 1) & (d.pred == 0)).sum()),
          "of", int(d.label.sum()), "| false alarms", int(((d.label == 0) & (d.pred == 1)).sum()), "of", int((d.label == 0).sum()))
print("SCORED_OK")
