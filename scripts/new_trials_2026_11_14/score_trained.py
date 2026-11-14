"""On grundy: score the 9 fine-tuned models and TF-IDF + logistic regression on the newer-trial lines.
Each model must first reproduce its published held-out false negatives / false positives exactly.
Usage: score_trained.py run_dir outcomes.csv eligibility.csv outdir"""
import sys, re, json, pathlib, numpy as np, pandas as pd, torch, joblib
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
run, oc_path, el_path, out = sys.argv[1:5]; RUN = pathlib.Path(run)
PUBLISHED = {"BiomedBERT-large": (0, 2), "BioLinkBERT": (1, 0), "SciBERT": (1, 3), "BERT-base": (2, 1), "DeBERTa-v3": (3, 0),
             "BioplusClinicalBERT": (3, 0), "ModernBERT": (4, 1), "PubMedBERT": (4, 1), "BioBERT": (5, 2), "TF-IDF + logistic regression": (4, 2)}
def clean_text(text):                               # verbatim from src/01_prepare_dataset.py
    text = re.sub(r"\['([^']+)'\]", r'\1', str(text)); text = re.sub(r"[\[\]'\"{}]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()
df = pd.read_csv(RUN / "outputs/dataset/labeled_dataset.csv"); te = np.load(RUN / "outputs/dataset/test_idx.npy")
xt, yt = df.clean_freeform.fillna("").values[te], df.has_biomarker.values[te]
oc = pd.read_csv(oc_path, dtype={"line_no": str}); el = pd.read_csv(el_path, dtype={"line_no": str})
oc["model_input"] = (oc.outcome_measure.fillna("").astype(str) + " " + oc.outcome_description.fillna("").astype(str)).apply(clean_text)
el["model_input"] = el.line_text.fillna("").astype(str).apply(clean_text)
class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__(); self.bert = AutoModel.from_config(config)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(config.hidden_size, 2))
    def forward(self, ids, mask):
        return self.head(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])
dev = torch.device("cuda")
def bert_probs(d, texts, bs=64):
    tok = AutoTokenizer.from_pretrained(d / "tokenizer"); m = Classifier(AutoConfig.from_pretrained(d)).to(dev).eval()
    m.load_state_dict(torch.load(d / "model_state.pt", map_location=dev)); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = [re.sub(r"\s+", " ", str(v)).strip().lower() for v in texts[i:i+bs]]
            enc = tok(b, truncation=True, max_length=128, padding="max_length", return_tensors="pt").to(dev)
            out += torch.softmax(m(enc["input_ids"], enc["attention_mask"]), dim=1)[:, 1].cpu().tolist()
    del m; torch.cuda.empty_cache(); return np.array(out)
lr = joblib.load(RUN / "export/bmo-fast/pipeline.joblib")
models = {n: (lambda t, n=n: bert_probs(RUN / "models/transformers" / n, t)) for n in PUBLISHED if n != "TF-IDF + logistic regression"}
models["TF-IDF + logistic regression"] = lambda t: lr.predict_proba(list(t))[:, 1]
for name, f in models.items():
    p = f(xt) >= 0.5; fn, fp = int(((yt == 1) & ~p).sum()), int(((yt == 0) & p).sum())
    ok = (fn, fp) == PUBLISHED[name]; print(f"{name}: held-out missed {fn}, false positives {fp} -> {'REPRODUCED' if ok else 'MISMATCH'}", flush=True)
    assert ok, name
    oc[name] = f(oc.model_input.values).round(6); el[name] = f(el.model_input.values).round(6)
oc.to_csv(f"{out}/outcomes_trained_scored.csv", index=False); el.to_csv(f"{out}/eligibility_trained_scored.csv", index=False)
print("TRAINED_DONE")
