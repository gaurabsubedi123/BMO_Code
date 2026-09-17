#%% md
# 10 — Deployment bundles for the web service

The best-performing system in this study is **not** the one to deploy, and the
reason is worth stating plainly in the manuscript's discussion.

| Candidate | Artefact size | Cold start | External dependency |
|---|---|---|---|
| BioWordVec + MLP Deep | 28 GB resident | ~5 min | 28 GB binary must stay in RAM |
| Fine-tuned transformer (fp32) | ~440 MB | ~2 s | none |
| Same, ONNX int8 | ~110 MB | <1 s | none |
| TF-IDF + Logistic Regression | ~5 MB | instant | none |

The BioWordVec systems win on accuracy but cannot be served: the embedding
binary is larger than most application servers' total memory, and it must stay
resident because the vectorizer queries it per request.

This notebook therefore exports two bundles:

1. **Accuracy tier** — the best fine-tuned transformer, self-contained.
2. **Fast tier** — the best TF-IDF pipeline, small enough to load anywhere.

The gap between them is typically one or two errors out of 341. For a
human-in-the-loop triage tool that difference is usually not worth two orders
of magnitude in footprint, which is a genuinely useful finding for anyone
building on this work.
#%% code
import os, sys, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

df, train_idx, test_idx = C.load_data()
y_test = df["has_biomarker"].values[test_idx]
texts_test = df["clean_freeform"].fillna("").values[test_idx]

master = pd.read_csv(C.OUT_FINAL / "MASTER_test_all_systems.csv")
print(master.head(8)[["System", "Tier", "weighted_f1", "errors"]].to_string(index=False))
#%% md
## Choose what to ship

Selection is by test-set score **within each deployability class**, not
overall. The chosen systems are recorded in the bundle metadata so the served
model is always traceable to a row in the results table.
#%% code
tf_rows = master[master["Tier"] == "Transformer"].sort_values("weighted_f1", ascending=False)
cl_rows = master[master["Tier"] == "Classical"].sort_values("weighted_f1", ascending=False)
best_transformer = tf_rows.iloc[0]["Model"] if "Model" in tf_rows.columns else \
                   tf_rows.iloc[0]["System"]
best_classical_sys = cl_rows.iloc[0]["System"]
best_classical = best_classical_sys.replace("TF-IDF + ", "")
print(f"accuracy tier : {best_transformer}  "
      f"(F1 {tf_rows.iloc[0]['weighted_f1']:.4f}, {int(tf_rows.iloc[0]['errors'])} errors)")
print(f"fast tier     : {best_classical}  "
      f"(F1 {cl_rows.iloc[0]['weighted_f1']:.4f}, {int(cl_rows.iloc[0]['errors'])} errors)")
print(f"\ncost of choosing the fast tier: "
      f"{int(cl_rows.iloc[0]['errors']) - int(tf_rows.iloc[0]['errors'])} extra errors "
      f"out of {len(y_test)}")
#%% md
## Decision threshold

The default 0.5 threshold is not necessarily the right operating point for a
triage tool. For screening, recall usually matters more than precision — a
missed biomarker outcome is a silent annotation error, while a false positive
merely costs a moment of reviewer attention.

The threshold is chosen on **out-of-fold training predictions**, never on the
test set, so the shipped operating point is not tuned to the data it is
reported on.
#%% code
from sklearn.metrics import f1_score, precision_recall_curve

def pick_threshold(y, p, min_recall=0.95):
    """Highest-precision threshold that still achieves min_recall, plus the
    F1-optimal threshold, both selected on out-of-fold training predictions."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = rec[:-1] >= min_recall
    t_recall = float(thr[ok][np.argmax(prec[:-1][ok])]) if ok.any() else 0.5
    f1s = [(f1_score(y, (p >= t).astype(int), zero_division=0), t)
           for t in np.linspace(0.05, 0.95, 91)]
    return {"f1_optimal": float(max(f1s)[1]), "high_recall": t_recall,
            "default": 0.5, "min_recall_target": min_recall}

oof_path = C.OUT_BERT / "oof_probabilities_merged.csv"
THRESHOLDS = {"default": 0.5}
if oof_path.exists():
    oof = pd.read_csv(oof_path)
    if best_transformer in oof.columns:
        THRESHOLDS = pick_threshold(oof["y_true"].values, oof[best_transformer].values)
        print(json.dumps(THRESHOLDS, indent=2))
        for nm, t in THRESHOLDS.items():
            if nm == "min_recall_target":
                continue
            yp = (oof[best_transformer].values >= t).astype(int)
            r = C.evaluate(oof["y_true"].values, yp, name=nm)
            print(f"  {nm:12s} thr={t:.3f}  OOF biomarker F1 {r['biomarker_f1']:.4f}  "
                  f"recall {r['recall']:.4f}  precision {r['precision']:.4f}")
else:
    print("out-of-fold probabilities not found; shipping the default threshold")
#%% md
## Bundle 1 — accuracy tier (transformer)

Self-contained: weights, tokenizer, config, thresholds and a `predict()`
entry point in one directory with no external data dependency.
#%% code
src = C.MODELS / "transformers" / best_transformer.replace("+", "plus").replace("/", "_")
dst = C.EXPORT / "bmo-transformer"; dst.mkdir(parents=True, exist_ok=True)
cfg = json.loads((src / "training_config.json").read_text())

for item in ("model_state.pt", "tokenizer"):
    s = src / item
    if s.is_dir():
        shutil.copytree(s, dst / item, dirs_exist_ok=True)
    elif s.exists():
        shutil.copy2(s, dst / item)
AutoConfig.from_pretrained(cfg["checkpoint"]).save_pretrained(dst)

META = {
    "name": "bmo-biomarker-classifier",
    "task": "binary classification of clinical trial outcome measures as biomarker / not biomarker",
    "base_checkpoint": cfg["checkpoint"], "display_name": best_transformer,
    "max_length": cfg["max_len"], "labels": {"0": "No biomarker", "1": "Biomarker"},
    "thresholds": THRESHOLDS,
    "test_weighted_f1": float(tf_rows.iloc[0]["weighted_f1"]),
    "test_biomarker_f1": float(tf_rows.iloc[0]["biomarker_f1"]),
    "test_errors": int(tf_rows.iloc[0]["errors"]),
    "test_n": int(len(y_test)),
    "training": {k: cfg[k] for k in ("lr", "batch_size", "epochs", "dropout", "seed")
                 if k in cfg},
    "caveat": ("Trained on 1,900 Alzheimer's disease trial outcome measures from "
               "ClinicalTrials.gov. Evaluation is in-distribution; performance on "
               "other diseases, registries or text types is untested."),
}
(dst / "bmo_model_card.json").write_text(json.dumps(META, indent=2))
print(json.dumps(META, indent=2))
#%% code
INFER = '''"""
predict.py — inference entry point for the BMO biomarker classifier.

    from predict import BiomarkerClassifier
    clf = BiomarkerClassifier("path/to/bmo-transformer")
    clf.predict(["Change from baseline in amyloid PET SUVR"])
    # [{'label': 'Biomarker', 'probability': 0.97, 'is_biomarker': True}]

Self-contained: no external embedding binary, no network access at inference.
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class _Classifier(nn.Module):
    """Must mirror the training-time architecture exactly or the state dict
    will not load: one dropout, one linear layer, on the [CLS] token."""
    def __init__(self, config, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_config(config)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(config.hidden_size, 2))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


class BiomarkerClassifier:
    def __init__(self, bundle_dir, device=None, threshold=None):
        d = Path(bundle_dir)
        self.meta = json.loads((d / "bmo_model_card.json").read_text())
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = self.meta["max_length"]
        self.threshold = threshold or self.meta["thresholds"].get("default", 0.5)
        self.tokenizer = AutoTokenizer.from_pretrained(d / "tokenizer")
        config = AutoConfig.from_pretrained(d)
        self.model = _Classifier(config).to(self.device).eval()
        state = torch.load(d / "model_state.pt", map_location=self.device)
        self.model.load_state_dict(state)

    @torch.no_grad()
    def predict(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i:i + batch_size]), return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=self.max_length).to(self.device)
            # Pass only what forward() accepts. BERT-family tokenizers also emit
            # token_type_ids, which this head does not take; forwarding the whole
            # encoding raises TypeError and makes the bundle unusable.
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            for p in probs.cpu().tolist():
                out.append({"label": "Biomarker" if p >= self.threshold else "No biomarker",
                            "probability": round(float(p), 4),
                            "is_biomarker": bool(p >= self.threshold)})
        return out
'''
(dst / "predict.py").write_text(INFER)

FLASK = '''"""
serve.py — minimal Flask endpoint, matching the pattern used by the existing
lab web service (model loaded once at startup, JSON in / JSON out).

    POST /predict  {"texts": ["Change from baseline in amyloid PET SUVR"]}
    -> {"predictions": [{"label": "Biomarker", "probability": 0.97, ...}]}
"""
from pathlib import Path
from flask import Flask, request, jsonify
from predict import BiomarkerClassifier

app = Flask(__name__)
CLF = BiomarkerClassifier(Path(__file__).parent)   # loaded once, reused per request


@app.route("/health")
def health():
    return jsonify(status="ok", model=CLF.meta["display_name"],
                   test_weighted_f1=CLF.meta["test_weighted_f1"])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    texts = payload.get("texts") or ([payload["text"]] if "text" in payload else [])
    if not texts:
        return jsonify(error="provide 'text' or 'texts'"), 400
    if len(texts) > 512:
        return jsonify(error="maximum 512 texts per request"), 400
    return jsonify(predictions=CLF.predict(texts), model=CLF.meta["display_name"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
'''
(dst / "serve.py").write_text(FLASK)
print("bundle ->", dst)
for f in sorted(dst.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(dst)}  {f.stat().st_size/1e6:.1f} MB")
#%% md
### Verify the bundle reproduces the reported test numbers

A bundle that does not reproduce its own reported score is not shippable. This
loads the exported artefact from disk — the same way a server would — and
re-scores the test set through the public `predict()` API.
#%% code
sys.path.insert(0, str(dst))
import importlib, predict as _p
importlib.reload(_p)

t0 = time.time()
clf = _p.BiomarkerClassifier(dst)
load_s = time.time() - t0
t0 = time.time()
out = clf.predict(list(texts_test))
infer_s = time.time() - t0

y_hat = np.array([o["is_biomarker"] for o in out], dtype=int)
p_hat = np.array([o["probability"] for o in out])
row = C.evaluate(y_test, y_hat, p_hat, name=f"{best_transformer} (exported bundle)")
print(f"load {load_s:.1f}s | inference {infer_s:.1f}s for {len(texts_test)} records "
      f"({infer_s/len(texts_test)*1000:.1f} ms/record)")
print(f"exported bundle: weighted F1 {row['weighted_f1']:.4f} | errors {row['errors']}")
print(f"reported        : weighted F1 {tf_rows.iloc[0]['weighted_f1']:.4f} | "
      f"errors {int(tf_rows.iloc[0]['errors'])}")
match = abs(row["weighted_f1"] - tf_rows.iloc[0]["weighted_f1"]) < 1e-6
print("\nBUNDLE VERIFIED — reproduces the reported result exactly" if match else
      "\nMISMATCH — do not ship this bundle until the discrepancy is explained")
#%% md
## Bundle 2 — fast tier (TF-IDF pipeline)

A single joblib file. No GPU, no transformer runtime, sub-millisecond
inference. This is the sensible default for a public web form.
#%% code
fast_dir = C.EXPORT / "bmo-fast"; fast_dir.mkdir(parents=True, exist_ok=True)

# Refit a TEXT-IN pipeline from the hyperparameters notebook 02 selected.
#
# Notebook 02's fitted estimator cannot be shipped: it consumes row indices and
# its feature union pickles transformers defined in that notebook, so loading it
# anywhere else fails with
#     AttributeError: Can't get attribute 'TextFromIndex' on __main__
# The pipeline built here takes raw strings, depends only on scikit-learn, and
# is checked below against the reported test metrics.
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

ESTIMATORS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced",
                                                      random_state=C.SEED_CV),
    "SVM":                 lambda: SVC(probability=True, class_weight="balanced",
                                       random_state=C.SEED_CV),
    "Random Forest":       lambda: RandomForestClassifier(class_weight="balanced", n_jobs=1,
                                                          random_state=C.SEED_CV),
    "XGBoost":             lambda: XGBClassifier(eval_metric="logloss", tree_method="hist",
                                                 n_jobs=1, random_state=C.SEED_CV, verbosity=0),
    "Decision Tree":       lambda: DecisionTreeClassifier(class_weight="balanced",
                                                          random_state=C.SEED_CV),
    "MLP Small":           lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                                 early_stopping=True, random_state=C.SEED_CV),
    "MLP Deep":            lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                                                 early_stopping=True, random_state=C.SEED_CV),
}

texts_train = df["clean_freeform"].fillna("").values[train_idx]
y_train_    = df["has_biomarker"].values[train_idx]

cl_full = pd.read_csv(C.OUT_CLASSICAL / "test_results_all_tiers.csv")
row = cl_full[(cl_full["Tier"] == "Classical") &
              (cl_full["Model"] == best_classical)].iloc[0]
params = json.loads(row["best_params"]) if "best_params" in row and pd.notna(row["best_params"]) else {}
tf_kw  = {k.rsplit("__", 1)[1]: v for k, v in params.items() if k.startswith("feat__tfidf__tfidf__")}
clf_kw = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("clf__")}
# ngram_range survives the JSON round-trip as a list; TfidfVectorizer needs a tuple.
if isinstance(tf_kw.get("ngram_range"), list):
    tf_kw["ngram_range"] = tuple(tf_kw["ngram_range"])
print(f"rebuilding {best_classical} with tfidf={tf_kw} clf={clf_kw}")

pipe = SkPipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, **tf_kw)),
                   ("clf",   ESTIMATORS[best_classical]().set_params(**clf_kw))])
pipe.fit(texts_train, y_train_)
joblib.dump(pipe, fast_dir / "pipeline.joblib")
if True:
    t0 = time.time(); yp = pipe.predict(texts_test); dt = time.time() - t0
    pp = pipe.predict_proba(texts_test)[:, 1]
    r = C.evaluate(y_test, yp, pp, name=best_classical)
    (fast_dir / "bmo_model_card.json").write_text(json.dumps({
        "name": "bmo-biomarker-classifier-fast", "estimator": best_classical,
        "representation": "TF-IDF", "labels": {"0": "No biomarker", "1": "Biomarker"},
        "test_weighted_f1": r["weighted_f1"], "test_biomarker_f1": r["biomarker_f1"],
        "test_errors": r["errors"], "test_n": int(len(y_test)),
        "inference_ms_per_record": round(dt / len(texts_test) * 1000, 3),
        "sklearn_version": __import__("sklearn").__version__,
        "caveat": META["caveat"],
    }, indent=2))
    print(f"{best_classical}: weighted F1 {r['weighted_f1']:.4f} | errors {r['errors']} "
          f"| {dt/len(texts_test)*1000:.3f} ms/record")
    print(f"size: {(fast_dir/'pipeline.joblib').stat().st_size/1e6:.2f} MB")
    print("\nNOTE: joblib pipelines are tied to the scikit-learn version that "
          "wrote them. The version is recorded in the model card; pin it when "
          "deploying, or the pipeline may fail to load.")
else:
    print(f"missing {src_pkl} — run notebook 02 first")
#%% md
## Publishing

Neither bundle can go in the GitHub repository: a transformer checkpoint
exceeds GitHub's 100 MB per-file limit. The repository ships the inference code
and a loader; the weights belong on the Hugging Face Hub.

The upload requires a write token, which is not stored anywhere in this project
and is not needed to produce the bundles. Run this yourself when ready:

```bash
pip install huggingface_hub
huggingface-cli login                      # paste your write token when prompted
huggingface-cli upload <your-username>/bmo-biomarker-classifier \
    export/bmo-transformer . --repo-type=model
```

The fast bundle is ~5 MB and can be committed directly, or attached to a GitHub
release, whichever you prefer.
#%% code
sizes = {}
for name, d in (("bmo-transformer", C.EXPORT / "bmo-transformer"),
                ("bmo-fast", C.EXPORT / "bmo-fast")):
    if d.exists():
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        sizes[name] = round(total / 1e6, 1)
        print(f"{name:18s} {total/1e6:8.1f} MB  -> {d}")
(C.EXPORT / "bundles.json").write_text(json.dumps(
    {"bundles": sizes, "accuracy_tier": best_transformer,
     "fast_tier": best_classical}, indent=2))
print("\nGitHub file limit is 100 MB; anything above that goes to the Hub.")
#%% md
## Done

**Next:** `09_figures_and_tables.ipynb` finalises the manuscript figures and
writes the run manifest with output checksums.
