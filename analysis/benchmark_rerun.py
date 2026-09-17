"""
Inference-timing benchmark for the 2026-08-27 re-run.

Written because the manuscript's cost table (tab:cost) has no source file: the
poster-run benchmark timed poster-run checkpoints, and its numbers differ from
the ones reported. This times the re-run checkpoints themselves.

  LATENCY    batch size 1, single CPU thread. The portable, conservative figure.
             Median + IQR over N_LAT records, best of REPEATS sweeps.
  CORPUS     the full 341-record held-out test partition in one pass.
  GPU        batch-32 pass over the same partition, transformers only.

Outputs $BMO_RUN_DIR/tables/inference_timing_rerun.csv
"""
import os

# Must precede numpy/torch import or the thread caps do not take effect.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModel, AutoConfig

torch.set_num_threads(1)


# The saved classical pipelines take ROW INDICES, not raw text: the text and the
# embedding matrix were precomputed and sliced inside the pipeline, because
# anything reading a runtime global inside GridSearchCV(n_jobs>1) either fails to
# pickle or silently yields zeros in the workers. Both classes are pickled by
# reference to __main__, so they must be redefined here, identically, to load.
class TextFromIndex(BaseEstimator, TransformerMixin):
    """Map row indices to their raw text, for the TF-IDF branch."""

    def __init__(self, texts=None):
        self.texts = texts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.texts[np.asarray(X).ravel().astype(int)]


class EmbFromIndex(BaseEstimator, TransformerMixin):
    """Slice rows out of a precomputed embedding matrix."""

    def __init__(self, E=None):
        self.E = E

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(self.E[np.asarray(X).ravel().astype(int)])

RUN = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
N_LAT = int(os.environ.get("N_LAT", 200))
REPEATS = int(os.environ.get("REPEATS", 3))
WARMUP = 10
BATCH = 32

ds = pd.read_csv(RUN / "outputs/dataset/labeled_dataset.csv")
test_idx = np.load(RUN / "outputs/dataset/test_idx.npy")
test_texts = ds.loc[test_idx, "clean_freeform"].astype(str).tolist()

rng = np.random.RandomState(42)
pick = rng.choice(len(test_texts), N_LAT, replace=False)
lat_texts = [test_texts[i] for i in pick]
# The classical pipelines are indexed against the full corpus, so they are timed
# on the corresponding original row indices rather than on the text itself.
lat_idx = [int(test_idx[i]) for i in pick]

print(f"test partition: {len(test_texts)} records | latency sample: {N_LAT} "
      f"| repeats: {REPEATS} | torch threads: {torch.get_num_threads()}")
print(f"cuda: {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'})")

rows = []


def sweep(fn, sample):
    """Per-record seconds, best (lowest median) of REPEATS sweeps."""
    for t in sample[:WARMUP]:
        fn(t)
    best = None
    for _ in range(REPEATS):
        per = []
        for t in sample:
            t0 = time.perf_counter()
            fn(t)
            per.append(time.perf_counter() - t0)
        per = np.array(per)
        if best is None or np.median(per) < np.median(best):
            best = per
    return best


def record(name, tier, artefact_bytes, params, load_s, lat, cpu_corpus_s,
           gpu_corpus_s=float("nan"), note=""):
    med = float(np.median(lat)) * 1e3
    q1 = float(np.percentile(lat, 25)) * 1e3
    q3 = float(np.percentile(lat, 75)) * 1e3
    rows.append(dict(
        System=name, Tier=tier,
        Params_M=round(params / 1e6, 1) if params else None,
        Artefact_MB=round(artefact_bytes / 1e6, 3) if artefact_bytes else None,
        Load_s=round(load_s, 2),
        Latency_ms_median=round(med, 3),
        Latency_ms_IQR_low=round(q1, 3),
        Latency_ms_IQR_high=round(q3, 3),
        Corpus_341_cpu_s=round(cpu_corpus_s, 3),
        Corpus_341_gpu_s=(round(gpu_corpus_s, 3) if gpu_corpus_s == gpu_corpus_s else None),
        Note=note))
    print(f"  {name:<32} load={load_s:6.2f}s  lat={med:8.3f} ms "
          f"[{q1:.3f}-{q3:.3f}]  corpus_cpu={cpu_corpus_s:7.2f}s  "
          f"corpus_gpu={gpu_corpus_s:6.2f}s  {note}", flush=True)


def dir_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


# ── classical, TF-IDF ────────────────────────────────────────────────────────
print("\n--- classical (TF-IDF) ---", flush=True)
CLASSICAL = ["Logistic_Regression", "SVM", "Random_Forest", "XGBoost",
             "Decision_Tree", "MLP_Small", "MLP_Deep"]
for stem in CLASSICAL:
    path = RUN / "models/classical" / f"{stem}_tfidf.joblib"
    if not path.exists():
        print(f"  {stem}: missing, skipped")
        continue
    try:
        t0 = time.perf_counter()
        mdl = joblib.load(path)
        load_s = time.perf_counter() - t0
        lat = sweep(lambda i: mdl.predict([i]), lat_idx)
        t0 = time.perf_counter()
        mdl.predict(list(test_idx))
        corpus_s = time.perf_counter() - t0
        record(f"TF-IDF + {stem.replace('_', ' ')}", "Classical",
               path.stat().st_size, None, load_s, lat, corpus_s,
               note="CPU 1 thread")
        del mdl
        gc.collect()
    except Exception as e:
        print(f"  {stem}: FAILED {type(e).__name__}: {e}")


# ── fine-tuned transformers ──────────────────────────────────────────────────
class Classifier(nn.Module):
    """Mirrors export/bmo-transformer/predict.py so the state dict loads strictly."""

    def __init__(self, config, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_config(config)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(config.hidden_size, 2))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


print("\n--- fine-tuned transformers ---", flush=True)
WANT = os.environ.get("BMO_BENCH_MODELS", "BioLinkBERT,BiomedBERT-large").split(",")
for name in [w.strip() for w in WANT if w.strip()]:
    d = RUN / "models/transformers" / name
    if not d.exists():
        print(f"  {name}: checkpoint missing, skipped")
        continue
    try:
        tc = json.loads((d / "training_config.json").read_text())
        max_len = tc.get("max_len", 128)
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        cfg = AutoConfig.from_pretrained(d)
        mdl = Classifier(cfg, tc.get("dropout", 0.3))
        mdl.load_state_dict(torch.load(d / "model_state.pt", map_location="cpu"))
        mdl.eval()
        load_s = time.perf_counter() - t0
        n_par = sum(p.numel() for p in mdl.parameters())

        def one(t, tok=tok, mdl=mdl, max_len=max_len):
            e = tok(t, truncation=True, max_length=max_len, padding="max_length",
                    return_tensors="pt")
            with torch.no_grad():
                mdl(e["input_ids"], e["attention_mask"])

        lat = sweep(one, lat_texts)

        t0 = time.perf_counter()
        for t in test_texts:
            one(t)
        cpu_corpus = time.perf_counter() - t0

        gpu_corpus = float("nan")
        if torch.cuda.is_available():
            g = mdl.cuda().eval()
            with torch.no_grad():
                for i in range(0, 2 * BATCH, BATCH):
                    e = tok(test_texts[i:i + BATCH], truncation=True,
                            max_length=max_len, padding="max_length",
                            return_tensors="pt")
                    g(e["input_ids"].cuda(), e["attention_mask"].cuda())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(0, len(test_texts), BATCH):
                    e = tok(test_texts[i:i + BATCH], truncation=True,
                            max_length=max_len, padding="max_length",
                            return_tensors="pt")
                    g(e["input_ids"].cuda(), e["attention_mask"].cuda())
                torch.cuda.synchronize()
                gpu_corpus = time.perf_counter() - t0
            mdl = g.cpu()
            torch.cuda.empty_cache()

        record(name, "Transformer", dir_bytes(d), n_par, load_s, lat,
               cpu_corpus, gpu_corpus,
               note=f"CPU 1 thread; GPU batch{BATCH}; max_len={max_len}")
        del mdl, tok
        gc.collect()
    except Exception as e:
        print(f"  {name}: FAILED {type(e).__name__}: {e}")

# ── exported deployment bundles ──────────────────────────────────────────────
# These, not the research checkpoints, are what a deployment would actually
# carry: the fast bundle is a rebuilt text pipeline (no stored corpus array) and
# the transformer bundle is the checkpoint plus its tokenizer. They are the
# artefacts the manuscript's cost table is about.
print("\n--- exported bundles ---", flush=True)
fast = RUN / "export/bmo-fast/pipeline.joblib"
if fast.exists():
    try:
        t0 = time.perf_counter()
        mdl = joblib.load(fast)
        load_s = time.perf_counter() - t0
        lat = sweep(lambda t: mdl.predict([t]), lat_texts)
        t0 = time.perf_counter()
        mdl.predict(test_texts)
        corpus_s = time.perf_counter() - t0
        record("bmo-fast (TF-IDF + logistic regression)", "Export bundle",
               fast.stat().st_size, None, load_s, lat, corpus_s,
               note="CPU 1 thread; raw text in")
        del mdl
        gc.collect()
    except Exception as e:
        print(f"  bmo-fast: FAILED {type(e).__name__}: {e}")

bundle = RUN / "export/bmo-transformer"
if bundle.exists():
    try:
        card = json.loads((bundle / "bmo_model_card.json").read_text())
        max_len = card["max_length"]
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(bundle / "tokenizer")
        cfg = AutoConfig.from_pretrained(bundle)
        mdl = Classifier(cfg)
        mdl.load_state_dict(torch.load(bundle / "model_state.pt", map_location="cpu"))
        mdl.eval()
        load_s = time.perf_counter() - t0
        n_par = sum(p.numel() for p in mdl.parameters())

        def one(t, tok=tok, mdl=mdl, max_len=max_len):
            e = tok(t, truncation=True, max_length=max_len, padding="max_length",
                    return_tensors="pt")
            with torch.no_grad():
                mdl(e["input_ids"], e["attention_mask"])

        lat = sweep(one, lat_texts)
        t0 = time.perf_counter()
        for t in test_texts:
            one(t)
        cpu_corpus = time.perf_counter() - t0

        gpu_corpus = float("nan")
        if torch.cuda.is_available():
            g = mdl.cuda().eval()
            with torch.no_grad():
                for i in range(0, 2 * BATCH, BATCH):
                    e = tok(test_texts[i:i + BATCH], truncation=True,
                            max_length=max_len, padding="max_length",
                            return_tensors="pt")
                    g(e["input_ids"].cuda(), e["attention_mask"].cuda())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(0, len(test_texts), BATCH):
                    e = tok(test_texts[i:i + BATCH], truncation=True,
                            max_length=max_len, padding="max_length",
                            return_tensors="pt")
                    g(e["input_ids"].cuda(), e["attention_mask"].cuda())
                torch.cuda.synchronize()
                gpu_corpus = time.perf_counter() - t0
            mdl = g.cpu()
            torch.cuda.empty_cache()

        record(f"bmo-transformer ({card['display_name']})", "Export bundle",
               dir_bytes(bundle), n_par, load_s, lat, cpu_corpus, gpu_corpus,
               note=f"CPU 1 thread; GPU batch{BATCH}; max_len={max_len}")
        del mdl, tok
        gc.collect()
    except Exception as e:
        print(f"  bmo-transformer: FAILED {type(e).__name__}: {e}")


out = pd.DataFrame(rows)
dest = RUN / "tables/inference_timing_rerun.csv"
out.to_csv(dest, index=False)
print(f"\nwrote {dest}  ({len(out)} systems)")
if len(out):
    print(out[["System", "Artefact_MB", "Latency_ms_median",
               "Corpus_341_cpu_s", "Corpus_341_gpu_s"]].to_string(index=False))
