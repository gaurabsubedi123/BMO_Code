"""
Inference-timing benchmark for the poster run.

Two numbers per system, because they answer different questions:

  LATENCY   -- batch size 1, single thread. "How long does one record take?"
               Single-threaded so the figure is portable and conservative
               (roughly what a laptop would see) rather than flattered by a
               128-core server. Reported as median + IQR over N records.

  THROUGHPUT-- whole corpus in one pass, default parallelism (all cores; GPU for
               the transformers where available). "How long to screen the
               pipeline?" Reported as records/second.

Model load time is measured separately and reported as a one-off cost, because
for BioWordVec it dominates everything (a 28 GB embedding binary).
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd

# ---------------------------------------------------------------------------
# Paths. Defaults follow the pipeline's own layout (outputs/, models/ beside
# this repo). Override with BMO_ROOT if your outputs live elsewhere:
#     BMO_ROOT=/path/to/pipeline python analysis/<script>.py
# ---------------------------------------------------------------------------
ROOT = os.environ.get('BMO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R    = os.path.join(ROOT, 'outputs')
M    = os.path.join(ROOT, 'models')
OUT  = os.path.join(ROOT, 'results')
FIG  = os.path.join(ROOT, 'results')
SRC  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)

N_LAT   = int(os.environ.get('N_LAT', 200))   # records timed individually
REPEATS = int(os.environ.get('REPEATS', 3))   # repeats of the latency sweep
WARMUP  = 10

df    = pd.read_csv(f'{R}/dataset/labeled_dataset.csv')
texts = df['clean_freeform'].astype(str).tolist()
rng   = np.random.RandomState(42)
lat_texts = [texts[i] for i in rng.choice(len(texts), N_LAT, replace=False)]
print(f'corpus: {len(texts)} records | latency sample: {N_LAT} | repeats: {REPEATS}')

results = []


def time_latency(fn, sample):
    """median/IQR seconds per single record, best of REPEATS sweeps."""
    for t in sample[:WARMUP]:
        fn(t)
    best = None
    for _ in range(REPEATS):
        per = []
        for t in sample:
            t0 = time.perf_counter(); fn(t); per.append(time.perf_counter() - t0)
        per = np.array(per)
        if best is None or np.median(per) < np.median(best):
            best = per
    return best


def record(name, tier, load_s, lat, thr_s, n_thr, note=''):
    med, q1, q3 = np.median(lat) * 1e3, np.percentile(lat, 25) * 1e3, np.percentile(lat, 75) * 1e3
    rps = n_thr / thr_s if thr_s else float('nan')
    results.append(dict(System=name, Tier=tier, Load_s=round(load_s, 2),
                        Latency_ms_median=round(med, 2),
                        Latency_ms_IQR_low=round(q1, 2), Latency_ms_IQR_high=round(q3, 2),
                        Throughput_rec_per_s=round(rps, 1),
                        Corpus_1900_s=round(1900 / rps, 2) if rps == rps else None,
                        Note=note))
    print(f'  {name:<34} load={load_s:7.2f}s  latency={med:8.3f} ms '
          f'[{q1:.3f}-{q3:.3f}]  throughput={rps:9.1f} rec/s  {note}')


# ── single-threaded for latency ──────────────────────────────────────────────
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'

import joblib
import torch
torch.set_num_threads(1)

# ══════════════════════════════════════════════════════════════════════════
# 1. Classical, TF-IDF
# ══════════════════════════════════════════════════════════════════════════
print('\n--- classical (TF-IDF) ---')
TFIDF = {'Logistic Regression': 'Logistic_Regression', 'SVM': 'SVM',
         'Random Forest': 'Random_Forest', 'XGBoost': 'XGBoost',
         'Decision Tree': 'Decision_Tree', 'MLP Small': 'MLP_Small', 'MLP Deep': 'MLP_Deep'}
for name, fn in TFIDF.items():
    path = f'{M}/classical_ml/{fn}.pkl'
    if not os.path.exists(path):
        print(f'  {name}: pkl missing, skipped'); continue
    try:
        t0 = time.perf_counter(); mdl = joblib.load(path); load_s = time.perf_counter() - t0
        lat = time_latency(lambda t: mdl.predict([t]), lat_texts)
        t0 = time.perf_counter(); mdl.predict(texts); thr = time.perf_counter() - t0
        record(name + ' (TF-IDF)', 'Classical ML', load_s, lat, thr, len(texts))
        del mdl; gc.collect()
    except Exception as e:
        print(f'  {name}: FAILED {type(e).__name__}: {e}')

# ══════════════════════════════════════════════════════════════════════════
# 2. Classical + BioWordVec
# ══════════════════════════════════════════════════════════════════════════
print('\n--- classical (TF-IDF + BioWordVec-200) ---')
BWV_PATH = os.environ.get('BIOWORDVEC_PATH', 'BioWordVec_PubMed_MIMICIII_d200.bin')
bwv_load = float('nan')
try:
    import embeddings as _emb
    import fasttext
    t0 = time.perf_counter(); _emb.BWV_MODEL = fasttext.load_model(BWV_PATH); bwv_load = time.perf_counter() - t0
    print(f'  BioWordVec binary loaded in {bwv_load:.1f}s')
    for name, fn in {'LR': 'LR', 'SVM': 'SVM', 'MLP Small': 'MLP_Small',
                     'MLP Deep': 'MLP_Deep', 'Random Forest': 'Random_Forest',
                     'XGBoost': 'XGBoost', 'Decision Tree': 'Decision_Tree'}.items():
        path = f'{M}/classical_ml_biowordvec/{fn}.pkl'
        if not os.path.exists(path):
            print(f'  {name}: pkl missing'); continue
        try:
            t0 = time.perf_counter(); mdl = joblib.load(path); load_s = time.perf_counter() - t0
            lat = time_latency(lambda t: mdl.predict([t]), lat_texts)
            t0 = time.perf_counter(); mdl.predict(texts); thr = time.perf_counter() - t0
            record(name + ' (BioWordVec)', 'BioWordVec ML', load_s, lat, thr, len(texts),
                   note=f'+{bwv_load:.0f}s one-off embedding load')
            del mdl; gc.collect()
        except Exception as e:
            print(f'  {name}: FAILED {type(e).__name__}: {e}')
except Exception as e:
    print(f'  BioWordVec tier unavailable: {type(e).__name__}: {e}')

# ══════════════════════════════════════════════════════════════════════════
# 3. Fine-tuned transformers  (CPU single-thread, then GPU)
# ══════════════════════════════════════════════════════════════════════════
print('\n--- fine-tuned transformers ---')
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

BERTS = {'BERT-base': 'bert_base', 'BioBERT': 'biobert', 'PubMedBERT': 'pubmedbert',
         'Bio+ClinicalBERT': 'bio_clinical_bert', 'BioLinkBERT': 'biolinkbert'}
MAX_LEN = 128


class BERTClassifier(nn.Module):
    """Identical to src/03_train_bert_models.py so state_dict loads strictly."""
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


for name, sub in BERTS.items():
    try:
        d = f'{M}/bert/{sub}'
        cfg = json.load(open(f'{d}/config.json'))
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(f'{d}/tokenizer')
        mdl = BERTClassifier(cfg['model_name'], cfg.get('dropout', 0.2))
        sd = torch.load(f'{d}/model_state.pt', map_location='cpu')
        mdl.load_state_dict(sd)              # strict: verifies architecture match
        mdl.eval()
        load_s = time.perf_counter() - t0
        n_par = sum(p.numel() for p in mdl.parameters())

        def one(t, tok=tok, mdl=mdl):
            e = tok(t, truncation=True, max_length=MAX_LEN, padding='max_length',
                    return_tensors='pt')
            with torch.no_grad():
                mdl(e['input_ids'], e['attention_mask'])
        lat = time_latency(one, lat_texts)

        thr, dev = float('nan'), 'cpu'
        if torch.cuda.is_available():
            g = mdl.cuda().eval(); dev = 'cuda(L40)'
            with torch.no_grad():
                for i in range(0, 64, 32):
                    e = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN,
                            padding='max_length', return_tensors='pt')
                    g(e['input_ids'].cuda(), e['attention_mask'].cuda())
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for i in range(0, len(texts), 32):
                    e = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN,
                            padding='max_length', return_tensors='pt')
                    g(e['input_ids'].cuda(), e['attention_mask'].cuda())
                torch.cuda.synchronize(); thr = time.perf_counter() - t0
            mdl = g.cpu(); torch.cuda.empty_cache()
        record(name, 'BERT fine-tuned', load_s, lat, thr, len(texts),
               note=f'{n_par/1e6:.0f}M params; latency CPU 1-thread; throughput {dev} batch32')
        del mdl, tok; gc.collect()
    except Exception as e:
        print(f'  {name}: FAILED {type(e).__name__}: {e}')

# ══════════════════════════════════════════════════════════════════════════
out = pd.DataFrame(results)
out.to_csv(f'{OUT}/inference_timing.csv', index=False)
print(f'\nwrote {OUT}/inference_timing.csv  ({len(out)} systems)')
if len(out):
    print()
    print(out[['System', 'Latency_ms_median', 'Throughput_rec_per_s', 'Corpus_1900_s']].to_string(index=False))
