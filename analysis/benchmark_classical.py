"""
Classical-tier inference timing (poster run).

Run under python3.11 with scikit-learn 1.7.2 -- the version stamped inside the
pickles (_sklearn_version). The pyannote env cannot be used: its numpy is 1.23
and the pickles were written with numpy>=2 (they reference numpy._core).

Same protocol as the transformer benchmark:
  latency    = batch 1, single thread, median + IQR over N_LAT records x REPEATS sweeps
  throughput = whole 1,900-record corpus in one predict() call, single thread
"""
import os, sys, time, gc, warnings
warnings.filterwarnings('ignore')
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

N_LAT   = int(os.environ.get('N_LAT', 100))
REPEATS = int(os.environ.get('REPEATS', 3))
WARMUP  = 10

import joblib, sklearn
print(f'python {sys.version.split()[0]} | sklearn {sklearn.__version__} | numpy {np.__version__}')

df    = pd.read_csv(f'{R}/dataset/labeled_dataset.csv')
texts = df['clean_freeform'].astype(str).tolist()
rng   = np.random.RandomState(42)
lat_texts = [texts[i] for i in rng.choice(len(texts), N_LAT, replace=False)]
print(f'corpus {len(texts)} | latency sample {N_LAT} x {REPEATS} sweeps')

results = []


def time_latency(fn, sample):
    for t in sample[:WARMUP]:
        fn(t)
    best = None
    for _ in range(REPEATS):
        per = np.array([(lambda t: (lambda s: (fn(t), time.perf_counter() - s)[1])(time.perf_counter()))(t)
                        for t in sample])
        if best is None or np.median(per) < np.median(best):
            best = per
    return best


def record(name, tier, load_s, lat, thr_s, n, note=''):
    med, q1, q3 = np.median(lat) * 1e3, np.percentile(lat, 25) * 1e3, np.percentile(lat, 75) * 1e3
    rps = n / thr_s
    results.append(dict(System=name, Tier=tier, Load_s=round(load_s, 2),
                        Latency_ms_median=round(med, 3),
                        Latency_ms_IQR_low=round(q1, 3), Latency_ms_IQR_high=round(q3, 3),
                        Throughput_rec_per_s=round(rps, 1),
                        Corpus_1900_s=round(1900 / rps, 2), Note=note))
    print(f'  {name:<30} load={load_s:8.2f}s  latency={med:8.3f} ms [{q1:.3f}-{q3:.3f}]  '
          f'{rps:9.1f} rec/s  corpus={1900/rps:7.2f}s  {note}', flush=True)


def bench_dir(subdir, tier, names, note=''):
    for disp, fn in names.items():
        path = f'{M}/{subdir}/{fn}.pkl'
        if not os.path.exists(path):
            print(f'  {disp}: missing'); continue
        try:
            t0 = time.perf_counter(); mdl = joblib.load(path); load_s = time.perf_counter() - t0
            lat = time_latency(lambda t: mdl.predict([t]), lat_texts)
            t0 = time.perf_counter(); mdl.predict(texts); thr = time.perf_counter() - t0
            record(disp, tier, load_s, lat, thr, len(texts), note)
            del mdl; gc.collect()
        except Exception as e:
            print(f'  {disp}: FAILED {type(e).__name__}: {e}', flush=True)


TFIDF = {'Logistic Regression (TF-IDF)': 'Logistic_Regression', 'SVM (TF-IDF)': 'SVM',
         'Random Forest (TF-IDF)': 'Random_Forest', 'XGBoost (TF-IDF)': 'XGBoost',
         'Decision Tree (TF-IDF)': 'Decision_Tree', 'MLP Small (TF-IDF)': 'MLP_Small',
         'MLP Deep (TF-IDF)': 'MLP_Deep'}
BWV = {'LR (BioWordVec)': 'LR', 'SVM (BioWordVec)': 'SVM',
       'Random Forest (BioWordVec)': 'Random_Forest', 'XGBoost (BioWordVec)': 'XGBoost',
       'Decision Tree (BioWordVec)': 'Decision_Tree', 'MLP Small (BioWordVec)': 'MLP_Small',
       'MLP Deep (BioWordVec)': 'MLP_Deep'}

print('\n--- classical (TF-IDF) ---', flush=True)
bench_dir('classical_ml', 'Classical ML', TFIDF)

print('\n--- classical (TF-IDF + BioWordVec-200) ---', flush=True)
try:
    import embeddings as _emb, fasttext
    t0 = time.perf_counter()
    _emb.BWV_MODEL = fasttext.load_model(os.environ.get('BIOWORDVEC_PATH', 'BioWordVec_PubMed_MIMICIII_d200.bin'))
    bwv = time.perf_counter() - t0
    print(f'  BioWordVec 28 GB binary loaded in {bwv:.1f}s (page cache warm)', flush=True)
    bench_dir('classical_ml_biowordvec', 'BioWordVec ML', BWV,
              note=f'+{bwv:.0f}s one-off embedding load')
except Exception as e:
    print(f'  BioWordVec tier unavailable: {type(e).__name__}: {e}')

out = pd.DataFrame(results)
out.to_csv(f'{OUT}/inference_timing_classical.csv', index=False)
print(f'\nwrote {OUT}/inference_timing_classical.csv ({len(out)} systems)')
print(out[['System', 'Latency_ms_median', 'Throughput_rec_per_s', 'Corpus_1900_s']].to_string(index=False))
