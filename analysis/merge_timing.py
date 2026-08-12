import os, pandas as pd, numpy as np
ROOT = os.environ.get('BMO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, 'results')
a = pd.read_csv(f'{D}/inference_timing_classical.csv')
b = pd.read_csv(f'{D}/inference_timing.csv')
b['Note'] = b['Note'].fillna('') + ' | timed under python3.8 / torch 2.4.1'
a['Note'] = a['Note'].fillna('') + ' | timed under python3.11 / scikit-learn 1.7.2'
t = pd.concat([a, b], ignore_index=True).sort_values('Latency_ms_median').reset_index(drop=True)

sz = {}
M = os.path.join(ROOT, 'models')
for n, p in [('SVM (BioWordVec)', 'classical_ml_biowordvec/SVM.pkl'),
             ('LR (BioWordVec)', 'classical_ml_biowordvec/LR.pkl'),
             ('SVM (TF-IDF)', 'classical_ml/SVM.pkl'),
             ('Logistic Regression (TF-IDF)', 'classical_ml/Logistic_Regression.pkl'),
             ('PubMedBERT', 'bert/pubmedbert/model_state.pt'),
             ('BioLinkBERT', 'bert/biolinkbert/model_state.pt')]:
    f = f'{M}/{p}'
    if os.path.exists(f): sz[n] = os.path.getsize(f) / 1e6
t['Model_file_MB'] = [round(sz[s], 1) if s in sz else None for s in t.System]
t.to_csv(f'{D}/inference_timing_all.csv', index=False)

print(t[['System', 'Latency_ms_median', 'Latency_ms_IQR_low', 'Latency_ms_IQR_high',
         'Throughput_rec_per_s', 'Corpus_1900_s', 'Model_file_MB']].to_string(index=False))

svm = t[t.System == 'SVM (BioWordVec)'].iloc[0]
lr  = t[t.System == 'LR (BioWordVec)'].iloc[0]
bert = t[t.Tier == 'BERT fine-tuned']
pmb = t[t.System == 'PubMedBERT'].iloc[0]
print('\n' + '=' * 70)
print('HEADLINE — like for like: batch 1, single-threaded CPU, same machine')
print('=' * 70)
print(f'  SVM + BioWordVec-200 : {svm.Latency_ms_median:8.3f} ms/record  '
      f'[IQR {svm.Latency_ms_IQR_low:.3f}-{svm.Latency_ms_IQR_high:.3f}]')
print(f'  LR  + BioWordVec-200 : {lr.Latency_ms_median:8.3f} ms/record')
print(f'  PubMedBERT           : {pmb.Latency_ms_median:8.3f} ms/record  '
      f'[IQR {pmb.Latency_ms_IQR_low:.3f}-{pmb.Latency_ms_IQR_high:.3f}]')
print(f'  transformer mean     : {bert.Latency_ms_median.mean():8.3f} ms/record')
print(f'\n  SVM+BioWordVec is {pmb.Latency_ms_median/svm.Latency_ms_median:.0f}x faster '
      f'than PubMedBERT per record ({bert.Latency_ms_median.mean()/svm.Latency_ms_median:.0f}x vs the transformer mean)')
print(f'  Model on disk: {sz.get("SVM (BioWordVec)",0):.1f} MB vs {sz.get("PubMedBERT",0):.0f} MB '
      f'({sz.get("PubMedBERT",1)/max(sz.get("SVM (BioWordVec)",1),1e-9):.0f}x smaller)')
print('\nAgainst the poster claim "~50 ms vs ~2-4 s" (~50x):')
print(f'  measured SVM  {svm.Latency_ms_median:.2f} ms  -> poster overstated it {50/svm.Latency_ms_median:.0f}x')
print(f'  measured BERT {pmb.Latency_ms_median:.0f} ms   -> poster overstated it '
      f'{3000/pmb.Latency_ms_median:.0f}x (vs a 3 s midpoint)')
print(f'  true ratio {pmb.Latency_ms_median/svm.Latency_ms_median:.0f}x, not 50x — the poster '
      f'understated the advantage')
print('\nWhole-corpus (1,900 records):')
print(f'  SVM+BioWordVec, 1 CPU thread : {svm.Corpus_1900_s:.2f} s  (+55 s one-off embedding load, 299 s cold)')
print(f'  transformers, L40 GPU batch32: {bert.Corpus_1900_s.mean():.2f} s  (+~5 s model load)')
