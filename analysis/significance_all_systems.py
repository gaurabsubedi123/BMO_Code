"""
Significance testing across every benchmarked system (poster run).

Replicates the statistical logic of src/07_significance_tests.py without needing
seaborn/statsmodels: McNemar's test, exact binomial when min(b,c) < 25, otherwise
asymptotic chi-square with continuity correction. Bonferroni over all pairs.

Two analyses:
  A. OOF (n=1,559)  -- 31 systems. The powered, primary analysis.
  B. Held-out test (n=341) -- all 33 systems, including the two feature-condition
     ensembles that exist only at test time. Reported, but underpowered by design:
     the leading systems make 3-6 errors each, so almost nothing can reach
     significance. Included for completeness, not for inference.
"""
import os, sys, numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import binom, chi2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def mcnemar(b, c):
    """Return (p, method). Mirrors statsmodels mcnemar(exact=min(b,c)<25)."""
    n = b + c
    if n == 0:
        return 1.0, 'no discordant pairs'
    if min(b, c) < 25:
        k = min(b, c)
        p = min(1.0, 2.0 * binom.cdf(k, n, 0.5))
        return p, 'exact'
    stat = (abs(b - c) - 1.0) ** 2 / n
    return float(chi2.sf(stat, 1)), 'asymptotic (continuity-corrected)'


def run(preds: pd.DataFrame, y, tag, note):
    models = list(preds.columns)
    pairs = list(combinations(range(len(models)), 2))
    bonf = 0.05 / len(pairs)
    print(f'\n=== {tag}: {len(models)} systems, {len(pairs)} pairs, n={len(y)} ===')
    print(f'Bonferroni threshold = {bonf:.3e}')

    rows, P = [], np.ones((len(models), len(models)))
    for i, j in pairs:
        p1 = preds[models[i]].values.astype(int)
        p2 = preds[models[j]].values.astype(int)
        c1, c2 = (p1 == y), (p2 == y)
        b = int(np.sum(c1 & ~c2))
        c = int(np.sum(~c1 & c2))
        p, method = mcnemar(b, c)
        winner = models[i] if b > c else (models[j] if c > b else 'tie')
        rows.append(dict(Model_1=models[i], Model_2=models[j],
                         only_1_correct=b, only_2_correct=c,
                         acc_1=round(c1.mean(), 4), acc_2=round(c2.mean(), 4),
                         p_value=p, method=method,
                         sig_uncorrected=p < 0.05, sig_bonferroni=p < bonf,
                         winner=winner))
        P[i, j] = P[j, i] = p

    df = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)
    df.to_csv(f'{OUT}/significance_{tag}.csv', index=False)
    print(f'  significant p<0.05      : {df.sig_uncorrected.sum()} / {len(df)}')
    print(f'  significant (Bonferroni): {df.sig_bonferroni.sum()} / {len(df)}')

    # heatmap of -log10(p)
    order = sorted(range(len(models)), key=lambda k: -(preds[models[k]].values == y).mean())
    M = -np.log10(np.clip(P[np.ix_(order, order)], 1e-300, 1))
    np.fill_diagonal(M, np.nan)
    lab = [models[k] for k in order]
    fig, ax = plt.subplots(figsize=(max(9, .38 * len(lab)), max(7.5, .34 * len(lab))), dpi=170)
    fig.patch.set_facecolor('#FCFCFB')
    im = ax.imshow(np.ma.masked_invalid(M), cmap='YlGnBu', vmin=0, vmax=12)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_title(f'McNemar pairwise significance — {tag}\n'
                 f'colour = $-\\log_{{10}}(p)$; Bonferroni threshold '
                 f'$-\\log_{{10}}$ = {-np.log10(bonf):.2f}', fontsize=10, loc='left')
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.outline.set_visible(False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_significance_{tag}.png', facecolor='#FCFCFB')
    plt.close(fig)
    return df, models, bonf, note


# ── A. OOF, 31 systems ───────────────────────────────────────────────────────
d = pd.read_csv(f'{R}/dataset/labeled_dataset.csv')
train_idx = np.load(f'{R}/dataset/train_idx.npy')
y_oof = d['has_biomarker'].values[train_idx]

cls = pd.read_csv(f'{R}/classical_ml/oof_predictions.csv', index_col=0).reset_index(drop=True)
brt = pd.read_csv(f'{R}/bert_models/oof_predictions.csv', index_col=0).reset_index(drop=True)
zs = pd.read_csv(f'{R}/zeroshot_llm/all_predictions.csv', index_col=0).iloc[train_idx].reset_index(drop=True)
enh = pd.read_csv(f'{R}/classical_ml_enhanced/oof_predictions.csv').reset_index(drop=True)
bwv = pd.read_csv(f'{R}/classical_ml_biowordvec/oof_predictions.csv').reset_index(drop=True)
enh.columns = [c + ' (FastText)' for c in enh.columns]
bwv.columns = [c + ' (BioWordVec)' for c in bwv.columns]
oof = pd.concat([cls, enh, bwv, brt, zs], axis=1)
assert len(oof) == len(y_oof)
df_oof, models_oof, bonf_oof, _ = run(oof, y_oof, 'OOF_31systems', '')

# ── B. Held-out test, all 33 systems ─────────────────────────────────────────
probs = pd.read_csv(f'{OUT}/all_test_probs.csv')
y_te = probs['y_true'].values.astype(int)
met = pd.read_csv(f'{OUT}/all_test_metrics.csv').set_index('Model')
cols = [c for c in probs.columns if c != 'y_true']

# Verify thresholding reproduces the published confusion matrices before using it.
te_pred, mismatch = {}, []
for c in cols:
    pr = (probs[c].values >= 0.5).astype(int)
    if c in met.index:
        tn = int(np.sum((pr == 0) & (y_te == 0))); fp = int(np.sum((pr == 1) & (y_te == 0)))
        fn = int(np.sum((pr == 0) & (y_te == 1))); tp = int(np.sum((pr == 1) & (y_te == 1)))
        exp = met.loc[c]
        if (tn, fp, fn, tp) != (int(exp.TN), int(exp.FP), int(exp.FN), int(exp.TP)):
            mismatch.append((c, (tn, fp, fn, tp), (int(exp.TN), int(exp.FP), int(exp.FN), int(exp.TP))))
    te_pred[c] = pr

print('\n--- test-set threshold verification (0.5) ---')
if mismatch:
    print(f'{len(mismatch)} of {len(cols)} systems do NOT reproduce their published matrix:')
    for m, got, want in mismatch:
        print(f'  {m:<34} got TN/FP/FN/TP={got}  published={want}')
    print('  -> these used a tuned threshold; excluded from analysis B.')
else:
    print(f'all {len(cols)} systems reproduce their published confusion matrix exactly.')

keep = [c for c in cols if c not in {m[0] for m in mismatch}]
te = pd.DataFrame({c: te_pred[c] for c in keep})
df_te, models_te, bonf_te, _ = run(te, y_te, f'TEST_{len(keep)}systems', '')

# ── summary ──────────────────────────────────────────────────────────────────
KEY = [('SVM (BioWordVec)', 'PubMedBERT'), ('SVM (BioWordVec)', 'BERT Ensemble'),
       ('SVM (BioWordVec)', 'BioLinkBERT'), ('LR (BioWordVec)', 'PubMedBERT'),
       ('SVM (BioWordVec)', 'SVM'), ('SVM (BioWordVec)', 'SVM (FastText)'),
       ('Logistic Regression', 'PubMedBERT'), ('BERT Ensemble', 'Logistic Regression'),
       ('MLP Deep (BioWordVec)', 'PubMedBERT')]

with open(f'{OUT}/significance_summary_all_systems.txt', 'w') as f:
    def w(s=''): f.write(s + '\n'); print(s)
    w('=' * 74); w("McNemar pairwise significance — poster run, all benchmarked systems"); w('=' * 74)
    w()
    w(f'A. OUT-OF-FOLD  (primary)   {len(models_oof)} systems | '
      f'{len(df_oof)} pairs | n=1,559 | Bonferroni alpha={bonf_oof:.3e}')
    w(f'   significant p<0.05: {df_oof.sig_uncorrected.sum()} | '
      f'after Bonferroni: {df_oof.sig_bonferroni.sum()}')
    w('   Excludes the two feature-condition ensembles, which are formed at test')
    w('   time and therefore have no out-of-fold predictions.')
    w()
    w(f'B. HELD-OUT TEST            {len(models_te)} systems | '
      f'{len(df_te)} pairs | n=341 | Bonferroni alpha={bonf_te:.3e}')
    w(f'   significant p<0.05: {df_te.sig_uncorrected.sum()} | '
      f'after Bonferroni: {df_te.sig_bonferroni.sum()}')
    w('   UNDERPOWERED BY CONSTRUCTION: the leading systems make 3-6 errors each in')
    w('   341 records, so discordant-pair counts are tiny and almost nothing can')
    w('   reach significance. Absence of significance here is not evidence of')
    w('   equivalence. Analysis A is the one to cite.')
    w()
    w('-' * 74); w('KEY COMPARISONS (out-of-fold)'); w('-' * 74)
    w(f'{"Model 1":<24}{"Model 2":<20}{"b":>5}{"c":>5}{"p":>12}  verdict')
    for a, b_ in KEY:
        r = df_oof[((df_oof.Model_1 == a) & (df_oof.Model_2 == b_)) |
                   ((df_oof.Model_1 == b_) & (df_oof.Model_2 == a))]
        if r.empty: continue
        r = r.iloc[0]
        v = ('*** Bonferroni' if r.sig_bonferroni else
             ('*  p<0.05' if r.sig_uncorrected else 'ns'))
        w(f'{r.Model_1:<24}{r.Model_2:<20}{r.only_1_correct:>5}{r.only_2_correct:>5}'
          f'{r.p_value:>12.3e}  {v}'
          + (f'  winner={r.winner}' if r.sig_uncorrected else ''))
    w()
    w('-' * 74); w('SYSTEMS NOT DISTINGUISHABLE FROM THE BEST (out-of-fold, p>=0.05)'); w('-' * 74)
    best = max(models_oof, key=lambda m: (oof[m].values == y_oof).mean())
    w(f'best by OOF accuracy: {best}')
    for _, r in df_oof.iterrows():
        if best in (r.Model_1, r.Model_2) and not r.sig_uncorrected:
            other = r.Model_2 if r.Model_1 == best else r.Model_1
            w(f'  {other:<34} p={r.p_value:.3f}')

print('\nwrote:')
for f_ in [f'{OUT}/significance_OOF_31systems.csv', f'{OUT}/significance_TEST_{len(keep)}systems.csv',
           f'{OUT}/significance_summary_all_systems.txt',
           f'{FIG}/fig_significance_OOF_31systems.png']:
    print(' ', f_)
