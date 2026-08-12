"""
07_significance_tests.py -- Statistical significance testing for all pairwise model comparisons.

Uses McNemar's test on per-sample OOF predictions (1559 training samples).

Evaluation context:
  - Supervised models (Classical ML, BERT): OOF predictions — each sample was
    predicted by a model that had NOT seen it during training.
  - Zero-shot models: filtered to train_idx from all_predictions.csv — valid
    because zero-shot models have no training dependency on any sample.

All supervised and zero-shot models are compared pairwise (105 pairs for 15 models;
231 pairs for 22 models if 02a was run; 351 pairs for 27 models if both 02a and 02b run).
Bonferroni correction: significance threshold = 0.05 / N_PAIRS (computed dynamically).

Outputs:
    outputs/final_results/significance_tests.csv       -- all 105 pairs, p-values, statistics
    outputs/final_results/fig_significance_heatmap.png -- p-value heatmap (−log10 scale)
    outputs/final_results/significance_summary.txt     -- human-readable summary
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUT_DATASET, OUT_CLASSICAL, OUT_ENHANCED, OUT_BIOWORDVEC, OUT_BERT,
                    OUT_ZEROSHOT, OUT_FINAL, SPLIT_TRAIN_IDX)

os.makedirs(OUT_FINAL, exist_ok=True)

print("-" * 60)
print("Step 7 of 7: Statistical significance testing (McNemar's test)")
print("-" * 60)

# ── Load true labels (training set only) ─────────────────────────────────────
df        = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
train_idx = np.load(SPLIT_TRAIN_IDX)
y_true    = df['has_biomarker'].values[train_idx]
print(f"\nTrue labels loaded: {len(y_true)} training samples")
print(f"Positive rate: {y_true.mean()*100:.1f}%")

# ── Load OOF predictions ──────────────────────────────────────────────────────
# Note: OOF files use original dataset row numbers as index (0–1899 with gaps).
# Reset to sequential 0..1558 on all frames before concat to ensure alignment.
classical_oof = pd.read_csv(
    os.path.join(OUT_CLASSICAL, 'oof_predictions.csv'), index_col=0
).reset_index(drop=True)
bert_oof = pd.read_csv(
    os.path.join(OUT_BERT, 'oof_predictions.csv'), index_col=0
).reset_index(drop=True)

# Zero-shot: all 1900 samples, filter to train_idx positions
zeroshot_all   = pd.read_csv(
    os.path.join(OUT_ZEROSHOT, 'all_predictions.csv'), index_col=0)
zeroshot_train = zeroshot_all.iloc[train_idx].reset_index(drop=True)

# Enhanced ML (step 2a) — optional; 02a saves OOF with index=False
frames = [classical_oof, bert_oof, zeroshot_train]
_enh_oof_path = os.path.join(OUT_ENHANCED, 'oof_predictions.csv')
if os.path.exists(_enh_oof_path):
    enh_oof = pd.read_csv(_enh_oof_path).reset_index(drop=True)
    # Add "(Enh)" suffix to avoid column name collision with classical ML models
    enh_oof.columns = [c + ' (Enh)' for c in enh_oof.columns]
    frames.append(enh_oof)
    print(f"Loaded Enhanced ML OOF: {enh_oof.shape[1]} models")
else:
    print("INFO: Enhanced ML OOF not found — run step 02a to include (skipping)")

# BioWordVec ML (step 2b) — optional; 02b saves OOF with index=False
_bwv_oof_path = os.path.join(OUT_BIOWORDVEC, 'oof_predictions.csv')
if os.path.exists(_bwv_oof_path):
    bwv_oof = pd.read_csv(_bwv_oof_path).reset_index(drop=True)
    # Add "(BioWordVec)" suffix to avoid column name collision
    bwv_oof.columns = [c + ' (BioWordVec)' for c in bwv_oof.columns]
    frames.append(bwv_oof)
    print(f"Loaded BioWordVec ML OOF: {bwv_oof.shape[1]} models")
else:
    print("INFO: BioWordVec ML OOF not found — run step 02b to include (skipping)")

# Combine into a single DataFrame (1559 rows × N models)
all_preds = pd.concat(frames, axis=1)
models    = all_preds.columns.tolist()
n_models  = len(models)
print(f"\nModels loaded: {n_models}")
for m in models:
    acc = (all_preds[m].values == y_true).mean()
    print(f"  {m:<30}  accuracy={acc:.4f}")

# ── McNemar's test (all pairwise) ─────────────────────────────────────────────
# Contingency table for pair (i, j):
#   n11 = both correct
#   n12 = model_i correct, model_j wrong    (b)
#   n21 = model_i wrong,   model_j correct  (c)
#   n22 = both wrong
# McNemar's statistic uses only b and c (discordant pairs).
# Use exact test when min(b, c) < 25, asymptotic otherwise.

try:
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
    USE_STATSMODELS = True
except ImportError:
    from scipy.stats import binom
    USE_STATSMODELS = False
    print("\n[Warning] statsmodels not available — using scipy exact binomial fallback")

N_PAIRS    = len(list(combinations(range(n_models), 2)))
ALPHA      = 0.05
BONFERRONI = ALPHA / N_PAIRS
print(f"\nTotal pairs: {N_PAIRS}")
print(f"Bonferroni threshold: {BONFERRONI:.6f}  (alpha={ALPHA} / {N_PAIRS})")

results = []
p_matrix  = np.ones((n_models, n_models))

for i, j in combinations(range(n_models), 2):
    m1, m2   = models[i], models[j]
    pred1    = all_preds[m1].values.astype(int)
    pred2    = all_preds[m2].values.astype(int)

    n11 = int(np.sum((pred1 == y_true) & (pred2 == y_true)))
    n12 = int(np.sum((pred1 == y_true) & (pred2 != y_true)))  # b
    n21 = int(np.sum((pred1 != y_true) & (pred2 == y_true)))  # c
    n22 = int(np.sum((pred1 != y_true) & (pred2 != y_true)))

    b, c = n12, n21
    use_exact = (min(b, c) < 25)

    if USE_STATSMODELS:
        table  = np.array([[n11, n12], [n21, n22]])
        result = mcnemar_test(table, exact=use_exact, correction=(not use_exact))
        p_val  = float(result.pvalue)
        stat   = float(result.statistic)
    else:
        # Exact binomial fallback: under H0, b ~ Binomial(b+c, 0.5)
        n_total = b + c
        if n_total == 0:
            p_val, stat = 1.0, 0.0
        else:
            # two-sided p-value
            p_val = float(2 * binom.cdf(min(b, c), n_total, 0.5))
            p_val = min(p_val, 1.0)
            stat  = float(b)

    p_matrix[i, j] = p_val
    p_matrix[j, i] = p_val

    # Which model is better (lower error = more correct)?
    better = m1 if b > c else (m2 if c > b else "Tie")

    results.append({
        'Model_1'                  : m1,
        'Model_2'                  : m2,
        'b (M1_right_M2_wrong)'    : b,
        'c (M1_wrong_M2_right)'    : c,
        'Discordant_pairs'         : b + c,
        'Test_type'                : 'Exact' if use_exact else 'Asymptotic',
        'Statistic'                : round(stat, 4),
        'P_value'                  : p_val,  # raw float; use P_value_display for reporting
        'P_value_display'          : ('< 2.2e-16' if p_val == 0.0
                                      else f'{p_val:.2e}' if p_val < 0.001
                                      else f'{p_val:.4f}'),
        'Significant_p05'          : p_val < ALPHA,
        'Significant_Bonferroni'   : p_val < BONFERRONI,
        'Better_model'             : better,
    })

results_df = pd.DataFrame(results).sort_values('P_value')
out_csv    = os.path.join(OUT_FINAL, 'significance_tests.csv')
results_df.to_csv(out_csv, index=False)
print(f"\nSaved significance tests to {out_csv}")
print(f"Significant pairs (p<0.05):        "
      f"{results_df['Significant_p05'].sum()} / {N_PAIRS}")
print(f"Significant pairs (Bonferroni):     "
      f"{results_df['Significant_Bonferroni'].sum()} / {N_PAIRS}")

# ── Heatmap: −log10(p-value) ──────────────────────────────────────────────────
np.fill_diagonal(p_matrix, 1.0)
log_p = -np.log10(np.clip(p_matrix, 1e-10, 1.0))

# Short labels for display
short = [m.replace('Zero-Shot', 'ZS').replace('Few-Shot', 'FS')
          .replace('Classical ', 'C-').replace('BERT Ensemble', 'BERT-Ens')
          .replace('Logistic Regression', 'LR')
          .replace('Decision Tree', 'DT')
          .replace('Random Forest', 'RF')
          for m in models]

fig, ax = plt.subplots(figsize=(13, 10))
bonf_logp = -np.log10(BONFERRONI)

mask = np.eye(n_models, dtype=bool)
sns.heatmap(
    log_p,
    ax=ax,
    mask=mask,
    annot=True, fmt='.1f',
    xticklabels=short, yticklabels=short,
    cmap='RdYlGn',
    vmin=0, vmax=max(bonf_logp * 1.5, log_p[~mask].max()),
    linewidths=0.5,
    cbar_kws={'label': '−log₁₀(p-value)'},
)
ax.set_title(
    f'McNemar\'s Test — Pairwise Significance\n'
    f'(−log₁₀ scale; Bonferroni threshold = {bonf_logp:.1f}; '
    f'green = more significant)',
    fontsize=11
)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
fig_path = os.path.join(OUT_FINAL, 'fig_significance_heatmap.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved heatmap to {fig_path}")

# ── Human-readable summary ────────────────────────────────────────────────────
summary_path = os.path.join(OUT_FINAL, 'significance_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MCNEMAR'S TEST — PAIRWISE SIGNIFICANCE SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Samples: {len(y_true)} (OOF on training set)\n")
    f.write(f"Models:  {n_models}\n")
    f.write(f"Pairs:   {N_PAIRS}\n")
    f.write(f"Method:  Exact McNemar when min(b,c)<25, asymptotic otherwise\n")
    f.write(f"Alpha:   {ALPHA} (uncorrected) | "
            f"{BONFERRONI:.6f} (Bonferroni)\n\n")
    f.write(f"Significant (p<0.05):      "
            f"{results_df['Significant_p05'].sum()} pairs\n")
    f.write(f"Significant (Bonferroni):  "
            f"{results_df['Significant_Bonferroni'].sum()} pairs\n\n")

    # Key tier comparisons
    key_pairs = [
        ('PubMedBERT',          'Logistic Regression'),
        ('BioLinkBERT',         'Logistic Regression'),
        ('BioBERT',             'Logistic Regression'),
        ('BERT Ensemble',       'Logistic Regression'),
        ('BERT Ensemble',       'Classical Ensemble'),
        ('Logistic Regression', 'Flan-T5 Few-Shot'),
        ('Logistic Regression', 'BART Zero-Shot'),
        ('Logistic Regression', 'BioGPT Zero-Shot'),
        ('PubMedBERT',          'BioLinkBERT'),
        ('PubMedBERT',          'BioBERT'),
    ]
    f.write("-" * 70 + "\n")
    f.write("KEY COMPARISONS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Model 1':<25} {'Model 2':<25} {'p-value':>10} "
            f"{'Bonf.':>8} {'Winner':<20}\n")
    f.write("-" * 70 + "\n")
    for m1_name, m2_name in key_pairs:
        row = results_df[
            ((results_df['Model_1'] == m1_name) & (results_df['Model_2'] == m2_name)) |
            ((results_df['Model_1'] == m2_name) & (results_df['Model_2'] == m1_name))
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        bonf_mark = '***' if row['Significant_Bonferroni'] else \
                    ('*' if row['Significant_p05'] else 'ns')
        f.write(f"{m1_name:<25} {m2_name:<25} {row['P_value_display']:>12} "
                f"{bonf_mark:>8} {row['Better_model']:<20}\n")

    f.write("\n" + "-" * 70 + "\n")
    f.write("ALL SIGNIFICANT PAIRS (Bonferroni-corrected)\n")
    f.write("-" * 70 + "\n")
    bonf_df = results_df[results_df['Significant_Bonferroni']].copy()
    if len(bonf_df) == 0:
        f.write("No pairs survive Bonferroni correction.\n")
    else:
        for _, row in bonf_df.iterrows():
            f.write(f"  {row['Model_1']:<28} vs {row['Model_2']:<28} "
                    f"p={row['P_value_display']}  winner={row['Better_model']}\n")

    f.write("\n" + "-" * 70 + "\n")
    f.write("ALL NON-SIGNIFICANT PAIRS (p >= 0.05)\n")
    f.write("-" * 70 + "\n")
    ns_df = results_df[~results_df['Significant_p05']].copy()
    if len(ns_df) == 0:
        f.write("All pairs are significant at p<0.05.\n")
    else:
        for _, row in ns_df.iterrows():
            f.write(f"  {row['Model_1']:<28} vs {row['Model_2']:<28} "
                    f"p={row['P_value_display']}\n")

print(f"Saved summary to {summary_path}")
print("\nStep 7 complete.")
