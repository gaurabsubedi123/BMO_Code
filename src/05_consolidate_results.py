"""
05_consolidate_results.py -- Merge all model results into one table and produce paper figures.

Reads the results CSVs produced by steps 2, 3, and 4, then combines them into
a single master table with three labeled tiers:
  - Classical ML (Logistic Regression, SVM, Random Forest, XGBoost, Decision Tree)
  - BERT Models (BioBERT, PubMedBERT, Bio+ClinicalBERT, BioLinkBERT)
  - Zero-Shot LLMs (BART, Flan-T5, BioGPT)

Per-class metrics (Flags 2 & 7 — reviewer request):
  - Has Biomarker F1 and No Biomarker F1 are computed from saved OOF predictions
    (classical ML and BERT) and all_predictions.csv filtered to training samples
    (zero-shot). All are leakage-free out-of-fold estimates.

Produces:
  - A bar chart comparing all 15 models by F1 and AUC side by side.
  - A performance heatmap showing all metrics including per-class F1.
  - A LaTeX table suitable for direct inclusion in a paper.
  - master_results.csv for reference.

Outputs:
    outputs/final_results/master_results.csv
    outputs/final_results/results_table.tex
    outputs/final_results/fig_all_models.png
    outputs/final_results/fig_heatmap.png
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUT_CLASSICAL, OUT_ENHANCED, OUT_BIOWORDVEC, OUT_BERT, OUT_ZEROSHOT,
                    OUT_FINAL, OUT_DATASET, SPLIT_TRAIN_IDX)

os.makedirs(OUT_FINAL, exist_ok=True)

print("-" * 60)
print("Step 5 of 7: Consolidating all model results")
print("-" * 60)


def safe_load(path):
    """Load a CSV, returning an empty DataFrame and printing a warning on failure."""
    try:
        return pd.read_csv(path, index_col=0)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return pd.DataFrame()


# ---- Load per-tier results ---------------------------------------------------
# Use full_results.csv for tiers 1 and 2 so F1 Std is available for error bars.
# Fall back to results.csv if full_results.csv doesn't exist yet.
def safe_load_full(full_path, fallback_path):
    df = safe_load(full_path)
    if df.empty:
        df = safe_load(fallback_path)
    return df

t1 = safe_load_full(os.path.join(OUT_CLASSICAL, 'full_results.csv'),
                    os.path.join(OUT_CLASSICAL, 'results.csv'))
t2 = safe_load_full(os.path.join(OUT_BERT,      'full_results.csv'),
                    os.path.join(OUT_BERT,      'results.csv'))
t4 = safe_load(os.path.join(OUT_ZEROSHOT, 'results.csv'))

# Enhanced ML (step 2a) — optional; only included if outputs exist.
t1b = safe_load_full(os.path.join(OUT_ENHANCED, 'full_results.csv'),
                     os.path.join(OUT_ENHANCED, 'results.csv'))
if not t1b.empty:
    t1b.index = t1b.index + ' (Enhanced)'

# BioWordVec ML (step 2b) — optional; only included if outputs exist.
t1c = safe_load_full(os.path.join(OUT_BIOWORDVEC, 'full_results.csv'),
                     os.path.join(OUT_BIOWORDVEC, 'results.csv'))
if not t1c.empty:
    t1c.index = t1c.index + ' (BioWordVec)'

for name, df in [('Classical ML', t1), ('Enhanced ML', t1b), ('BioWordVec ML', t1c),
                 ('BERT Models', t2), ('Zero-Shot LLMs', t4)]:
    if df.empty:
        if name == 'Enhanced ML':
            print(f"INFO: Enhanced ML results not found -- run step 02a to include them.")
        elif name == 'BioWordVec ML':
            print(f"INFO: BioWordVec ML results not found -- run step 02b to include them.")
        else:
            print(f"WARNING: {name} results are empty -- run the corresponding step first.")
    else:
        print(f"Loaded {name}: {len(df)} models")

# Assign tier labels using plain human-readable names.
t1['Tier'] = 'Classical ML'
t2['Tier'] = 'BERT Models'
t4['Tier'] = 'Zero-Shot LLMs'

concat_frames = [t1]
if not t1b.empty:
    t1b['Tier'] = 'Enhanced ML'
    concat_frames.append(t1b)
if not t1c.empty:
    t1c['Tier'] = 'BioWordVec ML'
    concat_frames.append(t1c)
concat_frames += [t2, t4]

master = pd.concat(concat_frames)
# Keep F1 Std if present; zero-shot uses bootstrap CI so may not have it.
keep_cols = ['Tier', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']
if 'F1 Std' in master.columns:
    keep_cols.append('F1 Std')
master = master[keep_cols].round(4)
# Drop models that completely failed to produce predictions.
master = master[master['F1-Score'] > 0]

print(f"\nMaster table: {len(master)} models across {master['Tier'].nunique()} tiers")
print("\nMaster results:")
print(master.to_string())

master_path = os.path.join(OUT_FINAL, 'master_results.csv')
master.to_csv(master_path)
print(f"\nSaved master results to {master_path}")

# =============================================================================
# Per-Class Metrics (Flags 2 & 7): compute from OOF predictions
# =============================================================================
# For Classical ML and BERT: OOF predictions are indexed over the train set only.
# For Zero-Shot: all_predictions.csv covers all 1900 samples; we filter to the
# train set (using SPLIT_TRAIN_IDX) so the estimate is not contaminated by the
# held-out test set.
# =============================================================================
print("\n" + "-" * 60)
print("Computing per-class metrics from OOF predictions")
print("-" * 60)

try:
    _df_full   = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
    _y_all     = _df_full['has_biomarker'].values
    _train_idx = np.load(SPLIT_TRAIN_IDX)
    _y_train   = _y_all[_train_idx]

    def _per_class(y_true, y_pred):
        bm_f1   = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        nobm_f1 = f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
        macro   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return bm_f1, nobm_f1, macro

    # Tier 1: Classical ML OOF
    _cl_oof_path = os.path.join(OUT_CLASSICAL, 'oof_predictions.csv')
    if os.path.exists(_cl_oof_path):
        _cl_oof = pd.read_csv(_cl_oof_path, index_col=0)
        for _col in _cl_oof.columns:
            if _col in master.index:
                _bm, _no, _mc = _per_class(_y_train, _cl_oof[_col].values.astype(int))
                master.loc[_col, 'BM F1 (OOF)']   = round(_bm, 4)
                master.loc[_col, 'NoBM F1 (OOF)']  = round(_no, 4)
                master.loc[_col, 'Macro F1 (OOF)'] = round(_mc, 4)
                print(f"  {_col:<26} BM-F1={_bm:.4f}  NoBM-F1={_no:.4f}  Macro={_mc:.4f}")
    else:
        print("  WARNING: classical ML OOF predictions not found")

    # Tier 1b: Enhanced ML OOF (02a saves with index=False, so load without index_col)
    _enh_oof_path = os.path.join(OUT_ENHANCED, 'oof_predictions.csv')
    if os.path.exists(_enh_oof_path):
        _enh_oof = pd.read_csv(_enh_oof_path)
        for _col in _enh_oof.columns:
            _master_col = _col + ' (Enhanced)'
            if _master_col in master.index:
                _bm, _no, _mc = _per_class(_y_train, _enh_oof[_col].values.astype(int))
                master.loc[_master_col, 'BM F1 (OOF)']   = round(_bm, 4)
                master.loc[_master_col, 'NoBM F1 (OOF)']  = round(_no, 4)
                master.loc[_master_col, 'Macro F1 (OOF)'] = round(_mc, 4)
                print(f"  {_master_col:<36} BM-F1={_bm:.4f}  NoBM-F1={_no:.4f}  Macro={_mc:.4f}")
    else:
        print("  INFO: Enhanced ML OOF predictions not found (run step 02a to include)")

    # Tier 2: BERT OOF
    _bert_oof_path = os.path.join(OUT_BERT, 'oof_predictions.csv')
    if os.path.exists(_bert_oof_path):
        _bert_oof = pd.read_csv(_bert_oof_path, index_col=0)
        for _col in _bert_oof.columns:
            if _col in master.index:
                _bm, _no, _mc = _per_class(_y_train, _bert_oof[_col].values.astype(int))
                master.loc[_col, 'BM F1 (OOF)']   = round(_bm, 4)
                master.loc[_col, 'NoBM F1 (OOF)']  = round(_no, 4)
                master.loc[_col, 'Macro F1 (OOF)'] = round(_mc, 4)
                print(f"  {_col:<26} BM-F1={_bm:.4f}  NoBM-F1={_no:.4f}  Macro={_mc:.4f}")
    else:
        print("  WARNING: BERT OOF predictions not found")

    # Tier 3: Zero-Shot — filter all_predictions.csv to train rows
    _zs_path = os.path.join(OUT_ZEROSHOT, 'all_predictions.csv')
    if os.path.exists(_zs_path):
        _zs_df    = pd.read_csv(_zs_path, index_col=0)
        _zs_train = _zs_df.values[_train_idx]   # positional indexing into 1900-row array
        for _i, _col in enumerate(_zs_df.columns):
            if _col in master.index:
                _bm, _no, _mc = _per_class(_y_train, _zs_train[:, _i].astype(int))
                master.loc[_col, 'BM F1 (OOF)']   = round(_bm, 4)
                master.loc[_col, 'NoBM F1 (OOF)']  = round(_no, 4)
                master.loc[_col, 'Macro F1 (OOF)'] = round(_mc, 4)
                print(f"  {_col:<26} BM-F1={_bm:.4f}  NoBM-F1={_no:.4f}  Macro={_mc:.4f}")
    else:
        print("  WARNING: zero-shot all_predictions.csv not found")

    # Re-save master with per-class columns
    master.to_csv(master_path)
    print(f"\nUpdated master_results.csv with per-class metrics")

except Exception as _e:
    print(f"  WARNING: Could not compute per-class metrics: {_e}")

# ---- Tier color scheme -------------------------------------------------------
# Grey for classical ML, purple for BERT, orange-red for zero-shot.
# This scheme carries through all figures consistently.
tier_colors = {
    'Classical ML':   '#888780',
    'Enhanced ML':    '#3A9E6B',
    'BioWordVec ML':  '#E8A020',
    'BERT Models':    '#7F77DD',
    'Zero-Shot LLMs': '#D85A30',
}
bar_colors = [tier_colors[t] for t in master['Tier']]
patches    = [mpatches.Patch(color=c, label=l) for l, c in tier_colors.items()]

# ---- Figure: Grand F1 + AUC bar comparison -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Biomarker Detection: All Models Comparison', fontsize=15, fontweight='bold')

x = np.arange(len(master))
f1_stds = master['F1 Std'].values if 'F1 Std' in master.columns else None

for ax, metric, ylabel in [
    (axes[0], 'F1-Score', 'Weighted F1-Score'),
    (axes[1], 'ROC-AUC',  'ROC-AUC Score'),
]:
    yerr = f1_stds if (metric == 'F1-Score' and f1_stds is not None) else None
    bars = ax.bar(x, master[metric], color=bar_colors, alpha=0.88,
                  edgecolor='white', lw=0.5,
                  yerr=yerr, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#333333'})
    ax.set_xticks(x)
    ax.set_xticklabels(master.index, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    title = f'{metric} by Model' + (' (error bars = ±1 std across CV folds)' if metric == 'F1-Score' and yerr is not None else '')
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_ylim(0, 1.18)
    for i, val in enumerate(master[metric]):
        ax.text(i, val + (f1_stds[i] if yerr is not None else 0) + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    # Add dashed vertical lines at tier boundaries to make it easy to compare groups.
    tier_list = master['Tier'].tolist()
    for i in range(1, len(tier_list)):
        if tier_list[i] != tier_list[i - 1]:
            ax.axvline(i - 0.5, color='black', lw=1.2, ls='--', alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(OUT_FINAL, 'fig_all_models.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure to {fig_path}")

# ---- Figure: Performance heatmap ---------------------------------------------
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']
for _extra in ['BM F1 (OOF)', 'NoBM F1 (OOF)', 'Macro F1 (OOF)']:
    if _extra in master.columns:
        metrics.append(_extra)
plot_df  = master[metrics].copy().astype(float)
vmin_val = max(0.0, float(plot_df.values.min()) - 0.05)

fig, ax = plt.subplots(figsize=(14, max(6, len(master) * 0.55)))
sns.heatmap(
    plot_df,
    annot=True,
    fmt='.3f',
    cmap='YlGnBu',
    vmin=vmin_val,
    vmax=1.0,
    linewidths=0.5,
    ax=ax,
    mask=plot_df.isnull(),
    cbar_kws={'label': 'Score'},
)
ax.set_title('All Models -- Performance Heatmap', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Metric', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)

# Color-code y-axis labels by tier so readers can immediately see the groupings.
for ytick in ax.get_yticklabels():
    name = ytick.get_text()
    if name in master.index:
        ytick.set_color(tier_colors.get(master.loc[name, 'Tier'], 'black'))
        ytick.set_fontweight('bold')

# Add horizontal lines at tier boundaries.
tier_list = master['Tier'].tolist()
for i in range(1, len(tier_list)):
    if tier_list[i] != tier_list[i - 1]:
        ax.axhline(i, color='black', lw=2.0)

plt.tight_layout()
heatmap_path = os.path.join(OUT_FINAL, 'fig_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved heatmap to {heatmap_path}")

# ---- LaTeX table -------------------------------------------------------------
def bold_max(series):
    mx = series.max()
    return series.apply(lambda v: f'\\textbf{{{v:.4f}}}' if v == mx else f'{v:.4f}')


# Build LaTeX table columns: standard metrics + optional F1 Std + per-class columns
has_std      = 'F1 Std' in master.columns
has_perclass_latex = 'BM F1 (OOF)' in master.columns
latex_metrics = ['Precision', 'Recall', 'F1-Score']
if has_std:
    latex_metrics.append('F1 Std')
latex_metrics += ['Accuracy', 'ROC-AUC']
if has_perclass_latex:
    latex_metrics += ['BM F1 (OOF)', 'NoBM F1 (OOF)', 'Macro F1 (OOF)']

col_lines = '|'.join(['c'] * len(latex_metrics))
std_note  = ' F1 Std is SD across CV folds (not available for Zero-Shot LLMs).' if has_std else ''
pc_note   = ' BM F1 = Has Biomarker F1; NoBM F1 = No Biomarker F1; all from OOF predictions.' if has_perclass_latex else ''
col_header_map = {
    'F1 Std':        'F1 Std ($\\sigma$)',
    'BM F1 (OOF)':  'BM F1',
    'NoBM F1 (OOF)':'NoBM F1',
    'Macro F1 (OOF)':'Macro F1',
}
lines = [
    '\\begin{table}[htbp]',
    '\\centering',
    '\\caption{Biomarker detection performance across all model tiers. '
    'Bold values indicate best performance per metric. '
    'Classical ML: 5-fold nested CV; BERT: 5-fold CV with early stopping; '
    f'Zero-Shot LLMs: evaluated on training portion of the dataset.{std_note}{pc_note}}}',
    '\\label{tab:results}',
    f'\\begin{{tabular}}{{l|{col_lines}}}',
    '\\hline',
    'Model & ' + ' & '.join(col_header_map.get(m, m) for m in latex_metrics) + ' \\\\',
    '\\hline\\hline',
]

scorable = [m for m in ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC',
                          'BM F1 (OOF)', 'NoBM F1 (OOF)', 'Macro F1 (OOF)']
            if m in master.columns]
best_vals = {m: master[m].max() for m in scorable}
prev_tier = None
for model_name, row in master.iterrows():
    tier = row['Tier']
    if tier != prev_tier:
        if prev_tier is not None:
            lines.append('\\hline')
        lines.append(f'\\multicolumn{{{len(latex_metrics)+1}}}{{l}}'
                     f'{{\\textit{{{tier}}}}} \\\\')
        lines.append('\\hline')
        prev_tier = tier
    cells = []
    for m in latex_metrics:
        if m == 'F1 Std':
            val = row.get('F1 Std', float('nan'))
            cells.append(f'{val:.4f}' if pd.notna(val) else '---')
        elif m in ('BM F1 (OOF)', 'NoBM F1 (OOF)', 'Macro F1 (OOF)'):
            val = row.get(m, float('nan'))
            if pd.notna(val):
                cells.append(f'\\textbf{{{val:.4f}}}' if val == best_vals.get(m) else f'{val:.4f}')
            else:
                cells.append('---')
        else:
            v = row[m]
            cells.append(f'\\textbf{{{v:.4f}}}' if v == best_vals.get(m) else f'{v:.4f}')
    lines.append(f'{model_name} & ' + ' & '.join(cells) + ' \\\\')

lines += ['\\hline', '\\end{tabular}', '\\end{table}']
latex_str = '\n'.join(lines)

latex_path = os.path.join(OUT_FINAL, 'results_table.tex')
with open(latex_path, 'w') as f:
    f.write(latex_str)
print(f"Saved LaTeX table to {latex_path}")

# ---- Summary stats -----------------------------------------------------------
print("\n" + "-" * 60)
print("Paper Summary Stats")
print("-" * 60)
best_model = master['F1-Score'].idxmax()
best_f1    = master['F1-Score'].max()
best_auc_m = master['ROC-AUC'].idxmax()
best_auc   = master['ROC-AUC'].max()
print(f"Best F1:  {best_model} -> {best_f1:.4f}")
print(f"Best AUC: {best_auc_m} -> {best_auc:.4f}")
print()
for tier_name, grp in master.groupby('Tier', sort=False):
    print(f"  {tier_name}:")
    print(f"    Best F1  = {grp['F1-Score'].max():.4f} ({grp['F1-Score'].idxmax()})")
    print(f"    Best AUC = {grp['ROC-AUC'].max():.4f} ({grp['ROC-AUC'].idxmax()})")
    print(f"    Mean F1  = {grp['F1-Score'].mean():.4f}")

# ---- evaluation_table.txt ---------------------------------------------------
from datetime import date
eval_lines = [
    "=" * 60,
    "FULL MODEL EVALUATION TABLE",
    "Biomarker Detection in Alzheimer's Clinical Trial Texts",
    "=" * 60,
    f"\nGenerated: {date.today()}",
    f"Dataset: {len(t1)+len(t1b)+len(t2)+len(t4)} models evaluated",
    "Text format: clean_freeform (natural language segments only)",
]

has_std    = 'F1 Std' in master.columns
has_perclass = 'BM F1 (OOF)' in master.columns
tier_labels  = {
    'Classical ML':   'TIER 1: CLASSICAL ML  (5-fold nested CV, weighted metrics)',
    'Enhanced ML':    'TIER 1b: ENHANCED ML  (TF-IDF + FastText, MLP, 5-fold nested CV)',
    'BERT Models':    'TIER 2: BERT MODELS  (5-fold CV, weighted metrics)',
    'Zero-Shot LLMs': 'TIER 3: ZERO-SHOT LLMs  (train-set OOF via index filter, no fine-tuning)',
}
for tier_name, tier_label in tier_labels.items():
    subset = master[master['Tier'] == tier_name]
    if subset.empty:
        continue
    header = f"  {'Model':<22} {'Prec':>6} {'Recall':>7} {'W-F1':>6}"
    if has_std:
        header += f" {'F1Std':>6}"
    if has_perclass:
        header += f" {'BM-F1':>6} {'NoBM':>6} {'MacroF1':>8}"
    header += f" {'Acc':>6} {'AUC':>7}"
    sep = "  " + "-" * (len(header) - 2)
    eval_lines += ["", "-" * 60, tier_label, "-" * 60, "", header, sep]
    for model_name, row in subset.iterrows():
        line = f"  {model_name:<22} {row['Precision']:>6.4f} {row['Recall']:>7.4f} {row['F1-Score']:>6.4f}"
        if has_std:
            val = row.get('F1 Std', float('nan'))
            line += f" {val:>6.4f}" if pd.notna(val) else "     --"
        if has_perclass:
            bm  = row.get('BM F1 (OOF)',   float('nan'))
            no  = row.get('NoBM F1 (OOF)', float('nan'))
            mc  = row.get('Macro F1 (OOF)', float('nan'))
            line += f" {bm:>6.4f}" if pd.notna(bm) else "     --"
            line += f" {no:>6.4f}" if pd.notna(no) else "     --"
            line += f" {mc:>8.4f}" if pd.notna(mc) else "       --"
        line += f" {row['Accuracy']:>6.4f} {row['ROC-AUC']:>7.4f}"
        eval_lines.append(line)

eval_lines += [
    "",
    "-" * 60,
    "OVERALL BEST MODELS",
    "-" * 60,
    "",
    f"  Best F1  : {master['F1-Score'].idxmax():<25} F1={master['F1-Score'].max():.4f}",
    f"  Best AUC : {master['ROC-AUC'].idxmax():<25} AUC={master['ROC-AUC'].max():.4f}",
    "",
    "  (Test set results: see outputs/final_results/test_set_results.csv)",
    "",
    "=" * 60,
    "END",
    "=" * 60,
]

eval_table_path = os.path.join(OUT_FINAL, 'evaluation_table.txt')
with open(eval_table_path, 'w') as f:
    f.write('\n'.join(eval_lines))
print(f"Saved evaluation table to {eval_table_path}")

print("\nStep 5 complete.")
