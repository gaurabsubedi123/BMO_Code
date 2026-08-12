"""
01_prepare_dataset.py -- Build outcome-level labeled dataset from biomarker annotations.

Reads the biomarker annotations Excel file and produces one row per outcome measure.
This is an outcome-level dataset (not trial-level): each row is a single clinical
trial outcome with its own label.

Label scheme (from the annotation column):
  has_biomarker = 0  if Annotation starts with 'ANALOG'
                     (non-biomarker outcomes: neuropsych evals, safety, QoL, etc.)
  has_biomarker = 1  if Annotation is any other non-null value
                     (biomarker outcomes: CSF, Plasma, MRI, PET, EEG, etc.)
  Rows with null Annotation are excluded (unannotatable / unclear endpoints).
  Rows with comma-separated multi-label annotations are treated as biomarker=1
  since at least one annotation is a biomarker type.

Input:
    data/raw/GOLD_FILE   (Excel, Outcomes sheet)

Output:
    outputs/dataset/labeled_dataset.csv
    outputs/dataset/fig_dataset_overview.png
"""

import os
import sys
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import GroupShuffleSplit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_RAW, OUT_DATASET, GOLD_FILE, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX

os.makedirs(OUT_DATASET, exist_ok=True)

print("-" * 60)
print("Step 1 of 7: Preparing dataset")
print("-" * 60)

# ---- Load annotations file ---------------------------------------------------
xl_path = os.path.join(DATA_RAW, GOLD_FILE)
df_raw  = pd.read_excel(xl_path, sheet_name='Outcomes')
print(f"\nLoaded {len(df_raw)} rows from Outcomes sheet")
print(f"Unique trials (NCTId): {df_raw['NCTId'].nunique()}")
print(f"Fields: {df_raw['Field'].value_counts().to_dict()}")

# ---- Exclude rows with null Annotation ---------------------------------------
n_before = len(df_raw)
df_raw   = df_raw[df_raw['Annotation'].notna()].copy()
n_after  = len(df_raw)
print(f"\nExcluded {n_before - n_after} rows with null annotations")
print(f"Working dataset: {n_after} rows")

# ---- Binary label ------------------------------------------------------------
# ANALOG_* = non-biomarker outcome (label 0).
# Everything else, including comma-separated multi-label rows, = biomarker (label 1).
df_raw['has_biomarker'] = (~df_raw['Annotation'].str.startswith('ANALOG', na=False)).astype(int)

pos = df_raw['has_biomarker'].sum()
neg = (df_raw['has_biomarker'] == 0).sum()
print(f"\nClass distribution:")
print(f"  has_biomarker=1 (biomarker):     {pos} ({pos/len(df_raw)*100:.1f}%)")
print(f"  has_biomarker=0 (non-biomarker): {neg} ({neg/len(df_raw)*100:.1f}%)")


# ---- Text cleaning -----------------------------------------------------------
def clean_text(text):
    """Normalize text: strip brackets/quotes, collapse whitespace, lowercase."""
    text = re.sub(r"\['([^']+)'\]", r'\1', str(text))
    text = re.sub(r"[\[\]'\"{}]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


# clean_freeform: concatenate Outcome Measure + Outcome Description.
# Both fields are already natural language, so no structured-code filtering needed.
df_raw['clean_freeform'] = (
    df_raw['Outcome Measure'].fillna('').astype(str) + ' ' +
    df_raw['Outcome Description'].fillna('').astype(str)
).apply(clean_text)

# ---- Keyword features (for baseline and error analysis) ----------------------
BIOMARKER_KEYWORDS = [
    'amyloid', 'abeta', 'a-beta', 'florbetapir', 'florbetaben', 'flutemetamol',
    'tau', 'p-tau', 'ptau', 'phospho-tau', 'p-tau181', 'p-tau217', 'p-tau231',
    't-tau', 'ttau', 'flortaucipir', 'mk-6240',
    'mri', 'fdg-pet', 'fdg pet', 'hippocampal', 'hippocampus', 'atrophy',
    'cortical thickness', 'white matter', 'ventricular',
    'csf', 'cerebrospinal fluid', 'plasma', 'serum',
    'neurofilament', 'nfl', 'neurogranin', 'gfap', 'ykl-40',
    'eeg', 'electroencephalogram', 'event-related potential',
    'apoe', 'apoe4',
    'biomarker', 'biological marker',
    'pet', 'spect', 'diffusion tensor', 'dti',
]

df_raw['matched_keywords'] = df_raw['clean_freeform'].apply(
    lambda t: [kw for kw in BIOMARKER_KEYWORDS if kw in t])
df_raw['keyword_count']    = df_raw['matched_keywords'].apply(len)
df_raw['word_count']       = df_raw['clean_freeform'].apply(lambda x: len(x.split()))
df_raw['text_length']      = df_raw['clean_freeform'].apply(len)

# ---- Print summary -----------------------------------------------------------
print(f"\nBy outcome field:")
for field, grp in df_raw.groupby('Field'):
    p = grp['has_biomarker'].sum()
    n = (grp['has_biomarker'] == 0).sum()
    print(f"  {field:<22}: {len(grp):5d} total | {p} positive / {n} negative")

print(f"\nTop 10 annotation types (biomarker=1):")
bm_annots = df_raw[df_raw['has_biomarker'] == 1]['Annotation'].value_counts().head(10)
for ann, cnt in bm_annots.items():
    print(f"  {ann:<45}: {cnt}")

print(f"\nTop 5 ANALOG types (biomarker=0):")
an_annots = df_raw[df_raw['has_biomarker'] == 0]['Annotation'].value_counts().head(5)
for ann, cnt in an_annots.items():
    print(f"  {ann:<45}: {cnt}")

all_kws = [kw for kws in df_raw['matched_keywords'] for kw in kws]
print(f"\nTop 10 matched keywords:")
for kw, cnt in Counter(all_kws).most_common(10):
    print(f"  {kw:<35}: {cnt}")

print(f"\nText length stats (words):")
print(f"  Mean: {df_raw['word_count'].mean():.1f}  "
      f"Median: {df_raw['word_count'].median():.0f}  "
      f"Min/Max: {df_raw['word_count'].min()} / {df_raw['word_count'].max()}")

print(f"\nTrial-level group stats:")
grp_sizes = df_raw.groupby('NCTId').size()
print(f"  Unique trials: {len(grp_sizes)}")
print(f"  Outcomes per trial — mean: {grp_sizes.mean():.1f}  "
      f"min: {grp_sizes.min()}  max: {grp_sizes.max()}")

# ---- Save dataset ------------------------------------------------------------
SAVE_COLS = [
    'RowId', 'NCTId', 'Field',
    'Outcome Measure', 'Outcome Description',
    'clean_freeform',
    'has_biomarker', 'Annotation',
    'matched_keywords', 'keyword_count',
    'word_count', 'text_length',
]
out_path = os.path.join(OUT_DATASET, 'labeled_dataset.csv')
df_raw[SAVE_COLS].to_csv(out_path, index=False)
print(f"\nSaved dataset to {out_path}  ({len(df_raw)} rows, {df_raw['NCTId'].nunique()} trials)")

# ---- 80/20 group-aware train/test split (created once, used by ALL downstream steps) ---
# GroupShuffleSplit keeps all outcomes from one trial together so no trial
# appears in both train and test. random_state=42 makes the split reproducible.
gss_main = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_y       = df_raw['has_biomarker'].values
_groups  = df_raw['NCTId'].values
global_tr_idx, global_te_idx = next(
    gss_main.split(np.arange(len(df_raw)), _y, groups=_groups)
)
np.save(SPLIT_TRAIN_IDX, global_tr_idx)
np.save(SPLIT_TEST_IDX,  global_te_idx)
n_tr_pos = int(_y[global_tr_idx].sum())
n_te_pos = int(_y[global_te_idx].sum())
print(f"\nSaved 80/20 group-aware split:")
print(f"  Train: {len(global_tr_idx)} samples "
      f"({n_tr_pos} positive / {len(global_tr_idx) - n_tr_pos} negative) | "
      f"{len(set(_groups[global_tr_idx]))} trials")
print(f"  Test:  {len(global_te_idx)} samples "
      f"({n_te_pos} positive / {len(global_te_idx) - n_te_pos} negative) | "
      f"{len(set(_groups[global_te_idx]))} trials")

# ---- Figures -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Dataset Overview -- Biomarker Annotations (Trial & Outcome Level)',
             fontsize=13, fontweight='bold')

# Panel 1: Class balance by trial
# A trial is "Has Biomarker" if ANY of its outcomes has has_biomarker=1,
# otherwise it is "No Biomarker".
trial_label = df_raw.groupby('NCTId')['has_biomarker'].max()
trial_pos   = int((trial_label == 1).sum())
trial_neg   = int((trial_label == 0).sum())
trial_counts = [trial_neg, trial_pos]
labels = ['No Biomarker', 'Has Biomarker']
colors = ['#B4B2A9', '#7F77DD']
bars   = axes[0].bar(labels, trial_counts, color=colors, alpha=0.85, width=0.5)
axes[0].set_title('Trial-Level Class Distribution', fontweight='bold')
axes[0].set_ylabel('Number of Trials')
for bar, val in zip(bars, trial_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Panel 2: Trial-level class balance by outcome field
# For each (trial, field) pair, a trial "Has Biomarker" in that field if any
# of its outcomes in that field are labelled 1.
trial_field = df_raw.groupby(['NCTId', 'Field'])['has_biomarker'].max().reset_index()
field_grp = trial_field.groupby(['Field', 'has_biomarker']).size().unstack(fill_value=0)
field_grp.columns = ['No Biomarker Trials', 'Has Biomarker Trials']
field_grp.plot(kind='bar', ax=axes[1], color=['#B4B2A9', '#7F77DD'], alpha=0.85)
axes[1].set_title('Trials with Biomarker by Outcome Field', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Number of Trials')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3: Word count distribution by label
for label, color, name in [(0, '#B4B2A9', 'No Biomarker'),
                            (1, '#7F77DD', 'Has Biomarker')]:
    sub = df_raw[df_raw['has_biomarker'] == label]['word_count']
    axes[2].hist(sub, bins=30, alpha=0.65, color=color, label=name)
axes[2].set_xlabel('Word count')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Text Length by Label', fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_DATASET, 'fig_dataset_overview.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved figure to {fig_path}")
print("\nStep 1 complete.")
