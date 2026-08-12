"""
config.py -- Central path configuration for the BMO classification pipeline.

Every other script imports paths from here. This means if you move the project
or rename an output folder, you only need to change it in one place.

Usage in other scripts:
    from config import OUT_DATASET, OUT_CLASSICAL, DATA_RAW, GOLD_FILE
"""

import os

# The project root is two directories above this file (src/ -> project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw input data lives here. Users must copy the two Excel files into this folder.
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')

# Output directories, one per pipeline step.
OUT_DATASET  = os.path.join(BASE_DIR, 'outputs', 'dataset')
OUT_CLASSICAL   = os.path.join(BASE_DIR, 'outputs', 'classical_ml')
OUT_ENHANCED    = os.path.join(BASE_DIR, 'outputs', 'classical_ml_enhanced')
OUT_BIOWORDVEC  = os.path.join(BASE_DIR, 'outputs', 'classical_ml_biowordvec')
OUT_BERT        = os.path.join(BASE_DIR, 'outputs', 'bert_models')
OUT_ZEROSHOT = os.path.join(BASE_DIR, 'outputs', 'zeroshot_llm')
OUT_FINAL    = os.path.join(BASE_DIR, 'outputs', 'final_results')

# Saved model directories. Classical ML models are pickles; BERT models get subfolders.
MODELS_CLASSICAL            = os.path.join(BASE_DIR, 'models', 'classical_ml')
MODELS_CLASSICAL_ENHANCED   = os.path.join(BASE_DIR, 'models', 'classical_ml_enhanced')
MODELS_CLASSICAL_BIOWORDVEC = os.path.join(BASE_DIR, 'models', 'classical_ml_biowordvec')
MODELS_BERT                 = os.path.join(BASE_DIR, 'models', 'bert')

# The metadata JSON is used by the demo to look up CV scores and model lists.
MODELS_META      = os.path.join(BASE_DIR, 'models', 'metadata.json')

# BERT ensemble members (soft vote). Bio+ClinicalBERT excluded (weakest by CV F1).
BERT_ENSEMBLE_MEMBERS = ['BioBERT', 'PubMedBERT', 'BioLinkBERT']

# 80/20 group-aware train/test split indices (created once in step 01, used by all steps).
SPLIT_TRAIN_IDX = os.path.join(OUT_DATASET, 'train_idx.npy')
SPLIT_TEST_IDX  = os.path.join(OUT_DATASET, 'test_idx.npy')

# Exact filename of the raw annotation Excel file.
# This must match what is placed in data/raw/ exactly.
# Note: CTO_FILE (combined trial metadata) is defined here for reference but is not
# used by any pipeline script — only GOLD_FILE is loaded by 01_prepare_dataset.py.
CTO_FILE  = '2026-02-11 Combined CTO and BMO 2026 Data - FINALL (2).xlsx'
GOLD_FILE = '2025-04-25 Biomarker Annotations (clean and current) - for CS 3-25-2026.xlsx'
