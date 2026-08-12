"""
demo/app.py -- Interactive Gradio demo for the biomarker detection pipeline.

Loads all trained models and provides four tabs:
  Tab 1: Single Model Prediction -- choose a model, enter text, get a prediction.
  Tab 2: Batch Excel Processing -- upload an Excel file, specify a column, run all rows.
  Tab 3: Compare All Models -- run all 15 models on the same text, vote on consensus.
  Tab 4: Model Info -- CV scores and descriptions for all models.

Classical ML models and BERT models are loaded at startup. The three zero-shot
LLMs (BART, Flan-T5, BioGPT) are lazy-loaded on first use to avoid consuming
GPU memory when not needed.

Usage:
    conda run -n biomarker-ad python demo/app.py
    conda run -n biomarker-ad python demo/app.py --share
"""

import os
import sys
import json
import pickle
import copy
import warnings
import argparse
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src/ to the path so we can import config from there.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC_DIR)
from config import MODELS_CLASSICAL, MODELS_BERT, MODELS_META, BERT_ENSEMBLE_MEMBERS, BASE_DIR

MODELS_CLASSICAL_ENHANCED   = os.path.join(BASE_DIR, 'models', 'classical_ml_enhanced')
OUT_ENHANCED                = os.path.join(BASE_DIR, 'outputs', 'classical_ml_enhanced')
MODELS_CLASSICAL_BIOWORDVEC = os.path.join(BASE_DIR, 'models', 'classical_ml_biowordvec')
OUT_BIOWORDVEC              = os.path.join(BASE_DIR, 'outputs', 'classical_ml_biowordvec')

# Import shared transformer class so pickled BioWordVec pipelines can be
# unpickled here (pickle looks for embeddings.BioWordVecVectorizer).
import embeddings as _emb
from embeddings import BioWordVecVectorizer, FastTextVectorizer  # noqa: F401

# Load BioWordVec model into the shared module if models exist.
_BWV_MODEL_PATH = os.environ.get(
    'BIOWORDVEC_PATH',
    os.path.join(os.path.expanduser('~'), 'BioWordVec_PubMed_MIMICIII_d200.bin'),
)
if os.path.isdir(MODELS_CLASSICAL_BIOWORDVEC) and os.path.exists(_BWV_MODEL_PATH):
    try:
        import fasttext as _ft_lib
        _emb.BWV_MODEL = _ft_lib.load_model(_BWV_MODEL_PATH)
        print(f"BioWordVec loaded for demo ({_emb.BWV_MODEL.get_dimension()}-dim)")
    except Exception as _bwv_exc:
        print(f"WARNING: Could not load BioWordVec for demo: {_bwv_exc}")
elif os.path.isdir(MODELS_CLASSICAL_BIOWORDVEC):
    print(f"WARNING: BioWordVec model file not found at {_BWV_MODEL_PATH}")
    print(f"  Set BIOWORDVEC_PATH env var to load BioWordVec models in the demo.")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# ---- Parse args --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true', help='Generate a public Gradio link')
parser.add_argument('--port', type=int, default=7860)
args, _ = parser.parse_known_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Check that models exist -------------------------------------------------
if not os.path.exists(MODELS_META):
    print("ERROR: models/metadata.json not found.")
    print("You need to run the full pipeline first:")
    print("  bash run_pipeline.sh")
    sys.exit(1)

with open(MODELS_META) as f:
    META = json.load(f)

print("-" * 60)
print("Biomarker Detection Demo")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("-" * 60)

# =============================================================================
# Load Classical ML models (fast -- sklearn pickles)
# =============================================================================
print("\nLoading classical ML models...")
CLASSICAL_MODELS = {}
classical_meta   = META.get('classical_ml', {})
for model_name in classical_meta.get('models', []):
    safe_name = model_name.replace(' ', '_')
    path = os.path.join(MODELS_CLASSICAL, f'{safe_name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            CLASSICAL_MODELS[model_name] = pickle.load(f)
        cv_f1 = classical_meta.get('cv_f1', {}).get(model_name, 0)
        print(f"  Loaded {model_name}  (CV F1={cv_f1:.4f})")
    else:
        print(f"  Not found: {path}")

# =============================================================================
# Load Enhanced Classical ML models (TF-IDF + FastText + MLP, from step 02a)
# =============================================================================
# Models are stored as sklearn Pipelines — same interface as classical ML.
# Names get an " (Enhanced)" suffix to distinguish them in the dropdown.
# CV scores are read from outputs/classical_ml_enhanced/results.csv if present.

print("\nLoading enhanced classical ML models (02a)...")
ENHANCED_MODELS  = {}
enhanced_cv_f1   = {}

_enhanced_results_path = os.path.join(OUT_ENHANCED, 'results.csv')
if os.path.exists(_enhanced_results_path):
    _edf = pd.read_csv(_enhanced_results_path, index_col=0)
    enhanced_cv_f1 = _edf['F1-Score'].to_dict() if 'F1-Score' in _edf.columns else {}

_enhanced_model_names = [
    'LR', 'SVM', 'Random Forest', 'XGBoost', 'Decision Tree',
    'MLP Small', 'MLP Deep', 'CatBoost',
]
for _mname in _enhanced_model_names:
    _safe  = _mname.replace(' ', '_')
    _path  = os.path.join(MODELS_CLASSICAL_ENHANCED, f'{_safe}.pkl')
    _label = f'{_mname} (Enhanced)'
    if os.path.exists(_path):
        with open(_path, 'rb') as _f:
            ENHANCED_MODELS[_label] = pickle.load(_f)
        _f1 = enhanced_cv_f1.get(_mname, 0)
        print(f"  Loaded {_label}  (CV F1={_f1:.4f})")
    else:
        print(f"  Not found (run 02a first): {_path}")

# =============================================================================
# Load BioWordVec Classical ML models (TF-IDF + BioWordVec + MLP, from step 02b)
# =============================================================================
print("\nLoading BioWordVec classical ML models (02b)...")
BIOWORDVEC_MODELS  = {}
biowordvec_cv_f1   = {}

_biowordvec_results_path = os.path.join(OUT_BIOWORDVEC, 'results.csv')
if os.path.exists(_biowordvec_results_path):
    _bdf = pd.read_csv(_biowordvec_results_path, index_col=0)
    biowordvec_cv_f1 = _bdf['F1-Score'].to_dict() if 'F1-Score' in _bdf.columns else {}

_biowordvec_model_names = [
    'LR', 'SVM', 'Random Forest', 'XGBoost', 'Decision Tree',
    'MLP Small', 'MLP Deep', 'CatBoost',
]
for _mname in _biowordvec_model_names:
    _safe  = _mname.replace(' ', '_')
    _path  = os.path.join(MODELS_CLASSICAL_BIOWORDVEC, f'{_safe}.pkl')
    _label = f'{_mname} (BioWordVec)'
    if os.path.exists(_path):
        with open(_path, 'rb') as _f:
            BIOWORDVEC_MODELS[_label] = pickle.load(_f)
        _f1 = biowordvec_cv_f1.get(_mname, 0)
        print(f"  Loaded {_label}  (CV F1={_f1:.4f})")
    else:
        print(f"  Not found (run 02b first): {_path}")

# =============================================================================
# BERT model class (matches architecture used in 03_train_bert_models.py)
# =============================================================================
class BERTClassifier(nn.Module):
    def __init__(self, model_name, hidden_size, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


# =============================================================================
# Load BERT models (heavier -- each is ~400MB on GPU)
# =============================================================================
print("\nLoading BERT models...")
BERT_MODELS     = {}
BERT_TOKENIZERS = {}
bert_meta = META.get('bert', {})

for model_name in bert_meta.get('models', []):
    subfolder = bert_meta.get('subfolders', {}).get(model_name)
    if not subfolder:
        continue
    model_dir = os.path.join(MODELS_BERT, subfolder)
    config_path = os.path.join(model_dir, 'config.json')
    state_path  = os.path.join(model_dir, 'model_state.pt')
    tok_dir     = os.path.join(model_dir, 'tokenizer')

    if not all(os.path.exists(p) for p in [config_path, state_path, tok_dir]):
        print(f"  Missing files for {model_name}, skipping")
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    try:
        tok   = AutoTokenizer.from_pretrained(tok_dir)
        model = BERTClassifier(cfg['model_name'], cfg['hidden_size']).to(DEVICE)
        state = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        BERT_MODELS[model_name]     = (model, cfg)
        BERT_TOKENIZERS[model_name] = tok
        cv_f1  = bert_meta.get('cv_f1', {}).get(model_name, 0)
        cv_auc = bert_meta.get('cv_auc', {}).get(model_name, 0)
        print(f"  Loaded {model_name}  (CV F1={cv_f1:.4f}, AUC={cv_auc:.4f})")
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")

# =============================================================================
# Zero-shot LLMs -- lazy loaded on first use to save GPU memory at startup
# =============================================================================
ZEROSHOT_CACHE = {}


def get_bart():
    """Load BART NLI pipeline on first call, then cache it."""
    if 'bart' not in ZEROSHOT_CACHE:
        print("Loading BART zero-shot model (first use)...")
        from transformers import pipeline as hf_pipeline
        ZEROSHOT_CACHE['bart'] = hf_pipeline(
            'zero-shot-classification',
            model='facebook/bart-large-mnli',
            device=0 if torch.cuda.is_available() else -1,
        )
    return ZEROSHOT_CACHE['bart']


def get_flant5():
    """Load Flan-T5 tokenizer and model on first call, then cache."""
    if 'flan' not in ZEROSHOT_CACHE:
        print("Loading Flan-T5 model (first use)...")
        from transformers import AutoTokenizer as AT, AutoModelForSeq2SeqLM
        tok = AT.from_pretrained('google/flan-t5-base')
        mod = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(DEVICE)
        mod.eval()
        ZEROSHOT_CACHE['flan'] = (tok, mod)
    return ZEROSHOT_CACHE['flan']


def get_biogpt():
    """Load BioGPT tokenizer and model on first call, then cache."""
    if 'biogpt' not in ZEROSHOT_CACHE:
        print("Loading BioGPT model (first use)...")
        from transformers import AutoTokenizer as AT, AutoModelForCausalLM
        tok = AT.from_pretrained('microsoft/biogpt')
        mod = AutoModelForCausalLM.from_pretrained('microsoft/biogpt').to(DEVICE)
        mod.eval()
        ZEROSHOT_CACHE['biogpt'] = (tok, mod)
    return ZEROSHOT_CACHE['biogpt']


# =============================================================================
# Prediction functions
# =============================================================================

def predict_classical(text, model_name):
    """Run a sklearn pipeline and return (label, confidence)."""
    pipe = CLASSICAL_MODELS[model_name]
    pred = int(pipe.predict([text])[0])
    prob = float(pipe.predict_proba([text])[0][pred])
    return pred, prob


def predict_bert(text, model_name):
    """Tokenize, forward pass, return (label, confidence)."""
    if model_name not in BERT_MODELS:
        return None, None
    model, cfg = BERT_MODELS[model_name]
    tok = BERT_TOKENIZERS[model_name]
    enc = tok(text, truncation=True, padding=True,
               max_length=cfg.get('max_len', 512), return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


def predict_bart(text):
    """BART NLI zero-shot classification."""
    bart = get_bart()
    labels = ['contains biomarker', 'no biomarker']
    r = bart(text[:512], labels)
    pred = 1 if r['labels'][0] == labels[0] else 0
    prob = float(r['scores'][0])
    return pred, prob


def predict_flant5(text):
    """Flan-T5 few-shot generation with Yes/No logit scoring."""
    tok, mod = get_flant5()
    prompt = (
        "Classify whether the following clinical trial text mentions any biomarkers "
        "(such as CSF tau, amyloid PET, plasma p-tau, MRI, EEG, or similar biological markers).\n\n"
        'Example 1: "CSF p-tau181 and amyloid PET eligibility criteria; plasma Abeta42/40 ratio measured at baseline" -> Yes\n'
        'Example 2: "Patients must be 65 years or older with mild cognitive impairment. No prior neurological surgery." -> No\n'
        'Example 3: "Plasma GFAP and NfL collected to assess neurodegeneration. Tau PET imaging at baseline and week 78." -> Yes\n'
        'Example 4: "Oral tablet administered twice daily for 52 weeks. Blood pressure monitored monthly." -> No\n\n'
        f'Now classify: "{text[:512]}"\nAnswer (Yes/No):'
    )
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    yes_id = tok.encode('Yes', add_special_tokens=False)[0]
    no_id  = tok.encode('No',  add_special_tokens=False)[0]
    with torch.no_grad():
        out = mod.generate(**inputs, max_new_tokens=5,
                           output_scores=True, return_dict_in_generate=True)
    decoded = tok.decode(out.sequences[0], skip_special_tokens=True)
    pred    = 1 if 'yes' in decoded.lower() else 0
    try:
        logits = out.scores[0][0]
        prob   = torch.softmax(
            torch.tensor([logits[no_id].item(), logits[yes_id].item()]), dim=0
        )[pred].item()
    except Exception:
        prob = 1.0 if pred == 1 else 0.0
    return pred, float(prob)


def predict_biogpt(text):
    """BioGPT perplexity scoring: compare LM loss for pos vs neg hypothesis."""
    tok, mod = get_biogpt()
    pos_hyp = 'This clinical trial text mentions biomarkers.'
    neg_hyp = 'This clinical trial text does not mention any biomarkers.'
    ctx = text[:250]

    def score(hypothesis):
        full = ctx + ' ' + hypothesis
        enc  = tok(full, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mod(**enc, labels=enc['input_ids'])
        return -out.loss.item()  # higher = more plausible

    s_pos = score(pos_hyp)
    s_neg = score(neg_hyp)
    pred  = 1 if s_pos > s_neg else 0
    prob  = float(torch.softmax(torch.tensor([s_neg, s_pos]), dim=0)[1].item())
    return pred, prob


# ---- Classical Ensemble (soft voting over all 7 classical models: LR, SVM, RF, XGBoost, DT, MLP Small, MLP Deep) -----------
def predict_classical_ensemble(text):
    """CV-F1-weighted soft vote over all loaded classical models."""
    if not CLASSICAL_MODELS:
        return None, None
    cv_f1     = classical_meta.get('cv_f1', {})
    prob_sum  = np.zeros(2)
    weight_sum = 0.0
    for name, pipe in CLASSICAL_MODELS.items():
        w = cv_f1.get(name, 1.0)
        prob_sum   += w * pipe.predict_proba([text])[0]
        weight_sum += w
    avg_probs = prob_sum / weight_sum
    pred = int(np.argmax(avg_probs))
    return pred, float(avg_probs[pred])


# ---- BERT ensemble (soft voting over BERT_ENSEMBLE_MEMBERS from config) ------
def predict_bert_ensemble(text):
    """Average softmax probabilities from the three strongest BERT models."""
    available = [m for m in BERT_ENSEMBLE_MEMBERS if m in BERT_MODELS]
    if not available:
        return None, None
    prob_sum = np.zeros(2)
    for model_name in available:
        model, cfg = BERT_MODELS[model_name]
        tok = BERT_TOKENIZERS[model_name]
        enc = tok(text, truncation=True, padding=True,
                  max_length=cfg.get('max_len', 512), return_tensors='pt')
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prob_sum += probs
    avg_probs = prob_sum / len(available)
    pred = int(np.argmax(avg_probs))
    return pred, float(avg_probs[pred])


# ---- Unified prediction dispatch ---------------------------------------------
ALL_MODEL_NAMES = (
    list(CLASSICAL_MODELS.keys()) +
    ['Classical Ensemble'] +
    list(ENHANCED_MODELS.keys()) +
    list(BIOWORDVEC_MODELS.keys()) +
    list(BERT_MODELS.keys()) +
    ['BERT Ensemble'] +
    ['BART Zero-Shot', 'Flan-T5 Few-Shot', 'BioGPT Zero-Shot']
)


def run_prediction(model_name, text):
    """Route a prediction to the correct function based on model type."""
    if model_name in CLASSICAL_MODELS:
        return predict_classical(text, model_name)
    elif model_name in ENHANCED_MODELS:
        pipe = ENHANCED_MODELS[model_name]
        pred = int(pipe.predict([text])[0])
        prob = float(pipe.predict_proba([text])[0][pred])
    elif model_name in BIOWORDVEC_MODELS:
        pipe = BIOWORDVEC_MODELS[model_name]
        pred = int(pipe.predict([text])[0])
        prob = float(pipe.predict_proba([text])[0][pred])
        return pred, prob
    elif model_name == 'Classical Ensemble':
        return predict_classical_ensemble(text)
    elif model_name in BERT_MODELS:
        return predict_bert(text, model_name)
    elif model_name == 'BERT Ensemble':
        return predict_bert_ensemble(text)
    elif model_name == 'BART Zero-Shot':
        return predict_bart(text)
    elif model_name == 'Flan-T5 Few-Shot':
        return predict_flant5(text)
    elif model_name == 'BioGPT Zero-Shot':
        return predict_biogpt(text)
    return None, None


# ---- CV score lookup ---------------------------------------------------------
def get_cv_score(model_name):
    if model_name in ENHANCED_MODELS:
        base = model_name.replace(' (Enhanced)', '')
        return enhanced_cv_f1.get(base, 0.0)
    if model_name in BIOWORDVEC_MODELS:
        base = model_name.replace(' (BioWordVec)', '')
        return biowordvec_cv_f1.get(base, 0.0)
    if model_name in classical_meta.get('cv_f1', {}):
        return classical_meta['cv_f1'][model_name]
    if model_name == 'Classical Ensemble':
        return classical_meta.get('ensemble_cv_f1', 0.8446)
    if model_name in bert_meta.get('cv_f1', {}):
        return bert_meta['cv_f1'][model_name]
    if model_name == 'BERT Ensemble':
        return bert_meta.get('ensemble_cv_f1', 0.0)
    zeroshot_scores = META.get('zeroshot', {}).get('scores', {})
    if model_name in zeroshot_scores:
        return zeroshot_scores[model_name].get('f1', 0)
    fallbacks = {
        'BART Zero-Shot':    0.6925,
        'Flan-T5 Few-Shot':  0.4465,
        'BioGPT Zero-Shot':  0.6122,
    }
    return fallbacks.get(model_name, 0.0)


def get_tier_label(model_name):
    if model_name in CLASSICAL_MODELS:
        return 'Classical ML'
    if model_name == 'Classical Ensemble':
        return 'Classical ML'
    if model_name in ENHANCED_MODELS:
        return 'Enhanced ML (TF-IDF + FastText)'
    if model_name in BIOWORDVEC_MODELS:
        return 'BioWordVec ML (TF-IDF + BioWordVec)'
    if model_name in BERT_MODELS:
        return 'BERT Models'
    if model_name == 'BERT Ensemble':
        return 'BERT Ensemble'
    return 'Zero-Shot LLM'


# =============================================================================
# Example texts for the demo
# =============================================================================
EXAMPLE_TEXTS = [
    "CSF p-tau181 and amyloid PET eligibility criteria; plasma Abeta42/40 ratio measured at baseline, week 26 and week 52.",
    "Patients must be 65 years or older with a diagnosis of mild cognitive impairment. No prior neurological surgery.",
    "FDG-PET scan and MRI volumetry used to assess hippocampal atrophy. APOE e4 genotyping performed at screening.",
    "Oral tablet administered twice daily for 52 weeks. Blood pressure and heart rate monitored monthly.",
    "Plasma GFAP and NfL biomarkers collected to assess neurodegeneration. Tau PET imaging at baseline and week 78.",
]

# =============================================================================
# Gradio interface functions
# =============================================================================

def single_predict(text, model_name):
    """Run one model and return formatted HTML result, confidence bar, and info text."""
    if not text.strip():
        return (
            "<div style='padding:20px;text-align:center;background:#f0f0f0;color:#555;border-radius:8px'>"
            "Enter text above and click Predict.</div>",
            "",
            "",
            "",
        )

    pred, conf = run_prediction(model_name, text)
    if pred is None:
        return (
            "<div style='padding:20px;text-align:center;background:#fff3cd;color:#856404;border-radius:8px'>"
            "Model not available. Run run_pipeline.sh first.</div>",
            "", "", "",
        )

    label     = "Has Biomarker" if pred == 1 else "No Biomarker"
    tier      = get_tier_label(model_name)
    cv_f1     = get_cv_score(model_name)
    conf_pct  = conf * 100

    if pred == 1:
        box_style  = 'background:#d4edda;color:#155724;border:2px solid #28a745;'
        bar_color  = '#28a745'
    else:
        box_style  = 'background:#f8d7da;color:#721c24;border:2px solid #dc3545;'
        bar_color  = '#dc3545'

    result_html = (
        f"<div style='padding:20px;text-align:center;font-size:1.4em;"
        f"font-weight:bold;border-radius:8px;{box_style}'>"
        f"{label}<br>"
        f"<span style='font-size:0.65em;font-weight:normal;color:#555'>{tier}</span>"
        f"</div>"
    )

    conf_html = (
        f"<div style='margin:8px 0'>"
        f"<div style='background:#e9ecef;border-radius:8px;height:24px;overflow:hidden'>"
        f"<div style='background:{bar_color};width:{conf_pct:.1f}%;height:100%'></div></div>"
        f"<div style='text-align:center;font-weight:bold;margin-top:4px'>"
        f"Confidence: {conf_pct:.1f}%</div>"
        f"</div>"
    )

    info = f"Model: **{model_name}** | {tier} | CV F1 = {cv_f1:.4f}"
    return result_html, conf_html, info, f"{conf_pct:.1f}%"


# =============================================================================
# Batch Excel processing
# =============================================================================

def batch_predict_excel(file_obj, column_name, model_name, progress=gr.Progress()):
    """
    Read an uploaded Excel file, run predictions on every row in `column_name`,
    insert a 'Has Biomarker' column immediately to the right of it, and return
    the annotated file as a downloadable path plus a summary HTML string.

    Cell colour is written via openpyxl after pandas builds the data, so the
    downloaded .xlsx has green / red / amber fills matching the UI preview.

    Returns
    -------
    output_path : str | None   -- path to the annotated .xlsx (None on error)
    summary_html : str         -- HTML snippet with stats + preview table
    """
    # ------------------------------------------------------------------ guards
    if file_obj is None:
        return None, _alert("Please upload an Excel file first.", "warn")

    col = column_name.strip()
    if not col:
        return None, _alert("Please enter the column name to process.", "warn")

    # ------------------------------------------------------------------ read
    try:
        df = pd.read_excel(file_obj.name)
    except Exception as exc:
        return None, _alert(f"Could not read file: {exc}", "warn")

    if col not in df.columns:
        available = ", ".join(str(c) for c in df.columns)
        return None, _alert(
            f'Column "{col}" not found in the file. '
            f'Available columns: {available}', "warn"
        )

    # ------------------------------------------------------------------ predict
    labels      = []
    confidences = []
    n = len(df)

    for i, cell_val in enumerate(df[col]):
        progress((i + 1) / n, desc=f"Processing row {i+1} / {n}...")
        text = str(cell_val).strip() if pd.notna(cell_val) else ""

        if not text:
            labels.append("ERROR")
            confidences.append(None)
            continue

        pred, conf = run_prediction(model_name, text)
        if pred is None:
            labels.append("ERROR")
            confidences.append(None)
        else:
            labels.append("Yes" if pred == 1 else "No")
            confidences.append(round(conf * 100, 1))

    # ------------------------------------------------------------------ insert column
    target_pos = df.columns.get_loc(col) + 1          # position right after target col

    has_biomarker_vals = []
    for lbl, conf_val in zip(labels, confidences):
        if lbl == "ERROR":
            has_biomarker_vals.append("Error — empty cell")
        elif lbl == "Yes":
            has_biomarker_vals.append(f"Yes ({conf_val}%)")
        else:
            has_biomarker_vals.append(f"No ({conf_val}%)")

    df.insert(target_pos, "Has Biomarker", has_biomarker_vals)

    # ------------------------------------------------------------------ save with colours
    import openpyxl
    from openpyxl.styles import PatternFill, Font

    fill_green  = PatternFill("solid", fgColor="D4EDDA")
    fill_red    = PatternFill("solid", fgColor="F8D7DA")
    fill_amber  = PatternFill("solid", fgColor="FFF3CD")
    font_green  = Font(color="155724", bold=True)
    font_red    = Font(color="721C24", bold=True)
    font_amber  = Font(color="856404", bold=True)


    import tempfile, os
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biomarker_results.xlsx")

    # output_path = file_obj.name.replace(".xlsx", "_biomarker_results.xlsx").replace(".xls", "_biomarker_results.xlsx")
    df.to_excel(output_path, index=False)

    wb = openpyxl.load_workbook(output_path)
    ws = wb.active

    # openpyxl is 1-indexed; row 1 = header, data starts at row 2
    result_col_idx = target_pos + 1   # 1-based column index in openpyxl

    for row_i, lbl in enumerate(labels, start=2):   # start=2 skips header
        cell = ws.cell(row=row_i, column=result_col_idx)
        if lbl == "Yes":
            cell.fill = fill_green
            cell.font = font_green
        elif lbl == "No":
            cell.fill = fill_red
            cell.font = font_red
        else:
            cell.fill = fill_amber
            cell.font = font_amber

    wb.save(output_path)

    # ------------------------------------------------------------------ summary HTML
    n_yes   = labels.count("Yes")
    n_no    = labels.count("No")
    n_err   = labels.count("ERROR")

    stats_html = (
        "<div style='display:flex;gap:10px;margin-bottom:14px'>"
        f"<div style='flex:1;background:#f8f9fa;border-radius:6px;padding:10px 14px'>"
        f"<div style='font-size:11px;color:#666'>Total rows</div>"
        f"<div style='font-size:22px;font-weight:600'>{n}</div></div>"
        f"<div style='flex:1;background:#d4edda;border-radius:6px;padding:10px 14px'>"
        f"<div style='font-size:11px;color:#155724'>Has biomarker</div>"
        f"<div style='font-size:22px;font-weight:600;color:#155724'>{n_yes}</div></div>"
        f"<div style='flex:1;background:#f8d7da;border-radius:6px;padding:10px 14px'>"
        f"<div style='font-size:11px;color:#721c24'>No biomarker</div>"
        f"<div style='font-size:22px;font-weight:600;color:#721c24'>{n_no}</div></div>"
        f"<div style='flex:1;background:#fff3cd;border-radius:6px;padding:10px 14px'>"
        f"<div style='font-size:11px;color:#856404'>Errors</div>"
        f"<div style='font-size:22px;font-weight:600;color:#856404'>{n_err}</div></div>"
        "</div>"
    )

    # Preview table: first 20 rows, only target col + result col
    preview_df = df[[col, "Has Biomarker"]].head(20)

    def _style_cell(val):
        if str(val).startswith("Yes"):
            return "background:#d4edda;color:#155724;font-weight:bold;padding:4px 8px;border-radius:4px"
        if str(val).startswith("No"):
            return "background:#f8d7da;color:#721c24;font-weight:bold;padding:4px 8px;border-radius:4px"
        return "background:#fff3cd;color:#856404;font-weight:bold;padding:4px 8px;border-radius:4px"

    rows_html = ""
    for idx, row in preview_df.iterrows():
        cell_text = str(row[col])[:120] + ("…" if len(str(row[col])) > 120 else "")
        result_val = str(row["Has Biomarker"])
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 10px;font-size:12px;color:#555;max-width:320px;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{cell_text}</td>"
            f"<td style='padding:6px 10px'>"
            f"<span style='{_style_cell(result_val)}'>{result_val}</span></td>"
            f"</tr>"
        )

    table_html = (
        "<div style='overflow-x:auto;border:1px solid #dee2e6;border-radius:6px'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<thead><tr>"
        f"<th style='text-align:left;padding:8px 10px;border-bottom:1px solid #dee2e6;"
        f"font-size:11px;color:#666'>{col}</th>"
        "<th style='text-align:left;padding:8px 10px;border-bottom:1px solid #dee2e6;"
        "font-size:11px;color:#27500A;background:#EAF3DE'>Has Biomarker</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
        f"<p style='font-size:11px;color:#888;margin-top:6px'>"
        f"Showing first 20 of {n} rows. Download the file for full results.</p>"
    )

    summary_html = stats_html + table_html
    return output_path, summary_html


def _alert(msg, level="warn"):
    """Return a small HTML alert box."""
    styles = {
        "warn":  "background:#fff3cd;color:#856404;border:1px solid #f0ad4e",
        "info":  "background:#d1ecf1;color:#0c5460;border:1px solid #bee5eb",
        "error": "background:#f8d7da;color:#721c24;border:1px solid #f5c6cb",
    }
    s = styles.get(level, styles["warn"])
    return f"<div style='padding:10px 14px;border-radius:6px;font-size:13px;{s}'>{msg}</div>"


def compare_all(text):
    """Run all available models and return a results DataFrame plus consensus line."""
    if not text.strip():
        return (
            None,
            "<p style='color:#888'>Enter text above then click Compare All.</p>",
        )

    rows = []
    for mname in ALL_MODEL_NAMES:
        pred, conf = run_prediction(mname, text)
        if pred is None:
            continue
        label  = "Has Biomarker" if pred == 1 else "No Biomarker"
        tier   = get_tier_label(mname)
        cv_f1  = get_cv_score(mname)
        rows.append({
            'Model':      mname,
            'Prediction': label,
            'Confidence': f"{conf*100:.1f}%",
            'Tier':       tier,
            'CV F1':      f"{cv_f1:.4f}",
        })

    result_df = pd.DataFrame(rows)

    preds_list = [1 if r['Prediction'] == 'Has Biomarker' else 0 for r in rows]
    vote  = sum(preds_list)
    total = len(preds_list)
    majority_label = "Has Biomarker" if vote > total / 2 else "No Biomarker"
    consensus = (
        f"<b>Consensus: {vote}/{total} models predict {majority_label}</b>"
    )
    return result_df, consensus


# =============================================================================
# Build the Gradio interface
# =============================================================================

with gr.Blocks(title="Biomarker Detection Demo") as demo:
    gr.Markdown("""
# Biomarker Detection in Alzheimer's Clinical Trials
### Multi-Tier NLP Classification Pipeline

Predicts whether a clinical trial text mentions biomarkers
(tau, amyloid, CSF, MRI, GFAP, NfL, EEG, APOE, and related).
""")

    with gr.Tabs():

        # -----------------------------------------------------------------
        # Tab 1: Single model prediction
        # -----------------------------------------------------------------
        with gr.Tab("Single Model Prediction"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input_1 = gr.Textbox(
                        label="Clinical Trial Text",
                        placeholder="Paste or type clinical trial text here...",
                        lines=6,
                    )
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=ALL_MODEL_NAMES,
                            value=ALL_MODEL_NAMES[0] if ALL_MODEL_NAMES else None,
                            label="Select Model",
                        )
                        predict_btn = gr.Button("Predict", variant="primary", scale=1)

                with gr.Column(scale=2):
                    result_html    = gr.HTML(label="Prediction")
                    conf_bar_html  = gr.HTML(label="Confidence")
                    info_text      = gr.Markdown()
                    conf_hidden    = gr.Textbox(visible=False)

            gr.Examples(
                examples=EXAMPLE_TEXTS,
                inputs=text_input_1,
                label="Example Texts (click to load)",
            )

            predict_btn.click(
                fn=single_predict,
                inputs=[text_input_1, model_dropdown],
                outputs=[result_html, conf_bar_html, info_text, conf_hidden],
            )
            text_input_1.submit(
                fn=single_predict,
                inputs=[text_input_1, model_dropdown],
                outputs=[result_html, conf_bar_html, info_text, conf_hidden],
            )

        # -----------------------------------------------------------------
        # Tab 2: Batch Excel Processing  ← NEW TAB
        # -----------------------------------------------------------------
        with gr.Tab("Batch Excel Processing"):
            gr.Markdown(
                "Upload an Excel file, specify the column containing your clinical trial text, "
                "select a model, and click **Run Batch**. "
                "A **Has Biomarker** column will be inserted immediately to the right of your "
                "target column — green for Yes, red for No, amber for empty cells. "
                "Download the annotated file when processing is complete."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    batch_file = gr.File(
                        label="Upload Excel file (.xlsx / .xls)",
                        file_types=[".xlsx", ".xls"],
                    )
                with gr.Column(scale=1):
                    batch_col_name = gr.Textbox(
                        label="Column name to process",
                        placeholder='e.g. text',
                        value="text",
                    )
                    batch_model_dropdown = gr.Dropdown(
                        choices=ALL_MODEL_NAMES,
                        value=ALL_MODEL_NAMES[0] if ALL_MODEL_NAMES else None,
                        label="Select Model",
                    )
                    batch_run_btn = gr.Button("Run Batch", variant="primary")

            batch_summary_html = gr.HTML(label="Results Summary & Preview")

            batch_download = gr.File(
                label="⬇️ Download annotated Excel file",
                visible=True,
                value=None,
            )

            gr.Markdown(
                "**Note:** Zero-shot LLMs (BART, Flan-T5, BioGPT) load on first use "
                "(30–60 s). For large files, BERT models are faster than zero-shot LLMs."
            )

            def _run_batch_and_show(file_obj, col, model, progress=gr.Progress()):
                output_path, summary_html = batch_predict_excel(
                    file_obj, col, model, progress
                )
                if output_path is None:
                    return summary_html, None
                return summary_html, output_path

            batch_run_btn.click(
                fn=_run_batch_and_show,
                inputs=[batch_file, batch_col_name, batch_model_dropdown],
                outputs=[batch_summary_html, batch_download],
            )

        # -----------------------------------------------------------------
        # Tab 3: Compare all models
        # -----------------------------------------------------------------
        with gr.Tab("Compare All Models"):
            gr.Markdown(
                "Runs all loaded models on the same text and shows results side by side."
            )
            with gr.Row():
                text_input_2 = gr.Textbox(
                    label="Clinical Trial Text",
                    placeholder="Paste or type clinical trial text here...",
                    lines=6,
                )
                compare_btn = gr.Button("Compare All Models", variant="primary")

            consensus_box  = gr.HTML()
            compare_table  = gr.Dataframe(
                headers=['Model', 'Prediction', 'Confidence', 'Tier', 'CV F1'],
                label="All Model Predictions",
            )

            gr.Markdown(
                "Note: Zero-shot LLMs (BART, Flan-T5, BioGPT) are loaded on first use, "
                "which may take 30-60 seconds."
            )

            gr.Examples(
                examples=EXAMPLE_TEXTS,
                inputs=text_input_2,
                label="Example Texts",
            )

            compare_btn.click(
                fn=compare_all,
                inputs=text_input_2,
                outputs=[compare_table, consensus_box],
            )

        # -----------------------------------------------------------------
        # Tab 4: Model information
        # -----------------------------------------------------------------
        with gr.Tab("Model Info"):
            n_samples    = META.get('dataset_size', 'unknown')
            pos_rate     = META.get('positive_rate', 0)
            classical_f1 = classical_meta.get('cv_f1', {})
            bert_f1      = bert_meta.get('cv_f1', {})
            bert_auc     = bert_meta.get('cv_auc', {})
            zs_scores    = META.get('zeroshot', {}).get('scores', {})
            enhanced_meta   = META.get('enhanced_ml', {})
            biowordvec_meta = META.get('biowordvec_ml', {})

            classical_rows = '\n'.join(
                f'| {m} | {classical_f1.get(m, 0):.4f} |'
                for m in classical_meta.get('models', [])
            )
            cl_ens_f1  = classical_meta.get('ensemble_cv_f1', 0.0)
            cl_ens_auc = classical_meta.get('ensemble_cv_auc', 0.0)
            if cl_ens_f1 > 0:
                classical_rows += f'\n| Classical Ensemble (CV-F1-weighted soft vote) | {cl_ens_f1:.4f} |'
            bert_rows = '\n'.join(
                f'| {m} | {bert_f1.get(m, 0):.4f} | {bert_auc.get(m, 0):.4f} |'
                for m in bert_meta.get('models', [])
            )
            ens_f1  = bert_meta.get('ensemble_cv_f1', 0.0)
            ens_auc = bert_meta.get('ensemble_cv_auc', 0.0)
            if ens_f1 > 0:
                bert_rows += f'\n| BERT Ensemble (BioBERT+PubMedBERT+BioLinkBERT) | {ens_f1:.4f} | {ens_auc:.4f} |'
            zs_rows = '\n'.join(
                f'| {m} | {v.get("f1", 0):.4f} |'
                for m, v in zs_scores.items()
            ) if zs_scores else ('| BART Zero-Shot | 0.6658 |\n'
                                  '| Flan-T5 Few-Shot | 0.4465 |\n'
                                  '| BioGPT Zero-Shot | 0.6122 |')

            _enh_f1   = enhanced_meta.get('cv_f1', {})
            _enh_auc  = enhanced_meta.get('cv_auc', {})
            _enh_feat = enhanced_meta.get('features', 'TF-IDF + FastText-300')
            _enh_rows = '\n'.join(
                f'| {m} | {_enh_f1.get(m, 0):.4f} | {_enh_auc.get(m, 0):.4f} |'
                for m in enhanced_meta.get('models', [])
            ) if enhanced_meta.get('models') else '_Run 02a_classical_ml_enhanced.py to populate._'

            _bwv_f1   = biowordvec_meta.get('cv_f1', {})
            _bwv_auc  = biowordvec_meta.get('cv_auc', {})
            _bwv_feat = biowordvec_meta.get('features', 'TF-IDF + BioWordVec-200')
            _bwv_rows = '\n'.join(
                f'| {m} | {_bwv_f1.get(m, 0):.4f} | {_bwv_auc.get(m, 0):.4f} |'
                for m in biowordvec_meta.get('models', [])
            ) if biowordvec_meta.get('models') else '_Run 02b_classical_ml_biowordvec.py to populate._'

            gr.Markdown(f"""
## Loaded Models

Trained on **{n_samples} samples** ({pos_rate*100:.1f}% positive).

### Classical ML (TF-IDF + Classifier)
5-fold nested cross-validation. Class imbalance handled via `class_weight='balanced'`.

| Model | CV F1 |
|---|---|
{classical_rows}

### Enhanced ML (TF-IDF + FastText + MLP)
Same 5-fold CV structure. Features: {_enh_feat}.
Includes neural network models (MLP Small, MLP Deep) with early stopping.

| Model | CV F1 | CV AUC |
|---|---|---|
{_enh_rows}

### BioWordVec ML (TF-IDF + BioWordVec + MLP)
Same 5-fold CV structure. Features: {_bwv_feat}.
BioWordVec trained on PubMed + MIMIC-III — domain-specific biomedical embeddings.

| Model | CV F1 | CV AUC |
|---|---|---|
{_bwv_rows}

### BERT Models (Fine-tuned, 5-fold CV)
Domain-specific BERT variants pretrained on biomedical literature.
Simple classifier head (Dropout -> Linear) with early stopping.

| Model | CV F1 | CV AUC |
|---|---|---|
{bert_rows}

### Zero-Shot LLMs (No training)
Used as-is from HuggingFace Hub. No labeled training data needed.
Loaded on first use from the Compare All tab.

| Model | Full-dataset F1 |
|---|---|
{zs_rows}

---
**Task:** Binary classification -- does the text mention biomarkers?

**Biomarkers included:** tau, amyloid-beta, p-tau, CSF, MRI, FDG-PET,
GFAP, NfL, APOE, EEG, hippocampal atrophy, and related terms.
""")

# ---- Launch ------------------------------------------------------------------
print("\n" + "-" * 60)
print("Launching demo...")
print(f"Port: {args.port}")
if args.share:
    print("Public sharing enabled.")
print("-" * 60)

demo.launch(
    server_name='0.0.0.0',
    server_port=args.port,
    share=args.share,
    show_error=True,
)
