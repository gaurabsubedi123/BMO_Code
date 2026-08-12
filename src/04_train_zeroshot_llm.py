"""
04_train_zeroshot_llm.py -- Zero-shot and few-shot LLM evaluation.

These models are used WITHOUT any fine-tuning on our dataset. That means no
overfitting risk, but also typically lower performance than fine-tuned BERT.
The value of this comparison is to understand how much labeled training data
helps versus just prompting a pretrained model.

Models:
  - BART (facebook/bart-large-mnli): NLI zero-shot classification.
    Scores entailment probability between the text and a label hypothesis.
  - Flan-T5 (google/flan-t5-base): Seq2Seq few-shot prompting.
    Given examples in the prompt, generates "Yes" or "No".
  - BioGPT (microsoft/biogpt): Causal LM perplexity scoring.
    Compares language model loss for positive vs negative hypothesis appended
    to the context. Whichever hypothesis the model assigns lower loss wins.

"Tuning" here means selecting the best prompt phrasing, not adjusting weights:
  Phase 1: Try 3 prompt/label variants on a fixed 20% holdout (seed=99).
  Phase 2: Evaluate the best variant on the full dataset.

Since there are no training folds, we use bootstrap confidence intervals
(1000 resamples, 95%) instead of CV to quantify uncertainty.

Outputs:
    outputs/zeroshot_llm/results.csv
    outputs/zeroshot_llm/analysis.txt
    outputs/zeroshot_llm/all_predictions.csv
    outputs/zeroshot_llm/all_probabilities.csv
    outputs/zeroshot_llm/fig_zeroshot_results.png
    outputs/zeroshot_llm/fig_cm.png
    outputs/zeroshot_llm/fig_test_eval.png
    models/metadata.json
"""

import os
import sys
import json
import warnings
import datetime
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (pipeline, AutoTokenizer,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM)
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUT_DATASET, OUT_CLASSICAL, OUT_BERT, OUT_ZEROSHOT,
                    MODELS_META, BERT_ENSEMBLE_MEMBERS, SPLIT_TRAIN_IDX, SPLIT_TEST_IDX)

os.makedirs(OUT_ZEROSHOT, exist_ok=True)

print("-" * 60)
print("Step 4 of 7: Zero-shot and few-shot LLM evaluation")
print("-" * 60)

DEVICE   = 0 if torch.cuda.is_available() else -1
DEVICE_T = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {'cuda -- ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# ---- Load dataset ------------------------------------------------------------
df         = pd.read_csv(os.path.join(OUT_DATASET, 'labeled_dataset.csv'))
texts_all  = df['clean_freeform'].fillna('').values
y_true_all = df['has_biomarker'].values
pos_rate   = y_true_all.mean()
print(f"\nLoaded {len(texts_all)} samples | {pos_rate*100:.1f}% positive")

# Prompt/threshold tuning uses 20% of the TRAINING set only (not test data).
# Load the 80/20 split from step 01, then take a 20% subset of train.
# Zero-shot evaluation still runs on all N samples (no training = no leakage).
groups           = df['NCTId'].values
global_train_idx = np.load(SPLIT_TRAIN_IDX)
global_test_idx  = np.load(SPLIT_TEST_IDX)
texts_train      = texts_all[global_train_idx]
y_train          = y_true_all[global_train_idx]
groups_train     = groups[global_train_idx]
texts_test_global  = texts_all[global_test_idx]
y_test_global      = y_true_all[global_test_idx]
gss_tune = GroupShuffleSplit(n_splits=1, test_size=0.8, random_state=99)
tune_idx, _ = next(gss_tune.split(texts_train, y_train, groups=groups_train))
texts_tune = texts_train[tune_idx]
y_tune     = y_train[tune_idx]


def truncate(text, max_chars=512):
    """Truncate to max_chars at a word boundary to avoid cutting mid-word."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0]


texts_trunc_all  = [truncate(t) for t in texts_all]
texts_trunc_tune = [truncate(t) for t in texts_tune]


# ---- Bootstrap CI ------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, seed=42):
    """Compute 95% bootstrap confidence interval for F1 and AUC."""
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    f1s, aucs = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        f1s.append(f1_score(y_true[idx], y_pred[idx],
                            average='weighted', zero_division=0))
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return {
        'F1_mean':     np.mean(f1s),
        'F1_ci_low':   np.percentile(f1s, 2.5),
        'F1_ci_high':  np.percentile(f1s, 97.5),
        'AUC_mean':    np.mean(aucs) if aucs else 0.5,
        'AUC_ci_low':  np.percentile(aucs, 2.5) if aucs else 0.5,
        'AUC_ci_high': np.percentile(aucs, 97.5) if aucs else 0.5,
    }


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Accuracy':  accuracy_score(y_true, y_pred),
        'ROC-AUC':   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


# ---- Baselines ---------------------------------------------------------------
print("\nComputing baselines...")
y_majority       = np.ones(len(y_true_all), dtype=int)
y_majority_prob  = np.ones(len(y_true_all))
baseline_majority = compute_metrics(y_true_all, y_majority, y_majority_prob)

if 'keyword_count' in df.columns:
    y_keyword = (df['keyword_count'] > 0).astype(int).values
else:
    y_keyword = y_majority.copy()
y_keyword_prob   = y_keyword.astype(float)
baseline_keyword = compute_metrics(y_true_all, y_keyword, y_keyword_prob)

print(f"  Majority class baseline:  F1={baseline_majority['F1-Score']:.4f}  "
      f"Acc={baseline_majority['Accuracy']:.4f}")
print(f"  Keyword silver baseline:  F1={baseline_keyword['F1-Score']:.4f}  "
      f"Acc={baseline_keyword['Accuracy']:.4f}")

all_results = {}
all_preds   = {}
all_probs   = {}
all_ci      = {}

# =============================================================================
# Model 1: BART Zero-Shot NLI
# =============================================================================
print("\n" + "-" * 60)
print("Model 1: BART (facebook/bart-large-mnli)")
print("-" * 60)

BART_LABEL_OPTIONS = {
    'Option A': ['contains biomarker', 'no biomarker'],
    'Option B': ['mentions biomarkers', 'does not mention biomarkers'],
    'Option C': ['biomarker present', 'biomarker absent'],
}

try:
    bart_pipe = pipeline('zero-shot-classification',
                         model='facebook/bart-large-mnli', device=DEVICE)

    print(f"\nPhase 1: label + threshold tuning on {len(texts_tune)} holdout samples")
    bart_best_f1, bart_best_labels, bart_best_thresh = -1, None, 0.5
    for opt_name, labels in BART_LABEL_OPTIONS.items():
        probs = []
        for text in texts_trunc_tune:
            r    = bart_pipe(text, labels)
            prob = r['scores'][0] if r['labels'][0] == labels[0] else 1 - r['scores'][0]
            probs.append(prob)
        probs = np.array(probs)
        # Search for the threshold that maximises weighted F1 on the holdout set.
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds_t = (probs >= thresh).astype(int)
            f1_t = f1_score(y_tune, preds_t, average='weighted', zero_division=0)
            if f1_t > bart_best_f1:
                bart_best_f1, bart_best_labels, bart_best_thresh = f1_t, labels, thresh
        print(f"  {opt_name}: best threshold={bart_best_thresh:.2f}  F1={bart_best_f1:.4f}")

    print(f"\nBest labels: {bart_best_labels}  threshold={bart_best_thresh:.2f}  "
          f"(holdout F1={bart_best_f1:.4f})")

    print(f"\nPhase 2: full evaluation on all {len(texts_all)} samples...")
    y_pred_bart, y_prob_bart = [], []
    for i, text in enumerate(texts_trunc_all):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(texts_trunc_all)}", end='\r')
        r    = bart_pipe(text, bart_best_labels)
        prob = r['scores'][0] if r['labels'][0] == bart_best_labels[0] else 1 - r['scores'][0]
        pred = 1 if prob >= bart_best_thresh else 0
        y_pred_bart.append(pred)
        y_prob_bart.append(prob)
    print()

    y_pred_bart = np.array(y_pred_bart)
    y_prob_bart = np.array(y_prob_bart)
    all_results['BART Zero-Shot'] = compute_metrics(y_true_all, y_pred_bart, y_prob_bart)
    all_preds['BART Zero-Shot']   = y_pred_bart
    all_probs['BART Zero-Shot']   = y_prob_bart
    all_ci['BART Zero-Shot']      = bootstrap_ci(y_true_all, y_pred_bart, y_prob_bart)

    r  = all_results['BART Zero-Shot']
    ci = all_ci['BART Zero-Shot']
    print(f"\nBART Zero-Shot results:")
    print(f"  F1={r['F1-Score']:.4f} (95% CI: {ci['F1_ci_low']:.4f}-{ci['F1_ci_high']:.4f})")
    print(f"  AUC={r['ROC-AUC']:.4f} | Acc={r['Accuracy']:.4f}")
    del bart_pipe
    torch.cuda.empty_cache()

except Exception as e:
    print(f"BART failed: {e}")
    all_results['BART Zero-Shot'] = {'Precision': 0, 'Recall': 0, 'F1-Score': 0,
                                      'Accuracy': 0, 'ROC-AUC': 0}
    all_preds['BART Zero-Shot'] = np.zeros(len(y_true_all))
    all_probs['BART Zero-Shot'] = np.zeros(len(y_true_all))
    all_ci['BART Zero-Shot']    = {}

# =============================================================================
# Model 2: Flan-T5 Few-Shot
# =============================================================================
print("\n" + "-" * 60)
print("Model 2: Flan-T5 (google/flan-t5-base)")
print("-" * 60)

FLAN_PROMPTS = {
    'Detailed': (
        "Classify whether the following clinical trial text mentions any biomarkers "
        "(such as CSF tau, amyloid PET, plasma p-tau, MRI, EEG, or similar biological markers).\n\n"
        'Example 1: "CSF p-tau181 and amyloid PET eligibility criteria; plasma Abeta42/40 ratio measured at baseline" -> Yes\n'
        'Example 2: "Patients must be 65 years or older with mild cognitive impairment. No prior neurological surgery." -> No\n'
        'Example 3: "Plasma p-tau217 and tau PET imaging used as co-primary endpoints at week 78." -> Yes\n'
        'Example 4: "Oral tablet administered twice daily for 52 weeks. Blood pressure monitored monthly." -> No\n\n'
        'Now classify: "{text}"\nAnswer (Yes/No):'
    ),
    'Concise': (
        "Does this clinical trial text mention biomarkers (tau, amyloid, MRI, CSF, EEG, GFAP, NfL, etc.)?\n"
        'Example: "amyloid PET scan and plasma GFAP collected at baseline" -> Yes\n'
        'Example: "oral medication twice daily, MMSE score monitored" -> No\n'
        'Text: "{text}"\nAnswer (Yes/No):'
    ),
    'Clinical': (
        "Task: Biomarker detection in Alzheimer's disease clinical trials.\n"
        "Biomarkers include: amyloid-beta, tau, p-tau, MRI, FDG-PET, CSF, GFAP, NfL, EEG, APOE.\n\n"
        'Example 1: "FDG-PET scan and MRI volumetry used to assess hippocampal atrophy. APOE e4 genotyping at screening." -> Yes\n'
        'Example 2: "Computerized cognitive training program. Primary outcome is change in MMSE score." -> No\n'
        'Example 3: "Plasma GFAP and NfL collected to assess neurodegeneration. Tau PET imaging at baseline and week 78." -> Yes\n'
        'Example 4: "Psychoeducation program for caregivers. Outcomes include Zarit Burden Interview and PHQ-9." -> No\n\n'
        'Text: "{text}"\nAnswer (Yes/No):'
    ),
}

try:
    flan_tok = AutoTokenizer.from_pretrained('google/flan-t5-base')
    flan_mod = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(DEVICE_T)
    flan_mod.eval()

    yes_id = flan_tok.encode('Yes', add_special_tokens=False)[0]
    no_id  = flan_tok.encode('No',  add_special_tokens=False)[0]

    def flan_predict(texts, prompt_template):
        preds, probs = [], []
        for text in texts:
            prompt = prompt_template.format(text=text[:512])
            inputs = flan_tok(prompt, return_tensors='pt',
                              truncation=True, max_length=512).to(DEVICE_T)
            with torch.no_grad():
                out = flan_mod.generate(
                    **inputs, max_new_tokens=5,
                    output_scores=True, return_dict_in_generate=True,
                )
            decoded = flan_tok.decode(out.sequences[0], skip_special_tokens=True)
            pred = 1 if 'yes' in decoded.lower() else 0
            try:
                logits    = out.scores[0][0]
                yes_logit = logits[yes_id].item()
                no_logit  = logits[no_id].item()
                prob = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)[1].item()
            except Exception:
                prob = float(pred)
            preds.append(pred)
            probs.append(prob)
        return np.array(preds), np.array(probs)

    print(f"\nPhase 1: prompt tuning on {len(texts_tune)} holdout samples")
    flan_best_f1, flan_best_prompt_name, flan_best_prompt = -1, None, None
    for pname, prompt in FLAN_PROMPTS.items():
        preds, _ = flan_predict(texts_trunc_tune, prompt)
        f1 = f1_score(y_tune, preds, average='weighted', zero_division=0)
        print(f"  {pname}: F1={f1:.4f}")
        if f1 > flan_best_f1:
            flan_best_f1, flan_best_prompt_name, flan_best_prompt = f1, pname, prompt

    print(f"\nBest prompt: {flan_best_prompt_name}  (holdout F1={flan_best_f1:.4f})")

    print(f"\nPhase 2: full evaluation on all {len(texts_all)} samples...")
    y_pred_flan, y_prob_flan = flan_predict(texts_trunc_all, flan_best_prompt)
    all_results['Flan-T5 Few-Shot'] = compute_metrics(y_true_all, y_pred_flan, y_prob_flan)
    all_preds['Flan-T5 Few-Shot']   = y_pred_flan
    all_probs['Flan-T5 Few-Shot']   = y_prob_flan
    all_ci['Flan-T5 Few-Shot']      = bootstrap_ci(y_true_all, y_pred_flan, y_prob_flan)

    r  = all_results['Flan-T5 Few-Shot']
    ci = all_ci['Flan-T5 Few-Shot']
    print(f"\nFlan-T5 results:")
    print(f"  F1={r['F1-Score']:.4f} (95% CI: {ci['F1_ci_low']:.4f}-{ci['F1_ci_high']:.4f})")
    print(f"  AUC={r['ROC-AUC']:.4f} | Acc={r['Accuracy']:.4f}")
    del flan_mod, flan_tok
    torch.cuda.empty_cache()

except Exception as e:
    print(f"Flan-T5 failed: {e}")
    all_results['Flan-T5 Few-Shot'] = {'Precision': 0, 'Recall': 0, 'F1-Score': 0,
                                        'Accuracy': 0, 'ROC-AUC': 0}
    all_preds['Flan-T5 Few-Shot'] = np.zeros(len(y_true_all))
    all_probs['Flan-T5 Few-Shot'] = np.zeros(len(y_true_all))
    all_ci['Flan-T5 Few-Shot']    = {}

# =============================================================================
# Model 3: BioGPT Zero-Shot (perplexity scoring)
# =============================================================================
print("\n" + "-" * 60)
print("Model 3: BioGPT (microsoft/biogpt) -- perplexity scoring")
print("-" * 60)

# BioGPT doesn't natively support NLI, so we use a perplexity trick:
# score two hypotheses by appending them to the context and comparing LM loss.
# The hypothesis the model finds more plausible (lower loss) wins.
BIOGPT_HYP_OPTIONS = {
    'Option A': (
        'This clinical trial text mentions biomarkers.',
        'This clinical trial text does not mention any biomarkers.',
    ),
    'Option B': (
        'Biomarkers such as tau, amyloid, or MRI are discussed in this text.',
        'No biomarkers are discussed in this text.',
    ),
    'Option C': (
        'This text includes biological markers for Alzheimer disease.',
        'This text does not include any biological markers.',
    ),
}

try:
    bio_tok = AutoTokenizer.from_pretrained('microsoft/biogpt')
    bio_mod = AutoModelForCausalLM.from_pretrained('microsoft/biogpt').to(DEVICE_T)
    bio_mod.eval()

    def biogpt_score(context, hypothesis):
        """Compute negative LM loss for the full text. Higher = model finds it more likely."""
        full = context + ' ' + hypothesis
        enc  = bio_tok(full, return_tensors='pt',
                       truncation=True, max_length=512).to(DEVICE_T)
        with torch.no_grad():
            out = bio_mod(**enc, labels=enc['input_ids'])
        return -out.loss.item()

    def biogpt_predict(texts, pos_hyp, neg_hyp):
        preds, probs = [], []
        for text in texts:
            ctx   = text[:250]
            s_pos = biogpt_score(ctx, pos_hyp)
            s_neg = biogpt_score(ctx, neg_hyp)
            prob  = torch.softmax(torch.tensor([s_neg, s_pos]), dim=0)[1].item()
            preds.append(1 if s_pos > s_neg else 0)
            probs.append(prob)
        return np.array(preds), np.array(probs)

    print(f"\nPhase 1: hypothesis tuning on {len(texts_tune)} holdout samples")
    bio_best_f1, bio_best_name, bio_best_pos, bio_best_neg = -1, None, None, None
    for opt_name, (pos_h, neg_h) in BIOGPT_HYP_OPTIONS.items():
        preds, _ = biogpt_predict(texts_trunc_tune, pos_h, neg_h)
        f1 = f1_score(y_tune, preds, average='weighted', zero_division=0)
        print(f"  {opt_name}: F1={f1:.4f}  | pos_hyp: '{pos_h[:55]}...'")
        if f1 > bio_best_f1:
            bio_best_f1, bio_best_name = f1, opt_name
            bio_best_pos, bio_best_neg = pos_h, neg_h

    print(f"\nBest hypothesis: {bio_best_name}  (holdout F1={bio_best_f1:.4f})")

    print(f"\nPhase 2: full evaluation on all {len(texts_all)} samples...")
    y_pred_bio, y_prob_bio = biogpt_predict(texts_trunc_all, bio_best_pos, bio_best_neg)
    all_results['BioGPT Zero-Shot'] = compute_metrics(y_true_all, y_pred_bio, y_prob_bio)
    all_preds['BioGPT Zero-Shot']   = y_pred_bio
    all_probs['BioGPT Zero-Shot']   = y_prob_bio
    all_ci['BioGPT Zero-Shot']      = bootstrap_ci(y_true_all, y_pred_bio, y_prob_bio)

    r  = all_results['BioGPT Zero-Shot']
    ci = all_ci['BioGPT Zero-Shot']
    print(f"\nBioGPT results:")
    print(f"  F1={r['F1-Score']:.4f} (95% CI: {ci['F1_ci_low']:.4f}-{ci['F1_ci_high']:.4f})")
    print(f"  AUC={r['ROC-AUC']:.4f} | Acc={r['Accuracy']:.4f}")
    del bio_mod, bio_tok
    torch.cuda.empty_cache()

except Exception as e:
    print(f"BioGPT failed: {e}")
    all_results['BioGPT Zero-Shot'] = {'Precision': 0, 'Recall': 0, 'F1-Score': 0,
                                        'Accuracy': 0, 'ROC-AUC': 0}
    all_preds['BioGPT Zero-Shot'] = np.zeros(len(y_true_all))
    all_probs['BioGPT Zero-Shot'] = np.zeros(len(y_true_all))
    all_ci['BioGPT Zero-Shot']    = {}

# ---- Results table -----------------------------------------------------------
print("\n" + "-" * 60)
print("Zero-Shot LLM Results Summary")
print("-" * 60)

results_df = pd.DataFrame(all_results).T.round(4)
results_df = results_df[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']]
print(results_df.to_string())
results_df.to_csv(os.path.join(OUT_ZEROSHOT, 'results.csv'))
print(f"\nSaved results to {os.path.join(OUT_ZEROSHOT, 'results.csv')}")
# Save raw predictions for combined CM in step 05
pd.DataFrame(all_preds, index=np.arange(len(y_true_all))).to_csv(
    os.path.join(OUT_ZEROSHOT, 'all_predictions.csv'))
pd.DataFrame(all_probs, index=np.arange(len(y_true_all))).to_csv(
    os.path.join(OUT_ZEROSHOT, 'all_probabilities.csv'))

print("\nBootstrap 95% Confidence Intervals (F1):")
for model_name, ci in all_ci.items():
    if ci:
        print(f"  {model_name:<22}: {ci['F1_mean']:.4f} "
              f"[{ci['F1_ci_low']:.4f} - {ci['F1_ci_high']:.4f}]")

# ---- Per-class breakdown -----------------------------------------------------
print("\n" + "-" * 60)
print("Per-class breakdown (Precision / Recall / F1)")
print("-" * 60)
for model_name in all_results:
    y_pred = all_preds[model_name]
    if y_pred.sum() == 0:
        continue
    print(f"\n{model_name}:")
    report = classification_report(
        y_true_all, y_pred,
        target_names=['No Biomarker', 'Has Biomarker'],
        digits=4, zero_division=0,
    )
    for line in report.split('\n'):
        print(f"  {line}")

    cm = confusion_matrix(y_true_all, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensitivity (recall pos): {tp/(tp+fn+1e-9):.4f}")
    print(f"  Specificity (recall neg): {tn/(tn+fp+1e-9):.4f}")

# ---- Comparison to baselines -------------------------------------------------
print("\n" + "-" * 60)
print("Comparison to baselines")
print("-" * 60)
print(f"  {'Model':<25} {'F1':>7}  {'AUC':>7}  {'Acc':>7}")
print(f"  {'-'*50}")
print(f"  {'Majority Class':<25} {baseline_majority['F1-Score']:>7.4f}  "
      f"{baseline_majority['ROC-AUC']:>7.4f}  {baseline_majority['Accuracy']:>7.4f}")
print(f"  {'Keyword Silver':<25} {baseline_keyword['F1-Score']:>7.4f}  "
      f"{baseline_keyword['ROC-AUC']:>7.4f}  {baseline_keyword['Accuracy']:>7.4f}")
for model_name, r in all_results.items():
    print(f"  {model_name:<25} {r['F1-Score']:>7.4f}  "
          f"{r['ROC-AUC']:>7.4f}  {r['Accuracy']:>7.4f}")

# ---- Figures -----------------------------------------------------------------
colors    = ['#7F77DD', '#1D9E75', '#D85A30']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Zero-Shot / Few-Shot LLM Evaluation',
             fontsize=14, fontweight='bold')

model_names = list(all_results.keys())

# F1 bar with 95% CI error bars.
f1_vals  = [all_results[m]['F1-Score'] for m in model_names]
f1_lower = [all_results[m]['F1-Score'] - all_ci[m].get('F1_ci_low', all_results[m]['F1-Score'])
            for m in model_names]
f1_upper = [all_ci[m].get('F1_ci_high', all_results[m]['F1-Score']) - all_results[m]['F1-Score']
            for m in model_names]
bars = axes[0].bar(range(len(model_names)), f1_vals, color=colors[:len(model_names)],
                   alpha=0.85, yerr=[f1_lower, f1_upper], capsize=6)
axes[0].axhline(baseline_majority['F1-Score'], color='gray', ls='--', lw=1.2,
                label=f"Majority baseline ({baseline_majority['F1-Score']:.3f})")
axes[0].axhline(baseline_keyword['F1-Score'], color='brown', ls=':', lw=1.2,
                label=f"Keyword baseline ({baseline_keyword['F1-Score']:.3f})")
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[0].set_ylabel('Weighted F1-Score')
axes[0].set_title('F1 with 95% Bootstrap CI', fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f'{val:.3f}', ha='center', fontsize=10)

# Multi-metric grouped bar.
metrics  = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']
x        = np.arange(len(model_names))
width    = 0.15
m_colors = ['#7F77DD', '#1D9E75', '#D85A30', '#BA7517', '#A32D2D']
for i, (metric, mc) in enumerate(zip(metrics, m_colors)):
    vals = [all_results[m][metric] for m in model_names]
    axes[1].bar(x + i * width, vals, width, label=metric, color=mc, alpha=0.82)
axes[1].set_xticks(x + width * 2)
axes[1].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[1].set_ylabel('Score')
axes[1].set_title('All Metrics Comparison', fontweight='bold')
axes[1].set_ylim(0, 1.15)
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3, axis='y')

# Confidence distribution for positive samples.
for ax_idx, (model_name, color) in enumerate(zip(model_names, colors)):
    y_prob = all_probs[model_name]
    axes[2].hist(y_prob[y_true_all == 1], bins=15, alpha=0.5, color=color,
                 label=f'{model_name} (pos)', density=True)
axes[2].set_xlabel('Predicted probability (positive class)')
axes[2].set_ylabel('Density')
axes[2].set_title('Confidence Distribution -- Positive Samples', fontweight='bold')
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_ZEROSHOT, 'fig_zeroshot_results.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved figure to {fig_path}")

# ---- Write analysis text file ------------------------------------------------
today      = datetime.date.today().strftime('%Y-%m-%d')
best_model = max(all_results, key=lambda m: all_results[m]['F1-Score'])
best_f1    = all_results[best_model]['F1-Score']
best_auc_m = max(all_results, key=lambda m: all_results[m]['ROC-AUC'])

analysis_path = os.path.join(OUT_ZEROSHOT, 'analysis.txt')
with open(analysis_path, 'w') as f:
    f.write("-" * 60 + "\n")
    f.write("ZERO-SHOT LLM -- ANALYSIS AND FINDINGS\n")
    f.write("Biomarker Detection in Alzheimer's Clinical Trial Texts\n")
    f.write(f"Date: {today}\n")
    f.write("-" * 60 + "\n\n")
    f.write(f"Dataset: {len(texts_all)} samples, {pos_rate*100:.1f}% positive\n\n")

    f.write("-" * 60 + "\n")
    f.write("METHODOLOGY\n")
    f.write("-" * 60 + "\n")
    f.write("Approach: Zero-shot / few-shot inference (no training on dataset)\n")
    f.write("No overfitting risk -- model weights are never updated.\n\n")
    f.write("Inference strategy:\n")
    f.write("  BART:   NLI entailment score for positive label.\n")
    f.write("  Flan-T5: Seq2Seq generation with few-shot examples in prompt.\n")
    f.write("  BioGPT: Perplexity scoring -- positive hypothesis vs negative hypothesis.\n\n")
    f.write("Tuning (prompt/label phrasing only):\n")
    f.write("  Phase 1: 3 variants on 20% holdout (seed=99)\n")
    f.write("  Phase 2: Best variant on full dataset\n\n")
    f.write("Flan-T5 prompt selection result:\n")
    if 'flan_best_prompt_name' in dir() and flan_best_prompt_name is not None:
        f.write(f"  Selected variant: {flan_best_prompt_name}\n")
        f.write(f"  Holdout F1 (seed=99): {flan_best_f1:.4f}\n")
        f.write(f"  Prompt template:\n")
        for line in flan_best_prompt.split('\n'):
            f.write(f"    {line}\n")
    else:
        f.write("  Flan-T5 did not run (model load error).\n")
    f.write("\n")
    f.write("Uncertainty: Bootstrap CI (1000 resamples, 95%)\n\n")

    f.write("-" * 60 + "\n")
    f.write("RESULTS TABLE\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Acc':>7} {'AUC':>7}\n")
    f.write("-" * 60 + "\n")
    for model_name in results_df.index:
        r = results_df.loc[model_name]
        f.write(f"{model_name:<22} {r['Precision']:>9.4f} {r['Recall']:>7.4f} "
                f"{r['F1-Score']:>7.4f} {r['Accuracy']:>7.4f} {r['ROC-AUC']:>7.4f}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("BOOTSTRAP 95% CI\n")
    f.write("-" * 60 + "\n")
    for model_name, ci in all_ci.items():
        if ci:
            f.write(f"  {model_name:<22}: F1={ci['F1_mean']:.4f} "
                    f"[{ci['F1_ci_low']:.4f} - {ci['F1_ci_high']:.4f}]\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("-" * 60 + "\n")
    f.write(f"\n1. Best F1:  {best_model} ({best_f1:.4f})\n")
    f.write(f"2. Best AUC: {best_auc_m} ({all_results[best_auc_m]['ROC-AUC']:.4f})\n\n")
    f.write("3. BART uses entailment which is well-suited to binary classification.\n")
    f.write("   Flan-T5 benefits from few-shot examples in the prompt.\n")
    f.write("   BioGPT uses domain pretraining, but perplexity scoring is indirect.\n")
    f.write("\n4. All three models are compared against two baselines:\n")
    f.write(f"   - Majority class: F1={baseline_majority['F1-Score']:.4f}\n")
    f.write(f"   - Keyword silver: F1={baseline_keyword['F1-Score']:.4f}\n")

print(f"Saved analysis to {analysis_path}")

# =============================================================================
# Write metadata.json for the demo app
# =============================================================================
# Reads CV scores from steps 02 and 03 output CSVs, combines with zero-shot
# scores already in memory, and writes a single JSON the demo reads on startup.
print("\n" + "-" * 60)
print("Writing models/metadata.json")
print("-" * 60)

os.makedirs(os.path.dirname(MODELS_META), exist_ok=True)

# -- Classical ML scores from step 02 --
cl_results_path = os.path.join(OUT_CLASSICAL, 'results.csv')
cl_models, cl_cv_f1, cl_cv_auc = [], {}, {}
cl_ens_cv_f1, cl_ens_cv_auc = 0.0, 0.0

if os.path.exists(cl_results_path):
    cl_df = pd.read_csv(cl_results_path, index_col=0)
    for name in cl_df.index:
        if name == 'Classical Ensemble':
            cl_ens_cv_f1  = float(cl_df.loc[name, 'F1-Score'])
            cl_ens_cv_auc = float(cl_df.loc[name, 'ROC-AUC'])
        else:
            cl_models.append(name)
            cl_cv_f1[name]  = float(cl_df.loc[name, 'F1-Score'])
            cl_cv_auc[name] = float(cl_df.loc[name, 'ROC-AUC'])
else:
    print("  WARNING: classical_ml/results.csv not found -- run step 02 first")

# -- BERT scores from step 03 --
bert_results_path = os.path.join(OUT_BERT, 'full_results.csv')
bert_simple_path  = os.path.join(OUT_BERT, 'results.csv')
bert_models, bert_cv_f1, bert_cv_auc = [], {}, {}
bert_subfolders = {
    'BERT-base':        'bert_base',
    'BioBERT':          'biobert',
    'PubMedBERT':       'pubmedbert',
    'Bio+ClinicalBERT': 'bio_clinical_bert',
    'BioLinkBERT':      'biolinkbert',
}
bert_ens_cv_f1, bert_ens_cv_auc = 0.0, 0.0

if os.path.exists(bert_results_path):
    bert_full = pd.read_csv(bert_results_path, index_col=0)
    for name in bert_subfolders:
        if name in bert_full.index:
            bert_models.append(name)
            bert_cv_f1[name]  = float(bert_full.loc[name, 'F1-Score'])
            bert_cv_auc[name] = float(bert_full.loc[name, 'ROC-AUC'])
else:
    print("  WARNING: bert_models/full_results.csv not found -- run step 03 first")

if os.path.exists(bert_simple_path):
    bert_simple = pd.read_csv(bert_simple_path, index_col=0)
    if 'BERT Ensemble' in bert_simple.index:
        bert_ens_cv_f1  = float(bert_simple.loc['BERT Ensemble', 'F1-Score'])
        bert_ens_cv_auc = float(bert_simple.loc['BERT Ensemble', 'ROC-AUC'])

# -- Zero-shot scores from memory --
zs_scores = {}
for model_name, r in all_results.items():
    zs_scores[model_name] = {
        'f1':  float(r['F1-Score']),
        'auc': float(r['ROC-AUC']),
    }

metadata = {
    'classical_ml': {
        'models':          cl_models,
        'cv_f1':           cl_cv_f1,
        'cv_auc':          cl_cv_auc,
        'ensemble_cv_f1':  cl_ens_cv_f1,
        'ensemble_cv_auc': cl_ens_cv_auc,
    },
    'bert': {
        'models':           bert_models,
        'subfolders':       {k: v for k, v in bert_subfolders.items() if k in bert_models},
        'cv_f1':            bert_cv_f1,
        'cv_auc':           bert_cv_auc,
        'ensemble_cv_f1':   bert_ens_cv_f1,
        'ensemble_cv_auc':  bert_ens_cv_auc,
        'ensemble_members': BERT_ENSEMBLE_MEMBERS,
    },
    'zeroshot': {
        'models': list(zs_scores.keys()),
        'scores': zs_scores,
    },
    'dataset_size':  int(len(y_true_all)),
    'positive_rate': float(y_true_all.mean()),
}

with open(MODELS_META, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {MODELS_META}")

# =============================================================================
# Confusion Matrices — Zero-Shot LLMs (full dataset, no leakage)
# =============================================================================
print("\n" + "-" * 60)
print("Confusion Matrices — Zero-Shot LLMs")
print("-" * 60)

import seaborn as sns
from sklearn.metrics import confusion_matrix as _cm2

_zs_model_names = list(all_results.keys())
_n_zs = len(_zs_model_names)
_ncols_zs = min(3, _n_zs)
_nrows_zs = (_n_zs + _ncols_zs - 1) // _ncols_zs
_fig_zs_cm, _axes_zs_cm = plt.subplots(_nrows_zs, _ncols_zs,
                                         figsize=(_ncols_zs * 4, _nrows_zs * 3.8))
if _n_zs == 1:
    _axes_zs_cm = np.array([[_axes_zs_cm]])
elif _nrows_zs == 1:
    _axes_zs_cm = _axes_zs_cm.reshape(1, -1)
_axes_zs_flat = _axes_zs_cm.flatten()

for _i_zs, _mname_zs in enumerate(_zs_model_names):
    _ax_zs = _axes_zs_flat[_i_zs]
    _preds_zs = all_preds[_mname_zs].astype(int)
    _cm_zs = _cm2(y_true_all, _preds_zs)
    _tn_zs, _fp_zs, _fn_zs, _tp_zs = _cm_zs.ravel()
    _f1_zs = f1_score(y_true_all, _preds_zs, average='weighted', zero_division=0)
    sns.heatmap(_cm_zs, annot=False, ax=_ax_zs, cbar=False,
                cmap=sns.light_palette('#C44E52', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for _r in range(2):
        for _c in range(2):
            _val = _cm_zs[_r, _c]
            _lbl = [['TN', 'FP'], ['FN', 'TP']][_r][_c]
            _ax_zs.text(_c + 0.5, _r + 0.5, f'{_lbl}\n{_val}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if _val > _cm_zs.max() * 0.5 else '#333333')
    _ax_zs.set_title(f'{_mname_zs}\nF1={_f1_zs:.4f}', fontsize=9, fontweight='bold')
    _ax_zs.set_xlabel('Predicted', fontsize=8)
    _ax_zs.set_ylabel('Actual', fontsize=8)
    _ax_zs.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    _ax_zs.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

for _j_zs in range(_i_zs + 1, len(_axes_zs_flat)):
    _axes_zs_flat[_j_zs].set_visible(False)
_fig_zs_cm.suptitle(f'Confusion Matrices — Zero-Shot LLMs (full dataset, N={len(y_true_all)})',
                    fontsize=13, fontweight='bold')
plt.tight_layout()
_zs_cm_path = os.path.join(OUT_ZEROSHOT, 'fig_cm.png')
plt.savefig(_zs_cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved zero-shot CM figure to {_zs_cm_path}")

# =============================================================================
# Test Set Evaluation — Zero-Shot LLMs (filter all_preds to test indices)
# =============================================================================
print("\n" + "-" * 60)
print("Test Set Evaluation — Zero-Shot LLMs (20% held-out)")
print("-" * 60)

_y_test_zs = y_test_global
_n_test_zs = len(_y_test_zs)
print(f"Test set: {_n_test_zs} samples | {_y_test_zs.sum()} positive / {(_y_test_zs==0).sum()} negative")

_n_zs2 = len(_zs_model_names)
_ncols_zs2 = min(3, _n_zs2)
_nrows_zs2 = (_n_zs2 + _ncols_zs2 - 1) // _ncols_zs2
_fig_zs_te = plt.figure(figsize=(_ncols_zs2 * 4.5, _nrows_zs2 * 3.8 + 4.5))
_cm_zs_axes = [_fig_zs_te.add_subplot(_nrows_zs2 + 1, _ncols_zs2, i + 1)
               for i in range(_n_zs2)]
_roc_zs_ax = _fig_zs_te.add_subplot(_nrows_zs2 + 1, 1, _nrows_zs2 + 1)

_zs_colors = ['#7F77DD', '#1D9E75', '#D85A30']
for _i_zs2, _mname_zs2 in enumerate(_zs_model_names):
    _preds_zs2 = all_preds[_mname_zs2][global_test_idx].astype(int)
    _probs_zs2 = all_probs[_mname_zs2][global_test_idx]
    _ax_zs2 = _cm_zs_axes[_i_zs2]
    _cm_zs2 = _cm2(_y_test_zs, _preds_zs2)
    _tn_zs2, _fp_zs2, _fn_zs2, _tp_zs2 = _cm_zs2.ravel()
    _f1_zs2 = f1_score(_y_test_zs, _preds_zs2, average='weighted', zero_division=0)
    print(f"  {_mname_zs2:<22}: test F1={_f1_zs2:.4f}")
    sns.heatmap(_cm_zs2, annot=False, ax=_ax_zs2, cbar=False,
                cmap=sns.light_palette('#C44E52', as_cmap=True),
                linewidths=0.5, linecolor='white')
    for _r in range(2):
        for _c in range(2):
            _val = _cm_zs2[_r, _c]
            _lbl = [['TN', 'FP'], ['FN', 'TP']][_r][_c]
            _ax_zs2.text(_c + 0.5, _r + 0.5, f'{_lbl}\n{_val}', ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         color='white' if _val > _cm_zs2.max() * 0.5 else '#333333')
    _ax_zs2.set_title(f'{_mname_zs2}\nF1={_f1_zs2:.4f}', fontsize=9, fontweight='bold')
    _ax_zs2.set_xlabel('Predicted', fontsize=8)
    _ax_zs2.set_ylabel('Actual', fontsize=8)
    _ax_zs2.set_xticklabels(['No BM', 'Has BM'], fontsize=8)
    _ax_zs2.set_yticklabels(['No BM', 'Has BM'], fontsize=8, rotation=0)

    if len(np.unique(_probs_zs2)) > 1:
        _fpr_zs2, _tpr_zs2, _ = roc_curve(_y_test_zs, _probs_zs2)
        _auc_zs2 = roc_auc_score(_y_test_zs, _probs_zs2)
        _roc_zs_ax.plot(_fpr_zs2, _tpr_zs2,
                        color=_zs_colors[_i_zs2 % len(_zs_colors)], lw=2,
                        label=f'{_mname_zs2} (AUC={_auc_zs2:.4f})')

_roc_zs_ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
_roc_zs_ax.set_xlabel('False Positive Rate', fontsize=11)
_roc_zs_ax.set_ylabel('True Positive Rate', fontsize=11)
_roc_zs_ax.set_title('ROC Curves — Zero-Shot Test Set', fontsize=12, fontweight='bold')
_roc_zs_ax.legend(fontsize=8, loc='lower right')
_roc_zs_ax.grid(True, alpha=0.3)

_fig_zs_te.suptitle(f'Test Set Evaluation — Zero-Shot LLMs (20% held-out, N={_n_test_zs})',
                    fontsize=14, fontweight='bold')
plt.tight_layout()
_zs_te_path = os.path.join(OUT_ZEROSHOT, 'fig_test_eval.png')
plt.savefig(_zs_te_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved zero-shot test evaluation figure to {_zs_te_path}")

print("\nStep 4 complete.")
