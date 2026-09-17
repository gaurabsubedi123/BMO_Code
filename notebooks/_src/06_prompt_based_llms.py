#%% md
# 06 — Prompt-based models (no task-specific training)

Five models used **without fine-tuning**, to measure what out-of-the-box
prompting achieves on this task.

| Short name | Checkpoint | Mechanism | Status |
|---|---|---|---|
| BART zero-shot | `facebook/bart-large-mnli` | NLI entailment scoring | existing |
| Flan-T5 few-shot | `google/flan-t5-base` | seq2seq generation | existing |
| BioGPT zero-shot | `microsoft/biogpt` | perplexity comparison | existing |
| **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | instruct chat, Yes/No logits | **new** |
| **OpenBioLLM-8B** | `aaditya/Llama3-OpenBioLLM-8B` | instruct chat, Yes/No logits | **new** |

## Why this tier needed the additions more than the encoder tier did

The three original models date from 2020–2022. BioGPT scored AUC 0.32 with
**zero true positives** — it predicted the negative class for every record.
Reporting "prompt-based models underperform" on that evidence invites the
obvious objection that the models tested were three generations old, and a
reviewer would be right to raise it.

Qwen2.5-7B-Instruct and OpenBioLLM-8B are current instruction-tuned models —
one general, one biomedical (Llama-3-based, tuned on medical corpora). If they
still lose to a BioWordVec support vector machine, that is a far stronger
result than the paper currently claims. If they win, that is a finding worth
reporting on its own terms. Either outcome improves the manuscript; the current
evidence supports neither.

## Protocol

Prompt selection is tuning, so it obeys the same rule as everything else:
candidate prompts are scored on the **inner holdout of the training set**
(`GroupShuffleSplit`, seed 99), and only the winning prompt is applied to the
test set. Because these models have no training folds, uncertainty is reported
as a 1,000-resample percentile bootstrap CI rather than fold standard deviation.

**Scoring instruct models.** Rather than parsing generated text — which is
brittle and throws away confidence information — we read the logits for the
first generated token and compare the mass assigned to "Yes" against "No".
That yields both a hard prediction and a continuous score, so AUC and
calibration are defined for these models exactly as they are for the others.
#%% code
import os, sys, json, time, gc, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, pipeline)
from sklearn.model_selection import GroupShuffleSplit

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print("device:", DEVICE, "| dtype:", DTYPE)

df, train_idx, test_idx = C.load_data()
texts_all  = df["clean_freeform"].fillna("").values
labels_all = df["has_biomarker"].values
groups_all = df["NCTId"].values
texts, labels, groups = texts_all[train_idx], labels_all[train_idx], groups_all[train_idx]
texts_te, labels_te   = texts_all[test_idx],  labels_all[test_idx]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED_TUNE)
_, tune_va = next(gss.split(texts, labels, groups=groups))
tune_texts, tune_labels = texts[tune_va], labels[tune_va]
print(f"prompt-selection holdout: {len(tune_texts)} records "
      f"({tune_labels.mean()*100:.1f}% positive) | test: {len(texts_te)}")
#%% md
## Task definition shared by every prompt

All five models are asked the same underlying question, phrased in whatever
form each architecture expects. Keeping the task statement fixed is what makes
the comparison about the model rather than about prompt engineering effort.
#%% code
TASK = ("A clinical trial outcome measure is a biomarker outcome if it measures a "
        "biological indicator of disease state or progression — for example amyloid "
        "PET, tau PET, MRI volumetrics, CSF or plasma p-tau, amyloid-beta, "
        "neurofilament light, or a genetic marker. It is not a biomarker outcome if "
        "it measures cognition, function, behaviour, quality of life, safety, or "
        "adverse events.")

PROMPT_VARIANTS = {
    "direct":   "{task}\n\nOutcome measure: \"{text}\"\n\nIs this a biomarker outcome? Answer Yes or No.",
    "role":     ("You are an expert annotator for an Alzheimer's disease clinical trial registry.\n\n"
                 "{task}\n\nOutcome measure: \"{text}\"\n\nAnswer Yes or No only."),
    "question": ('Question: Is "{text}" a measurement of a biological biomarker '
                 "(such as PET, MRI, CSF or blood protein levels)? Answer yes or no."),
    "fewshot":  ("{task}\n\n"
                 "Outcome measure: \"Change from baseline in amyloid PET SUVR\"\nAnswer: Yes\n"
                 "Outcome measure: \"Change from baseline in ADAS-Cog 13 score\"\nAnswer: No\n"
                 "Outcome measure: \"Plasma phosphorylated tau 217 concentration\"\nAnswer: Yes\n"
                 "Outcome measure: \"Incidence of treatment-emergent adverse events\"\nAnswer: No\n"
                 "Outcome measure: \"{text}\"\nAnswer:"),
}
for k, v in PROMPT_VARIANTS.items():
    print(f"--- {k} ---\n{v.format(task='[task definition]', text='[outcome text]')}\n")
#%% md
## Instruct models — Yes/No logit scoring

`score_instruct` returns P(Yes) for each record by softmaxing the first-token
logits restricted to the "Yes" and "No" token ids. Restricting to two tokens
matters: the raw softmax over a 150k vocabulary would put most mass on neither
option, and the resulting probabilities would be uninterpretable as confidence.
#%% code
def yes_no_ids(tok):
    """Token ids for Yes/No in the positions the model will actually emit them."""
    def first(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    yes = {first(v) for v in ("Yes", " Yes", "yes", " yes")} - {None}
    no  = {first(v) for v in ("No", " No", "no", " no")} - {None}
    return sorted(yes), sorted(no)

# Not every published checkpoint ships a chat template. OpenBioLLM-8B does not,
# and apply_chat_template then raises rather than falling back. Since it is a
# Llama-3 derivative, applying Llama-3's own template is the faithful choice --
# it is the format the model was instruction-tuned in.
LLAMA3_TEMPLATE = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   "{content}<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n")

def build_prompt(tok, content):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True)
    return LLAMA3_TEMPLATE.format(content=content)

@torch.no_grad()
def _score_batch(model, tok, chunk, variant, yes_ids, no_ids):
    prompts = [build_prompt(tok, PROMPT_VARIANTS[variant].format(task=TASK, text=str(t)))
               for t in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=768).to(model.device)
    logits = model(**enc).logits[:, -1, :].float()
    y = torch.logsumexp(logits[:, yes_ids], dim=-1)
    n = torch.logsumexp(logits[:, no_ids],  dim=-1)
    return torch.sigmoid(y - n).cpu().tolist()          # P(Yes) vs P(No) only

def score_instruct(model, tok, texts, variant, batch_size=8):
    """P(Yes) per record from first-token logits, via the model's chat template.

    This GPU is shared with other services, so the batch size is modest and the
    loop halves it on CUDA OOM rather than aborting the run. A crash here after
    an hour of encoder training would be an expensive way to learn that another
    process took the memory.
    """
    yes_ids, no_ids = yes_no_ids(tok)
    probs, i, bs = [], 0, batch_size
    while i < len(texts):
        try:
            probs.extend(_score_batch(model, tok, texts[i:i + bs], variant,
                                      yes_ids, no_ids))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    CUDA OOM -> reducing batch size to {bs}")
    return np.array(probs)

def free(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Decision threshold
#
# The default 0.5 cut is arbitrary for a prompted model. Qwen2.5-7B ranks this
# task almost perfectly (AUC ~0.999) yet loses badly at 0.5 because its
# probabilities are shifted, while OpenBioLLM-8B fails in the opposite
# direction. Reporting only the 0.5 numbers would attribute a *calibration*
# failure to *capability*, which is a different and weaker claim.
#
# So each model also gets a threshold chosen to maximise biomarker-class F1 --
# selected on the TRAINING HOLDOUT and then applied unchanged to the test set.
# This is the same discipline the supervised models get (they learn a decision
# boundary from training data); it gives prompting its fairest shot without
# ever letting the test set influence a choice.
# ---------------------------------------------------------------------------
from sklearn.metrics import f1_score as _f1

def tune_threshold(y_holdout, p_holdout, grid=None):
    """Threshold maximising biomarker-class F1 on the holdout. Never sees test."""
    grid = np.linspace(0.01, 0.99, 197) if grid is None else grid
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = _f1(y_holdout, (p_holdout >= t).astype(int), pos_label=1, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s

def record(name, y_te, p_te, y_ho, p_ho, extra=None):
    """Emit both the default-0.5 row and the holdout-tuned row for one model."""
    rows, preds = [], {}
    base = dict(extra or {})

    y_hat = (p_te >= 0.5).astype(int)
    lo, hi = C.bootstrap_ci(y_te, y_hat)
    rows.append(C.evaluate(y_te, y_hat, p_te, name=name, tier="Prompt-based",
                           extra={**base, "threshold": 0.5, "threshold_source": "default",
                                  "ci_low": lo, "ci_high": hi}))
    preds[name] = (y_hat, p_te)

    t, ho_f1 = tune_threshold(y_ho, p_ho)
    y_hat_t = (p_te >= t).astype(int)
    lo, hi = C.bootstrap_ci(y_te, y_hat_t)
    tname = f"{name} (tuned threshold)"
    rows.append(C.evaluate(y_te, y_hat_t, p_te, name=tname, tier="Prompt-based",
                           extra={**base, "threshold": round(t, 4),
                                  "threshold_source": "holdout",
                                  "holdout_biomarker_f1": round(ho_f1, 4),
                                  "ci_low": lo, "ci_high": hi}))
    preds[tname] = (y_hat_t, p_te)
    print(f"  {name}: default 0.5 -> {rows[0]['errors']} errors | "
          f"tuned {t:.3f} -> {rows[1]['errors']} errors "
          f"(threshold chosen on holdout, holdout biomarker F1 {ho_f1:.4f})")
    return rows, preds
#%% code
INSTRUCT_MODELS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "OpenBioLLM-8B":       "aaditya/Llama3-OpenBioLLM-8B",
}

instruct_rows, instruct_preds, prompt_search = [], {}, []
holdout_probs = {}
failed_models = []

def free_gpu_gb():
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b / 1e9

def wait_for_vram(need_gb, timeout_h=14, poll_s=60, sustain_polls=3):
    """Wait for a *sustained* window of free VRAM before loading a large model.

    Two other workloads share these cards: an ollama service that holds ~23 GB
    while a model is resident and releases it after a few minutes idle, and an
    evening speech-transcription job.

    `sustain_polls` is the important parameter. Transcription releases memory
    briefly between audio files, so a single lucky poll is not evidence the GPU
    is actually free -- grabbing 18 GB in that gap is exactly how this notebook
    would break someone else's job. Requiring the headroom to hold across
    several consecutive polls means we only start when the other job has
    genuinely finished, not when it happens to be between files.

    The timeout is hours rather than minutes so this stage can simply wait out
    the evening transcription and run afterwards, unattended.
    """
    deadline = time.time() + timeout_h * 3600
    streak, waited, announced = 0, 0, False
    while time.time() < deadline:
        free = free_gpu_gb()
        if free >= need_gb:
            streak += 1
            if streak >= sustain_polls:
                print(f"  proceeding: {free:.1f} GB free, sustained across "
                      f"{sustain_polls} consecutive checks")
                return True
        else:
            if streak:
                print(f"    headroom disappeared after {streak} check(s) "
                      f"({free:.1f} GB free) — another job is still active")
            streak = 0
            if not announced:
                print(f"  waiting for {need_gb:.0f} GB of sustained free VRAM "
                      f"(currently {free:.1f} GB). Will wait up to {timeout_h} h "
                      f"so this can run after the evening transcription.")
                announced = True
        time.sleep(poll_s); waited += poll_s
        if waited % 1800 == 0:
            print(f"    still waiting ({waited//3600} h {(waited%3600)//60} min, "
                  f"{free_gpu_gb():.1f} GB free)")
    print(f"  TIMED OUT after {timeout_h} h: {free_gpu_gb():.1f} GB free")
    return False

# bf16 weights plus activation headroom, per model.
VRAM_NEEDED_GB = {"Qwen2.5-7B-Instruct": 18.0, "OpenBioLLM-8B": 19.0}

# ---------------------------------------------------------------------------
# Evening curfew.
#
# A speech-transcription job (WhisperX large + pyannote diarization, roughly
# 8-12 GB of VRAM) runs on this node in the evening. This notebook is the only
# stage of the study that holds enough VRAM to starve it: a 7-8B model in bf16
# takes ~18 GB, which alongside the resident ollama service would leave too
# little for transcription to load.
#
# Every other stage is either small (encoders, ~4-8 GB) or CPU-only (the
# classical tier), so no other part of the pipeline needs this guard.
#
# Rather than risk breaking someone else's job, this stage yields after the
# curfew: it waits (for up to 14 hours) until the card is genuinely free --
# meaning the headroom holds across several consecutive checks, not just one
# lucky moment between audio files -- and then continues automatically.
# Nothing else in the study depends on when this stage runs.
# ---------------------------------------------------------------------------
import datetime as _dt
CURFEW_HOUR = int(os.environ.get("BMO_LLM_CURFEW_HOUR", "17"))   # 17:00 local

def past_curfew():
    return _dt.datetime.now().hour >= CURFEW_HOUR

print(f"local time now: {_dt.datetime.now():%H:%M} | "
      f"LLM curfew at {CURFEW_HOUR:02d}:00 "
      f"({'PAST CURFEW - will skip' if past_curfew() else 'clear to run'})")

for name, ckpt in INSTRUCT_MODELS.items():
    try:
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        need = VRAM_NEEDED_GB.get(name, 18.0)
        print(f"  free GPU memory: {free_gpu_gb():.1f} GB | need ~{need:.0f} GB")
        if past_curfew():
            print(f"  past the {CURFEW_HOUR:02d}:00 curfew — the evening transcription "
                  f"job has priority on this GPU.\n"
                  f"  This stage will WAIT for the transcription to finish and the "
                  f"card to be genuinely free, then continue automatically. "
                  f"No action needed.")
        if not wait_for_vram(need):
            print(f"  SKIPPING {name}: insufficient VRAM after waiting. Re-run this "
                  f"notebook when the GPU frees up; the rest of the study is "
                  f"unaffected and this model is simply absent from the tier.")
            continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(ckpt)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"          # required for last-position logit reads
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=DTYPE, device_map={"": 0}).eval()
        print(f"  loaded in {time.time()-t0:.0f}s")

        # --- prompt selection on the inner holdout only -------------------------
        best_variant, best_f1, best_holdout_p = None, -1.0, None
        for variant in PROMPT_VARIANTS:
            p = score_instruct(model, tok, tune_texts, variant)
            r = C.evaluate(tune_labels, (p >= 0.5).astype(int), p,
                           name=f"{name}/{variant}", tier="Prompt-based")
            prompt_search.append({"Model": name, "variant": variant,
                                  "holdout_weighted_f1": r["weighted_f1"],
                                  "holdout_biomarker_f1": r["biomarker_f1"],
                                  "holdout_auc": r.get("auc", np.nan)})
            print(f"  {variant:9s} holdout weighted F1 {r['weighted_f1']:.4f} "
                  f"| biomarker F1 {r['biomarker_f1']:.4f} | AUC {r.get('auc', float('nan')):.4f}")
            if r["weighted_f1"] > best_f1:
                best_variant, best_f1, best_holdout_p = variant, r["weighted_f1"], p
        print(f"  -> selected prompt: {best_variant}")

        # --- single test-set evaluation with the winning prompt -----------------
        p_te = score_instruct(model, tok, texts_te, best_variant)
        rows, preds = record(name, labels_te, p_te, tune_labels, best_holdout_p,
                             extra={"prompt_variant": best_variant,
                                    "n_prompts_tried": len(PROMPT_VARIANTS),
                                    "load_seconds": round(time.time() - t0, 1)})
        instruct_rows.extend(rows); instruct_preds.update(preds)
        holdout_probs[name] = best_holdout_p
        for rr in rows:
            C.plot_confusion(labels_te, (p_te >= rr["threshold"]).astype(int),
                             f"{rr['Model']}: test set",
                             C.FIGS / ("cm_test_" + rr["Model"].replace(".", "_")
                                       .replace("/", "_").replace(" ", "_")
                                       .replace("(", "").replace(")", "") + ".png"))
        free(model, tok)

    except Exception as exc:
        # Isolate per model: a checkpoint that cannot be loaded or prompted
        # should cost us that one row, not the whole tier. The failure is
        # recorded so it appears in the manuscript rather than vanishing.
        print(f'  FAILED {name}: {type(exc).__name__}: {exc}')
        failed_models.append({'Model': name, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        for _v in ('model', 'tok'):
            if _v in dir():
                pass
        gc.collect(); torch.cuda.empty_cache()
# saved again at the end of the tier, once every model has contributed
C.save_table(pd.DataFrame(prompt_search), C.OUT_LLM / "prompt_selection_instruct.csv")
#%% md
## Legacy prompt-based models

Re-run unchanged so the new results sit on the same corpus, split and metric
definitions as everything else. Their purpose in the paper is now different:
they are no longer the evidence for the claim about prompting, they are the
historical baseline the modern models are measured against.
#%% code
legacy_rows, legacy_preds = [], {}

# ---- BART: NLI entailment ------------------------------------------------
print("BART-large-MNLI (zero-shot NLI)")
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
              device=0 if DEVICE == "cuda" else -1)

# The legacy models originally got ONE hand-written prompt each while the two
# instruct models got a three-variant search. That asymmetry favours the new
# models over the baselines they are compared against, and it is not small: a
# diagnostic on Flan-T5 moved AUC from 0.45 to 0.80 by changing the prompt
# alone. Every prompt-based system now gets the same search on the same holdout.
BART_HYPS = {
    "biomarker-vs-clinical": (["a biomarker measurement",
                               "a clinical or cognitive measurement"],
                              "This outcome measure is {}."),
    "biological-vs-symptom": (["a biological or laboratory measurement",
                               "a symptom, function or safety measurement"],
                              "This outcome measure is {}."),
    "imaging-fluid-genetic": (["an imaging, fluid or genetic biomarker",
                               "a cognitive, functional or safety outcome"],
                              "This clinical trial endpoint measures {}."),
}

def bart_scores_for(labels_pair, template):
    def f(texts):
        out_p = []
        for i in range(0, len(texts), 32):
            res = zs(list(map(str, texts[i:i+32])), labels_pair,
                     hypothesis_template=template, multi_label=False)
            res = res if isinstance(res, list) else [res]
            for o in res:
                out_p.append(dict(zip(o["labels"], o["scores"]))[labels_pair[0]])
        return np.array(out_p)
    return f

best_v, best_s, best_fn = None, -1.0, None
for vname, (lp, tmpl) in BART_HYPS.items():
    fn = bart_scores_for(lp, tmpl)
    ph = fn(tune_texts)
    r = C.evaluate(tune_labels, (ph >= 0.5).astype(int), ph, name=f"BART/{vname}")
    prompt_search.append({"Model": "BART zero-shot", "variant": vname,
                          "holdout_weighted_f1": r["weighted_f1"],
                          "holdout_biomarker_f1": r["biomarker_f1"],
                          "holdout_auc": r.get("auc", np.nan)})
    print(f"  {vname:24s} holdout weighted F1 {r['weighted_f1']:.4f} "
          f"| AUC {r.get('auc', float('nan')):.4f}")
    if r["weighted_f1"] > best_s:
        best_v, best_s, best_fn = vname, r["weighted_f1"], fn
print(f"  -> selected hypothesis set: {best_v}")

rows, preds = record("BART zero-shot", labels_te, best_fn(texts_te),
                     tune_labels, best_fn(tune_texts),
                     extra={"prompt_variant": best_v,
                            "n_prompts_tried": len(BART_HYPS)})
legacy_rows.extend(rows); legacy_preds.update(preds)
free(zs)

#%% code
# ---- Flan-T5: few-shot generation ---------------------------------------
print("Flan-T5-base (few-shot)")
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
m = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",
                                          torch_dtype=torch.float32).to(DEVICE).eval()

# Score the two candidate answers as complete decoder sequences and compare
# their log-likelihoods. Reading a single decoder position -- as an earlier
# version did -- is unreliable for T5: the first decoded token need not be the
# answer token, so the comparison can be dominated by formatting. That produced
# a below-chance MCC, a symptom of a scoring bug rather than a weak model.
@torch.no_grad()
def t5_candidate_logprob(prompts, candidate):
    lab = tok([candidate] * len(prompts), return_tensors="pt").input_ids.to(DEVICE)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(DEVICE)
    logits = m(**enc, labels=lab).logits.float().log_softmax(-1)
    tokl = logits.gather(2, lab.unsqueeze(-1)).squeeze(-1)
    mask = (lab != tok.pad_token_id).float()
    return ((tokl * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu()

def t5_scores_for(variant, yes_word, no_word):
    def f(texts):
        out_p = []
        for i in range(0, len(texts), 32):
            prompts = [PROMPT_VARIANTS[variant].format(task=TASK, text=str(t))
                       for t in texts[i:i+32]]
            ly = t5_candidate_logprob(prompts, yes_word)
            ln = t5_candidate_logprob(prompts, no_word)
            out_p.extend(torch.sigmoid(ly - ln).tolist())
        return np.array(out_p)
    return f

# Both the prompt and the answer words are searched: the diagnostic showed the
# label pair alone swings AUC from 0.29 to 0.81 on this model, so fixing one
# arbitrary choice would understate what prompting can achieve.
T5_GRID = [(v, y, n) for v in ("question", "direct", "fewshot")
           for y, n in (("Yes", "No"), ("true", "false"))]
best_v, best_s, best_fn = None, -1.0, None
for variant, yw, nw in T5_GRID:
    fn = t5_scores_for(variant, yw, nw)
    ph = fn(tune_texts)
    r = C.evaluate(tune_labels, (ph >= 0.5).astype(int), ph, name="t5")
    tag = f"{variant}/{yw}-{nw}"
    prompt_search.append({"Model": "Flan-T5", "variant": tag,
                          "holdout_weighted_f1": r["weighted_f1"],
                          "holdout_biomarker_f1": r["biomarker_f1"],
                          "holdout_auc": r.get("auc", np.nan)})
    print(f"  {tag:22s} holdout weighted F1 {r['weighted_f1']:.4f} "
          f"| AUC {r.get('auc', float('nan')):.4f}")
    if r["weighted_f1"] > best_s:
        best_v, best_s, best_fn = tag, r["weighted_f1"], fn
print(f"  -> selected: {best_v}")

rows, preds = record("Flan-T5", labels_te, best_fn(texts_te),
                     tune_labels, best_fn(tune_texts),
                     extra={"prompt_variant": best_v, "n_prompts_tried": len(T5_GRID)})
legacy_rows.extend(rows); legacy_preds.update(preds)
free(m, tok)

#%% code
# ---- BioGPT: perplexity comparison --------------------------------------
print("BioGPT (zero-shot perplexity)")
tok = AutoTokenizer.from_pretrained("microsoft/biogpt")
m = AutoModelForCausalLM.from_pretrained("microsoft/biogpt",
                                         torch_dtype=torch.float32).to(DEVICE).eval()
POS_H = "This outcome measure is a biomarker measurement."
NEG_H = "This outcome measure is a clinical or cognitive measurement."

@torch.no_grad()
def seq_logprob(context, hypothesis):
    ids_c = tok(context, return_tensors="pt", truncation=True, max_length=384).input_ids
    ids_h = tok(" " + hypothesis, return_tensors="pt",
                add_special_tokens=False).input_ids
    ids = torch.cat([ids_c, ids_h], dim=1).to(DEVICE)
    lg = m(ids).logits[:, :-1, :].float().log_softmax(-1)
    lp = lg.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp[:, -ids_h.size(1):].mean().item()

# Same treatment as every other prompt-based system: search the hypothesis
# wording on the holdout rather than fixing one arbitrary phrasing.
BIOGPT_HYPS = {
    "biomarker-vs-clinical": ("This outcome measure is a biomarker measurement.",
                              "This outcome measure is a clinical or cognitive measurement."),
    "biological-vs-symptom": ("This endpoint measures a biological marker of disease.",
                              "This endpoint measures a symptom, function or safety outcome."),
    "modality-explicit":     ("This is an imaging, cerebrospinal fluid, blood or genetic biomarker.",
                              "This is a cognitive test, functional scale or adverse event count."),
}

def biogpt_scores_for(pos_h, neg_h):
    def f(texts):
        out_p = []
        for t in texts:
            ctx = f"Outcome measure: {t}."
            d = seq_logprob(ctx, pos_h) - seq_logprob(ctx, neg_h)
            out_p.append(1 / (1 + np.exp(-d)))
        return np.array(out_p)
    return f

best_v, best_s, best_fn = None, -1.0, None
for vname, (ph_, nh_) in BIOGPT_HYPS.items():
    fn = biogpt_scores_for(ph_, nh_)
    hp = fn(tune_texts)
    r = C.evaluate(tune_labels, (hp >= 0.5).astype(int), hp, name="biogpt")
    prompt_search.append({"Model": "BioGPT zero-shot", "variant": vname,
                          "holdout_weighted_f1": r["weighted_f1"],
                          "holdout_biomarker_f1": r["biomarker_f1"],
                          "holdout_auc": r.get("auc", np.nan)})
    print(f"  {vname:24s} holdout weighted F1 {r['weighted_f1']:.4f} "
          f"| AUC {r.get('auc', float('nan')):.4f}")
    if r["weighted_f1"] > best_s:
        best_v, best_s, best_fn = vname, r["weighted_f1"], fn
print(f"  -> selected hypothesis pair: {best_v}")

rows, preds = record("BioGPT zero-shot", labels_te, best_fn(texts_te),
                     tune_labels, best_fn(tune_texts),
                     extra={"prompt_variant": best_v,
                            "n_prompts_tried": len(BIOGPT_HYPS)})
legacy_rows.extend(rows); legacy_preds.update(preds)
free(m, tok)

#%% md
## Tier summary
#%% code
llm_df = pd.DataFrame(instruct_rows + legacy_rows).sort_values("weighted_f1", ascending=False)
C.save_table(llm_df, C.OUT_LLM / "prompt_llm_test_results.csv")

all_preds = {**instruct_preds, **legacy_preds}
pred_df = pd.DataFrame({m: v[0] for m, v in all_preds.items()}); pred_df.insert(0, "y_true", labels_te)
prob_df = pd.DataFrame({m: v[1] for m, v in all_preds.items()}); prob_df.insert(0, "y_true", labels_te)
C.save_table(pred_df, C.OUT_LLM / "test_predictions.csv")
C.save_table(prob_df, C.OUT_LLM / "test_probabilities.csv")

cols = ["Model", "weighted_f1", "biomarker_f1", "auc", "TN", "FP", "FN", "TP", "errors"]
print(llm_df[cols].to_string(index=False))

# Holdout probabilities are kept so the threshold choice is auditable: a reader
# can verify the cut was selected on training data and not on the test set.
C.save_table(pd.DataFrame(prompt_search), C.OUT_LLM / "prompt_selection_all.csv")

if holdout_probs:
    hp = pd.DataFrame(holdout_probs); hp.insert(0, "y_true", tune_labels)
    C.save_table(hp, C.OUT_LLM / "holdout_probabilities.csv")

if failed_models:
    C.save_table(pd.DataFrame(failed_models), C.OUT_LLM / "failed_models.csv")
    print("\nMODELS THAT FAILED (recorded, not silently dropped):")
    for f in failed_models:
        print(f"  {f['Model']}: {f['error']}")

C.plot_ranked_bars(llm_df.to_dict("records"), "weighted_f1",
                   "Prompt-based models, test set", "Weighted F1",
                   C.FIGS / "prompt_tier_ranked.png", xlim=(0.0, 1.0))
#%% md
### Default threshold versus holdout-tuned threshold

The comparison that separates *calibration* from *capability*. A model whose
error count falls sharply once the threshold is corrected was never bad at the
task — it was bad at the arbitrary 0.5 cut. That distinction matters for the
manuscript, because "prompting cannot do this" and "prompting needs
calibration" are different claims with different implications for practice.

AUC is invariant to the threshold, so it is the column to read when judging
whether a model can rank the task at all.
#%% code
base = llm_df[~llm_df["Model"].str.contains("tuned")].set_index("Model")
tuned = llm_df[llm_df["Model"].str.contains("tuned")].copy()
tuned["base_model"] = tuned["Model"].str.replace(" (tuned threshold)", "", regex=False)
tuned = tuned.set_index("base_model")

comp = pd.DataFrame({
    "AUC":              base["auc"].round(4),
    "errors @0.5":      base["errors"],
    "errors @tuned":    tuned["errors"],
    "threshold":        tuned["threshold"].round(3),
    "weighted_f1 @0.5": base["weighted_f1"].round(4),
    "weighted_f1 @tuned": tuned["weighted_f1"].round(4),
}).sort_values("errors @tuned")
comp["errors_saved"] = comp["errors @0.5"] - comp["errors @tuned"]
print(comp.to_string())
C.save_table(comp.reset_index(), C.OUT_LLM / "threshold_comparison.csv")

print("\nInterpretation:")
for m, r in comp.iterrows():
    if r["AUC"] > 0.95 and r["errors_saved"] > 5:
        print(f"  {m}: ranks the task well (AUC {r['AUC']:.3f}) and was mainly "
              f"mis-calibrated — {int(r['errors_saved'])} errors were threshold, not capability.")
    elif r["AUC"] < 0.7:
        print(f"  {m}: AUC {r['AUC']:.3f} — cannot rank the task; no threshold rescues it.")
    else:
        print(f"  {m}: AUC {r['AUC']:.3f}, {int(r['errors_saved'])} errors recovered by tuning.")

#%% md
## What this tier now supports

Read the table above before writing the manuscript paragraph, and let it
determine the claim rather than the reverse. Three outcomes are possible and
each implies different text:

1. **Modern instruct models remain well below the supervised systems.** The
   paper's existing claim survives and becomes considerably stronger: it now
   rests on current models rather than obsolete ones. State the gap in errors,
   not only in F1.
2. **They close most of the gap but do not beat the supervised tier.** The
   claim needs softening to something like "prompting alone does not yet match
   a task-tuned classifier on this corpus", with the trend noted.
3. **They match or beat the supervised systems.** That is a genuine finding and
   the paper's framing must change to report it. It would also alter the
   practical recommendation, since prompting requires no annotated corpus.

The cell below states which case holds, so the manuscript text is written
against the result rather than against expectation.
#%% code
try:
    bert_test = pd.read_csv(C.OUT_BERT / "test_results_merged.csv")
except FileNotFoundError:
    shards = sorted(C.OUT_BERT.glob("test_results_shard*.csv"))
    bert_test = pd.concat([pd.read_csv(p) for p in shards]) if shards else pd.DataFrame()

if len(bert_test):
    best_sup = bert_test.sort_values("weighted_f1", ascending=False).iloc[0]
    best_llm = llm_df.iloc[0]
    gap_f1  = best_sup["weighted_f1"] - best_llm["weighted_f1"]
    gap_err = int(best_llm["errors"]) - int(best_sup["errors"])
    print(f"best supervised : {best_sup['Model']:24s} F1 {best_sup['weighted_f1']:.4f}  "
          f"{int(best_sup['errors'])} errors")
    print(f"best prompt-only: {best_llm['Model']:24s} F1 {best_llm['weighted_f1']:.4f}  "
          f"{int(best_llm['errors'])} errors")
    print(f"\ngap: {gap_f1:+.4f} weighted F1, {gap_err:+d} errors")
    print("CASE:", "1 — prompting remains clearly behind" if gap_f1 > 0.05 else
                   "2 — prompting closes most of the gap" if gap_f1 > 0.0 else
                   "3 — prompting MATCHES OR BEATS supervised; reframe the paper")
else:
    print("Run notebook 05 first to compare against the supervised tier.")
#%% md
## Done

**Next:** `07_test_set_evaluation.ipynb` merges every tier into one table, then
`08` runs the pairwise significance tests that decide which differences here
are real.
