"""Qwen3.8-27B on the biomarker task, under the same protocol as Qwen2.5-7B.

Everything that decides the result is taken from notebook 06 or bmo_common,
not re-typed here: the task text and the four prompt variants (read from
notebook 06's source), the trial-grouped selection holdout (seed SEED_TUNE),
Yes/No first-token scoring, threshold tuning on the holdout, and the metric and
bootstrap code. The one addition is that Qwen3.8's thinking mode is switched
off, so it answers in one step like every other prompted model in the paper.

Qwen3.8 was released after the paper's pinned environment was frozen and needs
current transformers, so run it from its own venv and never from the study's.
Output goes to a new directory; the paper's run directory is only read.

    source /mnt/subedi-research/projects/bmo_rerun_2026/env.sh   # for BMO_RUN_DIR only
    python qwen38_zeroshot.py --check                 # render prompts, check scoring (~5 min)
    python qwen38_zeroshot.py                         # full run, bf16 across both GPUs
    python qwen38_zeroshot.py --precision fp8         # official FP8 checkpoint, one GPU

Run --check first and read it. It prints the end of the rendered prompt and
what the model actually generates. If the answer does not start at the next
token, first-token scoring is measuring the wrong thing and the run is invalid.
"""
import argparse
import datetime as dt
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C                                         # noqa: E402

OUT = Path(os.environ.get("BMO_QWEN38_OUT", "/mnt/subedi-research/projects/bmo_qwen38_2026"))
CHECKPOINTS = {"bf16": "Qwen/Qwen3.8-27B", "fp8": "Qwen/Qwen3.8-27B-FP8"}
NAME = "Qwen3.8-27B"
# Weights plus headroom for short prompts. bf16 is ~56 GB of weights and has to
# be split over both L40s; fp8 is ~28 GB and fits on one.
VRAM_NEEDED_GB = {"bf16": 62.0, "fp8": 34.0}
CURFEW_HOUR = int(os.environ.get("BMO_LLM_CURFEW_HOUR", "17"))


# ---------------------------------------------------------------- inputs
def load_prompts():
    """TASK and PROMPT_VARIANTS executed from notebook 06, so the text is identical."""
    src = (REPO / "notebooks" / "_src" / "06_prompt_based_llms.py").read_text()
    start = src.index("TASK = (")
    end = src.index("\n}\n", src.index("PROMPT_VARIANTS = {")) + 3
    ns = {}
    exec(src[start:end], ns)
    return ns["TASK"], ns["PROMPT_VARIANTS"]


TASK, PROMPT_VARIANTS = load_prompts()


def load_splits():
    """Same arrays notebook 06 builds: training holdout for selection, test for scoring."""
    df, train_idx, test_idx = C.load_data()
    texts_all = df["clean_freeform"].fillna("").values
    labels_all = df["has_biomarker"].values
    groups_all = df["NCTId"].values
    texts, labels, groups = texts_all[train_idx], labels_all[train_idx], groups_all[train_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED_TUNE)
    _, tune_va = next(gss.split(texts, labels, groups=groups))
    return (texts[tune_va], labels[tune_va],
            texts_all[test_idx], labels_all[test_idx])


# ---------------------------------------------------------------- GPU etiquette
def free_gb_all():
    return sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) / 1e9


def wait_for_vram(need_gb, timeout_h=14, poll_s=60, sustain_polls=3):
    """Start only when the headroom holds across several checks (see notebook 06).

    oceanus is shared with an ollama service (~24 GB per card while resident)
    and an evening transcription job, which frees memory briefly between files.
    A single lucky poll is not evidence the cards are free.
    """
    deadline, streak = time.time() + timeout_h * 3600, 0
    while time.time() < deadline:
        if dt.datetime.now().hour >= CURFEW_HOUR:
            streak = 0
        elif free_gb_all() >= need_gb:
            streak += 1
            if streak >= sustain_polls:
                print(f"proceeding: {free_gb_all():.1f} GB free across GPUs")
                return True
        else:
            streak = 0
        time.sleep(poll_s)
    print(f"TIMED OUT after {timeout_h} h with {free_gb_all():.1f} GB free")
    return False


# ---------------------------------------------------------------- model
def load_model(ckpt, precision, revision):
    tok = AutoTokenizer.from_pretrained(ckpt, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                          # last-position logit reads
    kw = dict(revision=revision, torch_dtype=torch.bfloat16)
    # bf16 does not fit one 48 GB card, so let accelerate split it; fp8 stays on one.
    kw["device_map"] = "auto" if precision == "bf16" else {"": 0}
    # Qwen3.8 is a vision-language checkpoint. Use the text-only causal LM class
    # when the architecture registers one; otherwise load the full model and
    # feed it text only, which is what the vision encoder sees as no images.
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(ckpt, **kw)
    return tok, model.eval()


def build_prompt(tok, content):
    # enable_thinking=False is the one departure from notebook 06's build_prompt:
    # without it the template opens a reasoning block and the first generated
    # token is <think>, not the answer.
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def yes_no_ids(tok):
    """Identical to notebook 06."""
    def first(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    yes = {first(v) for v in ("Yes", " Yes", "yes", " yes")} - {None}
    no = {first(v) for v in ("No", " No", "no", " no")} - {None}
    return sorted(yes), sorted(no)


def input_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def _score_batch(model, tok, chunk, variant, yes_ids, no_ids):
    prompts = [build_prompt(tok, PROMPT_VARIANTS[variant].format(task=TASK, text=str(t)))
               for t in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=768).to(input_device(model))
    logits = model(**enc).logits[:, -1, :].float()
    y = torch.logsumexp(logits[:, yes_ids], dim=-1)
    n = torch.logsumexp(logits[:, no_ids], dim=-1)
    return torch.sigmoid(y - n).cpu().tolist()


def score(model, tok, texts, variant, batch_size=8):
    """P(Yes) per record; halves the batch on OOM as notebook 06 does."""
    yes_ids, no_ids = yes_no_ids(tok)
    probs, i, bs = [], 0, batch_size
    while i < len(texts):
        try:
            probs.extend(_score_batch(model, tok, texts[i:i + bs], variant, yes_ids, no_ids))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"    CUDA OOM -> batch size {bs}")
    return np.array(probs)


def tune_threshold(y_holdout, p_holdout):
    """Identical to notebook 06: maximise biomarker-class F1 on the holdout."""
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        s = f1_score(y_holdout, (p_holdout >= t).astype(int), pos_label=1, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


# ---------------------------------------------------------------- modes
def check(tok, model, tune_texts, tune_labels):
    """Show that the answer really is the next token. Read this before a full run."""
    yes_ids, no_ids = yes_no_ids(tok)
    ok = True
    print("Yes ids", yes_ids, [tok.decode([i]) for i in yes_ids])
    print("No ids ", no_ids, [tok.decode([i]) for i in no_ids])
    for k in range(3):
        content = PROMPT_VARIANTS["direct"].format(task=TASK, text=str(tune_texts[k]))
        p = build_prompt(tok, content)
        print(f"\n--- record {k} (label {tune_labels[k]}) --- prompt ends with:\n{p[-160:]!r}")
        enc = tok([p], return_tensors="pt").to(input_device(model))
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
            top = torch.topk(torch.softmax(logits, -1), 5)
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False)
        print("top-5 next tokens:", [(tok.decode([int(i)]), round(float(v), 4))
                                     for v, i in zip(top.values, top.indices)])
        yn = float(torch.softmax(logits, -1)[yes_ids + no_ids].sum())
        print(f"probability mass on Yes/No tokens: {yn:.3f}  (should be close to 1)")
        print("greedy continuation:", repr(tok.decode(gen[0, enc.input_ids.shape[1]:])))
        # Gate for the unattended chain: the top token must be a Yes/No id and
        # carry most of the mass, or first-token scoring is reading noise.
        ok &= int(top.indices[0]) in yes_ids + no_ids and yn >= 0.9
    print("\nCHECK", "PASSED" if ok else "FAILED")
    return ok


def full_run(tok, model, precision, ckpt, revision, splits):
    tune_texts, tune_labels, texts_te, labels_te = splits
    search, holdout = [], {}
    for variant in PROMPT_VARIANTS:
        p = score(model, tok, tune_texts, variant)
        r = C.evaluate(tune_labels, (p >= 0.5).astype(int), p, name=f"{NAME}/{variant}",
                       tier="Prompt-based")
        search.append({"Model": NAME, "variant": variant,
                       "holdout_weighted_f1": r["weighted_f1"],
                       "holdout_biomarker_f1": r["biomarker_f1"],
                       "holdout_auc": r.get("auc", np.nan)})
        holdout[variant] = p
        print(f"  {variant:9s} holdout weighted F1 {r['weighted_f1']:.4f} | "
              f"AUC {r.get('auc', float('nan')):.4f}")
    # strict '>' in list order, as notebook 06, so ties go to the earlier variant
    best = None
    for s in search:
        if best is None or s["holdout_weighted_f1"] > best["holdout_weighted_f1"]:
            best = s
    variant = best["variant"]
    print(f"  -> selected prompt: {variant}")

    p_te = score(model, tok, texts_te, variant)
    rows, preds = [], {}
    extra = {"prompt_variant": variant, "n_prompts_tried": len(PROMPT_VARIANTS),
             "precision": precision, "thinking": "disabled"}
    for label, t, src in [(NAME, 0.5, "default"), (None, None, "holdout")]:
        if src == "holdout":
            t, ho_f1 = tune_threshold(tune_labels, holdout[variant])
            label = f"{NAME} (tuned threshold)"
            extra_t = {**extra, "holdout_biomarker_f1": round(ho_f1, 4)}
        else:
            extra_t = dict(extra)
        y_hat = (p_te >= t).astype(int)
        lo, hi = C.bootstrap_ci(labels_te, y_hat)
        rows.append(C.evaluate(labels_te, y_hat, p_te, name=label, tier="Prompt-based",
                               extra={**extra_t, "threshold": round(t, 4),
                                      "threshold_source": src, "ci_low": lo, "ci_high": hi}))
        preds[label] = y_hat
        print(f"  {label}: threshold {t:.3f} -> {rows[-1]['errors']} test errors")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search).to_csv(OUT / "prompt_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame({"y_true": labels_te, **preds, "p_yes": p_te}).to_csv(
        OUT / "test_predictions.csv", index=False)
    pd.DataFrame({v: p for v, p in holdout.items()}).assign(y_true=tune_labels).to_csv(
        OUT / "holdout_probs.csv", index=False)
    from huggingface_hub import model_info
    meta = {"checkpoint": ckpt, "revision": revision or model_info(ckpt).sha,
            "precision": precision, "thinking": "disabled", "date": dt.date.today().isoformat(),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "python": platform.python_version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--revision", default=None, help="pin a Hugging Face commit for the record")
    ap.add_argument("--check", action="store_true", help="3-record sanity check only")
    ap.add_argument("--no-wait", action="store_true", help="skip the VRAM wait and curfew")
    a = ap.parse_args()

    ckpt = CHECKPOINTS[a.precision]
    splits = load_splits()
    print(f"{ckpt} | holdout {len(splits[0])} | test {len(splits[2])} | "
          f"transformers {transformers.__version__}")
    assert len(splits[2]) == 341, "test partition is not the paper's 341 records"
    if not a.no_wait and not wait_for_vram(VRAM_NEEDED_GB[a.precision]):
        sys.exit(1)
    tok, model = load_model(ckpt, a.precision, a.revision)
    try:
        if a.check:
            if not check(tok, model, splits[0], splits[1]):
                sys.exit(2)
        else:
            full_run(tok, model, a.precision, ckpt, a.revision, splits)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
