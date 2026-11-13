"""Qwen3.8-27B with thinking ON, set up like the Claude reasoning runs.

The same prompt the other reasoning models get (the "direct" variant, read from
notebook 06), every one of the 1,900 records once plus a second pass over the
341 test records, and the same first-word rule applied to the answer that
follows the reasoning block. A reasoning model gives no usable probability, so
there is no threshold tuning and no AUC, as with the Claude models.

Sampling follows Qwen's guidance for thinking mode (temperature 0.6, top-p 0.95,
top-k 20; greedy decoding makes the reasoning loop). Each request has a fixed
seed, so the run can be repeated exactly. The second test pass uses different
seeds, which measures how stable the answers are.

    python qwen38_thinking.py --check    # 5 records: show prompt end, reasoning, parsed answer
    python qwen38_thinking.py            # full run

Environment: BMO_RUN_DIR (bmo_common and the dataset), BMO_OUT (output dir),
HF_HOME (the weights cache). Weights are read offline at a pinned revision.
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN_DIR = Path(os.environ["BMO_RUN_DIR"])
OUT = Path(os.environ["BMO_OUT"])
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C                                         # noqa: E402

CKPT = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"         # the snapshot the non-thinking run used
NAME = "Qwen3.8-27B (thinking)"
MAX_NEW = 8192                                                 # reasoning + answer
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)


def load_prompt():
    src = (REPO / "notebooks" / "_src" / "06_prompt_based_llms.py").read_text()
    start = src.index("TASK = (")
    end = src.index("\n}\n", src.index("PROMPT_VARIANTS = {")) + 3
    ns = {}
    exec(src[start:end], ns)
    return ns["TASK"], ns["PROMPT_VARIANTS"]["direct"]


TASK, TEMPLATE = load_prompt()


def parse(answer):
    """First word: 1 Yes, 0 No, -1 anything else. Same rule as claude_zeroshot.parse."""
    t = re.sub(r"[^a-z]", " ", answer.lower()).split()
    if t and t[0] in ("yes", "no"):
        return int(t[0] == "yes")
    return -1


def split_reply(text):
    """(reasoning, answer). No closing tag means the reasoning never finished."""
    if "</think>" not in text:
        return text, None
    head, tail = text.split("</think>", 1)
    return head.replace("<think>", "").strip(), tail.strip()


def load_data():
    df, tr, te = C.load_data()
    assert len(te) == 341 and len(df) == 1900, "not the paper's partition"
    return df["clean_freeform"].fillna("").values, df["has_biomarker"].values, tr, te


def build(tok, text):
    content = TEMPLATE.format(task=TASK, text=str(text))
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def engine():
    from huggingface_hub import snapshot_download
    from vllm import LLM
    path = snapshot_download(CKPT, revision=REVISION, local_files_only=True)
    llm = LLM(model=path, dtype="bfloat16", tensor_parallel_size=1, max_model_len=12288,
              gpu_memory_utilization=0.85, seed=0)
    return llm, llm.get_tokenizer(), path


def generate(llm, tok, jobs):
    """jobs: list of (custom_id, text, seed). Returns one record per job."""
    from vllm import SamplingParams
    prompts = [build(tok, t) for _, t, _ in jobs]
    params = [SamplingParams(max_tokens=MAX_NEW, seed=s, **SAMPLING) for _, _, s in jobs]
    outs = llm.generate(prompts, params)
    recs = []
    for (cid, _, seed), o in zip(jobs, outs):
        g = o.outputs[0]
        reasoning, answer = split_reply(g.text)
        recs.append({"custom_id": cid, "seed": seed, "pred": parse(answer) if answer is not None else -1,
                     "answer": answer, "finished_reasoning": answer is not None,
                     "out_tok": len(g.token_ids), "finish_reason": g.finish_reason,
                     "reasoning_chars": len(reasoning)})
    return recs, prompts


def check(llm, tok, texts, y, tr):
    rng = np.random.default_rng(7)
    rows = rng.choice(tr, 5, replace=False)
    jobs = [(f"check__{r}", texts[r], 1000 + int(r)) for r in rows]
    recs, prompts = generate(llm, tok, jobs)
    print("prompt ends with:", repr(prompts[0][-200:]))
    ok = True
    for r, rec in zip(rows, recs):
        print(f"row {r} label {y[r]} -> pred {rec['pred']} | answer {rec['answer']!r} | "
              f"{rec['out_tok']} tokens, finish {rec['finish_reason']}")
        ok &= rec["pred"] >= 0
    print("\nCHECK", "PASSED" if ok else "FAILED")
    return ok


def full(llm, tok, texts, y, tr, te, path):
    jobs = [(f"run1__{r}", texts[r], 1000 + r) for r in range(len(texts))]
    jobs += [(f"run2__{r}", texts[r], 50000 + int(r)) for r in te]
    recs, _ = generate(llm, tok, jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "replies.jsonl").open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    raw = pd.DataFrame(recs)
    raw[["run", "row"]] = raw.custom_id.str.split("__", expand=True)
    raw["row"] = raw.row.astype(int)
    g1 = raw[raw.run == "run1"].set_index("row")
    g2 = raw[raw.run == "run2"].set_index("row")
    p1, p2 = g1.pred.reindex(te).values, g2.pred.reindex(te).values
    ptr = g1.pred.reindex(tr).values
    summary = {"Model": NAME, "test_errors": int((p1 != y[te]).sum()),
               "missed_biomarkers": int(((y[te] == 1) & (p1 == 0)).sum()),
               "false_alarms": int(((y[te] == 0) & (p1 == 1)).sum()),
               "unparsed": int((p1 < 0).sum()),
               "unfinished_reasoning": int((~g1.reindex(te).finished_reasoning.astype(bool)).sum()),
               "run_agreement": round(float((p1 == p2).mean()), 4),
               "train_errors_of_1559": int((ptr != y[tr]).sum()),
               "median_out_tokens": int(raw.out_tok.median()), "max_out_tokens": int(raw.out_tok.max())}
    pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame({"row": te, "y_true": y[te], NAME: np.where(p1 < 0, 1 - y[te], p1)}).to_csv(
        OUT / "test_predictions.csv", index=False)
    import torch, transformers, vllm
    meta = {"checkpoint": CKPT, "revision": REVISION, "local_path": path, "precision": "bf16",
            "thinking": "enabled", "sampling": SAMPLING, "max_new_tokens": MAX_NEW,
            "seeds": "run1 1000+row, run2 50000+row", "date": dt.date.today().isoformat(),
            "vllm": vllm.__version__, "transformers": transformers.__version__,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(summary, indent=2))
    assert len(raw) == 1900 + 341
    print("VERIFY PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    texts, y, tr, te = load_data()
    llm, tok, path = engine()
    if a.check:
        sys.exit(0 if check(llm, tok, texts, y, tr) else 2)
    full(llm, tok, texts, y, tr, te, path)


if __name__ == "__main__":
    main()
