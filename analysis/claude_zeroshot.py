"""Zero-shot Claude models on the biomarker task, set up exactly like Qwen2.5-7B.

Same prompt as Qwen (the "direct" variant it selected on the training holdout),
read straight from notebook 06 so the text cannot drift. No reasoning step, so
every prompted model answers in one step. The API gives no probabilities, so
there is no AUC and no threshold: compare against Qwen at the default 0.5 cut.

    python claude_zeroshot.py pilot            # 20 training records per model, synchronous
    python claude_zeroshot.py submit           # Batch API: all 1,900 records + a second test-set run
    python claude_zeroshot.py collect          # fetch results, score, write tables

The key is read from ~/.anthropic_key unless ANTHROPIC_API_KEY is set.
"""
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN = REPO.parent.parent / "rerun_results"
DATA = RUN / "outputs" / "dataset"
OUT = RUN / "outputs" / "prompt_llm" / "claude"
OUT.mkdir(parents=True, exist_ok=True)

# Thinking off everywhere. Opus 5 and Sonnet 5 run adaptive thinking when the
# parameter is omitted, so they need it disabled explicitly; Haiku 4.5 has no
# thinking unless asked. No temperature: these models reject it.
MODELS = {
    "claude-opus-5":    {"thinking": {"type": "disabled"}},
    "claude-sonnet-5":  {"thinking": {"type": "disabled"}},
    "claude-haiku-4-5": {},
}
MAX_TOKENS = 50     # the answer is one word; room to see what else a model says


def load_prompt():
    """TASK and the 'direct' template, executed from notebook 06's own source."""
    src = (REPO / "notebooks" / "_src" / "06_prompt_based_llms.py").read_text()
    start = src.index("TASK = (")
    end = src.index("\n}\n", src.index("PROMPT_VARIANTS = {")) + 3
    ns = {}
    exec(src[start:end], ns)
    return ns["TASK"], ns["PROMPT_VARIANTS"]["direct"]


TASK, TEMPLATE = load_prompt()


def prompt(text):
    return TEMPLATE.format(task=TASK, text=str(text))


def load_data():
    df = pd.read_csv(DATA / "labeled_dataset.csv")
    tr, te = np.load(DATA / "train_idx.npy"), np.load(DATA / "test_idx.npy")
    assert not (set(df.NCTId.values[tr]) & set(df.NCTId.values[te]))
    df["text"] = df["clean_freeform"].fillna("")
    return df, tr, te


def client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = (Path.home() / ".anthropic_key").read_text().strip()
    return anthropic.Anthropic(max_retries=5)


def parse(text):
    """First word of the reply: 1 for Yes, 0 for No, -1 for anything else.

    Qwen was scored on its first generated token, so the first word is the
    matching rule; any explanation that follows is ignored. A reply that does
    not open with Yes or No is never guessed and counts as an error.
    """
    t = re.sub(r"[^a-z]", " ", text.lower()).split()
    if t and t[0] in ("yes", "no"):
        return int(t[0] == "yes")
    return -1


def params(model, text):
    return dict(model=model, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt(text)}], **MODELS[model])


def reply_fields(msg):
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return {"reply": text, "pred": parse(text), "stop_reason": msg.stop_reason,
            "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens}


# ---------------------------------------------------------------- pilot
def pilot(n=20):
    df, tr, _ = load_data()
    rng = np.random.default_rng(7)
    rows = rng.choice(tr, n, replace=False)          # training records only
    c, out = client(), []
    for model in MODELS:
        for r in rows:
            msg = c.messages.create(**params(model, df.text[r]))
            out.append({"model": model, "row": int(r), "y": int(df.has_biomarker[r]),
                        **reply_fields(msg)})
    p = pd.DataFrame(out)
    p.to_csv(OUT / "pilot.csv", index=False)
    for model, g in p.groupby("model", sort=False):
        ok = g.pred >= 0
        print(f"{model:18s} parsed {ok.sum()}/{len(g)}  correct {(g.pred == g.y).sum()}/{len(g)}  "
              f"tokens in {g.in_tok.mean():.0f} out {g.out_tok.mean():.1f} per call")
        for _, x in g[~ok].iterrows():
            print(f"   unparsed row {x.row}: {x.reply[:80]!r} ({x.stop_reason})")
    print("\nprompt sent (first record):\n" + prompt(df.text[rows[0]]))


# ---------------------------------------------------------------- batch
def submit():
    df, _, te = load_data()
    reqs = []
    for model in MODELS:
        for r in range(len(df)):
            reqs.append(Request(custom_id=f"{model}__run1__{r}",
                                params=MessageCreateParamsNonStreaming(**params(model, df.text[r]))))
        for r in te:
            reqs.append(Request(custom_id=f"{model}__run2__{r}",
                                params=MessageCreateParamsNonStreaming(**params(model, df.text[r]))))
    batch = client().messages.batches.create(requests=reqs)
    meta = {"batch_id": batch.id, "submitted": date.today().isoformat(),
            "n_requests": len(reqs), "models": list(MODELS), "max_tokens": MAX_TOKENS}
    (OUT / "batch.json").write_text(json.dumps(meta, indent=2))
    print(f"submitted {len(reqs)} requests as {batch.id}")


def collect():
    meta = json.loads((OUT / "batch.json").read_text())
    c = client()
    while True:
        b = c.messages.batches.retrieve(meta["batch_id"])
        if b.processing_status == "ended":
            break
        print(f"{b.processing_status}: {b.request_counts.processing} still processing")
        time.sleep(60)
    rows = []
    for res in c.messages.batches.results(meta["batch_id"]):
        model, run, r = res.custom_id.split("__")
        rec = {"model": model, "run": run, "row": int(r), "result": res.result.type}
        if res.result.type == "succeeded":
            rec.update(reply_fields(res.result.message))
        rows.append(rec)
    raw = pd.DataFrame(rows).sort_values(["model", "run", "row"])
    raw.to_csv(OUT / "raw_replies.csv", index=False)
    print(raw.groupby(["model", "run"]).result.value_counts().to_string())
    score(raw)


def score(raw):
    df, tr, te = load_data()
    y = df.has_biomarker.values
    qwen = pd.read_csv(RUN / "outputs" / "prompt_llm" / "test_predictions.csv")
    assert (qwen.y_true.values == y[te]).all()
    pred = {}
    for model in MODELS:
        for run in ("run1", "run2"):
            g = raw[(raw.model == model) & (raw.run == run)].set_index("row")
            pred[model, run] = g.pred.reindex(te).fillna(-1).astype(int).values
    lines = []
    for model in MODELS:
        p1, p2 = pred[model, "run1"], pred[model, "run2"]
        flagged = int((p1 < 0).sum())
        wrong = int(((p1 != y[te]) & (p1 >= 0)).sum()) + flagged   # an unparsed reply counts as an error
        fn = int(((y[te] == 1) & (p1 == 0)).sum())
        fp = int(((y[te] == 0) & (p1 == 1)).sum())
        agree = float((p1 == p2).mean())
        g = raw[(raw.model == model) & (raw.run == "run1")].set_index("row")
        pa = g.pred.reindex(tr).fillna(-1).astype(int).values
        wrong_tr = int((pa != y[tr]).sum())
        lines.append({"Model": model, "test_errors": wrong, "missed_biomarkers": fn,
                      "false_alarms": fp, "unparsed": flagged, "run_agreement": round(agree, 4),
                      "train_errors_of_1559": wrong_tr})
        qwen[f"{model} (zero-shot)"] = np.where(p1 < 0, 1 - y[te], p1)   # unparsed scored as wrong
    tab = pd.DataFrame(lines)
    q = int((qwen["Qwen2.5-7B-Instruct"] != qwen.y_true).sum())
    print(tab.to_string(index=False))
    print(f"Qwen2.5-7B-Instruct at default threshold: {q} test errors")
    tab.to_csv(OUT / "claude_zeroshot_summary.csv", index=False)
    qwen.to_csv(OUT / "test_predictions_with_claude.csv", index=False)


if __name__ == "__main__":
    {"pilot": pilot, "submit": submit, "collect": collect}[sys.argv[1]]()
