"""Reasoning ("thinking") Claude models on the biomarker task.

Same prompt and records as claude_zeroshot.py (Qwen2.5's selected "direct"
prompt, all 1,900 records plus a second pass over the 341 test records), but
the model reasons before it answers. Opus 5.5 and Fable 5.2 cannot answer
without reasoning; effort is set to "high" for both (Opus 5.5 defaults to
"medium", so it is set explicitly).

No server-side fallback is enabled: if a model declines a request, the
reply is recorded as a refusal and scored as an error, so every scored answer
comes from the model named in the paper.

    python claude_reasoning.py pilot     # 20 training records per model, prints cost per call
    python claude_reasoning.py run       # every request; safe to re-run, skips answered ones
    python claude_reasoning.py score     # tables from what has been collected

Key from ~/.anthropic_key unless ANTHROPIC_API_KEY is set.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import anthropic
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_zeroshot as z   # prompt, data loading and the first-word parser, unchanged

OUT = z.RUN / "outputs" / "prompt_llm" / "claude_reasoning"
OUT.mkdir(parents=True, exist_ok=True)
EFFORT = os.environ.get("BMO_EFFORT", "high")
MODELS = ["claude-opus-5-5", "claude-fable-5-2"]
PRICE = {"claude-opus-5-5": (4, 20), "claude-fable-5-2": (8, 40)}   # $ per million tokens
BUDGET = float(os.environ.get("BMO_BUDGET_USD", "35"))   # hard stop; estimate is about $20
MAX_TOKENS = 16000          # thinking counts toward this; the answer itself is one word


def params(model, text):
    return dict(model=model, max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": z.prompt(text)}])


def call(c, model, text):
    msg = c.messages.create(**params(model, text))
    text_out = "".join(b.text for b in msg.content if b.type == "text").strip()
    pred = z.parse(text_out) if msg.stop_reason != "refusal" else -1
    return {"reply": text_out, "pred": pred, "stop_reason": msg.stop_reason,
            "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens}


def cost(rec, model):
    pin, pout = PRICE[model]
    return (rec["in_tok"] * pin + rec["out_tok"] * pout) / 1e6


def pilot(n=20):
    df, tr, _ = z.load_data()
    rows = np.random.default_rng(7).choice(tr, n, replace=False)   # same 20 as the earlier pilot
    c, out = z.client(), []
    for model in MODELS:
        for r in rows:
            rec = call(c, model, df.text[r])
            out.append({"model": model, "row": int(r), "y": int(df.has_biomarker[r]),
                        "cost": cost(rec, model), **rec})
    p = pd.DataFrame(out)
    p.to_csv(OUT / f"pilot_{EFFORT}.csv", index=False)
    for model, g in p.groupby("model", sort=False):
        per = g.cost.mean()
        print(f"{model:17s} effort={EFFORT} correct {(g.pred == g.y).sum()}/{len(g)} "
              f"unparsed {(g.pred < 0).sum()}  out_tok mean {g.out_tok.mean():.0f} max {g.out_tok.max()}  "
              f"${per:.4f}/call -> about ${per * 2241:.0f} for the full run")
        for _, x in g[g.pred < 0].iterrows():
            print(f"   unparsed row {x.row}: {x.reply[:80]!r} ({x.stop_reason})")


def jobs():
    df, _, te = z.load_data()
    out = []
    for model in MODELS:
        out += [(f"{model}__run1__{r}", model, df.text[r]) for r in range(len(df))]
        out += [(f"{model}__run2__{r}", model, df.text[r]) for r in te]
    return out


def log_path():
    return OUT / f"replies_{EFFORT}.jsonl"


def run(workers=8):
    log = log_path()
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec["result"] == "succeeded":
                done.add(rec["custom_id"])
    todo = [j for j in jobs() if j[0] not in done]
    print(f"{len(done)} already answered, {len(todo)} to send", flush=True)
    c, lock, spent, n = z.client(), Lock(), [0.0], [0]

    def one(job):
        cid, model, text = job
        with lock:
            if spent[0] >= BUDGET:
                return                               # over budget: send nothing more
        try:
            rec = {"custom_id": cid, "result": "succeeded", **call(c, model, text)}
        except anthropic.APIError as e:
            rec = {"custom_id": cid, "result": "errored", "error": f"{type(e).__name__}: {e}"}
        with lock:
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["result"] == "succeeded":
                spent[0] += cost(rec, model)
            n[0] += 1
            if n[0] % 200 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)} sent, ${spent[0]:.2f} so far", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in todo]):
            f.result()
    print(f"session cost ${spent[0]:.2f}")
    if spent[0] >= BUDGET:
        print(f"STOPPED at the ${BUDGET:.0f} budget; nothing more was sent")
    score()


def score():
    recs = {}
    for line in log_path().read_text().splitlines():
        rec = json.loads(line)
        if rec["result"] == "succeeded" or rec["custom_id"] not in recs:
            recs[rec["custom_id"]] = rec
    rows = []
    for cid, rec in recs.items():
        model, rn, r = cid.split("__")
        rows.append({"model": model, "run": rn, "row": int(r),
                     **{k: v for k, v in rec.items() if k != "custom_id"}})
    raw = pd.DataFrame(rows).sort_values(["model", "run", "row"])
    raw.to_csv(OUT / f"raw_replies_{EFFORT}.csv", index=False)
    bad = raw.result.ne("succeeded").sum()
    if bad:
        print(f"{bad} requests errored; run 'run' again to retry only those")
    df, tr, te = z.load_data()
    y = df.has_biomarker.values
    lines, preds = [], {}
    for model in MODELS:
        g1 = raw[(raw.model == model) & (raw.run == "run1")].set_index("row")
        g2 = raw[(raw.model == model) & (raw.run == "run2")].set_index("row")
        p1 = g1.pred.reindex(te).fillna(-1).astype(int).values
        p2 = g2.pred.reindex(te).fillna(-1).astype(int).values
        ptr = g1.pred.reindex(tr).fillna(-1).astype(int).values
        unp = int((p1 < 0).sum())
        lines.append({"Model": model, "effort": EFFORT,
                      "test_errors": int((p1 != y[te]).sum()),
                      "missed_biomarkers": int(((y[te] == 1) & (p1 == 0)).sum()),
                      "false_alarms": int(((y[te] == 0) & (p1 == 1)).sum()),
                      "unparsed": unp,
                      "refusals": int((g1.reindex(te).stop_reason == "refusal").sum()),
                      "run_agreement": round(float((p1 == p2).mean()), 4),
                      "train_errors_of_1559": int((ptr != y[tr]).sum()),
                      "cost_usd": round(sum(cost(r, model) for r in raw[raw.model == model]
                                            .dropna(subset=["in_tok"]).to_dict("records")), 2)})
        preds[f"{model} ({EFFORT})"] = np.where(p1 < 0, 1 - y[te], p1)   # unparsed scored as wrong
    tab = pd.DataFrame(lines)
    print(tab.to_string(index=False))
    tab.to_csv(OUT / f"summary_{EFFORT}.csv", index=False)
    pd.DataFrame({"row": te, "y_true": y[te], **preds}).to_csv(
        OUT / f"test_predictions_{EFFORT}.csv", index=False)


if __name__ == "__main__":
    {"pilot": pilot, "run": run, "score": score}[sys.argv[1]]()
