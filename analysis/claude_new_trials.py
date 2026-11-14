"""The five Claude models on the 268 newer CTO trials (outcome measures and eligibility criteria).

Each model is run exactly as in the paper: the three one-step models with the settings of
claude_zeroshot.py, the two reasoning models with those of claude_reasoning.py, all with the
'direct' prompt selected for Qwen2.5-7B-Instruct and the same first-word parser. The input is
the text every local model read on these trials (column model_input of the *_all_scored.csv
files written by scripts/new_trials_2026_11_14/), so all systems see identical text.

    python claude_new_trials.py dryrun            # fake API: whole pipeline, no key, no cost
    python claude_new_trials.py pilot             # 20 outcome + 10 eligibility lines per model
    python claude_new_trials.py run [outcomes]    # every request (or one part); resumable
    python claude_new_trials.py score             # tables from what has been collected

Safety: every reply is appended to the log as it arrives; a re-run skips anything already
answered, so an interruption never pays twice. The budget (BMO_BUDGET_USD, default 125) is
counted across ALL sessions from the log, and no request is sent once the running total plus
the worst case for one more request would pass it. No server-side fallback model is used.

Environment: BMO_RERUN_DIR (paper run dir), BMO_NEW_TRIALS_DIR (folder with
outcomes_all_scored.csv and eligibility_all_scored.csv). Key from ANTHROPIC_API_KEY or
~/.anthropic_key.
"""
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_zeroshot as z   # prompt, parser, client, one-step settings: unchanged

NEW = Path(os.environ.get("BMO_NEW_TRIALS_DIR", z.RUN.parent / "Paper/Nov_13_rewrite/new_trials_test"))
PARTS = {"outcomes": NEW / "outcomes_all_scored.csv", "eligibility": NEW / "eligibility_all_scored.csv"}
ONE_STEP = dict(z.MODELS)                       # thinking disabled / absent, max_tokens 50
REASONING = ["claude-opus-5-5", "claude-fable-5-2"]
MODELS = list(ONE_STEP) + REASONING
PRICE = {**z.PRICE, "claude-opus-5-5": (4, 20), "claude-fable-5-2": (8, 40)}   # $ per million tokens
BUDGET = float(os.environ.get("BMO_BUDGET_USD", "125"))
R_MAX_TOKENS = 16000                            # as claude_reasoning.py; thinking counts toward it
EFFORT = "high"


def params(model, text):
    if model in REASONING:
        return dict(model=model, max_tokens=R_MAX_TOKENS, thinking={"type": "adaptive"},
                    output_config={"effort": EFFORT},
                    messages=[{"role": "user", "content": z.prompt(text)}])
    return z.params(model, text)


def worst_case(model, text):
    """Most one request can cost: its prompt plus a full max_tokens reply."""
    pin, pout = PRICE[model]
    n_in = len(z.prompt(text)) / 2.5            # generous: real prompts run ~3.7 characters per token
    return (n_in * pin + (R_MAX_TOKENS if model in REASONING else z.MAX_TOKENS) * pout) / 1e6


def cost(rec, model):
    pin, pout = PRICE[model]
    return (rec["in_tok"] * pin + rec["out_tok"] * pout) / 1e6


def call(c, model, text):
    msg = c.messages.create(**params(model, text))
    out = "".join(b.text for b in msg.content if b.type == "text").strip()
    pred = z.parse(out) if msg.stop_reason != "refusal" else -1
    return {"reply": out, "pred": pred, "stop_reason": msg.stop_reason,
            "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens}


# ---------------------------------------------------------------- fake API for the dry run
class FakeError(Exception):
    pass


class FakeClient:
    """Answers like the API does, including the awkward cases the real run must survive:
    markdown-wrapped answers, a thinking block before the text, unreadable replies,
    refusals and transient errors."""
    def __init__(self, seed=0):
        self.rng, self.lock, self.calls = random.Random(seed), Lock(), 0
        self.messages = self

    def create(self, **kw):
        with self.lock:
            self.calls += 1
            u = self.rng.random()
        if u < 0.01:
            raise FakeError("simulated overloaded error")
        text = kw["messages"][0]["content"]
        assert kw["model"] in MODELS and "Yes" in text or "yes" in text.lower()
        ans = self.rng.choice(["Yes", "No", "**Yes.**", "No, this is a cognitive scale."])
        stop = "end_turn"
        if u < 0.02:
            ans = "I think it depends on the context."     # unreadable
        elif u < 0.025:
            ans, stop = "", "refusal"
        blocks = [SimpleNamespace(type="text", text=ans)]
        if kw["model"] in REASONING:
            blocks.insert(0, SimpleNamespace(type="thinking", thinking="..."))
            assert kw["thinking"] == {"type": "adaptive"} and kw["max_tokens"] == R_MAX_TOKENS
        else:
            assert kw["max_tokens"] == z.MAX_TOKENS
        n_in = int(len(text) / 3.7)
        n_out = {"claude-opus-5": 5, "claude-sonnet-5": 16, "claude-haiku-4-5": 46,
                 "claude-opus-5-5": 133, "claude-fable-5-2": 83}[kw["model"]]
        return SimpleNamespace(content=blocks, stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=n_in, output_tokens=n_out))


# ---------------------------------------------------------------- data and jobs
def load(part):
    d = pd.read_csv(PARTS[part], dtype={"line_no": str})
    assert len(d) == {"outcomes": 2754, "eligibility": 6022}[part], (part, len(d))
    assert d.model_input.notna().all() and (d.model_input.str.len() > 0).all()
    return d


def jobs(parts):
    out = []
    for part in parts:
        texts = load(part).model_input.tolist()
        for model in MODELS:
            out += [(f"{model}__{part}__{i}", model, t) for i, t in enumerate(texts)]
    return out


def read_log(log):
    recs = {}
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec["result"] == "succeeded" or rec["custom_id"] not in recs:
                recs[rec["custom_id"]] = rec
    return recs


# ---------------------------------------------------------------- run
def run(parts, out_dir, client, workers=8, budget=BUDGET):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "replies.jsonl"
    prior = read_log(log)
    done = {k for k, v in prior.items() if v["result"] == "succeeded"}
    spent = [sum(cost(v, k.split("__")[0]) for k, v in prior.items() if v["result"] == "succeeded")]
    todo = [j for j in jobs(parts) if j[0] not in done]
    print(f"{len(done)} already answered (${spent[0]:.2f} spent), {len(todo)} to send, budget ${budget:.2f}", flush=True)
    lock, n, stopped, reserved = Lock(), [0], [False], [0.0]

    def one(job):
        cid, model, text = job
        wc = worst_case(model, text)
        with lock:   # reserve the worst case first, so requests in flight can never add up past the budget
            if spent[0] + reserved[0] + wc > budget:
                stopped[0] = True
                return
            reserved[0] += wc
        try:
            rec = {"custom_id": cid, "result": "succeeded", **call(client, model, text)}
        except Exception as e:                              # retries are exhausted by now
            rec = {"custom_id": cid, "result": "errored", "error": f"{type(e).__name__}: {e}"}
        with lock:
            reserved[0] -= wc
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["result"] == "succeeded":
                spent[0] += cost(rec, model)
            n[0] += 1
            if n[0] % 1000 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)} sent, ${spent[0]:.2f} total", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in todo]):
            f.result()
    print(f"total spent ${spent[0]:.2f}")
    if stopped[0]:
        print(f"STOPPED before the ${budget:.2f} budget; unsent requests remain. Raise BMO_BUDGET_USD to continue.")
    return spent[0]


# ---------------------------------------------------------------- score
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, (len(p) - r) * p[i]); adj[i] = min(1.0, run_)
    return adj


def drug_mask(texts):
    g = Path(__file__).resolve().parents[1] / "scripts" / "new_trials_2026_11_14"
    ns = {}
    exec(open(g / "analyze2.py").read().split("c = pd.read_csv")[0]
         .replace("pathlib.Path(__file__).with_name('analyze.py')", repr(str(g / "analyze.py"))), ns)
    return ns["drug"](pd.Series(texts)).values


def score(out_dir, parts=("outcomes", "eligibility")):
    recs = read_log(out_dir / "replies.jsonl")
    rows = []
    for part in parts:
        d = load(part); y = d.label.values; dr = drug_mask(d.model_input)
        ref = d["BiomedBERT-large"].values >= 0.5
        for model in MODELS:
            got = [recs.get(f"{model}__{part}__{i}") for i in range(len(d))]
            ok = np.array([g is not None and g["result"] == "succeeded" for g in got])
            if not ok.any():
                continue
            pred = np.array([g["pred"] if g is not None and g["result"] == "succeeded" else -1 for g in got])
            # anything not a clean yes/no (unreadable, refusal, missing) is scored as wrong
            p = np.where(pred < 0, 1 - y, pred).astype(bool)
            P, N = int((y == 1).sum()), int((y == 0).sum())
            fn = int(((y == 1) & ~p).sum()); fp = int(((y == 0) & p).sum())
            b = int(((y == 1) & ref & ~p).sum()); c = int(((y == 1) & ~ref & p).sum())
            rows.append(dict(part=part, model=model, answered=int(ok.sum()), of=len(d),
                             unreadable=int(((pred < 0) & ok).sum()),
                             refusals=int(sum(g is not None and g.get("stop_reason") == "refusal" for g in got)),
                             missed=fn, missed_pct=round(100 * fn / P, 1), fp=fp,
                             fp_no_drug=int(((y == 0) & p & ~dr).sum()), flagged_pct=round(100 * p.mean(), 1),
                             p_misses_vs_biomedbert=mcnemar(b, c),
                             cost_usd=round(sum(cost(g, model) for g in got if g is not None and g["result"] == "succeeded"), 2)))
    t = pd.DataFrame(rows)
    if len(t):
        for part, g in t.groupby("part"):
            t.loc[g.index, "p_holm"] = holm(g.p_misses_vs_biomedbert.values)
        t.to_csv(out_dir / "summary.csv", index=False)
        print(t.to_string(index=False))
    return t


# ---------------------------------------------------------------- modes
def dryrun():
    # fake replies are throwaway: keep them on a local disk (a Windows mount makes 44k appends slow)
    out = Path(os.environ.get("BMO_DRYRUN_DIR", "/tmp/claude_new_trials_DRYRUN"))
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    parts = list(PARTS)
    j = jobs(parts)
    assert len(j) == 5 * (2754 + 6022) and len({x[0] for x in j}) == len(j)
    print(f"jobs: {len(j)} requests ({', '.join(f'{p} {len(load(p))} lines' for p in parts)}) x {len(MODELS)} models")
    est = {m: 0.0 for m in MODELS}
    for cid, m, t in j:
        pin, pout = PRICE[m]
        n_out = {"claude-opus-5": 5, "claude-sonnet-5": 16, "claude-haiku-4-5": 46, "claude-opus-5-5": 133, "claude-fable-5-2": 83}[m]
        est[m] += (len(z.prompt(t)) / 3.7 * pin + n_out * pout) / 1e6
    print("estimated cost at the earlier runs' reply lengths:", {m: round(v, 2) for m, v in est.items()},
          f"total ${sum(est.values()):.2f}")
    # 1. whole run against the fake API
    fake = FakeClient()
    spent = run(parts, out, fake, budget=1e9)
    recs = read_log(out / "replies.jsonl")
    errored = sum(r["result"] == "errored" for r in recs.values())
    print(f"after first pass: {len(recs)} logged, {errored} errored (simulated)")
    # 2. resume: only the errored requests are sent again, nothing already answered
    before = fake.calls
    run(parts, out, fake, budget=1e9)
    resent = fake.calls - before
    assert resent == errored, (resent, errored)
    print(f"resume check passed: re-run sent exactly the {resent} errored requests and nothing else")
    # 3. budget: a fresh log with a $1 budget must stop without passing it
    bud = out / "budget_test"
    s = run(["outcomes"], bud, FakeClient(1), budget=1.0)
    assert s <= 1.0, s
    print(f"budget check passed: stopped at ${s:.2f} of a $1.00 budget")
    # 4. parser on real reply shapes seen in the paper's runs
    for t, want in [("Yes", 1), ("**Yes.**", 1), ("No.", 0), ("No, this is a scale", 0), ("It depends", -1), ("", -1)]:
        assert z.parse(t) == want, t
    print("parser check passed")
    # 5. scoring end to end
    t = score(out)
    assert set(t.model) == set(MODELS) and len(t) == 10
    assert (t.answered + 0 <= t["of"]).all()
    print("\nDRY RUN PASSED: no API key used, no money spent")


def pilot():
    out = z.RUN / "outputs" / "prompt_llm" / "claude_new_trials"
    out.mkdir(parents=True, exist_ok=True)
    c, rows = z.client(), []
    for part, n in (("outcomes", 20), ("eligibility", 10)):
        d = load(part)
        idx = np.random.default_rng(11).choice(len(d), n, replace=False)
        for model in MODELS:
            for i in idx:
                rec = call(c, model, d.model_input.iloc[i])
                rows.append({"part": part, "model": model, "i": int(i), "y": int(d.label.iloc[i]),
                             "cost": cost(rec, model), **rec})
    p = pd.DataFrame(rows); p.to_csv(out / "pilot.csv", index=False)
    full = {"outcomes": 2754, "eligibility": 6022}
    tot = 0.0
    for (part, model), g in p.groupby(["part", "model"], sort=False):
        proj = g.cost.mean() * full[part]; tot += proj
        print(f"{part:11s} {model:17s} correct {(g.pred == g.y).sum()}/{len(g)}  unreadable {(g.pred < 0).sum()}  "
              f"out_tok mean {g.out_tok.mean():.0f} max {g.out_tok.max()}  ${g.cost.mean():.5f}/line -> ${proj:.2f}")
        for _, x in g[g.pred < 0].iterrows():
            print(f"     unreadable: {x.reply[:80]!r} ({x.stop_reason})")
    print(f"pilot cost ${p.cost.sum():.2f}; projected full run ${tot:.2f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dryrun":
        dryrun()
    elif mode == "pilot":
        pilot()
    elif mode == "run":
        parts = sys.argv[2:] or list(PARTS)
        out = z.RUN / "outputs" / "prompt_llm" / "claude_new_trials"
        run(parts, out, z.client())
        score(out, parts)
    elif mode == "score":
        score(z.RUN / "outputs" / "prompt_llm" / "claude_new_trials")
