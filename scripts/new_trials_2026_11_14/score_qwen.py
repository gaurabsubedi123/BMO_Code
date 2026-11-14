"""On grundy: Qwen3.8-27B (reasoning off), exactly the paper's protocol (prompt 'direct', first-token Yes/No,
cut-offs 0.5 and 0.10), on the newer-trial lines. Gate: must reproduce 6 / 2 held-out misses and 0 false positives.
Usage (env.sh sourced so BMO_RUN_DIR is set): score_qwen.py repo_analysis_dir outcomes.csv eligibility.csv outdir"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import qwen38_zeroshot as q
oc_path, el_path, out = sys.argv[2:5]
def clean_text(text):
    text = re.sub(r"\['([^']+)'\]", r'\1', str(text)); text = re.sub(r"[\[\]'\"{}]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()
_, _, xt, yt = q.load_splits()
tok, model = q.load_model("Qwen/Qwen3.8-27B", "bf16", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
p = q.score(model, tok, list(xt), "direct")
for thr, want in ((0.5, 6), (0.10, 2)):
    fn, fp = int(((yt == 1) & (p < thr)).sum()), int(((yt == 0) & (p >= thr)).sum())
    print(f"held-out at {thr}: missed {fn}, false positives {fp}", flush=True); assert (fn, fp) == (want, 0)
oc = pd.read_csv(oc_path, dtype={"line_no": str}); el = pd.read_csv(el_path, dtype={"line_no": str})
oc["model_input"] = (oc.outcome_measure.fillna("").astype(str) + " " + oc.outcome_description.fillna("").astype(str)).apply(clean_text)
el["model_input"] = el.line_text.fillna("").astype(str).apply(clean_text)
oc["Qwen3.8-27B"] = q.score(model, tok, list(oc.model_input), "direct").round(6); oc.to_csv(f"{out}/outcomes_qwen_scored.csv", index=False)
print("QWEN_OUTCOMES_DONE", flush=True)
el["Qwen3.8-27B"] = q.score(model, tok, list(el.model_input), "direct").round(6); el.to_csv(f"{out}/eligibility_qwen_scored.csv", index=False)
print("QWEN_DONE")
