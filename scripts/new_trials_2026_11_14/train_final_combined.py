"""Train the released BiomedBERT-large on outcomes + eligibility, and test it on the paper's 341 held-out outcomes.

Training data: the paper's 1,559 training outcomes + all 5,987 eligibility lines of the 268 newer trials (drug-level
lines set aside, as in eligibility_cv.py). Recipe identical to the cross-validation (paper final-fit settings).
Its expected performance on eligibility and on newer-trial outcomes is the cross-validated estimate from
eligibility_cv.py; the 341 held-out outcomes are a direct test because they are never trained on.

Usage: train_final_combined.py RUN_DIR NEW_TRIALS_DIR OUTDIR
Writes OUTDIR/model/{model_state.pt, config.json, tokenizer/, training_config.json} and OUTDIR/heldout_predictions.csv
"""
import json, pathlib, sys
import numpy as np, pandas as pd, torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from drug_level import drug  # noqa: E402
from eligibility_cv import RECIPE, SEED, clean_text, probs, seed_all, train  # noqa: E402

run, new, out = map(pathlib.Path, sys.argv[1:4]); (out / "model").mkdir(parents=True, exist_ok=True)
df = pd.read_csv(run / "outputs/dataset/labeled_dataset.csv")
tr_i, te_i = np.load(run / "outputs/dataset/train_idx.npy"), np.load(run / "outputs/dataset/test_idx.npy")
cx, cy = df.clean_freeform.fillna("").values, df.has_biomarker.values
el = pd.read_csv(new / "new_eligibility_lines.csv", dtype={"line_no": str})
el["model_input"] = el.line_text.fillna("").astype(str).apply(clean_text)
el = el[~drug(el.model_input)].reset_index(drop=True)
assert (len(tr_i), len(te_i), len(el), int(el.label.sum())) == (1559, 341, 5987, 373)
assert not set(el.nct) & set(df.NCTId)

name = "BiomedBERT-large"; r = RECIPE[name]; tok = AutoTokenizer.from_pretrained(r["ckpt"])
X, Y = list(cx[tr_i]) + list(el.model_input), list(cy[tr_i]) + list(el.label)
seed_all(SEED); m = train(r, tok, X, Y, r["ep"])
p = probs(m, tok, list(cx[te_i])); yt = cy[te_i]; yp = p >= 0.5
fn, fp = int(((yt == 1) & ~yp).sum()), int(((yt == 0) & yp).sum())
tp, tn = int(((yt == 1) & yp).sum()), int(((yt == 0) & ~yp).sum())
print(json.dumps({"train_lines": len(X), "train_biomarker": int(sum(Y)), "heldout_missed": fn, "heldout_fp": fp,
                  "heldout_tp": tp, "heldout_tn": tn}), flush=True)
pd.DataFrame({"row": te_i, "NCTId": df.NCTId.values[te_i], "y_true": yt, "prob": p.round(6)}).to_csv(
    out / "heldout_predictions.csv", index=False)
d = out / "model"
torch.save(m.state_dict(), d / "model_state.pt"); tok.save_pretrained(d / "tokenizer")
AutoConfig.from_pretrained(r["ckpt"]).save_pretrained(d)
(d / "training_config.json").write_text(json.dumps(
    {"checkpoint": r["ckpt"], "lr": r["lr"], "batch_size": r["bs"], "epochs": r["ep"], "dropout": r["dp"],
     "weight_decay": r["wd"], "max_len": 128, "seed": SEED, "training_data": "1,559 outcomes (paper train split) + "
     "5,987 eligibility lines from 268 newer trials (drug-level lines set aside)",
     "heldout_341": {"missed": fn, "false_positives": fp}}, indent=2))
print("FINAL_DONE", flush=True)
