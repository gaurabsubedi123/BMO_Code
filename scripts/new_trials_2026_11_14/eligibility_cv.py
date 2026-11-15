"""Does training on eligibility criteria close the eligibility gap?

Trial-grouped 5-fold cross-validation on the eligibility lines of the 268 newer trials (drug-level lines set aside,
as in the paper). Every line is predicted by a model that never saw any line of its trial.

Two training sets per fold:
  A  eligibility only     : eligibility lines of the four training folds
  B  outcomes+eligibility : the paper's 1,559 training outcomes + the same eligibility lines
Model B is also scored on (i) the paper's 341 held-out outcomes and (ii) the outcome lines of the held-out fold's
trials, so the outcome side gets out-of-fold predictions for all 268 trials and we see whether adding eligibility
costs anything on outcomes.

Each model uses its paper recipe unchanged (lr, batch size, dropout, weight decay, epoch count of the final fit,
layer-wise LR decay, class-weighted loss, warmup, threshold 0.5). Nothing is tuned on these lines.

Usage: eligibility_cv.py RUN_DIR NEW_TRIALS_DIR OUTDIR [--smoke] [--models a,b] [--variants A,B]
"""
import argparse, copy, json, os, pathlib, random, re, sys, time
import numpy as np, pandas as pd, torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from drug_level import drug  # noqa: E402

# Paper recipes (rerun_results test_results_merged.csv; epochs = final fit)
RECIPE = {
    "BiomedBERT-large": dict(ckpt="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract", lr=2e-5, bs=8, ep=5, dp=0.5, wd=0.10),
    "SciBERT":          dict(ckpt="allenai/scibert_scivocab_uncased", lr=3e-5, bs=16, ep=3, dp=0.5, wd=0.01),
    "BioLinkBERT":      dict(ckpt="michiyasunaga/BioLinkBERT-base", lr=5e-5, bs=32, ep=2, dp=0.3, wd=0.01),
}
MAX_LEN, WARMUP, LAYER_DECAY, SEED = 128, 0.1, 0.95, 42
DEV = torch.device("cuda")


def clean_text(text):  # verbatim from src/01_prepare_dataset.py
    text = re.sub(r"\['([^']+)'\]", r'\1', str(text)); text = re.sub(r"[\[\]'\"{}]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, texts, labels, tok):
        self.t, self.y, self.tok = list(texts), np.asarray(labels, dtype=int), tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(re.sub(r"\s+", " ", str(self.t[i])).strip().lower(), max_length=MAX_LEN, truncation=True,
                     padding="max_length", return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0], torch.tensor(self.y[i])


class Clf(nn.Module):  # same head as the paper (CLS token, dropout, linear)
    def __init__(self, ckpt, dp):
        super().__init__(); cfg = AutoConfig.from_pretrained(ckpt)
        if hasattr(cfg, "reference_compile"): cfg.reference_compile = False
        self.bert = AutoModel.from_pretrained(ckpt, config=cfg)
        self.head = nn.Sequential(nn.Dropout(dp), nn.Linear(self.bert.config.hidden_size, 2))
    def forward(self, ids, mask):
        return self.head(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])


def optimizer(model, lr, wd):  # layer-wise LR decay + AdamW exclusions, as notebook 05
    nod = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"); groups = []
    def add(named, l):
        named = list(named)
        d = [p for n, p in named if not any(k in n for k in nod)]; o = [p for n, p in named if any(k in n for k in nod)]
        if d: groups.append({"params": d, "lr": l, "weight_decay": wd})
        if o: groups.append({"params": o, "lr": l, "weight_decay": 0.0})
    layers = model.bert.encoder.layer; n = len(layers)
    add(model.head.named_parameters(), lr)
    for i in range(n - 1, -1, -1): add(layers[i].named_parameters(), lr * LAYER_DECAY ** (n - i))
    add(model.bert.embeddings.named_parameters(), lr * LAYER_DECAY ** (n + 1))
    covered = sum(p.numel() for g in groups for p in g["params"])
    total = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith("bert.pooler"))
    assert covered == total, f"optimizer misses parameters: {covered} vs {total}"
    return AdamW(groups)


def train(r, tok, texts, y, epochs):
    model = Clf(r["ckpt"], r["dp"]).to(DEV)
    dl = DataLoader(DS(texts, y, tok), batch_size=r["bs"], shuffle=True, num_workers=2)
    opt = optimizer(model, r["lr"], r["wd"]); steps = len(dl) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    y = np.asarray(y); w = torch.tensor([1.0, (y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w); model.train()
    for _ in range(epochs):
        for ids, mask, lab in dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(ids.to(DEV), mask.to(DEV)), lab.to(DEV)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    model.eval(); return model


@torch.no_grad()
def probs(model, tok, texts):
    if len(texts) == 0: return np.array([])
    dl = DataLoader(DS(texts, np.zeros(len(texts)), tok), batch_size=128, shuffle=False, num_workers=2); out = []
    for ids, mask, _ in dl:
        out += torch.softmax(model(ids.to(DEV), mask.to(DEV)).float(), 1)[:, 1].cpu().tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", default=",".join(RECIPE))
    ap.add_argument("--variants", default="A,B"); a = ap.parse_args()
    run, new, out = map(pathlib.Path, (a.run, a.new, a.out)); out.mkdir(parents=True, exist_ok=True)

    # Paper corpus: 1,559 training / 341 held-out outcomes
    df = pd.read_csv(run / "outputs/dataset/labeled_dataset.csv")
    tr_i, te_i = np.load(run / "outputs/dataset/train_idx.npy"), np.load(run / "outputs/dataset/test_idx.npy")
    cx, cy = df.clean_freeform.fillna("").values, df.has_biomarker.values
    assert (len(tr_i), len(te_i)) == (1559, 341)

    # Newer trials, 2025 definition (drug-level lines set aside on wording)
    el = pd.read_csv(new / "new_eligibility_lines.csv", dtype={"line_no": str})
    el["model_input"] = el.line_text.fillna("").astype(str).apply(clean_text)
    el = el[~drug(el.model_input)].reset_index(drop=True)
    oc = pd.read_csv(new / "new_outcome_lines.csv", dtype={"line_no": str})
    oc["model_input"] = (oc.outcome_measure.fillna("").astype(str) + " " +
                         oc.outcome_description.fillna("").astype(str)).apply(clean_text)
    oc = oc[~drug(oc.model_input)].reset_index(drop=True)
    print(f"eligibility {len(el)} lines / {int(el.label.sum())} biomarker; outcomes {len(oc)} / {int(oc.label.sum())}",
          flush=True)
    assert (len(el), int(el.label.sum()), len(oc), int(oc.label.sum())) == (5987, 373, 2361, 780), "set-aside drifted"
    assert not set(el.nct) & set(df.NCTId), "newer trials overlap the paper corpus"

    folds = np.zeros(len(el), dtype=int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(el, el.label, el.nct)):
        folds[va] = k
    el["fold"] = folds
    fold_of_trial = el.groupby("nct").fold.first()
    oc["fold"] = oc.nct.map(fold_of_trial).fillna(-1).astype(int)  # trials with no eligibility lines: -1
    nfold = 1 if a.smoke else 5
    if a.smoke:
        print("SMOKE: 1 fold, 1 epoch, 400 eligibility training lines", flush=True)

    rows = []
    for name in a.models.split(","):
        r = RECIPE[name]; tok = AutoTokenizer.from_pretrained(r["ckpt"]); ep = 1 if a.smoke else r["ep"]
        for var in a.variants.split(","):
            el[f"{name}|{var}"] = np.nan
            if var == "B": oc[f"{name}|{var}"] = np.nan
            for k in range(nfold):
                t0 = time.time(); seed_all(SEED + k)
                trn = el[el.fold != k]
                if a.smoke: trn = trn.sample(400, random_state=0)
                X, Y = list(trn.model_input), list(trn.label)
                if var == "B": X, Y = list(cx[tr_i]) + X, list(cy[tr_i]) + Y
                assert not set(trn.nct) & set(el.nct[el.fold == k]), "fold leaks trials"
                m = train(r, tok, X, Y, ep)
                el.loc[el.fold == k, f"{name}|{var}"] = probs(m, tok, list(el.model_input[el.fold == k]))
                row = {"model": name, "variant": var, "fold": k, "train_lines": len(X), "seconds": 0}
                if var == "B":
                    sel = (oc.fold == k) | ((k == 0) & (oc.fold == -1))  # unseen-eligibility trials go with fold 0
                    oc.loc[sel, f"{name}|{var}"] = probs(m, tok, list(oc.model_input[sel]))
                    p = probs(m, tok, list(cx[te_i])) >= 0.5
                    row.update(heldout_missed=int(((cy[te_i] == 1) & ~p).sum()), heldout_fp=int(((cy[te_i] == 0) & p).sum()))
                row["seconds"] = round(time.time() - t0)
                v = el[el.fold == k]; pp = v[f"{name}|{var}"] >= 0.5
                row.update(elig_missed=int(((v.label == 1) & ~pp).sum()), elig_pos=int(v.label.sum()),
                           elig_fp=int(((v.label == 0) & pp).sum()))
                rows.append(row); print(json.dumps(row), flush=True)
                del m; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False)
            el.to_csv(out / "eligibility_cv_scored.csv", index=False); oc.to_csv(out / "outcomes_cv_scored.csv", index=False)
    if not a.smoke:
        for c in [c for c in el.columns if "|" in c]: assert el[c].notna().all(), f"{c} has unscored lines"
        for c in [c for c in oc.columns if "|" in c]: assert oc[c].notna().all(), f"{c} has unscored lines"
    print("CV_DONE", flush=True)


if __name__ == "__main__":
    main()
