#%% md
# 12 — Seed stability of the transformer results

Across three runs of an identical protocol on the same train/test split, four
different models have held first place — SciBERT, BioLinkBERT, Bio+ClinicalBERT
and BiomedBERT-large — all separated by one to three errors out of 341.

Because the split never changed, that movement is not split variance. It is
**training stochasticity**: weight initialisation, batch ordering, and
non-deterministic cuDNN kernels. This notebook measures it directly, by refitting
each model's selected configuration under several torch seeds and reporting the
spread of test-set results.

## Why this rather than repeated train/test splits

Repeated splits (~15 h) would measure a different quantity — sensitivity to the
partition — and would not answer the question the results actually raise. The
instability observed here appeared *within* a fixed partition, so seeds are the
variable to vary. This costs about 40 minutes.

Together with two measurements we already have, uncertainty is then characterised
from three independent directions:

| Source | Measured by |
|---|---|
| Test-sample variation | bootstrap CIs over test records (notebook 07) |
| Train-split variation | standard deviation across the 5 CV folds (notebook 05) |
| **Training stochasticity** | **this notebook** |

Nothing here touches model selection: every configuration is fixed, read from
the completed run. The test set is used only to score already-chosen models.
#%% code
import os, sys, json, time, copy, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch._dynamo; torch._dynamo.config.suppress_errors = True

GPU_CAP_GB = float(os.environ.get("BMO_GPU_CAP_GB", "10"))
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(0.95, GPU_CAP_GB/total), 0)
    print(f"device {torch.cuda.get_device_name(0)} | VRAM cap {GPU_CAP_GB:.0f} GB")

SEEDS = [int(x) for x in os.environ.get("BMO_SEEDS", "42,1337,2024,7,99").split(",")]
if os.environ.get("BMO_SMOKE") == "1":
    SEEDS = SEEDS[:2]
    C.OUT_BERT = C.OUT_BERT.parent / "bert_models_smoke"
    C.OUT_BERT.mkdir(parents=True, exist_ok=True)
print("seeds:", SEEDS)
#%% md
## Recover each model's selected configuration

Read from the completed run rather than re-derived, so this measures the
variability of the *chosen* models and cannot accidentally re-select anything.
#%% code
CKPT = {
    "BERT-base": "bert-base-uncased",
    "BioBERT": "dmis-lab/biobert-v1.1",
    "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "Bio+ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BioLinkBERT": "michiyasunaga/BioLinkBERT-base",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "DeBERTa-v3": "microsoft/deberta-v3-base",
    "ModernBERT": "answerdotai/ModernBERT-base",
    "BiomedBERT-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
}
src = sorted((C.OUT_BERT.parent / "bert_models").glob("test_results_shard*.csv"))
cfg_df = pd.concat([pd.read_csv(f) for f in src], ignore_index=True)
need = ["Model", "lr", "batch_size", "epochs", "dropout", "weight_decay", "errors"]
CONFIGS = {r["Model"]: r for _, r in cfg_df.iterrows() if r["Model"] in CKPT}
print(cfg_df[[c for c in need if c in cfg_df.columns]].round(5).to_string(index=False))
print(f"\n{len(CONFIGS)} configurations recovered")
#%% code
MAX_LEN, WARMUP_RATIO, LR_LAYER_DECAY = 128, 0.1, 0.95
df, train_idx, test_idx = C.load_data()
texts_all = df["clean_freeform"].fillna("").values
labels_all = df["has_biomarker"].values
texts, labels = texts_all[train_idx], labels_all[train_idx]
texts_te, labels_te = texts_all[test_idx], labels_all[test_idx]
print(f"train {len(texts):,} | test {len(texts_te):,}")

class DS(Dataset):
    def __init__(self, t, y, tok): self.t, self.y, self.tok = t, y, tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), max_length=MAX_LEN, truncation=True,
                     padding="max_length", return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(int(self.y[i]), dtype=torch.long)}

class Clf(nn.Module):
    def __init__(self, name, dropout):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        if hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False       # needs CPython headers this node lacks
        self.bert = AutoModel.from_pretrained(name, config=cfg)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.bert.config.hidden_size, 2))
    def forward(self, ids, mask):
        return self.head(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])

def layers_of(b):
    for path in (("encoder", "layer"), ("layers",), ("encoder", "layers")):
        o = b
        for a in path:
            o = getattr(o, a, None)
            if o is None: break
        if o is not None and hasattr(o, "__len__") and len(o): return o
    raise AttributeError(type(b).__name__)

def make_opt(m, lr, wd):
    nd = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = []
    def add(named, l):
        named = list(named)
        d = [p for n, p in named if not any(k in n for k in nd) and p.requires_grad]
        x = [p for n, p in named if any(k in n for k in nd) and p.requires_grad]
        if d: groups.append({"params": d, "lr": l, "weight_decay": wd})
        if x: groups.append({"params": x, "lr": l, "weight_decay": 0.0})
    L = layers_of(m.bert); n = len(L)
    add(m.head.named_parameters(), lr)
    for i in range(n - 1, -1, -1):
        add(L[i].named_parameters(), lr * (LR_LAYER_DECAY ** (n - i)))
    emb = getattr(m.bert, "embeddings", None)
    if emb is not None: add(emb.named_parameters(), lr * (LR_LAYER_DECAY ** (n + 1)))
    return AdamW(groups)
#%% md
## Refit each model once per seed

Only the seed changes. Configuration, data, split and epoch budget are fixed, so
the spread of results isolates training stochasticity.
#%% code
def fit_and_score(name, ckpt, lr, bs, epochs, dropout, wd, seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = Clf(ckpt, dropout).to(DEVICE)
    tl = DataLoader(DS(texts, labels, tok), batch_size=bs, shuffle=True,
                    num_workers=2, pin_memory=True,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(DS(texts_te, labels_te, tok), batch_size=max(bs, 64),
                    shuffle=False, num_workers=2, pin_memory=True)
    opt = make_opt(m, lr, wd)
    steps = max(1, len(tl)) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * steps), steps)
    neg, pos = int((labels == 0).sum()), int((labels == 1).sum())
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg/max(pos,1)],
                                                   dtype=torch.float, device=DEVICE))
    m.train()
    for _ in range(epochs):
        for b in tl:
            opt.zero_grad(set_to_none=True)
            loss = crit(m(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)),
                        b["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
    m.eval(); preds, probs = [], []
    with torch.no_grad():
        for b in vl:
            p = torch.softmax(m(b["input_ids"].to(DEVICE),
                                b["attention_mask"].to(DEVICE)).float(), 1)[:, 1]
            probs.extend(p.cpu().tolist()); preds.extend(p.ge(0.5).long().cpu().tolist())
    del m; torch.cuda.empty_cache()
    return np.array(preds), np.array(probs)

SHARD = os.environ.get("BMO_SHARD", "all")
models = list(CONFIGS)
if SHARD != "all":
    k, n = int(SHARD), int(os.environ.get("BMO_NSHARD", "2"))
    models = [m for i, m in enumerate(models) if i % n == k]
print("this process handles:", models)

rows = []
for name in models:
    c = CONFIGS[name]
    lr, bs = float(c["lr"]), int(c["batch_size"])
    ep = int(c.get("epochs", 4) or 4)
    dp = float(c.get("dropout", 0.3) or 0.3)
    wd = float(c.get("weight_decay", 0.01) or 0.01)
    print(f"\n{'='*70}\n{name}  lr={lr:g} bs={bs} epochs={ep} dropout={dp} wd={wd}\n{'='*70}")
    for sd in SEEDS:
        t0 = time.time()
        yp, pp = fit_and_score(name, CKPT[name], lr, bs, ep, dp, wd, sd)
        r = C.evaluate(labels_te, yp, pp, name=name, tier="Transformer",
                       extra={"seed": sd, "seconds": round(time.time()-t0, 1)})
        rows.append(r)
        print(f"  seed {sd:<6d} weighted F1 {r['weighted_f1']:.4f} | "
              f"errors {r['errors']:<3d} | FP {r['FP']} FN {r['FN']} | {r['seconds']:.0f}s")

seed_df = pd.DataFrame(rows)
C.save_table(seed_df, C.OUT_BERT / f"seed_stability_shard{SHARD}.csv")
#%% md
## Spread across seeds

`errors_sd` is the number that answers a reviewer asking how reproducible the
ranking is. If it approaches the gap between adjacent systems in the main table,
the ordering between them is not a real distinction.
#%% code
if len(seed_df):
    agg = (seed_df.groupby("Model")
           .agg(n_seeds=("seed", "size"),
                errors_mean=("errors", "mean"), errors_sd=("errors", "std"),
                errors_min=("errors", "min"), errors_max=("errors", "max"),
                wf1_mean=("weighted_f1", "mean"), wf1_sd=("weighted_f1", "std"))
           .round(4).sort_values("errors_mean").reset_index())
    print(agg.to_string(index=False))
    C.save_table(agg, C.OUT_BERT / f"seed_stability_summary_shard{SHARD}.csv")
    print(f"\nmean spread in errors across seeds: {agg['errors_sd'].mean():.2f}")
    print(f"largest single-model range: "
          f"{int((agg['errors_max'] - agg['errors_min']).max())} errors")
    print("\nInterpretation: any two systems whose mean error counts differ by less "
          "than roughly this spread cannot be ordered from a single run.")
