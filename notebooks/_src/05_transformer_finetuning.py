#%% md
# 05 — Fine-tuning the transformer tier

Nine encoder models, fine-tuned on the BMO corpus under one identical protocol.

| Short name | Checkpoint | Domain | Layers | Hidden | Status |
|---|---|---|---|---|---|
| BERT-base | `bert-base-uncased` | general | 12 | 768 | existing |
| BioBERT | `dmis-lab/biobert-v1.1` | biomedical | 12 | 768 | existing |
| PubMedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | biomedical | 12 | 768 | existing |
| Bio+ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | clinical | 12 | 768 | existing |
| BioLinkBERT | `michiyasunaga/BioLinkBERT-base` | biomedical | 12 | 768 | existing |
| **SciBERT** | `allenai/scibert_scivocab_uncased` | scientific | 12 | 768 | **new** |
| **DeBERTa-v3** | `microsoft/deberta-v3-base` | general | 12 | 768 | **new** |
| **ModernBERT** | `answerdotai/ModernBERT-base` | general | 22 | 768 | **new** |
| **BiomedBERT-large** | `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` | biomedical | 24 | 1024 | **new** |

## Why these four additions, and not four more biomedical BERT-base variants

The five original models produce *identical* confusion matrices on the test
set. Adding a sixth biomedical BERT-base would tie with them and add a row
without adding an argument. Each addition here answers a question the original
design could not:

- **ModernBERT** — the obvious reviewer question, "did you try a current
  architecture?". Released December 2024; a genuinely different encoder design
  (RoPE, GeGLU, alternating local/global attention), not a re-pretrained BERT.
- **DeBERTa-v3** — the strongest general-domain encoder of the previous
  generation. Sharpens the paper's central contrast: if a *general* model with
  a better architecture still loses to a *biomedical* model with a worse one,
  that is direct evidence for the representation-over-architecture claim.
- **BiomedBERT-large** — a scale control. It shares PubMedBERT's pretraining
  corpus at roughly 3× the parameters, so the comparison isolates model scale
  from domain vocabulary. The original design confounded the two and could not
  separate them.
- **SciBERT** — scientific rather than strictly biomedical vocabulary; a
  standard comparator in this literature that reviewers look for.

One naming trap worth recording: **PubMedBERT and BiomedBERT are the same
model.** Microsoft renamed the checkpoint. `BiomedBERT-large` here is the
*large* variant, not a rebrand of the base model already in the table.

## Protocol (unchanged from the original study, deliberately)

The existing protocol was already sound and is preserved exactly:

- Tuning holdout: `GroupShuffleSplit(test_size=0.2, random_state=99)` on train only.
- Cross-validation: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- Both are **trial-grouped**: all outcome measures from one `NCTId` stay on one
  side of every split. This matters because a single trial contributes many
  near-identically phrased outcomes.
- The test set is never touched until the final cell.

## What changed, and why

1. **Wider hyperparameter search.** The original searched three learning rates.
   Here we search learning rate × batch size, with a per-model learning-rate
   range, because ModernBERT is documented to need a substantially higher rate
   than BERT and would otherwise be evaluated in an underfit state — which would
   look like a fair comparison but would not be one.
2. **Architecture-agnostic layer-wise LR decay.** The original addressed
   `model.bert.encoder.layer[i]`, which exists on BERT but not on ModernBERT
   (`model.layers[i]`). Left unfixed, decay would silently not apply to the one
   model it was most likely to matter for.
3. **Larger batches.** Batch size 4 at 128 tokens uses a few percent of an L40.
   Batch size is now searched rather than fixed, which is both faster and fairer.
4. **Recorded diagnostics.** Per-epoch train/validation loss and F1, the
   selected epoch, the train-validation gap, and the number of configurations
   tried per model — so the paper's overfitting claim is checkable.

Sequence length stays at **128 tokens** for every model, including ModernBERT
(which supports 8192). Outcome measures are short, and holding the budget
constant keeps the comparison about the model rather than the context window.
#%% code
import os, sys, json, time, copy, math, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

# This node has no CPython development headers, so Triton cannot build its
# helper extension and any torch.compile path fails hard. Fall back to eager
# rather than aborting: eager execution is numerically identical, just slower.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
# Hard VRAM cap.
#
# These GPUs are shared three ways. The budget on a 46 GB L40 is:
#     ollama (llama3.3:70b, resident)   ~23 GB
#     evening speech transcription      ~12 GB
#     this study                        <=10 GB   <- enforced here
#                                        -------
#                                         45 GB of 46 GB
#
# set_per_process_memory_fraction makes the cap a property of the process
# rather than a promise in a comment: exceeding it raises CUDA OOM inside this
# notebook instead of silently taking memory another job depends on. The
# training loop below catches that and halves the batch size, so the cap
# degrades throughput rather than breaking the run.
# ---------------------------------------------------------------------------
GPU_CAP_GB = float(os.environ.get("BMO_GPU_CAP_GB", "10"))
if torch.cuda.is_available():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    frac = min(0.95, GPU_CAP_GB / total_gb)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"VRAM cap: {GPU_CAP_GB:.0f} GB of {total_gb:.0f} GB "
          f"(fraction {frac:.3f}) — leaves the rest for other users")
#%% md
### GPU sharding

Both L40s are used by running this notebook twice, once per shard, with
`CUDA_VISIBLE_DEVICES` and `BMO_SHARD` set by the driver script. Each process
sees exactly one GPU, so there is no cross-process contention and no
distributed-training complexity — the models are small and independent, which
makes model-level parallelism both simpler and more efficient than data
parallelism here.

`BMO_SHARD=all` (the default) runs every model in one process, which is what
you want when running the notebook interactively.
#%% code
BERT_MODELS = {
    "BERT-base":        "bert-base-uncased",
    "BioBERT":          "dmis-lab/biobert-v1.1",
    "PubMedBERT":       "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "Bio+ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BioLinkBERT":      "michiyasunaga/BioLinkBERT-base",
    "SciBERT":          "allenai/scibert_scivocab_uncased",
    "DeBERTa-v3":       "microsoft/deberta-v3-base",
    "ModernBERT":       "answerdotai/ModernBERT-base",
    "BiomedBERT-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
}

SHARD = os.environ.get("BMO_SHARD", "all")
if SHARD != "all":
    names = list(BERT_MODELS)
    k, n = int(SHARD), int(os.environ.get("BMO_NSHARD", "2"))
    BERT_MODELS = {m: BERT_MODELS[m] for i, m in enumerate(names) if i % n == k}
print(f"shard={SHARD}  models this process: {list(BERT_MODELS)}")
#%% md
### Hyperparameter grid

Learning-rate ranges are per-architecture, which is a fairness requirement
rather than a convenience. BERT-family encoders fine-tune at 1e-5–3e-5;
ModernBERT's authors report needing roughly 5e-5–8e-5, and large models
generally want a lower rate than their base counterparts. Searching one grid
for all of them would systematically disadvantage whichever model the grid
does not suit.

Every configuration is scored on the **inner tuning holdout only**. The CV
folds never influence the choice.
#%% code
MAX_LEN, MAX_EPOCHS, PATIENCE = 128, 12, 4
DROPOUT, WEIGHT_DECAY, WARMUP_RATIO, LR_LAYER_DECAY = 0.3, 0.01, 0.1, 0.95
N_FOLDS = C.N_FOLDS

LR_GRID = {
    "default":          [1e-5, 2e-5, 3e-5, 5e-5],
    "ModernBERT":       [3e-5, 5e-5, 8e-5],
    "BiomedBERT-large": [5e-6, 1e-5, 2e-5],
    "DeBERTa-v3":       [1e-5, 2e-5, 3e-5],
}
BATCH_GRID = {
    "default":          [16, 32],
    "BiomedBERT-large": [8, 16],
}

# ---------------------------------------------------------------------------
# Stage 2: regularisation search (BMO_STAGE2=1)
#
# Stage 1 searched learning rate and batch size only, 6-8 configurations per
# model, while the classical tier searched 48-768. That asymmetry matters: if a
# classical model outscores a transformer, part of the difference could be
# search budget rather than merit. Stage 2 closes it by searching the two
# knobs that actually govern generalisation on a corpus this small.
#
# It also exists to fix a specific finding: ModernBERT was the only model to
# exceed the 0.04 train-validation gap (0.052, flagged WATCH) *and* was tied
# for worst. Those two facts point at under-regularisation rather than a bad
# architecture, and dropout / weight decay are the levers for it.
#
# SELECTION RULE -- this is the important part.
# Configurations are ranked by validation F1 *among those whose train-validation
# gap is below the 0.04 threshold*. A configuration that scores higher by
# overfitting harder does not win. If no configuration passes the threshold, the
# smallest-gap one is taken and the model is flagged in the output, so the
# manuscript reports an honest failure rather than a flattering number.
# Everything is measured on the inner holdout; the test set is untouched.
# ---------------------------------------------------------------------------
STAGE2       = os.environ.get("BMO_STAGE2") == "1"
DROPOUT_GRID = [0.1, 0.3, 0.5]
WD_GRID      = [0.01, 0.1]
GAP_THRESHOLD = 0.04

# Smoke mode: one config, two epochs, one fold. Exercises every path end to end
# in a couple of minutes so runtime errors surface before an hour of GPU time is
# committed. Output goes to a separate directory and is never mixed with real
# results.
if os.environ.get("BMO_SMOKE") == "1":
    DROPOUT_GRID, WD_GRID = [0.3], [0.01]
    LR_GRID = {"default": [2e-5]}
    BATCH_GRID = {"default": [16]}
    MAX_EPOCHS, PATIENCE, N_FOLDS = 2, 1, 2
    import shutil as _sh
    _real = C.OUT_BERT
    C.OUT_BERT = C.OUT_BERT.parent / "bert_models_smoke"
    C.OUT_BERT.mkdir(parents=True, exist_ok=True)
    C.MODELS = C.MODELS / "smoke"
    # Copy the stage-1 caches across, otherwise the stage-2 lookup misses and
    # the smoke run silently exercises the wrong code path -- testing nothing.
    for _f in _real.glob("hyperparameter_search_shard*.csv"):
        _sh.copy2(_f, C.OUT_BERT / _f.name)
    print(f"SMOKE MODE: 1 config, {MAX_EPOCHS} epochs, {N_FOLDS} folds -> {C.OUT_BERT}")
    print(f"  stage-1 caches copied: {[f.name for f in C.OUT_BERT.glob('hyper*')]}")
def grid_for(name):
    lrs = LR_GRID.get(name, LR_GRID["default"])
    bss = BATCH_GRID.get(name, BATCH_GRID["default"])
    return [(lr, bs) for lr in lrs for bs in bss]

for m in BERT_MODELS:
    g = grid_for(m)
    print(f"{m:18s} {len(g):2d} configs  lr={sorted({c[0] for c in g})}  bs={sorted({c[1] for c in g})}")
#%% md
### Model, data and the architecture-agnostic optimiser

The classifier head is deliberately minimal — one dropout, one linear layer.
On ~1,500 training records a deeper head memorises rather than generalises.

`transformer_layers()` below is the fix for the layer-decay bug: it locates the
list of encoder blocks by trying the known attribute paths in turn, so the same
decay schedule applies to BERT-family models, DeBERTa and ModernBERT alike. If
it cannot find them it raises rather than silently skipping decay, because a
silent skip would produce a plausible-looking but differently-regularised model.
#%% code
class BiomarkerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(self.labels[i]), dtype=torch.long)}

class BERTClassifier(nn.Module):
    def __init__(self, model_name, dropout=DROPOUT):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        # ModernBERT's HuggingFace implementation compiles its MLP with
        # torch.compile by default (`reference_compile=True`). That routes
        # through Triton, which shells out to gcc and needs the CPython
        # development headers -- absent on this node, and not installable
        # without root. Left enabled it raises BackendCompilerFailed and takes
        # the whole model down. Disabling it costs a little throughput and
        # changes no numerics: it is a compilation strategy, not a modelling
        # choice, so the comparison against the other encoders stays fair.
        if hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False
        self.bert = AutoModel.from_pretrained(model_name, config=cfg)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.bert.config.hidden_size, 2))
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])   # [CLS]

def transformer_layers(bert):
    """Return the ModuleList of encoder blocks, whatever the architecture calls it.

    BERT / SciBERT / BioBERT / DeBERTa -> bert.encoder.layer
    ModernBERT                         -> bert.layers
    Raises rather than returning None: silently skipping layer-wise decay would
    change the effective regularisation without any visible signal.
    """
    for path in (("encoder", "layer"), ("layers",), ("encoder", "layers"),
                 ("transformer", "layer")):
        obj = bert
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise AttributeError(f"cannot locate encoder blocks on {type(bert).__name__}")

def embedding_module(bert):
    for attr in ("embeddings", "embed_tokens", "word_embeddings"):
        m = getattr(bert, attr, None)
        if m is not None:
            return m
    raise AttributeError(f"cannot locate embeddings on {type(bert).__name__}")

def make_optimizer(model, base_lr, weight_decay=WEIGHT_DECAY):
    """Layer-wise LR decay + correct AdamW weight-decay exclusions."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = []
    def add(named, lr):
        named = list(named)
        dec = [p for n, p in named if not any(k in n for k in no_decay) and p.requires_grad]
        nod = [p for n, p in named if any(k in n for k in no_decay) and p.requires_grad]
        if dec: groups.append({"params": dec, "lr": lr, "weight_decay": weight_decay})
        if nod: groups.append({"params": nod, "lr": lr, "weight_decay": 0.0})

    layers = transformer_layers(model.bert)
    n = len(layers)
    add(model.head.named_parameters(), base_lr)                    # head: full LR
    for i in range(n - 1, -1, -1):                                 # top -> bottom
        add(layers[i].named_parameters(), base_lr * (LR_LAYER_DECAY ** (n - i)))
    add(embedding_module(model.bert).named_parameters(),
        base_lr * (LR_LAYER_DECAY ** (n + 1)))                     # embeddings: lowest
    return AdamW(groups)

def class_weights(labels):
    """Inverse-frequency weights for the ~73/27 imbalance (no resampling)."""
    neg, pos = int((labels == 0).sum()), int((labels == 1).sum())
    return torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float, device=DEVICE)
#%% code
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None):
    """One pass. Training when optimizer is given, evaluation otherwise."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total, preds, trues, probs = 0.0, [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            ids  = b["input_ids"].to(DEVICE, non_blocking=True)
            mask = b["attention_mask"].to(DEVICE, non_blocking=True)
            y    = b["label"].to(DEVICE, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
            total += loss.item() * len(y)
            p = torch.softmax(logits.float(), dim=1)[:, 1]
            preds.extend(p.ge(0.5).long().cpu().tolist())
            probs.extend(p.cpu().tolist()); trues.extend(y.cpu().tolist())
    return (total / max(len(trues), 1),
            f1_score(trues, preds, average="weighted", zero_division=0),
            np.array(trues), np.array(preds), np.array(probs))

def fit(model_name, tr_texts, tr_y, va_texts, va_y, lr, bs, tokenizer,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, track_history=True,
        dropout=DROPOUT, weight_decay=WEIGHT_DECAY):
    """Fine-tune once with early stopping on validation weighted F1.

    Retries at a halved batch size if the VRAM cap is hit, so the cap costs
    throughput rather than failing the run. The learning rate is left unchanged
    on retry: altering it would silently evaluate a different configuration
    from the one the search selected.

    Returns the best-epoch state, the validation predictions at that epoch, and
    the full per-epoch history used for the overfitting figures.
    """
    while True:
        try:
            return _fit_once(model_name, tr_texts, tr_y, va_texts, va_y, lr, bs,
                             tokenizer, max_epochs, patience, track_history,
                             dropout, weight_decay)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 2:
                raise
            bs = max(2, bs // 2)
            print(f"    VRAM cap hit -> retrying at batch size {bs}")

def _fit_once(model_name, tr_texts, tr_y, va_texts, va_y, lr, bs, tokenizer,
              max_epochs, patience, track_history, dropout=DROPOUT,
              weight_decay=WEIGHT_DECAY):
    model = BERTClassifier(model_name, dropout=dropout).to(DEVICE)
    tr_loader = DataLoader(BiomarkerDataset(tr_texts, tr_y, tokenizer),
                           batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    # Separate loader for *scoring* the training set. Metrics collected during
    # the training pass are not a valid measure of fit: dropout is active and
    # the weights change between the first batch and the last, so the resulting
    # F1 understates true training performance. Since the train-minus-validation
    # gap is the paper's evidence for absence of overfitting, biasing it
    # downwards would make every model look better regularised than it is.
    # This loader re-scores the same data in eval() mode with frozen weights.
    tr_eval_loader = DataLoader(BiomarkerDataset(tr_texts, tr_y, tokenizer),
                                batch_size=max(bs, 64), shuffle=False,
                                num_workers=2, pin_memory=True)
    va_loader = DataLoader(BiomarkerDataset(va_texts, va_y, tokenizer),
                           batch_size=max(bs, 64), shuffle=False, num_workers=2, pin_memory=True)
    opt = make_optimizer(model, lr, weight_decay)
    steps = max(1, len(tr_loader)) * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * steps), steps)
    crit = nn.CrossEntropyLoss(weight=class_weights(np.asarray(tr_y)))

    hist = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best = {"f1": -1.0, "epoch": 0, "state": None, "val": None, "train_f1": 0.0}
    bad = 0
    for ep in range(1, max_epochs + 1):
        run_epoch(model, tr_loader, crit, opt, sched)          # the actual update pass
        trl, trf, *_ = run_epoch(model, tr_eval_loader, crit)  # clean train metrics
        val, vaf, yt, yp, pr = run_epoch(model, va_loader, crit)
        if track_history:
            hist["train_loss"].append(trl); hist["val_loss"].append(val)
            hist["train_f1"].append(trf);   hist["val_f1"].append(vaf)
        if vaf > best["f1"] + 1e-6:
            best.update(f1=vaf, epoch=ep, train_f1=trf,
                        state=copy.deepcopy(model.state_dict()),
                        val=(yt, yp, pr))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    hist["best_epoch"] = best["epoch"]
    del model; torch.cuda.empty_cache()
    return best, hist
#%% md
## Load data and build the two splits

`texts`/`labels` below are the **training portion only**. The test portion is
held in separate variables that no tuning or CV code touches.
#%% code
df, train_idx, test_idx = C.load_data()
texts_all  = df["clean_freeform"].fillna("").values
labels_all = df["has_biomarker"].values
groups_all = df["NCTId"].values

texts,  labels,  groups  = texts_all[train_idx], labels_all[train_idx], groups_all[train_idx]
texts_te, labels_te      = texts_all[test_idx],  labels_all[test_idx]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED_TUNE)
tune_tr, tune_va = next(gss.split(texts, labels, groups=groups))
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED_CV)

print(f"train {len(texts):,} | test {len(texts_te):,}")
print(f"tuning holdout: fit {len(tune_tr):,} / score {len(tune_va):,} (seed {C.SEED_TUNE})")
assert not (set(groups[tune_tr]) & set(groups[tune_va])), "tuning holdout leaks trials"
print("tuning holdout is trial-disjoint: OK")
#%% md
## Phase 1 — hyperparameter search on the inner holdout

Every configuration is fit on 80% of the training set and scored on the
remaining 20%, grouped by trial. The winner per model is carried into Phase 2.
Nothing here sees a CV fold or the test set.
#%% code
# If a completed Phase 1 already exists for this shard, reuse it. Hyperparameter
# selection depends only on validation scores, which the clean-train-metric fix
# does not change, so re-searching would burn GPU hours to arrive at exactly the
# same configurations.
# Read EVERY shard's stage-1 cache, not just this shard's. A model's stage-1
# winner is a property of the model, not of which process happened to fit it,
# and reading only the local file silently falls back to a full re-search
# whenever the shard split differs between runs.
cache_files = sorted(C.OUT_BERT.glob("hyperparameter_search_shard*.csv"))
cached = None
if cache_files:
    d = pd.concat([pd.read_csv(f) for f in cache_files], ignore_index=True)
    if set(BERT_MODELS) <= set(d["Model"]):
        cached = d
        print(f"reusing cached Phase 1 from {[f.name for f in cache_files]} "
              f"({len(d)} configurations across {d['Model'].nunique()} models)")
    else:
        missing = sorted(set(BERT_MODELS) - set(d["Model"]))
        print(f"cache incomplete (missing {missing}) — re-running the full search")
else:
    print("no stage-1 cache found — running the full search")

tuning_rows, best_config = [], {}
t_start = time.time()

if cached is not None and STAGE2:
    # Stage 2: hold the stage-1 winner's lr/batch, search regularisation.
    print(f"STAGE 2: regularisation search over dropout={DROPOUT_GRID} x "
          f"weight_decay={WD_GRID}, selecting the best configuration with "
          f"train-val gap < {GAP_THRESHOLD}")
    stage2_rows = []
    for name, ckpt in BERT_MODELS.items():
        sub = cached[cached["Model"] == name]
        base = sub.loc[sub["val_weighted_f1"].idxmax()].to_dict()
        lr, bs = float(base["lr"]), int(base["batch_size"])
        tok = AutoTokenizer.from_pretrained(ckpt)
        print(f"\n{'='*72}\n{name}  (lr={lr:g}, bs={bs} from stage 1)\n{'='*72}")
        cand = []
        for dp in DROPOUT_GRID:
            for wd in WD_GRID:
                t0 = time.time()
                best, _ = fit(ckpt, texts[tune_tr], labels[tune_tr],
                              texts[tune_va], labels[tune_va], lr, bs, tok,
                              track_history=False, dropout=dp, weight_decay=wd)
                gap = best["train_f1"] - best["f1"]
                r = {"Model": name, "lr": lr, "batch_size": bs, "dropout": dp,
                     "weight_decay": wd, "val_weighted_f1": best["f1"],
                     "train_weighted_f1": best["train_f1"], "gap": gap,
                     "best_epoch": best["epoch"], "seconds": round(time.time()-t0, 1),
                     "passes_gap": bool(gap < GAP_THRESHOLD)}
                cand.append(r); stage2_rows.append(r)
                print(f"  dropout={dp:<4} wd={wd:<5} val F1 {best['f1']:.4f} "
                      f"gap {gap:+.4f} {'OK' if gap < GAP_THRESHOLD else 'over'} "
                      f"({r['seconds']:.0f}s)")
        ok = [c for c in cand if c["passes_gap"]]
        if ok:
            pick = max(ok, key=lambda c: c["val_weighted_f1"])
            note = f"best of {len(ok)}/{len(cand)} configs within the gap threshold"
        else:
            pick = min(cand, key=lambda c: c["gap"])
            note = ("NO configuration met the gap threshold; taking the smallest "
                    "gap and flagging this model")
        pick["selection_note"] = note
        best_config[name] = pick
        print(f"  -> dropout={pick['dropout']}, wd={pick['weight_decay']}, "
              f"val F1 {pick['val_weighted_f1']:.4f}, gap {pick['gap']:+.4f}")
        print(f"     {note}")
    tuning_rows = stage2_rows
    C.save_table(pd.DataFrame(stage2_rows),
                 C.OUT_BERT / f"stage2_regularisation_shard{SHARD}.csv")

elif cached is not None:
    tuning_rows = cached.to_dict("records")
    for name in BERT_MODELS:
        sub = cached[cached["Model"] == name]
        row = sub.loc[sub["val_weighted_f1"].idxmax()].to_dict()
        # CSV round-trip turns these into numpy scalars; cast back so they are
        # plain Python values wherever they reach DataLoader or get serialised.
        row["lr"] = float(row["lr"])
        row["batch_size"] = int(row["batch_size"])
        row["best_epoch"] = int(row["best_epoch"])
        best_config[name] = row
        print(f"  {name:18s} lr={best_config[name]['lr']:<8g} "
              f"bs={int(best_config[name]['batch_size']):<3d} "
              f"val F1 {best_config[name]['val_weighted_f1']:.4f}")

for name, ckpt in ({} if cached is not None else BERT_MODELS).items():
    tok = AutoTokenizer.from_pretrained(ckpt)
    grid = grid_for(name)
    print(f"\n{'='*72}\n{name}  ({ckpt})\n  {len(grid)} configurations\n{'='*72}")
    best_here = None
    for lr, bs in grid:
        t0 = time.time()
        best, _ = fit(ckpt, texts[tune_tr], labels[tune_tr],
                      texts[tune_va], labels[tune_va], lr, bs, tok,
                      track_history=False)
        row = {"Model": name, "lr": lr, "batch_size": bs,
               "val_weighted_f1": best["f1"], "train_weighted_f1": best["train_f1"],
               "gap": best["train_f1"] - best["f1"],
               "best_epoch": best["epoch"], "seconds": round(time.time() - t0, 1)}
        tuning_rows.append(row)
        print(f"  lr={lr:<7g} bs={bs:<3d} val_F1={best['f1']:.4f} "
              f"train_F1={best['train_f1']:.4f} gap={row['gap']:+.4f} "
              f"ep={best['epoch']:<2d} {row['seconds']:.0f}s")
        if best_here is None or best["f1"] > best_here["val_weighted_f1"]:
            best_here = row
    best_config[name] = best_here
    print(f"  -> selected lr={best_here['lr']:g} bs={best_here['batch_size']} "
          f"(val F1 {best_here['val_weighted_f1']:.4f}, {len(grid)} configs tried)")

tuning_df = pd.DataFrame(tuning_rows)
C.save_table(tuning_df, C.OUT_BERT / f"hyperparameter_search_shard{SHARD}.csv")
print(f"\nPhase 1 total: {(time.time()-t_start)/60:.1f} min")
#%% md
### Tuning budget

The number of configurations evaluated per model, reported so that "we tuned
carefully" is a checkable claim. A model that wins after 8 configurations while
another had 6 has had more chances; stating the budget lets a reader judge
whether the margin is meaningful.
#%% code
budget = (tuning_df.groupby("Model")
          .agg(configs_tried=("val_weighted_f1", "size"),
               best_val_f1=("val_weighted_f1", "max"),
               worst_val_f1=("val_weighted_f1", "min"),
               search_minutes=("seconds", lambda s: round(s.sum() / 60, 1)))
          .reset_index()
          .sort_values("best_val_f1", ascending=False))
budget["spread"] = (budget["best_val_f1"] - budget["worst_val_f1"]).round(4)
print(budget.to_string(index=False))
C.save_table(budget, C.OUT_BERT / f"tuning_budget_shard{SHARD}.csv")
#%% md
`spread` is worth reading carefully. A model with a large spread was sensitive
to hyperparameters, meaning its reported score depends heavily on the search
having found the right corner. A model with near-zero spread is insensitive,
and its score is correspondingly more trustworthy. This is the kind of detail
that belongs in the manuscript's methods discussion.
#%% md
## Phase 2 — 5-fold trial-grouped cross-validation

The selected configuration is refit across five trial-disjoint folds. This is
the number the paper reports as the honest performance estimate. Out-of-fold
predictions are retained for the McNemar tests in notebook 08.
#%% code
# ---------------------------------------------------------------------------
# Nested selection (BMO_NESTED=1)
#
# Fixes an inconsistency in the stage-2 design: regularisation was chosen using
# the gap measured on ONE holdout split, while the manuscript reports the gap
# measured across FIVE cross-validation folds. Those are different estimators of
# the same quantity, so constraining the first does not constrain the second --
# ModernBERT was selected at a holdout gap of 0.032 and then reported a CV gap
# of 0.046, still above threshold.
#
# Under nested selection the regularisation search runs *inside* each outer
# fold, on an inner split of that fold's training portion. The outer fold is
# never used to choose anything, so the cross-validated score and gap are honest
# estimates of the whole procedure -- selection included. This is exactly what
# the classical tier already does (outer 5-fold, inner 3-fold GridSearchCV), so
# it also makes the two tiers methodologically comparable, which matters because
# the paper's central claim is a comparison between them.
#
# Cost: 5 outer folds x 6 configurations + 5 refits per model, against 5 fits
# before. The configuration chosen per fold is recorded, because disagreement
# between folds is itself worth reporting -- it measures how stable the
# regularisation choice actually is.
# ---------------------------------------------------------------------------
NESTED = os.environ.get("BMO_NESTED") == "1"
if NESTED:
    print(f"NESTED SELECTION: dropout={DROPOUT_GRID} x weight_decay={WD_GRID} "
          f"chosen independently inside each of the {N_FOLDS} outer folds")

cv_rows, oof_store, hist_store, fold_rows = [], {}, {}, []
nested_choices = []

for name, ckpt in BERT_MODELS.items():
    cfg = best_config[name]
    lr, bs = cfg["lr"], cfg["batch_size"]
    dp = cfg.get("dropout", DROPOUT); wd = cfg.get("weight_decay", WEIGHT_DECAY)
    tok = AutoTokenizer.from_pretrained(ckpt)
    oof_pred = np.zeros(len(texts), dtype=int)
    oof_prob = np.zeros(len(texts), dtype=float)
    per_fold, gaps = [], []
    print(f"\n{'='*72}\n{name}  CV  (lr={lr:g}, bs={bs})\n{'='*72}")

    for k, (tr_i, va_i) in enumerate(cv.split(texts, labels, groups=groups), 1):
        assert not (set(groups[tr_i]) & set(groups[va_i])), f"fold {k} leaks trials"
        t0 = time.time()

        if NESTED:
            # Choose regularisation using ONLY this fold's training portion.
            inner = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                      random_state=C.SEED_TUNE + k)
            i_tr, i_va = next(inner.split(texts[tr_i], labels[tr_i],
                                          groups=groups[tr_i]))
            abs_tr, abs_va = tr_i[i_tr], tr_i[i_va]
            assert not (set(groups[abs_tr]) & set(groups[abs_va])), "inner split leaks"
            assert not (set(groups[abs_tr]) & set(groups[va_i])), "inner train touches outer val"
            cands = []
            for dpc in DROPOUT_GRID:
                for wdc in WD_GRID:
                    b_in, _ = fit(ckpt, texts[abs_tr], labels[abs_tr],
                                  texts[abs_va], labels[abs_va], lr, bs, tok,
                                  track_history=False, dropout=dpc, weight_decay=wdc)
                    g = b_in["train_f1"] - b_in["f1"]
                    cands.append({"dropout": dpc, "weight_decay": wdc,
                                  "val": b_in["f1"], "gap": g,
                                  "ok": bool(g < GAP_THRESHOLD)})
            passing = [c for c in cands if c["ok"]]
            pick = (max(passing, key=lambda c: c["val"]) if passing
                    else min(cands, key=lambda c: c["gap"]))
            dp, wd = pick["dropout"], pick["weight_decay"]
            nested_choices.append({"Model": name, "fold": k, "dropout": dp,
                                   "weight_decay": wd, "inner_val_f1": pick["val"],
                                   "inner_gap": pick["gap"],
                                   "n_passing": len(passing), "n_tried": len(cands)})
            print(f"  fold {k}: selected dropout={dp}, wd={wd} "
                  f"(inner val {pick['val']:.4f}, inner gap {pick['gap']:+.4f}, "
                  f"{len(passing)}/{len(cands)} passed)")

        best, hist = fit(ckpt, texts[tr_i], labels[tr_i], texts[va_i], labels[va_i],
                         lr, bs, tok, dropout=dp, weight_decay=wd)
        yt, yp, pr = best["val"]
        oof_pred[va_i], oof_prob[va_i] = yp, pr
        per_fold.append(best["f1"]); gaps.append(best["train_f1"] - best["f1"])
        fold_rows.append({"Model": name, "fold": k, "val_weighted_f1": best["f1"],
                          "train_weighted_f1": best["train_f1"],
                          "gap": best["train_f1"] - best["f1"],
                          "best_epoch": best["epoch"], "dropout": dp,
                          "weight_decay": wd,
                          "seconds": round(time.time() - t0, 1)})
        if k == 1:
            hist_store[name] = hist       # fold 1 curves stand in for the figure
        print(f"  fold {k}: val F1 {best['f1']:.4f}  gap {gaps[-1]:+.4f}  "
              f"epoch {best['epoch']}  {time.time()-t0:.0f}s")

    oof_store[name] = (oof_pred, oof_prob)
    row = C.evaluate(labels, oof_pred, oof_prob, name=name, tier="Transformer",
                     extra={"cv_mean_f1": float(np.mean(per_fold)),
                            "cv_std_f1": float(np.std(per_fold)),
                            "mean_gap": float(np.mean(gaps)),
                            "lr": lr, "batch_size": bs,
                            "dropout": dp, "weight_decay": wd,
                            "configs_tried": int(cfg and len(grid_for(name)))})
    cv_rows.append(row)
    print(f"  OOF weighted F1 {row['weighted_f1']:.4f} | "
          f"CV {np.mean(per_fold):.4f} +/- {np.std(per_fold):.4f} | "
          f"mean gap {np.mean(gaps):+.4f}")

if nested_choices:
    nc = pd.DataFrame(nested_choices)
    C.save_table(nc, C.OUT_BERT / f"nested_choices_shard{SHARD}.csv")
    print("\nregularisation stability across folds "
          "(how often the folds agreed on a configuration):")
    for m, g in nc.groupby("Model"):
        combos = g.groupby(["dropout", "weight_decay"]).size().sort_values(ascending=False)
        top = combos.index[0]
        print(f"  {m:18s} modal choice dropout={top[0]}, wd={top[1]} "
              f"in {combos.iloc[0]}/{len(g)} folds "
              f"({len(combos)} distinct choice(s) across folds)")

cv_df = pd.DataFrame(cv_rows).sort_values("weighted_f1", ascending=False)
C.save_table(cv_df, C.OUT_BERT / f"cv_results_shard{SHARD}.csv")
C.save_table(pd.DataFrame(fold_rows), C.OUT_BERT / f"per_fold_shard{SHARD}.csv")
print(cv_df[["Model", "weighted_f1", "biomarker_f1", "cv_mean_f1", "cv_std_f1",
             "mean_gap", "errors"]].to_string(index=False))
#%% md
## Overfitting audit

The manuscript claims every language model kept its train-to-validation gap
below 0.04. This table is what makes that claim verifiable, per model, with the
tuning budget beside it. Any model flagged `WATCH` or `OVERFIT` must be
discussed in the text rather than quietly reported.
#%% code
audit = pd.DataFrame([
    C.overfit_report(r["cv_mean_f1"] + r["mean_gap"], r["cv_mean_f1"],
                     name=r["Model"], n_configs=r["configs_tried"])
    for _, r in cv_df.iterrows()
]).sort_values("gap")
print(audit.to_string(index=False))
C.save_table(audit, C.OUT_BERT / f"overfitting_audit_shard{SHARD}.csv")

flagged = audit[audit["status"] != "OK"]
if len(flagged):
    print("\nMODELS REQUIRING DISCUSSION IN THE MANUSCRIPT:")
    print(flagged.to_string(index=False))
else:
    print("\nAll models within the 0.04 gap threshold.")
#%% code
# Loss / F1 curves per model — the visual evidence behind the audit table.
for name, hist in hist_store.items():
    C.plot_curves(hist, f"{name}: fold 1 training dynamics",
                  C.FIGS / f"curves_{name.replace('+','plus').replace('/','_')}.png")
#%% md
## Persist out-of-fold predictions

Notebook 08 needs these for the pairwise McNemar tests. Storing predictions
rather than only metrics is what makes significance testing between 40 systems
possible after the fact.
#%% code
oof_pred_df = pd.DataFrame({m: v[0] for m, v in oof_store.items()})
oof_prob_df = pd.DataFrame({m: v[1] for m, v in oof_store.items()})
oof_pred_df.insert(0, "y_true", labels)
oof_prob_df.insert(0, "y_true", labels)
C.save_table(oof_pred_df, C.OUT_BERT / f"oof_predictions_shard{SHARD}.csv")
C.save_table(oof_prob_df, C.OUT_BERT / f"oof_probabilities_shard{SHARD}.csv")
#%% md
## Phase 3 — final fit and the single test-set evaluation

Each model is refit on the **entire** training set with its selected
configuration, using the median best-epoch count observed across the CV folds
(there is no validation set left to early-stop against, so the epoch budget is
inherited from cross-validation rather than chosen on test).

Weights are written to the NFS mount, and this is the first and only point at
which the test set is used.
#%% code
final_rows, test_pred_store = [], {}
MODELS_DIR = C.MODELS / "transformers"; MODELS_DIR.mkdir(parents=True, exist_ok=True)
fold_df = pd.DataFrame(fold_rows)

for name, ckpt in BERT_MODELS.items():
    cfg = best_config[name]
    lr, bs = cfg["lr"], cfg["batch_size"]
    dp = cfg.get("dropout", DROPOUT); wd = cfg.get("weight_decay", WEIGHT_DECAY)
    n_ep = int(np.median(fold_df.loc[fold_df["Model"] == name, "best_epoch"]))
    tok = AutoTokenizer.from_pretrained(ckpt)
    print(f"\n{name}: final fit on {len(texts):,} records, {n_ep} epochs "
          f"(median of CV folds), lr={lr:g}, bs={bs}")

    best, hist = fit(ckpt, texts, labels, texts_te, labels_te, lr, bs, tok,
                     max_epochs=n_ep, patience=n_ep, track_history=True,
                     dropout=dp, weight_decay=wd)
    yt, yp, pr = best["val"]          # <- the held-out test set, used once
    row = C.evaluate(labels_te, yp, pr, name=name, tier="Transformer",
                     extra={"lr": lr, "batch_size": bs, "epochs": n_ep,
                            "dropout": dp, "weight_decay": wd})
    final_rows.append(row); test_pred_store[name] = (yp, pr)
    print(f"  test weighted F1 {row['weighted_f1']:.4f} | "
          f"biomarker F1 {row['biomarker_f1']:.4f} | AUC {row.get('auc', float('nan')):.4f} "
          f"| errors {row['errors']}")

    d = MODELS_DIR / name.replace("+", "plus").replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], d / "model_state.pt")
    tok.save_pretrained(d / "tokenizer")
    AutoConfig.from_pretrained(ckpt).save_pretrained(d)
    (d / "training_config.json").write_text(json.dumps(
        {"checkpoint": ckpt, "lr": lr, "batch_size": bs, "epochs": n_ep,
         "max_len": MAX_LEN, "dropout": DROPOUT, "seed": C.SEED_CV,
         "test_weighted_f1": row["weighted_f1"],
         "test_biomarker_f1": row["biomarker_f1"]}, indent=2))
    C.plot_confusion(labels_te, yp, f"{name}: test set",
                     C.FIGS / f"cm_test_{name.replace('+','plus')}.png")

test_df = pd.DataFrame(final_rows).sort_values("weighted_f1", ascending=False)
C.save_table(test_df, C.OUT_BERT / f"test_results_shard{SHARD}.csv")
tp = pd.DataFrame({m: v[0] for m, v in test_pred_store.items()}); tp.insert(0, "y_true", labels_te)
pp = pd.DataFrame({m: v[1] for m, v in test_pred_store.items()}); pp.insert(0, "y_true", labels_te)
C.save_table(tp, C.OUT_BERT / f"test_predictions_shard{SHARD}.csv")
C.save_table(pp, C.OUT_BERT / f"test_probabilities_shard{SHARD}.csv")
print("\n" + test_df[["Model", "weighted_f1", "biomarker_f1", "auc",
                      "TN", "FP", "FN", "TP", "errors"]].to_string(index=False))
#%% md
## Reading these results honestly

Two cautions to carry into the manuscript:

**The test set has 341 records.** One record is worth roughly 0.003 weighted F1.
Differences of one or two errors between models are within noise, and the
McNemar tests in notebook 08 — not this ranking — decide what is real. Ranking
tables invite over-reading; the significance matrix exists to prevent it.

**A model that wins here after a larger search has had more opportunities to
win.** The tuning-budget table above is what lets a reader discount that. If the
top model's margin over the runner-up is smaller than its own hyperparameter
spread, the ranking between them should be described as a tie in the text.
#%% code
if len(test_df) > 1:
    top, second = test_df.iloc[0], test_df.iloc[1]
    spread = float(budget.loc[budget["Model"] == top["Model"], "spread"].iloc[0])
    margin = top["weighted_f1"] - second["weighted_f1"]
    print(f"top:     {top['Model']}  {top['weighted_f1']:.4f}  ({top['errors']} errors)")
    print(f"second:  {second['Model']}  {second['weighted_f1']:.4f}  ({second['errors']} errors)")
    print(f"margin:  {margin:.4f}   |   top model's tuning spread: {spread:.4f}")
    print("\nVERDICT:", "margin exceeds tuning spread — the ordering is defensible"
          if margin > spread else
          "margin is WITHIN the tuning spread — report these as tied, not ranked")
#%% md
## Done

Outputs written to `outputs/bert_models/` and weights to
`models/transformers/`. If this ran as two shards, notebook 07 merges the
`_shard0` / `_shard1` files before the combined evaluation.

**Next:** `06_prompt_based_llms.ipynb` — the prompt-based tier, including the
two modern instruct models that replace the paper's weakest claim.
