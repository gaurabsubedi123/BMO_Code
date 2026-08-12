# Results reference

What each committed table contains, and which file is authoritative for which number. All
files here are aggregate metrics — no trial text, no per-record labels, no `NCTId`s.

## Start here

**`results/MASTER_all_systems_poster_run.csv`** — all 33 systems, one row each, with
cross-validated and held-out test metrics side by side. Built by joining the per-tier
`results.csv` files to `all_test_metrics.csv`. Every number in the paper traces to this file
or to one of the three below.

| File | Contents |
|---|---|
| `all_test_metrics.csv` | Held-out test set: exact TN/FP/FN/TP plus derived metrics, all 33 systems |
| `inference_timing_all.csv` | Latency, throughput, load time and model size, 19 timed systems |
| `significance_OOF_31systems.csv` | McNemar, 465 pairs, out-of-fold (n=1,559) — **the powered analysis** |
| `significance_TEST_33systems.csv` | McNemar on the test set (n=341) — underpowered, see below |
| `significance_summary_all_systems.txt` | Human-readable summary of both |
| `cv_*_results.csv` | Per-tier cross-validation metrics as emitted by each pipeline step |

## Two caveats a reader will otherwise trip over

**1. `outputs/final_results/test_set_results.csv` is unreliable for the FastText and
BioWordVec tiers.** If you run the pipeline and step 06 executes before steps 02a/02b have
produced their models, those systems are evaluated before their embedding model is loaded, and
score against zero vectors — producing absurd values (SVM+BioWordVec at 0.59 weighted F1,
several at exactly 0.0 biomarker-class F1). The authoritative test-set numbers come from a
later re-evaluation and are what `all_test_metrics.csv` contains. **Run 02a and 02b before 06
and 07.**

**2. The test-set significance analysis is underpowered by construction.** The leading systems
make 3–6 errors each in 341 records, so discordant-pair counts are tiny and almost nothing can
reach significance. Absence of significance there is *not* evidence of equivalence. Cite
`significance_OOF_31systems.csv` (n=1,559).

## Coverage limits

- **Significance covers 31 systems, not 33.** The FastText and BioWordVec tier ensembles are
  constructed at test time and have no out-of-fold predictions, so they cannot enter the
  out-of-fold analysis. They appear in the 33-system test-set variant.
- **Timing covers 19 systems.** The 14 classical systems and the 5 transformers. Ensembles are
  not timed separately — their cost is the sum of their members'.
- **`per_fold_f1.csv` is missing its logistic-regression row** for the FastText and BioWordVec
  tiers in the original run (six of seven models present).

## Metric definitions

- **Weighted F1** — per-class F1 averaged by support. The primary metric, and the criterion
  used for hyperparameter selection, so tuning and reporting are consistent.
- **Biomarker-class F1** — F1 for the positive class alone. The class of interest for triage.
- **AUC** — threshold-free, so it separates a model that ranks well but is badly calibrated
  from one that ranks badly. BART is the clear example: AUC 0.87 with F1 0.56.
- **Latency** — batch size 1, single CPU thread, median and IQR over 100 records, best of three
  sweeps. Single-threaded deliberately, so the figure is portable rather than flattered by a
  128-core server.
- **Throughput** — full 1,900-record corpus in one pass; single-threaded CPU for classical
  models, L40 GPU at batch 32 for transformers.

Timing hardware: Intel Xeon Gold 6448Y (2×32 cores, up to 4.1 GHz), 503 GB RAM, NVIDIA L40.
Classical systems timed under Python 3.11 / scikit-learn 1.7.2; transformers under Python 3.8 /
PyTorch 2.4.1. Same machine, identical protocol.
