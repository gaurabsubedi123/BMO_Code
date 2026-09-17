# Biomarker or Not

Code accompanying **"Biomarker or Not: Automated Detection of Biomarker Outcome Measures in
Alzheimer's Disease Clinical Trial Text."**

The task is deliberately narrow: given one registered outcome measure from an Alzheimer's
disease trial on ClinicalTrials.gov, decide whether it concerns a biomarker. The purpose is
triage — removing the ~70% of outcomes that are cognitive, functional or administrative from
an expert annotator's queue, so expert time goes only where judgment is needed.

This repository benchmarks **40 systems** on that task: seven classical classifiers under
three text representations, nine fine-tuned encoders, five models prompted without
task-specific training, ensembles, and two reference baselines.

> **Two runs are included.** `results/poster_run_2026-08/` holds the analysis behind the
> AAIC 2026 poster (33 systems). `results/rerun_2026-08-28/` holds the full re-run reported
> in the journal manuscript (40 systems). They are **not interchangeable** — see
> [`results/README.md`](results/README.md).

---

## Headline result

**No individual system is demonstrably best.** Among the twelve strongest, none of the 28
pairwise comparisons reaches significance after correction for multiple testing; every
bootstrap confidence interval overlaps that of the leader; and refitting the encoders under
five random seeds moves error counts by a standard deviation of **1.55**, exceeding the
differences between adjacently ranked systems.

| System | Test weighted F1 | 95% CI | Errors / 341 | Latency (CPU) | Size |
|---|---|---|---|---|---|
| BioLinkBERT | 0.9971 | [0.9912, 1.0000] | 1 | 140.9 ms | 434 MB |
| BiomedBERT-large | 0.9941 | [0.9854, 1.0000] | 2 | 494.9 ms | 1.3 GB |
| BioWordVec + SVM | 0.9883 | [0.9764, 0.9971] | 4 | — | 28.4 GB dep. |
| **TF-IDF + logistic regression** | **0.9824** | [0.9675, 0.9941] | **6** | **0.32 ms** | **0.09 MB** |
| Qwen2.5-7B-Instruct (tuned threshold) | 0.9763 | — | 8 | — | 15 GB |
| Keyword baseline (no training) | 0.9158 | [0.8860, 0.9423] | 29 | — | — |

The last two rows of that table are the point: a linear model over word counts is
statistically indistinguishable from a fine-tuned biomedical transformer **4,748× its size**.

### What does separate systems

**Text representation, not architecture or scale.** Biomedical embeddings improved six of
seven classical classifiers (random forest 15 errors → 5; decision tree 41 → 14), while
general-domain embeddings of *larger* dimension delivered almost nothing. The same holds
inside the transformer tier: ModernBERT (2024) does not outperform BERT-base (2018), and
neither matches SciBERT, which differs chiefly in vocabulary. Tripling parameters within an
identical pretraining corpus changed nothing.

**Biomedical vocabulary changes what is learnable, not only what is scored.** Under TF-IDF
features, neither neural network reaches a train–validation gap below 0.04 in *any* of 72
configurations in *any* fold. With BioWordVec embeddings, both reach it in all five.

### Where the systems fail

Recall exceeds 0.94 on conventional modalities but is **0.32 on digital assessments**, which
form 4.4% of the biomarker class yet **43.6% of false negatives** — a figure that replicates
an earlier independent analysis of this corpus to within a percentage point. Every false
positive is a clinical outcome carrying biological vocabulary. Both error types turn on
*context of use* rather than measurement type, which the binary label cannot express.

### Exported models

Two inference bundles are produced by `notebooks/10_deployment_export.ipynb`.
They are provided so the trained systems can be used and inspected, not as a
recommendation: this study measured accuracy and cost, it did not evaluate any
system in production, and which trade-off is appropriate depends on how a
service is run and on the relative cost of the two error types in that setting.

| Bundle | Model | Errors / 341 | Missed | Latency | Size |
|---|---|---|---|---|---|
| `bmo-transformer` | BiomedBERT-large | 2 | **0** | 6.9 ms (GPU) / 495 ms (CPU) | 1.3 GB |
| `bmo-fast` | TF-IDF + logistic regression | 6 | 4 | 0.32 ms (CPU) | 0.09 MB |

The two are selected on different criteria, which is itself worth noting: the
weighted-F1 leader (BioLinkBERT, 1 error) misses one biomarker, whereas
BiomedBERT-large recovers all 109 at the cost of two false alarms. Ranking by
aggregate accuracy and ranking by recall do not agree here, and the systems are
not statistically distinguishable on the former.

Each bundle carries a model card recording its test metrics, false-negative
count, the reason it was selected, and a caveat that evaluation was
in-distribution. `predict.py` exposes a `BiomarkerClassifier.predict()` entry
point and `serve.py` a minimal Flask endpoint.

**Weights are not in this repository.** The transformer checkpoint exceeds
GitHub's file-size limit, and the nine research checkpoints (5.0 GB) are
reproducible from the notebooks. The BioWordVec systems are excluded from the
bundles entirely: they require a 28.4 GB embedding binary resident in memory.

---

## Reproducibility

- 13 executed notebooks with figures and diagnostics inline (`notebooks/executed/`)
- `11_verification.ipynb` re-derives every reported metric from raw prediction files and
  fails loudly on any mismatch. Verdict for the reported run: **PASS**, 34 checks, 0 failures
- Input and output checksums, seeds and full environment in
  `results/rerun_2026-08-28/run_manifest.json`
- Transformer metrics reproduce to ≈±0.005 weighted F1, not bit-for-bit; cuDNN determinism
  was not enforced. `per_tier/seed_stability_*.csv` quantifies the consequence

---

## Data availability

**The annotated corpus is not distributed with this repository.** The 1,900 expert-annotated
outcome measures from 234 trials were produced by the UNLV Clinical Trial Observatory /
Chambers-Grundy Center annotation effort and are not ours to release. Contact the
corresponding author regarding access.

The underlying trial text is public: all outcome measures come from ClinicalTrials.gov and can
be retrieved via its API. What is not public is the expert AT(N) annotation from which the
binary label is derived.

Consequence: **cloning this repository will not reproduce the paper's numbers.** The pipeline
runs end-to-end only with the annotation workbook in `data/raw/`. To let readers check the
reported figures without it, every aggregate result table is committed under `results/` — see
[`docs/RESULTS.md`](docs/RESULTS.md).

## Model availability

Trained models are **not** distributed. The nine research checkpoints total 5.0 GB and the
exported transformer bundle a further 1.3 GB, both beyond what belongs in a git repository.
They regenerate by running the notebooks; see the runtimes below.

---

## Installation

```bash
conda create -n biomarker-ad python=3.10 -y
conda activate biomarker-ad
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**`scikit-learn==1.7.2` is pinned deliberately.** The saved pipelines record that version
internally; newer releases fail to unpickle them (`AttributeError: 'SVC' object has no
attribute '_effective_probability'`). `numpy>=2` is required for the same reason — the
pipelines reference `numpy._core`.

The BioWordVec tier additionally needs the BioWordVec binary (~28 GB). It must stay resident
in memory: a cold load takes about five minutes.

```bash
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
export BIOWORDVEC_BIN=/path/to/BioWordVec_PubMed_MIMICIII_d200.bin   # notebooks
export BIOWORDVEC_PATH=$BIOWORDVEC_BIN                               # archived src/ pipeline
```

The two pipelines read different variable names; setting both costs nothing.

---

## Running

With the annotation workbook in place this repository runs end to end and produces
every table and figure in the manuscript. Follow the six steps in order.

**Read this first: a clone cannot reproduce the paper's exact numbers, and the reason is the
corpus, not the code.** The study was run on a snapshot of the annotation workbook taken
**2026-03-29**, containing 1,900 records from 234 trials. The annotation effort continued
afterwards. Running step 2 on a later revision of the workbook yields a *superset* -- the
revision six days later produces 1,933 records from 237 trials, the 33 additional rows all
positive and annotated with coarse modality labels (`MRI`, `Imaging`, `EEG`) that are almost
absent from the snapshot. A different record set means a different trial-grouped 80/20 split,
a test partition that is no longer 341 records, and different numbers everywhere downstream.
That is a difference in data, not drift in computation. If you need the exact corpus, contact
the corresponding author for the 2026-03-29 snapshot; with any other revision, treat this
pipeline as reproducing the study's *method* rather than its *numbers*.

### 1. Provide the corpus

Place the annotation workbook in `data/raw/`, under exactly this filename — step 2 looks it
up by name (`GOLD_FILE` in `src/config.py`):

```
data/raw/2025-04-25 Biomarker Annotations (clean and current) - for CS 3-25-2026.xlsx
```

The workbook is not distributed here; see [Data availability](#data-availability). Nothing
below this line runs without it.

### 2. Build the corpus and the split

```bash
python src/01_prepare_dataset.py
```

This is the **one step shared by both pipelines**, and the notebooks do not do it: they load
the prepared dataset rather than recreate it, so that every tier provably sees the same rows
and the same partition. It writes

```
outputs/dataset/labeled_dataset.csv     the corpus, one row per outcome measure
outputs/dataset/train_idx.npy           80% of records, grouped by trial
outputs/dataset/test_idx.npy            20% of records, trial-disjoint from train
```

On the 2026-03-29 snapshot this is 1,900 records from 234 trials, 29.68% positive, split
1,559 / 341. **Check what you actually get**: the script prints its record and trial counts,
and if they are not 1,900 and 234 you are on a different workbook revision and your numbers
will not match the paper's. See the note at the top of this section.

Labels are rule-derived: `has_biomarker = 0` where the expert annotation begins with
`ANALOG`, else 1. Records with no annotation (726 of 2,626 in the snapshot) are dropped.

### 3. Point the pipeline at your working directory

The notebooks read their configuration from the environment and default to the research mount
this study ran on, which will not exist on your machine. `BMO_RUN_DIR` must be the repository
root, because that is where step 2 wrote `outputs/dataset/`:

```bash
export BMO_RUN_DIR=$PWD                       # required: the repo root
export HF_HOME=/path/to/huggingface/cache     # transformer + prompt tiers
export BIOWORDVEC_BIN=/path/to/BioWordVec_PubMed_MIMICIII_d200.bin
export BMO_N_JOBS=32          # CPU workers          (default 32)
export BMO_GPU_CAP_GB=10      # per-process GPU cap  (default 10)
export BMO_NSHARD=2           # GPUs to shard across (default 2)
```

### 4. Smoke test first

```bash
BMO_SMOKE=1 jupyter nbconvert --to notebook --execute notebooks/00_setup_and_environment.ipynb
BMO_SMOKE=1 jupyter nbconvert --to notebook --execute notebooks/02_classical_tiers.ipynb
```

`BMO_SMOKE=1` collapses every search grid to a single point and writes to a separate output
directory, exercising the same code paths in about two minutes. This is not optional
politeness: several multi-hour runs during development were launched carrying defects a
two-minute execution would have exposed, and `nbconvert --allow-errors` exits 0 even when
every cell raised, so a clean exit code is not evidence that anything worked.

### 5. Run the pipeline

Run the notebooks in this order. **`00_setup_and_environment` must run first and is not
optional**: it authors `bmo_common.py` into `$BMO_RUN_DIR` and writes it out from there, and
every other notebook does `import bmo_common as C` for paths, seeds, input assertions and the
shared plotting helpers. The module is generated rather than committed, so nothing downstream
runs until 00 has executed.

```bash
for nb in 00_setup_and_environment 01_dataset_and_corpus 02_classical_tiers \
          05_transformer_finetuning 06_prompt_based_llms 07_consolidated_evaluation \
          08_significance_and_errors 09_figures_and_tables 10_deployment_export \
          11_verification 12_seed_stability; do
    jupyter nbconvert --to notebook --execute --allow-errors \
        --ExecutePreprocessor.timeout=-1 --output-dir=notebooks/executed \
        "notebooks/$nb.ipynb" || break
done
```

| Notebook | Produces | Runtime |
|---|---|---|
| `00_setup_and_environment` | `bmo_common.py`, seeds, input checksums | < 1 min |
| `01_dataset_and_corpus` | corpus statistics, modality taxonomy, corpus figure | < 1 min |
| `02_classical_tiers` | 7 estimators x 3 representations, nested CV | ~1 h |
| `05_transformer_finetuning` | 9 encoders, two-stage nested hyperparameter selection | **~3 h (2 GPUs)** |
| `06_prompt_based_llms` | 5 prompt-based systems, equalised prompt search, threshold tuning | ~8 min |
| `07_consolidated_evaluation` | all tiers merged, bootstrap CIs, integrity checks | < 1 min |
| `08_significance_and_errors` | 780 pairwise McNemar tests with Holm and BH, error analysis | < 1 min |
| `09_figures_and_tables` | manuscript figures and tables, run manifest | < 1 min |
| `10_deployment_export` | the two inference bundles | < 1 min |
| `11_verification` | independent audit; re-derives every metric from raw predictions | < 1 min |
| `12_seed_stability` | each encoder refit under five seeds | ~21 min (2 GPUs) |

Runtimes are wall clock from the executed copies in `notebooks/executed/`, on 2x NVIDIA L40
with a Xeon Gold 6448Y. Notebooks 05 and 12 shard their model roster across the two GPUs and
run both shards in parallel, so the figure quoted is the slower shard; on one GPU, expect
roughly the sum. Notebooks 03 and 04 do not exist: the numbering follows the original scripts,
and those stages were folded into 02 and 06.

Notebooks are authored as plain text in `notebooks/_src/` and built with `_src/build_nb.py`.
**Edit the `_src/` file, not the `.ipynb`.** See [`notebooks/README.md`](notebooks/README.md).

The shell drivers in `notebooks/_src/` (`run_all.sh`, `gated_launch.sh`, `nested_chain.sh` and
the rest) are the orchestration this study actually ran, under tmux, including the smoke gate
and the per-notebook error-cell count. Their paths are hardcoded to the research mount and
they source an `env.sh` that is not distributed, so read them as a record of how the run was
sequenced rather than as turn-key scripts.

### 6. Verify you reproduced it

`11_verification` is the check that matters. It re-derives every reported metric from the raw
prediction files, confirms all 40 systems are present, that every confusion matrix sums to
341, that all 780 pairwise tests exist, and that corrected p-values never fall below raw ones.
It ends on `assert not PROBLEMS`, so it fails loudly. The reported run scored **PASS, 34
checks, 0 failures**.

Compare your output against the committed copy of the same run:

| Your output | Committed reference | In the paper |
|---|---|---|
| `tables/table1_main_results.csv` | `results/rerun_2026-08-28/table1_main_results.csv` | Table 1, Fig. ranking |
| `tables/table2_representation_effect.csv` | `results/rerun_2026-08-28/table2_representation_effect.csv` | Table 2, Fig. features |
| `tables/table3_overfitting_audit.csv` | `results/rerun_2026-08-28/table3_overfitting_audit.csv` | Appendix `tab:overfit` |
| `tables/table_modality_composition.csv` | `results/rerun_2026-08-28/table_modality_composition.csv` | `tab:modality` |
| `outputs/final_results/significance_pairwise_test.csv` | `results/rerun_2026-08-28/significance_pairwise_test.csv` | Fig. significance |
| `outputs/provenance/run_manifest.json` | `results/rerun_2026-08-28/run_manifest.json` | Methods |

**This comparison is only meaningful if step 2 gave you 1,900 records from 234 trials.** On
any other workbook revision the corpus and the split differ, and so will every number below;
nothing is wrong with the pipeline in that case.

**On the same corpus, expect agreement to about ±0.005 weighted F1 on the transformer tiers,
not bit-for-bit.** Seeds are fixed throughout but deterministic cuDNN kernels were not
enforced. The classical, BioWordVec and FastText tiers are deterministic and should match
exactly.
`per_tier/seed_stability_*.csv` quantifies the consequence: a standard deviation of 1.55
errors out of 341, which exceeds the gap between adjacently ranked systems. **A different
system topping your leaderboard is an expected outcome of this study, not a failure to
reproduce** — that instability is one of the paper's findings.

---

## Other code in this repository

### `src/` — dataset preparation, plus the archived poster-run pipeline

**Step 01 is live and required** — `src/01_prepare_dataset.py` is what builds
`outputs/dataset/` from the annotation workbook for both pipelines, as described above.

**Steps 02-08 are archived.** They are the earlier script pipeline behind
`results/poster_run_2026-08/` and the AAIC 2026 poster: eight steps, five fine-tuned
transformers, 33 systems, single-stage hyperparameter selection. **They do not reproduce the
manuscript's numbers** and are kept only so the poster results stay auditable. Do not run
`run_pipeline.sh` expecting the paper's figures; it runs the archived path.

```bash
bash run_pipeline.sh                  # the archived poster-run path, all steps
python src/03_train_bert_models.py    # or a single archived step
```

| Step | Script | Produces | Runtime |
|---|---|---|---|
| 01 | `01_prepare_dataset.py` | corpus + trial-grouped 80/20 split | seconds |
| 02 | `02_train_classical_ml.py` | 7 classifiers, TF-IDF | ~10 min |
| 02a | `02a_classical_ml_enhanced.py` | 7 classifiers, TF-IDF + FastText-300 | ~20-40 min |
| 02b | `02b_classical_ml_biowordvec.py` | 7 classifiers, TF-IDF + BioWordVec-200 | ~20-40 min |
| 03 | `03_train_bert_models.py` | 5 fine-tuned transformers + ensemble | **6-8 h (GPU)** |
| 04 | `04_train_zeroshot_llm.py` | BART / Flan-T5 / BioGPT, no fine-tuning | ~30 min |
| 05 | `05_consolidate_results.py` | cross-tier comparison | ~1 min |
| 06 | `06_test_set_evaluation.py` | held-out test evaluation | ~5-10 min |
| 07 | `07_significance_tests.py` | McNemar, all pairs | ~1-2 min |
| 08 | `08_full_predictions.py` | per-record prediction workbook | ~5-10 min |

Steps 02a and 02b must run before 06 and 07, or the FastText and BioWordVec systems are
silently omitted from the comparison - see the caveat in `docs/RESULTS.md`.

### `analysis/` — standalone scripts

Written alongside the poster run. `significance_all_systems.py` is superseded by
`08_significance_and_errors.ipynb`, which covers all 40 systems and reports Holm and
Benjamini-Hochberg corrections alongside the raw p-values; the manuscript's significance
counts come from the notebook, not from this script. The timing scripts are still the tools
that produce the deployment-cost figures.

| Script | Purpose |
|---|---|
| `recall_by_modality.py` | **Current.** Recall on the biomarker class broken down by measurement modality, using the taxonomy from `01_dataset_and_corpus`. Defaults to out-of-fold predictions over the training partition, because the held-out partition has too few positives per modality to estimate recall (digital assessments have one). Produces `results/rerun_2026-08-28/recall_by_modality_oof.csv`. |
| `baselines.py` | **Current.** Recomputes the two reference baselines on the held-out partition: the training-free 46-term AT(N) keyword rule and the majority-class floor. Both are deterministic given the corpus, with bootstrap intervals matching `07_consolidated_evaluation`. Produces `results/rerun_2026-08-28/baselines.csv`. |
| `benchmark_rerun.py` | **Current.** Inference timing for the re-run checkpoints and the exported bundles: single-thread CPU latency (median of 200 held-out records, best of three sweeps), a full 341-record CPU pass, and a batch-32 GPU pass. Produces `results/rerun_2026-08-28/inference_timing_rerun.csv`, which is the source for the manuscript's cost table. |
| `significance_all_systems.py` | Superseded. McNemar across the 31 poster-run systems with out-of-fold predictions (465 pairs), plus a 33-system test-set variant, uncorrected. Standalone: needs only numpy/scipy/pandas/matplotlib. |
| `benchmark_classical.py` | Superseded by `benchmark_rerun.py`. Poster-run timing, classical tiers. Run under Python 3.11 + scikit-learn 1.7.2. |
| `benchmark_transformers.py` | Superseded by `benchmark_rerun.py`. Poster-run timing, transformer tier. |
| `merge_timing.py` | Merges the two poster-run timing runs and computes the ratios reported in the poster. |

---

## Method notes that matter for reproduction

**Trial-level grouping everywhere.** Every partition — the 80/20 split, all outer and inner
cross-validation folds, the transformer learning-rate search, and zero-shot prompt selection —
is grouped by `NCTId`. A single trial contributes many outcome measures that share phrasing;
splitting record-by-record would leak and inflate every figure. Verified: zero trial overlap
between train and test.

**Seeds.** 42 for all splits and folds; 99 for the learning-rate search and prompt selection,
so tuning decisions are provably independent of the evaluation folds.

**Labels are rule-derived.** `has_biomarker = 0` if the expert annotation begins with
`ANALOG`, else 1. Records with no annotation (726 of 2,626) are excluded.

**Class balance.** 564 of 1,900 records (29.7%) are biomarker outcomes. The frequently quoted
61.9% is *trial*-level — the share of the 234 trials with at least one biomarker outcome. Use
29.7% for anything to do with class imbalance.

---

## Repository layout

```
notebooks/  the pipeline the manuscript reports
            _src/      plain-text sources and shell drivers - edit these
            executed/  the same notebooks with figures and diagnostics inline
figures/    manuscript figures as published
results/    aggregate metric tables, poster run and re-run kept separate
            (no trial text, no labels, no NCTIds)
src/        archived poster-run pipeline, steps 01-08 - superseded
analysis/   standalone significance and inference-timing scripts
demo/       Gradio app for interactive inspection
docs/       RESULTS.md - what each number is and which file is authoritative
```

## Citing

See `CITATION.cff`. Licensed MIT.
