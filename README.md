# Biomarker or Not

Code accompanying **"Biomarker or Not: Automated Detection of Biomarker Outcome Measures in
Alzheimer's Disease Clinical Trial Text."**

The task is deliberately narrow: given one registered outcome measure from an Alzheimer's
disease trial on ClinicalTrials.gov, decide whether it concerns a biomarker. The purpose is
triage — removing the ~70% of outcomes that are cognitive, functional or administrative from
an expert annotator's queue, so expert time goes only where judgment is needed.

This repository benchmarks **33 systems** on that task: seven classical classifiers under
three text representations, five fine-tuned biomedical transformers, three generative models
prompted without task-specific training, four ensembles, and two reference baselines.

---

## Headline result

| System | CV weighted F1 | Test weighted F1 | Latency / record | Model size |
|---|---|---|---|---|
| Soft-vote transformer ensemble | **0.9702** | 0.9852 | ~191 ms | ~435 MB |
| BioLinkBERT | 0.9695 | 0.9852 | 192 ms | 433 MB |
| PubMedBERT | 0.9660 | 0.9852 | 191 ms | 438 MB |
| **SVM + BioWordVec-200** | **0.9658** | **0.9883** | **1.11 ms** | **0.8 MB** |
| Logistic regression + TF-IDF | 0.9469 | 0.9824 | 0.30 ms | 0.1 MB |
| Keyword baseline (no training) | 0.8985 | — | — | — |
| Best prompted model (Flan-T5) | 0.6283 | 0.4787 | — | — |

Two findings drive the paper:

1. **Representation beats architecture.** Swapping the same classifier from TF-IDF to
   biomedical embeddings is statistically significant (McNemar p = 1.2×10⁻⁶, Bonferroni over
   465 pairs). Swapping to a transformer is not — across all 30 of its pairwise comparisons,
   **no system significantly outperforms the BioWordVec SVM** (vs PubMedBERT p = 0.90).
   General-domain FastText embeddings gave no reliable benefit at all.
2. **The cheap model is ~172× faster.** 1.11 ms/record versus 190.6 ms for PubMedBERT, from a
   model file 518× smaller, measured on identical hardware at batch size 1, single-threaded.

Residual errors concentrate at one boundary: recall is 95.9% on conventional biomarkers but
30% on digital cognitive assessments, and every false positive is an outcome where a
biological instrument measures a behavioural or safety construct.

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

Trained models are **not** distributed (~11 GB, dominated by five transformer checkpoints).
They regenerate by running the pipeline; see runtimes below.

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

Step 02b additionally needs the BioWordVec binary (~28 GB):

```bash
wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
export BIOWORDVEC_PATH=/path/to/BioWordVec_PubMed_MIMICIII_d200.bin
```

---

## Running

```bash
bash run_pipeline.sh          # all steps in order
python src/03_train_bert_models.py   # or a single step
```

| Step | Script | Produces | Runtime |
|---|---|---|---|
| 01 | `01_prepare_dataset.py` | corpus + trial-grouped 80/20 split | seconds |
| 02 | `02_train_classical_ml.py` | 7 classifiers, TF-IDF | ~10 min |
| 02a | `02a_classical_ml_enhanced.py` | 7 classifiers, TF-IDF + FastText-300 | ~20–40 min |
| 02b | `02b_classical_ml_biowordvec.py` | 7 classifiers, TF-IDF + BioWordVec-200 | ~20–40 min |
| 03 | `03_train_bert_models.py` | 5 fine-tuned transformers + ensemble | **6–8 h (GPU)** |
| 04 | `04_train_zeroshot_llm.py` | BART / Flan-T5 / BioGPT, no fine-tuning | ~30 min |
| 05 | `05_consolidate_results.py` | cross-tier comparison | ~1 min |
| 06 | `06_test_set_evaluation.py` | held-out test evaluation | ~5–10 min |
| 07 | `07_significance_tests.py` | McNemar, all pairs | ~1–2 min |
| 08 | `08_full_predictions.py` | per-record prediction workbook | ~5–10 min |

Steps 02a and 02b must run before 06 and 07, or the FastText and BioWordVec systems are
silently omitted from the comparison — see the caveat in `docs/RESULTS.md`.

### `analysis/` — scripts written for the paper

| Script | Purpose |
|---|---|
| `significance_all_systems.py` | McNemar across all 31 systems with out-of-fold predictions (465 pairs), plus a 33-system test-set variant. Standalone: needs only numpy/scipy/pandas/matplotlib. |
| `benchmark_classical.py` | Inference timing, classical tiers. Run under Python 3.11 + scikit-learn 1.7.2. |
| `benchmark_transformers.py` | Inference timing, transformer tier (CPU latency, GPU throughput). |
| `merge_timing.py` | Merges the two timing runs and computes the reported ratios. |

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
src/        pipeline steps 01-08, config.py, embeddings.py
analysis/   significance and timing scripts written for the paper
demo/       Gradio app for interactive inspection
results/    aggregate metric tables (no trial text, no labels)
docs/       RESULTS.md - what each number is and which file is authoritative
```

## Citing

See `CITATION.cff`. Licensed MIT.
