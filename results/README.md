# Results

This directory holds **two independent runs of the study**, kept separate rather
than merged. They are not interchangeable and their numbers should never be
quoted together.

## `poster_run_2026-08/`

The run behind the AAIC 2026 poster and submitted abstract. 33 systems.
Preserved unchanged so the presented figures remain reconstructable.

Known issues, documented rather than corrected:

- Pairwise significance was reported **uncorrected** across 465 comparisons, where
  roughly 23 significant results are expected by chance at α = 0.05.
- The FastText and BioWordVec tiers in `all_test_metrics.csv` were scored before
  their embedding model finished loading in some configurations, so those rows
  reflect zero-vector features.

## `rerun_2026-08-28/`

The full re-run reported in the journal manuscript. 40 systems, including four
encoders (SciBERT, DeBERTa-v3, ModernBERT, BiomedBERT-large) and two
instruction-tuned models (Qwen2.5-7B-Instruct, OpenBioLLM-8B) not present in the
poster run.

Differences from the poster run that matter when comparing:

- Every model was retrained from scratch under one protocol; no numbers are
  carried over.
- Significance is reported raw, Holm–Bonferroni corrected and Benjamini–Hochberg
  corrected across all 780 pairs. **The corrected values are the ones to use.**
- Every system carries a percentile bootstrap confidence interval over 2,000
  resamples of the test partition.
- Each fine-tuned encoder was refit under five random seeds to quantify training
  stochasticity (`per_tier/seed_stability_*.csv`).
- The prompt-based models are re-implementations. Prompt search was equalised
  across all five, which changes their numbers substantially — Flan-T5 moves from
  0.31 to 0.77 weighted F1 relative to a single-prompt evaluation. **These rows
  are not comparable to the poster run.**

### Verification

`verification_report.json` records the result of an independent audit that
re-derives every reported metric from the raw prediction files, checks that all
40 systems are present, that every confusion matrix sums to the test-set size,
that the significance matrix is complete, and that corrected p-values never fall
below raw ones. Verdict: **PASS** (34 checks, 0 failures).

`run_manifest.json` and `CHECKSUMS.txt` record the software environment, random
seeds, input checksums and output checksums for the run.

One field in `run_manifest.json` needs reading carefully: `n_bootstrap` is `1000`,
which is the default of `bmo_common.bootstrap_ci`, used only for the prompt tier's
intermediate `ci_low` / `ci_high` columns. It is **not** the figure behind any
published interval. Every confidence interval in the manuscript and in
`MASTER_test_all_systems_full.csv` comes from the `wf1_lo` / `wf1_hi` and
`bf1_lo` / `bf1_hi` columns, recomputed for all 40 systems in
`07_consolidated_evaluation` at **2,000** resamples.

### The corpus is a snapshot

Both runs use the annotation workbook as it stood on **2026-03-29**: 1,900 records from 234
trials, 29.68% positive, split 1,559 / 341 by trial. The annotation effort continued after
that date, and later revisions of the workbook contain additional annotated records, so
rebuilding the dataset from a current workbook will not give the record set behind these
tables. The workbook is not distributed here in any revision; see Data availability in the
top-level README.

### Reproducibility tolerance

Seeds are fixed throughout, but deterministic GPU kernels were not enforced.
Transformer metrics reproduce to approximately ±0.005 weighted F1, not
bit-for-bit. The seed-stability files quantify what this means in practice: a
standard deviation of 1.55 errors out of 341, which exceeds the differences
between adjacently ranked systems.
