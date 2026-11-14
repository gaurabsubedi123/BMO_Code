# Test on newer CTO annotations (14 November 2026)

Scores the released BiomedBERT-large on trials the CTO annotated after the paper's data were taken.
The annotation export itself is not public and is not in this repository.

1. `prepare.py export.xlsx labeled_dataset.csv outdir` keeps trials not in the paper's 234, matches each outcome
   line to its ClinicalTrials.gov (API v2) title and description, and writes outcome and eligibility lines.
2. `score_new.py classifier_dir run_dir outcomes.csv eligibility.csv outdir` first reproduces the paper's held-out
   result (0 missed, 2 false positives on 341 records; it stops if not), then scores both sets.
3. `analyze.py` / `analyze2.py` (run in the folder holding the `*_scored.csv` files; set `BMO_RERUN_DIR`) report
   false negatives and false positives with exact 95% intervals, with and without drug-level measures.
