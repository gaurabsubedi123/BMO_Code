# Test on newer CTO annotations (14 November 2026)

Scores the released BiomedBERT-large on trials the CTO annotated after the paper's data were taken.
The annotation export itself is not public and is not in this repository.

1. `prepare.py export.xlsx labeled_dataset.csv outdir` keeps trials not in the paper's 234, matches each outcome
   line to its ClinicalTrials.gov (API v2) title and description, and writes outcome and eligibility lines.
2. `score_new.py classifier_dir run_dir outcomes.csv eligibility.csv outdir` first reproduces the paper's held-out
   result (0 missed, 2 false positives on 341 records; it stops if not), then scores both sets.
3. `analyze.py` / `analyze2.py` (run in the folder holding the `*_scored.csv` files; set `BMO_RERUN_DIR`) report
   false negatives and false positives with exact 95% intervals, with and without drug-level measures.

## Comparing models on the newer trials

4. `score_trained.py run_dir outcomes.csv eligibility.csv outdir` scores the nine fine-tuned models and TF-IDF +
   logistic regression; each must first reproduce its published held-out false negatives and false positives.
5. `score_qwen.py repo_analysis_dir outcomes.csv eligibility.csv outdir` does the same for Qwen3.8-27B (reasoning off,
   prompt 'direct', cut-offs 0.5 and 0.10), with the same reproduction gate.
6. `compare.py *_all_scored.csv` compares every model's misses with BiomedBERT-large (exact McNemar on the biomarker
   lines, Holm-corrected).
7. `../../analysis/claude_new_trials.py` runs the five Claude models on the outcome lines (paid API; `dryrun` mode
   tests the whole pipeline with a fake client first). `add_claude.py NEW_TRIALS_DIR replies.jsonl` merges them and
   recomputes the comparison across all 16 systems.
8. `digital_rule.py NEW_TRIALS_DIR` measures what the deployed tool's digital/EEG review rule adds.
9. `compare_2025_definition.py NEW_TRIALS_DIR replies.jsonl` is the comparison reported in the paper. The newer
   annotations label drug concentration and anti-drug antibody measures as non-biomarkers, unlike the training data,
   so lines describing them (`drug_level.py`, decided from wording only) are set aside and every system is compared
   on the rest. Results on all lines come from steps 6 and 7.
10. `eligibility_cv.py RUN_DIR NEW_TRIALS_DIR OUTDIR` retrains BiomedBERT-large, SciBERT and BioLinkBERT with
    eligibility criteria, by trial-grouped five-fold cross-validation over the 268 newer trials (drug-level lines set
    aside), on eligibility lines alone and together with the 1,559 training outcomes, using each model's settings from
    the main study. `--smoke` runs one fold for one epoch to test every path first. The combined models are also scored
    on the newer outcome lines and the 341 held-out outcomes. `eligibility_cv_compare.py NEW_TRIALS_DIR OUTDIR` pairs
    the results with the outcome-trained models on the same lines (Supplementary Table S8).
11. `train_final_combined.py RUN_DIR NEW_TRIALS_DIR OUTDIR` trains the released model (BiomedBERT-large on all 1,559
    training outcomes and 5,987 eligibility lines) and tests it on the 341 held-out outcomes (0 missed, 1 false positive).
