# Qwen3.8-27B and the Claude models (25 September 2026)

Five systems that complete the study's 45: Qwen3.8-27B at the default and tuned
threshold, and Claude Opus 5, Sonnet 5 and Haiku 4.5. They are part of every
count, figure and correction in the paper, including the 990-pair McNemar family.

| Path | Contents |
|---|---|
| `significance_pairwise_test_45.csv` | exact McNemar for all 990 pairs of the 45 systems, with Holm and BH-FDR corrections |
| `per_system_errors_45.csv` | errors and unique errors for all 45 systems |
| `qwen38_claude_test_metrics.csv` | weighted F1, bootstrap interval (notebook 07 method), biomarker F1, misses and false alarms for the five |
| `qwen38/` | Qwen3.8-27B under notebook 06's protocol (`analysis/qwen38_zeroshot.py`): prompt search, holdout probabilities, test predictions, run metadata (commit `1d4bf0f2`, bf16, 2x L40, thinking off), and the batch-1 vs batch-8 padding check |
| `claude/` | Claude summary and test predictions (`analysis/claude_zeroshot.py direct`) |

Produced by `analysis/combined_significance.py`, which first checks that it
reproduces the published 40-system family (780 pairs: 427 raw, 233 Holm) and
two published Table 4 intervals. The oceanus launch scripts are in
`scripts/oceanus_qwen38/`. Python 3.12 is required there, because only 3.12 has
the headers Triton needs.

The raw Claude replies are not released: they quote the outcome-measure text,
and the annotated corpus is not ours to distribute. The Claude runs cost US$5.74.
