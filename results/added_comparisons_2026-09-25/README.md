# Added comparisons: Qwen3.8-27B and three Claude models (25 September 2026)

Both were added after the paper's forty systems were fixed. They are reported in
Table 4 and Supplementary Table S7, with their own Holm family: each added system
against BioLinkBERT and against Qwen2.5-7B with a tuned threshold (ten tests).

| Path | Contents |
|---|---|
| `added_test_metrics.csv` | weighted F1, bootstrap interval (notebook 07 method), biomarker F1, errors for all five added rows |
| `added_mcnemar_holm.csv` | exact McNemar and Holm-adjusted p for the ten tests |
| `qwen38/` | Qwen3.8-27B under notebook 06's protocol (`analysis/qwen38_zeroshot.py`): prompt search, holdout probabilities, test predictions, run metadata (commit `1d4bf0f2`, bf16, 2x L40, thinking off) |
| `qwen38/padding_check_test.csv` | batch-1 vs batch-8 test probabilities (`analysis/qwen38_padding_check.py`): no prediction changed |
| `claude/` | Claude Opus 5, Sonnet 5, Haiku 4.5 (`analysis/claude_zeroshot.py direct`): summary and test predictions |

Produced by `analysis/added_comparisons_significance.py`, which first reproduces
two published Table 4 rows exactly. The oceanus launch scripts are in
`scripts/oceanus_qwen38/`: Python 3.12 is required there, because only 3.12 has the
headers Triton needs.

The raw Claude replies are not released: they quote the outcome-measure text, and
the annotated corpus is not ours to distribute. The Claude runs cost US$5.74.
