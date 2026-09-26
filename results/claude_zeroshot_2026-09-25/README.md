# Added comparison: Claude models, zero-shot (25 September 2026)

Three Claude models were prompted through the Anthropic API (standard API, not
the Batch API) with Qwen2.5-7B's selected prompt and no reasoning step. They sit
outside the paper's forty systems and are reported in Table 4 and Supplementary
Table S7. Produced by `analysis/claude_zeroshot.py direct` and
`analysis/claude_significance.py`.

| File | Contents |
|---|---|
| `claude_zeroshot_summary.csv` | errors, misses, false alarms, unreadable replies, run-to-run agreement, training-set errors |
| `claude_test_metrics.csv` | weighted F1, bootstrap interval (notebook 07 method), biomarker F1 on the 341 test records |
| `claude_mcnemar_holm.csv` | exact McNemar vs BioLinkBERT and tuned Qwen2.5-7B, Holm over the six tests |
| `test_predictions.csv` | test-set predictions in the same row order as `rerun_2026-08-28/per_tier/test_predictions.csv`; unreadable replies are scored as wrong |

The raw replies are not released: they quote the outcome-measure text, and the
annotated corpus is not ours to distribute. Cost: US$5.74 in total. The first
batch submission was cancelled after 85 minutes with nothing processed and
nothing billed.
