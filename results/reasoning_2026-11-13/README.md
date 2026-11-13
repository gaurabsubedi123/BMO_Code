# Reasoning models and error overlap (13 November 2026)

Summary metrics only. No outcome text, labels or trial identifiers are included.

| File | Produced by | What it is |
|---|---|---|
| `claude_reasoning_summary.csv` | `analysis/claude_reasoning.py` | Claude Opus 5.5 and Claude Fable 5.2, adaptive thinking, effort `high`, no fallbacks. Test errors on the 341 held-out records, misses, false alarms, agreement between two passes, cost. |
| `qwen38_thinking_summary.csv` | `analysis/qwen38_thinking.py` | Qwen3.8-27B with thinking on (vLLM, bf16, one H200; temperature 0.6, top-p 0.95, top-k 20, fixed seeds). |
| `qwen38_thinking_run_meta.json` | same | Checkpoint revision, sampling settings, library versions, GPU. |
| `leading_systems_shared_errors.csv` | `analysis/error_overlap.py` | For each pair of leading systems (at most 5 test errors), how many errors they share. |

All reasoning models used the same "direct" prompt as the other prompted models, read from notebook 06.
Answers were scored by the first word of the reply after reasoning; an unreadable reply counts as an error.

Run scripts for grundy are in `scripts/grundy_qwen38_thinking/`.
