# Figures as published (TRCI manuscript, 25 September 2026)

| File | In the paper |
|---|---|
| `ranking.pdf` | Figure 1: systems reaching 0.95 weighted F1 with bootstrap intervals, plus the added comparisons (Qwen3.8-27B, three Claude models) as hollow markers below a gap |
| `modality_recall.pdf` | Figure 2: out-of-fold recall by biomarker modality |
| `significance.pdf` | Supplementary Figure S1: pairwise McNemar matrix for the forty systems, Holm-corrected |
| `confusion.pdf` | Supplementary Figure S2: confusion matrices for the leading system in each group, plus Qwen3.8-27B |
| `corpus_overview.pdf` | Supplementary Figure S3: the annotated corpus |

Each figure also has a 300 dpi PNG. Regenerate them with
`BMO_RERUN_DIR=/path/to/rerun_results python3 analysis/make_figures.py`. This needs
the unreleased annotated corpus for Figures S2 and S3; the others need only
released result tables and predictions. `superseded_2026-08-28/` holds the
earlier poster-era figures, kept for reference.
