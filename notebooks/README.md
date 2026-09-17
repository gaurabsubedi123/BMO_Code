# Notebooks

The analysis pipeline. Each notebook is executed with `nbconvert`, so the copies
in `executed/` carry their figures and printed diagnostics inline and can be read
without re-running anything.

| Notebook | Purpose |
|---|---|
| `00_setup_and_environment` | environment capture, seeds, input checksums, shared helpers |
| `01_dataset_and_corpus` | corpus statistics, modality taxonomy, corpus figure |
| `02_classical_tiers` | 7 estimators x 3 representations, nested CV |
| `05_transformer_finetuning` | 9 encoders, two-stage nested hyperparameter selection |
| `06_prompt_based_llms` | 5 prompt-based systems, equalised prompt search, threshold tuning |
| `07_consolidated_evaluation` | merge all tiers, bootstrap CIs, integrity checks |
| `08_significance_and_errors` | pairwise McNemar with Holm and BH correction, error analysis |
| `09_figures_and_tables` | manuscript figures and tables, run manifest |
| `10_deployment_export` | deployable inference bundles |
| `11_verification` | independent audit; re-derives every metric from raw predictions |
| `12_seed_stability` | refits each encoder under five seeds |

Notebooks 03 and 04 do not exist: the numbering follows the original pipeline
scripts, and those stages were folded into `02` and `06` respectively.

## Authoring

Notebooks are written as plain text in `_src/` (cells separated by `#%% md` and
`#%% code` markers) and built with `_src/build_nb.py`. This keeps them diffable
in review. **Edit the `_src/` file, not the `.ipynb`.**

```bash
python notebooks/_src/build_nb.py notebooks/_src/02_classical_tiers.py
```

## Running

Every notebook honours `BMO_SMOKE=1`, which collapses all grids to a single
point and writes to a separate output directory. A full run takes hours; a smoke
run takes minutes and exercises the same code paths. The driver scripts in
`_src/` gate the real run behind a passing smoke test, because several long runs
during development failed on defects a two-minute execution would have caught.

`nbconvert --allow-errors` exits 0 even when every cell raised, so the drivers
count error cells explicitly after each notebook rather than trusting the exit
code.
