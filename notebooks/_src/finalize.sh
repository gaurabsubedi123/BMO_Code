#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run the LLM tier with holdout-tuned decision thresholds, then re-run the
# downstream consolidation once BOTH the LLM tier and the classical tier have
# finished.
#
# The recovery script may already run 07-10 when the classical tier completes.
# If the LLM re-run is still going at that point those outputs are stale, so
# this pass repeats them against the final data. Re-running consolidation is
# cheap (seconds); getting it wrong is not.
# ---------------------------------------------------------------------------
set -u
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW" || exit 1
JUP="$NEW/venv/bin/jupyter"

run_nb () {
    local nb="$1" log="$2"; shift 2
    echo "[$(date +%F_%T)] START  $nb"
    local t0=$SECONDS
    env "$@" "$JUP" nbconvert --to notebook --execute --allow-errors \
        --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
        "$nb" > "logs/${log}.log" 2>&1
    echo "[$(date +%F_%T)] DONE   $nb ($((SECONDS-t0))s rc=$?)"
}

run_nb notebooks/06_prompt_based_llms.ipynb 06_threshold CUDA_VISIBLE_DEVICES=0

echo "[$(date +%F_%T)] waiting for the classical tier to finish..."
while pgrep -f "02_classical_tiers.ipynb" > /dev/null 2>&1; do sleep 30; done
echo "[$(date +%F_%T)] classical tier finished"

# Let any downstream run started by the recovery script drain first, so we are
# not writing the same files from two processes at once.
while pgrep -f "0[789]_\|10_deployment" > /dev/null 2>&1; do sleep 20; done

run_nb notebooks/07_consolidated_evaluation.ipynb 07_final
run_nb notebooks/08_significance_and_errors.ipynb 08_final
run_nb notebooks/10_deployment_export.ipynb       10_final CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_final

echo "=== FINAL MASTER TABLE ==="
"$NEW/venv/bin/python" - <<'PY'
import pandas as pd, pathlib
f = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026/outputs/final_results/MASTER_test_all_systems.csv")
if f.exists():
    d = pd.read_csv(f)
    print(f"{len(d)} systems\n")
    print(d[["System","Tier","weighted_f1","biomarker_f1","errors"]].to_string(index=False))
else:
    print("MASTER table missing")
PY
echo "FINALIZE_COMPLETE"
