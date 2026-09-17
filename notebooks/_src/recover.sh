#!/bin/bash
# ---------------------------------------------------------------------------
# Recovery run after four bugs found in the first pass:
#
#  1. Notebook 02 never forwarded `groups` to GridSearchCV, so the inner
#     StratifiedGroupKFold raised on the first fold and every classical tier
#     produced nothing.
#  2. OpenBioLLM-8B ships no chat template; apply_chat_template raised and the
#     model was dropped from the tier.
#  3. BioGPT's tokenizer needs sacremoses, which was not installed.
#  4. Flan-T5 was scored from a single decoder position, giving a below-chance
#     MCC -- a scoring bug, not a weak model.
#
# All four are fixed. This re-runs the affected notebooks and then the
# downstream consolidation, which previously ran on missing data.
# ---------------------------------------------------------------------------
set -u
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW" || exit 1
mkdir -p logs

JUP="$NEW/venv/bin/jupyter"
run_nb () {   # run_nb <notebook> <logname> [env assignments...]
    local nb="$1" log="$2"; shift 2
    echo "[$(date +%F_%T)] START  $nb $*"
    local t0=$SECONDS
    env "$@" "$JUP" nbconvert --to notebook --execute --allow-errors \
        --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
        "$nb" > "logs/${log}.log" 2>&1
    local rc=$?
    echo "[$(date +%F_%T)] DONE   $nb ($((SECONDS-t0))s rc=$rc)"
}

# The classical tier is CPU-only, so it can start immediately and run alongside
# whatever is still using the GPUs.
run_nb notebooks/02_classical_tiers.ipynb 02_classical_retry &
CLASSICAL=$!

# The LLM tier needs a GPU, so wait for the transformer refit to release one.
echo "[$(date +%F_%T)] waiting for the transformer refit to release the GPU..."
while pgrep -f "05_transformer_finetuning_shard" > /dev/null 2>&1; do sleep 30; done
echo "[$(date +%F_%T)] GPU free"
run_nb notebooks/06_prompt_based_llms.ipynb 06_llm_retry CUDA_VISIBLE_DEVICES=0

echo "[$(date +%F_%T)] waiting for the classical tier..."
wait $CLASSICAL
echo "[$(date +%F_%T)] classical tier finished"

# Downstream stages previously ran against missing data; redo them all.
run_nb notebooks/07_consolidated_evaluation.ipynb 07_retry
run_nb notebooks/08_significance_and_errors.ipynb 08_retry
run_nb notebooks/10_deployment_export.ipynb       10_retry CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_retry

echo "=== FINAL MASTER TABLE ==="
"$NEW/venv/bin/python" - <<'PY'
import pandas as pd, pathlib
f = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026/outputs/final_results/MASTER_test_all_systems.csv")
if f.exists():
    d = pd.read_csv(f)
    print(f"{len(d)} systems\n")
    print(d[["System","Tier","weighted_f1","biomarker_f1","errors"]].to_string(index=False))
else:
    print("MASTER table missing -- consolidation still failing")
PY
echo "RECOVERY_COMPLETE"
