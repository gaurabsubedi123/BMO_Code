#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run the transformer CV and final fits with corrected training metrics.
#
# The first pass scored the training set *during* the update pass, with dropout
# active and the weights changing between the first batch and the last. That
# understates training F1, and therefore understates the train-minus-validation
# gap -- which is the paper's evidence against overfitting. Biasing that gap
# downwards makes every model look better regularised than it really is.
#
# This re-run scores the training set in eval() mode with frozen weights.
#
# Phase 1 (hyperparameter search) is NOT repeated: selection depends only on
# validation scores, which this fix does not affect. Both shards read their
# cached search results, so the cost is Phase 2 + Phase 3 only.
# ---------------------------------------------------------------------------
set -u
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW" || exit 1
mkdir -p logs

wait_for () {   # wait_for <pattern> <label>
    echo "[$(date +%F_%T)] waiting for $2 ..."
    while pgrep -f "$1" > /dev/null 2>&1; do sleep 30; done
    echo "[$(date +%F_%T)] $2 done"
}

# Shard 1's first pass must finish so its hyperparameter cache exists, and the
# LLM notebook must finish so GPU 0 is free.
wait_for "05_transformer_finetuning_shard1.ipynb" "shard 1 first pass"
wait_for "06_prompt_based_llms.ipynb"             "LLM notebook"

for sh in 0 1; do
    f="$NEW/outputs/bert_models/hyperparameter_search_shard${sh}.csv"
    if [ ! -f "$f" ]; then
        echo "[$(date +%F_%T)] ERROR: missing tuning cache for shard ${sh} ($f)"
        echo "  refusing to continue -- a re-run without the cache would repeat"
        echo "  the full hyperparameter search and is probably not what is wanted."
        exit 1
    fi
    echo "[$(date +%F_%T)] tuning cache present for shard ${sh}"
done

echo "[$(date +%F_%T)] re-running both shards (Phase 2 + 3, clean train metrics)"
JUP="$NEW/venv/bin/jupyter"

env CUDA_VISIBLE_DEVICES=0 BMO_SHARD=0 BMO_NSHARD=2 \
    "$JUP" nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
    --output 05_transformer_finetuning_shard0.ipynb \
    notebooks/05_transformer_finetuning.ipynb > logs/05_shard0_clean.log 2>&1 &
A=$!

env CUDA_VISIBLE_DEVICES=1 BMO_SHARD=1 BMO_NSHARD=2 \
    "$JUP" nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
    --output 05_transformer_finetuning_shard1.ipynb \
    notebooks/05_transformer_finetuning.ipynb > logs/05_shard1_clean.log 2>&1 &
B=$!

wait $A; RA=$?
wait $B; RB=$?
echo "[$(date +%F_%T)] refit complete  shard0=$RA shard1=$RB"

echo "=== corrected overfitting audit ==="
cat "$NEW"/outputs/bert_models/overfitting_audit_shard*.csv 2>/dev/null
echo "=== corrected test results ==="
cat "$NEW"/outputs/bert_models/test_results_shard*.csv 2>/dev/null
echo "REFIT_CLEAN_COMPLETE"
