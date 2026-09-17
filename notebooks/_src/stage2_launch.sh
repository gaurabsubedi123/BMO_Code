#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2: transformer regularisation search, gated behind its smoke test.
#
# Stage 1 searched only learning rate and batch size (6-8 configs per model)
# while the classical tier searched 48-768. Stage 2 adds dropout and weight
# decay at each model's stage-1 winning lr/batch, and -- crucially -- selects
# the best configuration *among those whose train-validation gap is under
# 0.04*, so a model cannot win by overfitting harder.
#
# Runs on both GPUs, in parallel with the CPU-only classical tier, so it costs
# no additional wall-clock time.
# ---------------------------------------------------------------------------
set -u
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW" || exit 1
JUP="$NEW/venv/bin/jupyter"

echo "[$(date +%F_%T)] waiting for the stage-2 smoke test..."
while pgrep -f "smoke05" > /dev/null 2>&1; do sleep 15; done

ERRS=$("$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke05.ipynb 2>/dev/null | grep -c "^### CELL")
STAGE2FILE=$(ls "$NEW"/outputs/bert_models_smoke/stage2_regularisation_shard*.csv 2>/dev/null | head -1)
echo "[$(date +%F_%T)] smoke error cells: ${ERRS:-unknown}"
echo "[$(date +%F_%T)] stage2 file from smoke: ${STAGE2FILE:-MISSING}"
if [ "${ERRS:-1}" -ne 0 ] || [ -z "$STAGE2FILE" ]; then
    echo "[$(date +%F_%T)] ABORT: smoke did not pass, or the stage-2 path never ran"
    "$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke05.ipynb 2>/dev/null | head -20
    exit 1
fi
echo "[$(date +%F_%T)] smoke validated; stage-2 path confirmed to execute"
rm -rf "$NEW/outputs/bert_models_smoke" "$NEW/models/smoke"

echo "[$(date +%F_%T)] launching stage 2 on both GPUs"
for SH in 0 1; do
  ( env CUDA_VISIBLE_DEVICES=$SH BMO_STAGE2=1 BMO_SHARD=$SH BMO_NSHARD=2 \
      "$JUP" nbconvert --to notebook --execute --allow-errors \
      --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
      --output 05_transformer_finetuning_shard${SH}.ipynb \
      notebooks/05_transformer_finetuning.ipynb > logs/05_stage2_shard${SH}.log 2>&1 ) &
done
wait
echo "[$(date +%F_%T)] stage 2 finished"
for SH in 0 1; do
  N=$("$NEW/venv/bin/python" /tmp/nberr2.py \
      "$NEW/notebooks_executed/05_transformer_finetuning_shard${SH}.ipynb" 2>/dev/null \
      | grep -c "^### CELL")
  echo "  shard${SH} error cells: ${N:-unknown}"
done
echo "=== STAGE 2 SELECTED CONFIGURATIONS ==="
cat "$NEW"/outputs/bert_models/stage2_regularisation_shard*.csv 2>/dev/null | head -60
echo "=== OVERFITTING AUDIT AFTER STAGE 2 ==="
cat "$NEW"/outputs/bert_models/overfitting_audit_shard*.csv 2>/dev/null
echo "=== TEST RESULTS AFTER STAGE 2 ==="
cat "$NEW"/outputs/bert_models/test_results_shard*.csv 2>/dev/null
echo "STAGE2_COMPLETE"
