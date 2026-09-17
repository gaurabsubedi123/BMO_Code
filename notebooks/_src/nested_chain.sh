#!/bin/bash
# ---------------------------------------------------------------------------
# Nested regularisation selection, then verification -- as one chain.
#
# Chained rather than run as independent watchers because the verification pass
# waits on "no 05_transformer_finetuning process running". Between shard 0's
# stage-2 run ending and the nested run starting there is a gap in which that
# condition is briefly true, and verification would fire against stage-2 results
# that nested is about to overwrite. One sequential chain removes the race.
#
# Steps: wait for stage 2 -> smoke the nested path -> run nested on both GPUs
#        -> clean stale outputs -> 07, 08, 10, 09 -> 11 (verification).
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
    echo "[$(date +%F_%T)] DONE   $nb ($((SECONDS-t0))s)"
    local n
    n=$("$NEW/venv/bin/python" /tmp/nberr2.py \
        "$NEW/notebooks_executed/$(basename "$nb")" 2>/dev/null | grep -c "^### CELL")
    echo "[$(date +%F_%T)]        error cells: ${n:-unknown}"
}

echo "[$(date +%F_%T)] waiting for the stage-2 run to finish..."
while pgrep -f "05_transformer_finetuning" > /dev/null 2>&1; do sleep 30; done
echo "[$(date +%F_%T)] stage 2 done; archiving its results for comparison"
mkdir -p "$NEW/outputs/bert_models_stage2_flat"
cp "$NEW"/outputs/bert_models/*.csv "$NEW/outputs/bert_models_stage2_flat/" 2>/dev/null
echo "[$(date +%F_%T)]   archived $(ls "$NEW/outputs/bert_models_stage2_flat" | wc -l) files"

# ---- smoke the nested path -------------------------------------------------
echo "[$(date +%F_%T)] smoke-testing the nested path"
rm -rf "$NEW/outputs/bert_models_smoke" /tmp/smoke_nested.ipynb
env CUDA_VISIBLE_DEVICES=1 BMO_SMOKE=1 BMO_STAGE2=1 BMO_NESTED=1 \
    BMO_SHARD=0 BMO_NSHARD=9 \
    "$JUP" nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=-1 --output-dir=/tmp \
    --output smoke_nested.ipynb notebooks/05_transformer_finetuning.ipynb \
    > /tmp/smoke_nested.log 2>&1
ERRS=$("$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke_nested.ipynb 2>/dev/null | grep -c "^### CELL")
NESTFILE=$(ls "$NEW"/outputs/bert_models_smoke/nested_choices_shard*.csv 2>/dev/null | head -1)
echo "[$(date +%F_%T)] smoke error cells: ${ERRS:-?} | nested_choices: ${NESTFILE:-MISSING}"
if [ "${ERRS:-1}" -ne 0 ] || [ -z "$NESTFILE" ]; then
    echo "[$(date +%F_%T)] ABORT: nested smoke failed or the nested path never ran"
    "$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke_nested.ipynb 2>/dev/null | head -20
    exit 1
fi
echo "[$(date +%F_%T)] nested path confirmed:"
cat "$NESTFILE"
rm -rf "$NEW/outputs/bert_models_smoke" "$NEW/models/smoke"

# ---- the real nested run ---------------------------------------------------
echo "[$(date +%F_%T)] launching nested selection on both GPUs (~2.8 h)"
for SH in 0 1; do
  ( env CUDA_VISIBLE_DEVICES=$SH BMO_STAGE2=1 BMO_NESTED=1 BMO_SHARD=$SH BMO_NSHARD=2 \
      "$JUP" nbconvert --to notebook --execute --allow-errors \
      --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
      --output 05_transformer_finetuning_shard${SH}.ipynb \
      notebooks/05_transformer_finetuning.ipynb > logs/05_nested_shard${SH}.log 2>&1 ) &
done
wait
echo "[$(date +%F_%T)] nested selection finished"
for SH in 0 1; do
  N=$("$NEW/venv/bin/python" /tmp/nberr2.py \
      "$NEW/notebooks_executed/05_transformer_finetuning_shard${SH}.ipynb" 2>/dev/null \
      | grep -c "^### CELL")
  echo "  shard${SH} error cells: ${N:-unknown}"
done

echo "=== PER-FOLD REGULARISATION CHOICES ==="
cat "$NEW"/outputs/bert_models/nested_choices_shard*.csv 2>/dev/null | head -60
echo "=== OVERFITTING AUDIT (nested) ==="
cat "$NEW"/outputs/bert_models/overfitting_audit_shard*.csv 2>/dev/null

# ---- clean, re-derive, verify ---------------------------------------------
echo "[$(date +%F_%T)] removing stale downstream outputs"
"$NEW/venv/bin/python" - <<'PY'
import pathlib
run = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026")
files = [f for d in (run/"outputs"/"final_results", run/"tables") if d.exists()
         for f in d.glob("*")]
if files:
    newest = max(f.stat().st_mtime for f in files)
    stale = [f for f in files if newest - f.stat().st_mtime > 3600]
    for f in stale:
        print("  removing", f.name); f.unlink()
    print(f"  removed {len(stale)} stale file(s)")
PY

run_nb notebooks/07_consolidated_evaluation.ipynb 07_nested
run_nb notebooks/08_significance_and_errors.ipynb 08_nested
run_nb notebooks/10_deployment_export.ipynb       10_nested CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_nested
run_nb notebooks/11_verification.ipynb            11_nested

echo "=== VERIFICATION VERDICT ==="
"$NEW/venv/bin/python" -c "
import json,pathlib
p=pathlib.Path('$NEW/outputs/provenance/verification_report.json')
if p.exists():
    r=json.load(open(p))
    print('verdict:',r['verdict'],'| passed',r['n_passed'],'| warnings',r['n_warnings'],'| failed',r['n_failed'])
    for x in r['failed']: print('  FAIL:',x)
    for x in r['warnings']: print('  WARN:',x)
else: print('no verification report')
"
echo "NESTED_CHAIN_COMPLETE"
