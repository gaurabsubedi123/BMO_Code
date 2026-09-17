#!/bin/bash
# ---------------------------------------------------------------------------
# Two improvements, gated behind a smoke test:
#
#   1. Bootstrap confidence intervals on all 40 systems (notebook 07). Only the
#      prompt-based tier had them; without intervals the main table invites the
#      reader to trust an ordering whose top has changed between every run.
#
#   2. Gap achievability diagnostic for the classical tier (notebook 02).
#      Constraining selection on the gap was tested and REJECTED: it cost 0.017
#      weighted F1 on average and still left five of seven models above 0.04,
#      because for most of them almost no candidate satisfies it (the MLPs have
#      none in any fold). Selection is therefore by score, and the notebook
#      instead records how many candidates COULD have met the threshold -- the
#      finding being that for several classical families, none can.
#
# Both are CPU-bound; the GPUs are not involved.
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

echo "[$(date +%F_%T)] smoke-testing the classical tier"
rm -rf "$NEW/outputs/classical_ml_smoke" /tmp/smoke02b.ipynb
env CUDA_VISIBLE_DEVICES="" BMO_SMOKE=1 "$JUP" nbconvert --to notebook --execute \
    --allow-errors --ExecutePreprocessor.timeout=-1 --output-dir=/tmp \
    --output smoke02b.ipynb notebooks/02_classical_tiers.ipynb > /tmp/smoke02b.log 2>&1
ERRS=$("$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke02b.ipynb 2>/dev/null | grep -c "^### CELL")
SELFILE=$(ls "$NEW"/outputs/classical_ml_smoke/gap_achievability_*.csv 2>/dev/null | head -1)
echo "[$(date +%F_%T)] smoke error cells: ${ERRS:-?} | achievability file: ${SELFILE:-MISSING}"
if [ "${ERRS:-1}" -ne 0 ] || [ -z "$SELFILE" ]; then
    echo "[$(date +%F_%T)] ABORT: smoke failed or the achievability diagnostic never ran"
    "$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke02b.ipynb 2>/dev/null | head -20
    exit 1
fi
echo "[$(date +%F_%T)] achievability diagnostic confirmed:"; head -4 "$SELFILE"
rm -rf "$NEW/outputs/classical_ml_smoke"

run_nb notebooks/02_classical_tiers.ipynb        02_gapsel CUDA_VISIBLE_DEVICES=""
run_nb notebooks/07_consolidated_evaluation.ipynb 07_ci
run_nb notebooks/08_significance_and_errors.ipynb 08_ci
run_nb notebooks/10_deployment_export.ipynb       10_ci CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_ci
run_nb notebooks/11_verification.ipynb            11_ci

echo "=== CLASSICAL OVERFITTING (score-based selection) ==="
head -25 "$NEW/outputs/classical_ml/overfitting_audit.csv" 2>/dev/null
echo "=== COULD THE 0.04 GAP HAVE BEEN MET AT ALL? ==="
"$NEW/venv/bin/python" - <<'PY'
import pandas as pd, glob
fs = sorted(glob.glob("/mnt/subedi-research/projects/bmo_rerun_2026/outputs/classical_ml/gap_achievability_*.csv"))
if fs:
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = d.groupby(["Tier","Model"]).agg(folds=("fold","size"),
                                        folds_with_a_passing_candidate=("n_passing", lambda s:(s>0).sum()),
                                        candidates=("n_candidates","max"),
                                        selected_gap=("selected_gap","mean"),
                                        best_achievable_gap=("best_achievable_gap","mean")).round(4)
    print(g.to_string())
PY
echo "=== VERIFICATION ==="
"$NEW/venv/bin/python" -c "
import json,pathlib
p=pathlib.Path('$NEW/outputs/provenance/verification_report.json')
r=json.load(open(p))
print('verdict:',r['verdict'],'| passed',r['n_passed'],'| warnings',r['n_warnings'],'| failed',r['n_failed'])
for x in r['failed']: print('  FAIL:',x)
for x in r['warnings']: print('  WARN:',x[:200])
"
echo "IMPROVE_CHAIN_COMPLETE"
