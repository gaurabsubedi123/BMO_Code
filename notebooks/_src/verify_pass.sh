#!/bin/bash
# ---------------------------------------------------------------------------
# Final pass: clean, re-derive, verify.
#
# Waits for the classical tier and the stage-2 transformer search to finish,
# then:
#   1. DELETES every result file older than the newest one by more than an
#      hour. Aborted runs earlier today wrote outputs from incomplete data, and
#      a surviving stale file is precisely how two runs get silently pooled --
#      the failure that started this project.
#   2. Re-runs consolidation, significance, export and figures from clean.
#   3. Runs notebook 11, which re-derives every metric from raw predictions and
#      refuses to pass if anything disagrees.
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

echo "[$(date +%F_%T)] waiting for the classical tier and stage 2..."
while pgrep -f "02_classical_tiers.ipynb|05_transformer_finetuning" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%F_%T)] both finished"

echo "[$(date +%F_%T)] removing stale result files"
"$NEW/venv/bin/python" - <<'PY'
import pathlib, time
run = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026")
dirs = [run/"outputs"/d for d in ("final_results",)] + [run/"tables"]
files = [f for d in dirs if d.exists() for f in d.glob("*")] 
if files:
    newest = max(f.stat().st_mtime for f in files)
    stale = [f for f in files if newest - f.stat().st_mtime > 3600]
    for f in stale:
        print(f"  removing stale {f.name} ({(newest-f.stat().st_mtime)/3600:.1f} h old)")
        f.unlink()
    print(f"  removed {len(stale)} stale file(s)")
else:
    print("  nothing to clean")
PY

run_nb notebooks/07_consolidated_evaluation.ipynb 07_final
run_nb notebooks/08_significance_and_errors.ipynb 08_final
run_nb notebooks/10_deployment_export.ipynb       10_final CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_final
run_nb notebooks/11_verification.ipynb            11_verify

echo "=== VERIFICATION VERDICT ==="
"$NEW/venv/bin/python" -c "
import json,pathlib
p=pathlib.Path('$NEW/outputs/provenance/verification_report.json')
if p.exists():
    r=json.load(open(p))
    print('verdict:',r['verdict'],'| passed',r['n_passed'],'| warnings',r['n_warnings'],'| failed',r['n_failed'])
    for x in r['failed']:   print('  FAIL:',x)
    for x in r['warnings']: print('  WARN:',x)
else:
    print('no verification report produced')
"
echo "=== MASTER TABLE ==="
"$NEW/venv/bin/python" -c "
import pandas as pd,pathlib
f=pathlib.Path('$NEW/outputs/final_results/MASTER_test_all_systems.csv')
if f.exists():
    d=pd.read_csv(f); print(len(d),'systems'); print(d[['System','Tier','weighted_f1','biomarker_f1','errors']].round(4).to_string(index=False))
"
echo "VERIFY_PASS_COMPLETE"
