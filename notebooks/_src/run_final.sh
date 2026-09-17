#!/bin/bash
# ---------------------------------------------------------------------------
# Final pass: the classical tier (restructured to precompute embeddings) and
# every downstream stage.
#
# The earlier recovery/finalize scripts waited on the classical process; when it
# was killed for restructuring they proceeded and produced consolidation output
# from incomplete data. Those stages are therefore repeated here against the
# complete result set.
#
# CUDA_VISIBLE_DEVICES="" on the classical tier: it is CPU-only, and leaving the
# GPUs visible makes each of the 32 joblib workers reserve ~426 MB of VRAM.
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
    local rc=$?
    echo "[$(date +%F_%T)] DONE   $nb ($((SECONDS-t0))s rc=$rc)"
    # nbconvert exits 0 even when cells raise, so count failures explicitly.
    local errs
    errs=$("$NEW/venv/bin/python" /tmp/nberr2.py \
           "$NEW/notebooks_executed/$(basename "$nb")" 2>/dev/null \
           | grep -c "^### CELL")
    echo "[$(date +%F_%T)]        error cells: ${errs:-unknown}"
}

# CPU-only, and GPUs hidden. XGBoost's CUDA path was benchmarked and rejected:
# it is slower than correctly-parallelised CPU and returns a different F1 on
# sparse input. Hiding the devices also stops joblib workers creating stray
# CUDA contexts (~426 MB each) for a tier that never needs them.
run_nb notebooks/02_classical_tiers.ipynb        02_final CUDA_VISIBLE_DEVICES=""
run_nb notebooks/07_consolidated_evaluation.ipynb 07_v2
run_nb notebooks/08_significance_and_errors.ipynb 08_v2
run_nb notebooks/10_deployment_export.ipynb       10_v2 CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_v2

echo "=== FINAL MASTER TABLE ==="
"$NEW/venv/bin/python" - <<'PY'
import pandas as pd, pathlib
f = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026/outputs/final_results/MASTER_test_all_systems.csv")
if f.exists():
    d = pd.read_csv(f)
    print(f"{len(d)} systems\n")
    print(d[["System","Tier","weighted_f1","biomarker_f1","errors"]].round(4).to_string(index=False))
else:
    print("MASTER table missing")
PY
echo "=== REPRESENTATION EFFECT ==="
cat "$NEW/outputs/classical_ml/representation_effect.csv" 2>/dev/null || echo "(missing)"
echo "RUN_FINAL_COMPLETE"
