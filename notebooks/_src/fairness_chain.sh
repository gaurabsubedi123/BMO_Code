#!/bin/bash
# ---------------------------------------------------------------------------
# Two fairness/robustness additions, queued behind the improvement chain.
#
#  A. Prompt-search parity (notebook 06). The two new instruct models were given
#     a three-variant prompt search while BART, Flan-T5 and BioGPT each got a
#     single hand-written prompt -- an asymmetry favouring the new models over
#     the baselines they are compared against. A diagnostic showed the effect is
#     large: Flan-T5 moved from AUC 0.45 to 0.80 on prompt choice alone, and its
#     label words alone swung it from 0.29 to 0.81. Every prompt-based system
#     now gets the same search on the same holdout.
#
#  B. Seed stability (notebook 12). Across three runs of an identical protocol
#     on a fixed split, four different models have led. Each transformer is
#     refit under five seeds with everything else held constant, so the spread
#     isolates training stochasticity and can be reported as a number.
#
# Then the downstream stages and verification are re-run over the new results.
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

echo "[$(date +%F_%T)] waiting for the improvement chain to finish..."
while pgrep -f "improve_chain" > /dev/null 2>&1; do sleep 30; done
echo "[$(date +%F_%T)] improvement chain done"

# ---- A. prompt parity ------------------------------------------------------
run_nb notebooks/06_prompt_based_llms.ipynb 06_fair CUDA_VISIBLE_DEVICES=0
echo "=== PROMPT SEARCH, ALL SYSTEMS ==="
"$NEW/venv/bin/python" -c "
import pandas as pd, pathlib
f=pathlib.Path('$NEW/outputs/prompt_llm/prompt_selection_all.csv')
if f.exists():
    d=pd.read_csv(f)
    for m,g in d.groupby('Model'):
        b=g.loc[g.holdout_weighted_f1.idxmax()]
        print(f'  {m:24s} {len(g)} variants | best {b.variant} '
              f'(holdout F1 {b.holdout_weighted_f1:.4f}, AUC {b.holdout_auc:.4f})')
"

# ---- B. seed stability, sharded across both GPUs ---------------------------
echo "[$(date +%F_%T)] seed stability on both GPUs"
for SH in 0 1; do
  ( env CUDA_VISIBLE_DEVICES=$SH BMO_SHARD=$SH BMO_NSHARD=2 \
      "$JUP" nbconvert --to notebook --execute --allow-errors \
      --ExecutePreprocessor.timeout=-1 --output-dir="$NEW/notebooks_executed" \
      --output 12_seed_stability_shard${SH}.ipynb \
      notebooks/12_seed_stability.ipynb > logs/12_seed_shard${SH}.log 2>&1 ) &
done
wait
for SH in 0 1; do
  N=$("$NEW/venv/bin/python" /tmp/nberr2.py \
      "$NEW/notebooks_executed/12_seed_stability_shard${SH}.ipynb" 2>/dev/null | grep -c "^### CELL")
  echo "  seed-stability shard${SH} error cells: ${N:-unknown}"
done
echo "=== SEED STABILITY ==="
"$NEW/venv/bin/python" -c "
import pandas as pd, glob
fs=sorted(glob.glob('$NEW/outputs/bert_models/seed_stability_summary_shard*.csv'))
if fs:
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).sort_values('errors_mean')
    print(d.round(4).to_string(index=False))
    print()
    print('mean sd across seeds: %.2f errors' % d.errors_sd.mean())
    print('largest range       : %d errors' % (d.errors_max-d.errors_min).max())
"

# ---- downstream + verification --------------------------------------------
run_nb notebooks/07_consolidated_evaluation.ipynb 07_fair
run_nb notebooks/08_significance_and_errors.ipynb 08_fair
run_nb notebooks/10_deployment_export.ipynb       10_fair CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb      09_fair
run_nb notebooks/11_verification.ipynb            11_fair

echo "=== VERIFICATION ==="
"$NEW/venv/bin/python" -c "
import json,pathlib
r=json.load(open('$NEW/outputs/provenance/verification_report.json'))
print('verdict:',r['verdict'],'| passed',r['n_passed'],'| warnings',r['n_warnings'],'| failed',r['n_failed'])
for x in r['failed']: print('  FAIL:',x)
for x in r['warnings']: print('  WARN:',x[:200])
"
echo "FAIRNESS_CHAIN_COMPLETE"
