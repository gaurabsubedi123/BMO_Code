#!/bin/bash
# ---------------------------------------------------------------------------
# BMO re-run driver.
#
# Executes the notebooks in order with nbconvert, so each committed notebook
# carries its own figures and printed diagnostics inline.
#
# Notebook 05 (transformer fine-tuning) is split across both L40s: two
# processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES, each handling
# alternating models from the roster. Model-level parallelism rather than
# data parallelism -- the models are small and independent, so this is both
# simpler and better utilised than DDP would be.
#
# Designed to run detached under tmux and survive disconnection.
#   tmux new-session -d -s bmo "bash run_all.sh"
#   tmux attach -t bmo          # to watch
#   tail -f logs/*.log          # or just read the logs
# ---------------------------------------------------------------------------
set -o pipefail
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW"
mkdir -p logs notebooks_executed

NB="$NEW/venv/bin/jupyter nbconvert --to notebook --execute --allow-errors
    --ExecutePreprocessor.timeout=-1 --output-dir=$NEW/notebooks_executed"

stamp () { date +%F_%T; }

run_nb () {           # run_nb <notebook> [extra env assignments...]
    local nb="$1"; shift
    local base; base=$(basename "$nb" .ipynb)
    echo "[$(stamp)] START  $base $*"
    local t0=$SECONDS
    # `env VAR= cmd` sets VAR to the empty string, which is what hides GPUs
    # from a child process; quoting matters here.
    env "$@" $NB "$nb" > "logs/${base}.log" 2>&1
    local rc=$?
    echo "[$(stamp)] $([ $rc -eq 0 ] && echo DONE || echo FAILED) $base  ($((SECONDS-t0))s, rc=$rc)"
    return $rc
}

echo "=========================================================="
echo " BMO re-run  |  started $(stamp)"
echo " run dir: $NEW"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "=========================================================="

run_nb notebooks/00_setup_and_environment.ipynb || exit 1
run_nb notebooks/01_dataset_and_corpus.ipynb    || exit 1

# ---------------------------------------------------------------------------
# ORDERING NOTE: all GPU work is deliberately front-loaded.
#
# This node's GPUs are shared -- an ollama service holds ~23 GB per card when a
# model is resident, and another user runs GPU work in the evening. The
# classical tier is pure CPU (GridSearchCV over TF-IDF and averaged embeddings)
# and therefore has no GPU contention at all, so it is moved to the END of the
# pipeline. That way the transformer and LLM stages, which are the only ones
# that need significant VRAM, complete as early as possible and release the
# cards well before the evening.
#
# The dependency order still holds: notebook 07 consumes the outputs of 02, 05
# and 06, and it runs after all three.
# ---------------------------------------------------------------------------

# Transformers: two shards, one per GPU, in parallel.
echo "[$(stamp)] START  05_transformer_finetuning (2 shards, GPU 0 + GPU 1)"
t0=$SECONDS
( env CUDA_VISIBLE_DEVICES=0 BMO_SHARD=0 BMO_NSHARD=2 \
      $NB --output 05_transformer_finetuning_shard0.ipynb \
      notebooks/05_transformer_finetuning.ipynb > logs/05_shard0.log 2>&1 ) &
P0=$!
( env CUDA_VISIBLE_DEVICES=1 BMO_SHARD=1 BMO_NSHARD=2 \
      $NB --output 05_transformer_finetuning_shard1.ipynb \
      notebooks/05_transformer_finetuning.ipynb > logs/05_shard1.log 2>&1 ) &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[$(stamp)] DONE   05_transformer_finetuning ($((SECONDS-t0))s, shard0=$R0 shard1=$R1)"
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] || echo "WARNING: a transformer shard failed; see logs/05_shard*.log"

# Prompt-based LLMs: 7B/8B models, one GPU is enough and keeps memory simple.
run_nb notebooks/06_prompt_based_llms.ipynb CUDA_VISIBLE_DEVICES=0

# GPU work is now finished. The classical tier is CPU-only, so it runs last and
# cannot collide with anything else using the cards.
#
# CUDA_VISIBLE_DEVICES="" is deliberate and not cosmetic. GridSearchCV spawns
# 32 joblib workers, and something in the import chain (xgboost's CUDA probe)
# initialises a CUDA context in every one of them. Each empty context costs
# ~426 MB, so 32 workers silently reserve ~14 GB of VRAM to run scikit-learn on
# the CPU. On a shared node that is memory taken from other people for no
# benefit. Hiding the devices makes the workers CPU-only, as intended.
echo "[$(stamp)] GPU stages complete; releasing cards"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
run_nb notebooks/02_classical_tiers.ipynb CUDA_VISIBLE_DEVICES="" || exit 1
echo "[$(stamp)] classical tier done; VRAM held by this run:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run_nb notebooks/07_consolidated_evaluation.ipynb || exit 1
run_nb notebooks/08_significance_and_errors.ipynb || exit 1
run_nb notebooks/10_deployment_export.ipynb CUDA_VISIBLE_DEVICES=0
run_nb notebooks/09_figures_and_tables.ipynb

echo "=========================================================="
echo " BMO re-run  |  finished $(stamp)"
echo "=========================================================="
if [ -f outputs/final_results/MASTER_test_all_systems.csv ]; then
    "$NEW/venv/bin/python" - <<'PY'
import pandas as pd
d = pd.read_csv("outputs/final_results/MASTER_test_all_systems.csv")
print(f"\n{len(d)} systems evaluated. Top 10 by weighted F1:\n")
print(d.head(10)[["System","Tier","weighted_f1","biomarker_f1","errors"]].to_string(index=False))
PY
fi
echo "RUN_ALL_COMPLETE"
