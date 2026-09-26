#!/bin/bash
# Qwen3.8-27B chain: wait for setup, then check, then full run, then verify outputs.
# Each step gates the next. qwen38_zeroshot.py itself waits for the 17:00 curfew
# to pass and for GPU memory to stay free before it loads the model.
set -uo pipefail
P=/mnt/subedi-research/projects/bmo_qwen38_2026
export HF_HOME=/mnt/subedi-research/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1                      # use only the pinned, downloaded snapshot
export BMO_RUN_DIR=/mnt/subedi-research/projects/bmo_rerun_2026
export BMO_QWEN38_OUT=$P/outputs
export PYTHONDONTWRITEBYTECODE=1             # write nothing into the paper's run dir
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                   # logs show progress live, not at exit
export BMO_LLM_CURFEW_HOUR=24               # curfew off (user, 2026-09-25); the free-VRAM wait still applies
PY=$P/venv312/bin/python
cd $P/analysis

echo "$(date) waiting for setup"
until grep -q SETUP312_DONE $P/logs/setup312.log 2>/dev/null; do
    if ! tmux has-session -t qwen38_setup312 2>/dev/null; then
        echo "$(date) setup session ended without SETUP312_DONE; see setup312.log"; exit 1
    fi
    sleep 60
done
REV=$(cat $P/logs/revision.txt)
echo "$(date) setup done, revision $REV"

echo "$(date) CHECK"
$PY qwen38_zeroshot.py --check --revision "$REV" > $P/logs/check.log 2>&1
rc=$?
cat $P/logs/check.log | tail -30
if [ $rc -ne 0 ] || ! grep -q "CHECK PASSED" $P/logs/check.log; then
    echo "$(date) CHECK did not pass (exit $rc); full run NOT started"; exit 1
fi

echo "$(date) FULL RUN"
$PY qwen38_zeroshot.py --revision "$REV" > $P/logs/full.log 2>&1
rc=$?
tail -20 $P/logs/full.log

# an exit code of 0 is not evidence: require the outputs themselves
$PY - <<'PYV'
import json, pandas as pd, sys
o = "/mnt/subedi-research/projects/bmo_qwen38_2026/outputs"
p = pd.read_csv(f"{o}/test_predictions.csv"); r = pd.read_csv(f"{o}/test_results.csv")
s = pd.read_csv(f"{o}/prompt_selection.csv"); m = json.load(open(f"{o}/run_meta.json"))
assert len(p) == 341 and len(r) == 2 and len(s) == 4, (len(p), len(r), len(s))
assert p.p_yes.between(0, 1).all() and p.p_yes.std() > 0.05, "probabilities look degenerate"
assert m["thinking"] == "disabled"
print("VERIFY PASSED")
print(r[["Model", "threshold", "errors", "weighted_f1"]].to_string(index=False))
PYV
echo "$(date) chain finished (full-run exit $rc)"
