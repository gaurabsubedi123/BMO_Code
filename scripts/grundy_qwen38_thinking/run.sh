#!/bin/bash
# Waits for the copy and the venv, then --check (must pass), then the full run.
B=$HOME/Nov_13_BMO; Q=$B/qwen38_thinking; source $B/env.sh
export PATH=$B/venv_vllm/bin:$PATH
export BMO_OUT=$Q/outputs
until grep -q COPY_DONE $B/copy.log 2>/dev/null && grep -q SETUP_DONE $Q/logs/setup.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
cd $Q/code/analysis
$B/venv_vllm/bin/python qwen38_thinking.py --check > $Q/logs/check.log 2>&1
grep -q "CHECK PASSED" $Q/logs/check.log || { echo "CHECK FAILED, no full run" >> $Q/logs/check.log; exit 2; }
$B/venv_vllm/bin/python qwen38_thinking.py > $Q/logs/run.log 2>&1
test -s $BMO_OUT/summary.csv && test -s $BMO_OUT/test_predictions.csv && grep -q "VERIFY PASSED" $Q/logs/run.log \
  && echo "ALL DONE $(date)" >> $Q/logs/run.log || echo "RUN INCOMPLETE $(date)" >> $Q/logs/run.log
