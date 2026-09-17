#!/bin/bash
# ---------------------------------------------------------------------------
# Gate the expensive run behind the smoke test.
#
# Three long runs today were launched carrying a defect that a two-minute
# execution would have exposed. This waits for the smoke run (all grids reduced
# to a single point) to finish, refuses to proceed unless it completed with zero
# error cells, and only then starts the real thing.
# ---------------------------------------------------------------------------
set -u
NEW=/mnt/subedi-research/projects/bmo_rerun_2026
. "$NEW/env.sh"
cd "$NEW" || exit 1

echo "[$(date +%F_%T)] waiting for the smoke run..."
while pgrep -f "smoke02" > /dev/null 2>&1; do sleep 15; done

if [ ! -f /tmp/smoke02.ipynb ]; then
    echo "[$(date +%F_%T)] ABORT: smoke run produced no notebook"; exit 1
fi
ERRS=$("$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke02.ipynb 2>/dev/null | grep -c "^### CELL")
echo "[$(date +%F_%T)] smoke run error cells: ${ERRS:-unknown}"
if [ "${ERRS:-1}" -ne 0 ]; then
    echo "[$(date +%F_%T)] ABORT: smoke run failed -- not launching the full run"
    "$NEW/venv/bin/python" /tmp/nberr2.py /tmp/smoke02.ipynb 2>/dev/null | head -25
    exit 1
fi

# Confirm the smoke run actually produced all three tiers, not just one.
"$NEW/venv/bin/python" - <<'PY'
import pandas as pd, pathlib, sys
f = pathlib.Path("/mnt/subedi-research/projects/bmo_rerun_2026/outputs/classical_ml_smoke/cv_results_all_tiers.csv")
if not f.exists():
    print("ABORT: smoke run did not write cv_results_all_tiers.csv"); sys.exit(1)
d = pd.read_csv(f)
tiers = sorted(d["Tier"].unique())
print("smoke tiers:", tiers, "| rows:", len(d))
if len(tiers) < 3:
    print("ABORT: expected 3 tiers"); sys.exit(1)
print("smoke run validated -- all tiers present")
PY
[ $? -ne 0 ] && exit 1

rm -rf "$NEW/outputs/classical_ml_smoke"
echo "[$(date +%F_%T)] smoke passed; launching the full run"
exec bash "$NEW/run_final.sh"
