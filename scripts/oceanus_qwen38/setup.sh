#!/bin/bash
# Rebuild on Python 3.12: oceanus has headers (Python.h) only for 3.12, and
# Triton compiles a C helper against them the first time a kernel runs.
set -euo pipefail
P=/mnt/subedi-research/projects/bmo_qwen38_2026
/usr/bin/python3.12 -m venv $P/venv312
$P/venv312/bin/pip install -q --upgrade pip
$P/venv312/bin/pip install -q torch torchvision transformers accelerate huggingface_hub \
    numpy pandas matplotlib scikit-learn pillow
$P/venv312/bin/pip freeze > $P/logs/pip_freeze_py312.txt
# prove the failing step works before any GPU time is spent
$P/venv312/bin/python - <<'PY'
import torch, transformers, triton
from triton.runtime.driver import driver
driver.active.get_current_target()          # compiles cuda_utils; this is what failed on 3.11
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "transformers", transformers.__version__, "triton", triton.__version__, "triton driver OK")
PY
echo SETUP312_DONE
