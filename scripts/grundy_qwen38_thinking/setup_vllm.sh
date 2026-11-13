#!/bin/bash
# Build the venv for the Qwen3.8 thinking run. Ends with SETUP_DONE only if vLLM imports.
set -e
B=$HOME/Nov_13_BMO; cd $B/qwen38_thinking
python3.12 -m venv $B/venv_vllm
$B/venv_vllm/bin/pip install -q --upgrade pip
$B/venv_vllm/bin/pip install -q vllm pandas scikit-learn huggingface_hub
$B/venv_vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_DONE
