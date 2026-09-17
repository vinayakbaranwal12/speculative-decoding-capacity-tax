#!/usr/bin/env bash
# MKTG-3389 Phase 0 bootstrap — pin exact versions from the execution prompt.
set -euo pipefail
exec > >(tee -a /root/mktg3389/logs/00-environment-verification.log) 2>&1
mkdir -p /root/mktg3389/{logs,results,traces,env,harness,analysis}
cd /root/mktg3389

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) BOOTSTRAP START ====="
echo "=== nvidia-smi ==="
nvidia-smi
echo "=== GPU query ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# AI/ML Ready usually has CUDA + python. Create venv for pinned stack.
python3 --version
python3 -m venv /root/mktg3389/venv
source /root/mktg3389/venv/bin/activate
pip install -U pip wheel setuptools

# Install exact pins from prompt
echo "=== installing vLLM 0.27.1 + torch pins ==="
pip install 'vllm==0.27.1'
python - <<'PY'
import torch, torchvision, importlib.metadata as m
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
try:
    print('triton', m.version('triton'))
except Exception as e:
    print('triton_err', e)
import vllm
print('vllm', vllm.__version__)
PY

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) BOOTSTRAP PIP DONE ====="
