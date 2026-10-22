#!/bin/bash
# One-off environment setup. Run it on the login node:  bash slurm/setup_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/config.sh
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CONDA_ENV} "; then
    conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
fi
conda activate "$CONDA_ENV"

# torch with CUDA (2080 = Turing, sm_75, supported by the cu126 wheels)
pip install "torch==2.8.0" --index-url "$TORCH_INDEX"
pip install -r slurm/requirements-server.txt
# only needed for --mem tight (8-bit AdamW):  pip install bitsandbytes

python scripts/check_env.py
