#!/bin/bash
# Generic job. Do not call directly, use slurm/submit.sh:
#   slurm/submit.sh pilot --lang ayr_Latn --seed 27
# Everything after the script name is passed to scripts/run_stage.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/config.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less memory fragmentation on 11 GB cards
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "args: $*"

EXTRA=()
if [ "$OFFLINE" = "1" ]; then EXTRA+=(--offline); fi
python scripts/run_stage.py "$@" ${EXTRA[@]+"${EXTRA[@]}"}
