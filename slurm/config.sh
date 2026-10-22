# Cluster settings used by setup_env.sh / submit.sh. EDIT THIS FILE for your cluster, nothing else needs touching.
# (Anything can also be overridden from the shell: PARTITION=gpu TIME=12:00:00 slurm/submit.sh ...)

export CONDA_ENV="${CONDA_ENV:-mtrl}"          # name of the conda env created by setup_env.sh
export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"   # cu118 if the driver is old

export PARTITION="${PARTITION:-}"              # e.g. gpu   (empty = cluster default)
export ACCOUNT="${ACCOUNT:-}"                  # empty = none
export GRES="${GRES:-gpu:2}"                   # 2 GPUs only. If the cluster needs the type: gpu:rtx2080:2 (see `sinfo -o "%G"`)
export CPUS="${CPUS:-8}"
export MEM="${MEM:-32G}"                       # host RAM (the model is loaded twice: policy + reference)
export TIME="${TIME:-04:00:00}"                # wall-time limit; pass a larger one for long runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # put it on a disk with >=10 GB free
export OFFLINE="${OFFLINE:-0}"                 # 1 if compute nodes have no internet (run prefetch.py on the login node first)
