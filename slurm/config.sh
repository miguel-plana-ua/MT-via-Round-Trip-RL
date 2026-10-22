# Cluster settings used by setup_env.sh / submit.sh. EDIT THIS FILE for your cluster, nothing else needs touching.
# (Anything can also be overridden from the shell: PARTITION=gpu TIME=12:00:00 slurm/submit.sh ...)

export CONDA_ENV="${CONDA_ENV:-mtrl}"          # name of the conda env created by setup_env.sh
export PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"   # cu118 if the driver is old

export PARTITION="${PARTITION:-compute}"       # abaco: single partition "compute" (default)
export ACCOUNT="${ACCOUNT:-}"                  # empty = none
export GRES="${GRES:-gpu:2}"                   # abaco: 8 untyped GPUs (gpu:8), we ask for 2
export CPUS="${CPUS:-6}"                       # abaco has 20 cores shared with colleagues
export MEM="${MEM:-32G}"                       # abaco has 125G in total; the model is loaded twice (policy + reference)
export TIME="${TIME:-24:00:00}"                # partition has no limit; be polite and ask only for what the stage needs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # put it on a disk with >=10 GB free
export OFFLINE="${OFFLINE:-0}"                 # 1 if compute nodes have no internet (run prefetch.py on the login node first)
