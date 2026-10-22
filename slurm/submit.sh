#!/bin/bash
# Submit one stage to Slurm:  slurm/submit.sh <stage> [run_stage.py options]
#   slurm/submit.sh smoke --lang ayr_Latn
#   TIME=08:00:00 slurm/submit.sh pilot --lang ayr_Latn --seed 27
# Logs go to slurm/logs/. Edit slurm/config.sh for partition/account/GPU request.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/config.sh
mkdir -p slurm/logs

# All our jobs share the name "mtrl" + --dependency=singleton: they run ONE AT A TIME, so we never hold more than
# $GRES GPUs even if several jobs are submitted at once (abaco has no per-user limit and 8 shared GPUs).
FLAGS=(--job-name=mtrl --dependency=singleton --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
       --output="slurm/logs/mtrl-$1-%j.out" --error="slurm/logs/mtrl-$1-%j.err" --export=ALL)
[ -n "$PARTITION" ] && FLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ] && FLAGS+=(--account="$ACCOUNT")
sbatch "${FLAGS[@]}" slurm/job.sbatch "$@"
