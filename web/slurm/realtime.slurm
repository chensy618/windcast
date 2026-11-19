#!/bin/bash
#SBATCH --job-name=wc-realtime
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/realtime_%j.out
# One real-time cycle of the website (web/pipeline/run_latest.py): newest run, downloads,
# forecasts of every configuration, observations. Scheduled by web/slurm/scrontab.txt; by hand:
#   sbatch web/slurm/realtime.slurm [--init 2026-11-18T12]
# WINDCAST_DEPLOY (optional): shell command run afterwards, e.g. an rsync of web/site to a server.
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO:/cluster/projects/nn8106k/siyan/envs/wfno-arm312-extra" OMP_NUM_THREADS=8 TORCH_DISABLE_NATIVE_JIT=1
srun "$VENV/bin/python" -u web/pipeline/run_latest.py "$@"
