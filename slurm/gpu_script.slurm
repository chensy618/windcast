#!/bin/bash
#SBATCH --job-name=wc-gpu
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/gpu_%j.out
# Runs one Python script of this repo on a GPU node with the training environment:
#   sbatch slurm/gpu_script.slurm scripts/predict_terrain_grid.py --run sfno_terrain
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO:/cluster/projects/nn8106k/siyan/envs/wfno-arm312-extra" OMP_NUM_THREADS=8 TORCH_DISABLE_NATIVE_JIT=1
srun "$VENV/bin/python" -u "$@"
