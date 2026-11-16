#!/bin/bash
#SBATCH --job-name=wc-train
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/train_%j.out
# GH200 (aarch64) node, wfno-arm312 venv. Arguments go to windcast.training.train:
#   sbatch slurm/train.slurm --config configs/f1_meps.yaml --run-name f1_fno_s0
#   sbatch slurm/train.slurm --config configs/f1_meps.yaml --run-name f1_unet_s0 --set model.name=unet
#   sbatch --time=02:00:00 slurm/train.slurm --config configs/f1_meps.yaml --baseline --split test
set -euo pipefail
ulimit -c 0                          # this venv can crash at exit and dump a ~40 GB core
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=4 TORCH_DISABLE_NATIVE_JIT=1
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" || exit 2
srun "$VENV/bin/python" -u -m windcast.training.train "$@"
