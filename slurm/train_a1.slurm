#!/bin/bash
#SBATCH --job-name=wc-a1
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/a1_%j.out
# A1 multi-model station correction with neural operators. Each line of the arguments is one
# run, executed one after another on the same GPU:
#   sbatch slurm/train_a1.slurm "--model fno --setting including" "--model gino --setting excluding --fold 0"
set -euo pipefail
ulimit -c 0
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO:/cluster/projects/nn8106k/siyan/envs/wfno-arm312-extra" OMP_NUM_THREADS=4 TORCH_DISABLE_NATIVE_JIT=1
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" || exit 2
for args in "$@"; do
  echo "=== $(date -Is) $args"
  srun "$VENV/bin/python" -u -m windcast.training.train_station_operator $args || echo "=== FAILED: $args"
done
