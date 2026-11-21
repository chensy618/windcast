#!/bin/bash
#SBATCH --job-name=wc-mngrid
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/mngrid_%j.out
# Gridded evaluation against MET Nordic (scripts/eval_metnordic_grid.py); arguments as one string:
#   sbatch slurm/eval_metnordic.slurm "--step 2"
set -euo pipefail
ulimit -c 0
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO:/cluster/projects/nn8106k/siyan/envs/wfno-arm312-extra" OMP_NUM_THREADS=4 TORCH_DISABLE_NATIVE_JIT=1
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" || exit 2
for args in "$@"; do
  echo "=== $(date -Is) $args"
  srun "$VENV/bin/python" -u scripts/eval_metnordic_grid.py $args || echo "=== FAILED: $args"
done
