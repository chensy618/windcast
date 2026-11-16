#!/bin/bash
#SBATCH --job-name=wc-smoke
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/smoke_%j.out
# End-to-end pipeline check: train every model 2 epochs on a few days, evaluate, baselines.
set -uo pipefail
ulimit -c 0
REPO=/cluster/home/siyan/github/windcast
VENV=/cluster/projects/nn8106k/siyan/envs/wfno-arm312
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=4 TORCH_DISABLE_NATIVE_JIT=1
PY="$VENV/bin/python -u -m windcast.training.train --config configs/smoke.yaml"
R=/cluster/work/projects/nn8106k/siyan/windcast/runs_smoke
for m in unet fno ufno; do
  rm -rf $R/smoke_$m
  $PY --run-name smoke_$m --set model.name=$m && $PY --eval $R/smoke_$m/best.pt --split test | tail -n 4
  echo "== $m exit $?"
done
$PY --baseline --split test
