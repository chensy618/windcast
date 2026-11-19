#!/bin/bash
#SBATCH --job-name=wc-cpu
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/cpu_%j.out
# Runs Python scripts of this repo on a CPU node with the windcast-x86 env, one after the other
# (each command is one quoted argument):
#   sbatch slurm/cpu_script.slurm "scripts/build_multimodel_samples.py --start 2024-03-01 --end 2025-09-12 --nordic"
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
# windcast-x86 links to the zen4 Python of the login nodes; the CPU nodes are zen5 (see
# slurm/run_notebook.slurm): a throw-away venv around the zen5 build of the same Python
ENV=${WC_ENV:-/cluster/projects/nn8106k/siyan/envs/windcast-x86}   # WC_ENV=.../dl-x86 for eccodes
ARCH=$(ls -d /cluster/software/NRIS/zen5 2>/dev/null || echo /cluster/software/NRIS/zen4)
BASE=$ARCH/software/Python/3.11.5-GCCcore-13.2.0/bin
V=${TMPDIR:-/tmp}/wc-venv-$SLURM_JOB_ID
mkdir -p "$V/bin"
ln -sf "$BASE/python3.11" "$V/bin/python"
ln -sfn "$ENV/lib" "$V/lib"
sed "s#^home = .*#home = $BASE#; s#^executable = .*#executable = $BASE/python3.11#" "$ENV/pyvenv.cfg" > "$V/pyvenv.cfg"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for c in "$@"; do
  echo "=== $(date -Is) $c"
  # shellcheck disable=SC2086
  "$V/bin/python" -u $c
done
