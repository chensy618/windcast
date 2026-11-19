#!/bin/bash
#SBATCH --job-name=wc-nb
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/nb_%j.out
# Executes notebooks in place on a CPU node (no cell timeout):
#   sbatch slurm/run_notebook.slurm notebooks/02_experiments.ipynb
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
# windcast-x86 links to the zen4 Python of the login nodes; the CPU nodes are zen5, so build a
# throw-away venv around the zen5 build of the same Python that reuses the env's site-packages
ENV=/cluster/projects/nn8106k/siyan/envs/windcast-x86
ARCH=$(ls -d /cluster/software/NRIS/zen5 2>/dev/null || echo /cluster/software/NRIS/zen4)
BASE=$ARCH/software/Python/3.11.5-GCCcore-13.2.0/bin
V=${TMPDIR:-/tmp}/wc-venv-$SLURM_JOB_ID
mkdir -p "$V/bin"
ln -sf "$BASE/python3.11" "$V/bin/python"
ln -sfn "$ENV/lib" "$V/lib"
sed "s#^home = .*#home = $BASE#; s#^executable = .*#executable = $BASE/python3.11#" "$ENV/pyvenv.cfg" > "$V/pyvenv.cfg"
PY=$V/bin/python
"$PY" -c "import sklearn, nbconvert, numpy; print('python ok', numpy.__version__, sklearn.__version__)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for nb in "$@"; do
  echo "=== $(date -Is) $nb"
  cd "$REPO/$(dirname "$nb")"
  "$PY" -m nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 "$(basename "$nb")" || echo "=== FAILED: $nb"
done
