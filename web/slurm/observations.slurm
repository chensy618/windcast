#!/bin/bash
#SBATCH --job-name=wc-obs
#SBATCH --account=nn8106k
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/realtime_obs_%j.out
# Hourly refresh of the station observations shown on the website (web/pipeline/observations.py).
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
# windcast-x86 links to the zen4 Python of the login nodes; the CPU nodes are zen5: a throw-away
# venv around the zen5 build of the same Python (as slurm/cpu_script.slurm)
ENV=/cluster/projects/nn8106k/siyan/envs/windcast-x86
ARCH=$(ls -d /cluster/software/NRIS/zen5 2>/dev/null || echo /cluster/software/NRIS/zen4)
BASE=$ARCH/software/Python/3.11.5-GCCcore-13.2.0/bin
V=${TMPDIR:-/tmp}/wc-venv-$SLURM_JOB_ID
mkdir -p "$V/bin"
ln -sf "$BASE/python3.11" "$V/bin/python"
ln -sfn "$ENV/lib" "$V/lib"
sed "s#^home = .*#home = $BASE#; s#^executable = .*#executable = $BASE/python3.11#" "$ENV/pyvenv.cfg" > "$V/pyvenv.cfg"
cd "$REPO"
"$V/bin/python" -u web/pipeline/observations.py
[ -n "${WINDCAST_DEPLOY:-}" ] && bash -c "$WINDCAST_DEPLOY" || true
