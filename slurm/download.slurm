#!/bin/bash
#SBATCH --job-name=wc-download
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/projects/nn8106k/siyan/windcast/logs/download_%j.out
# All windcast downloads in one GPU job (accel nodes reach the internet through the uan03
# proxy). One stream per archive: thredds (MEPS, MET Nordic) and the ECMWF GCS mirror
# rate-limit, and the proxy keeps the login node's address. Every downloader skips what is
# already on disk, so a stream that has not printed "done:" when the job ends is resumed by
# a resubmitted job, at most MAX_RESUBMIT times.
#   sbatch slurm/download.slurm            # all streams
#   sbatch slurm/download.slurm meps ifs   # some streams
#   sbatch slurm/download.slurm aiwp arome gfs ifsno aifsno   # MLWP / NWP comparison (not default)
set -uo pipefail
ulimit -c 0
REPO=/cluster/home/siyan/github/windcast
PY=/cluster/projects/nn8106k/siyan/envs/wfno-arm312/bin/python
W=/cluster/work/projects/nn8106k/siyan/windcast
MAX_RESUBMIT=${MAX_RESUBMIT:-5}
N=${RESUBMIT_COUNT:-0}
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
STREAMS=("$@")
[ ${#STREAMS[@]} -eq 0 ] && STREAMS=(meps metnordic ifs aifs)

declare -A CMD LOG
CMD[meps]="scripts/download_meps.py --start 2018-01-01 --end 2025-12-31"
LOG[meps]=$W/meps/logs/dl_all.log
CMD[metnordic]="scripts/download_metnordic.py --start 2018-01-01 --end 2025-12-31"
LOG[metnordic]=$W/metnordic/logs/dl_all.log
CMD[ifs]="scripts/download_ecmwf.py --model ifs --start 2023-07-12 --end 2025-12-31"
LOG[ifs]=$W/ecmwf/logs/dl_ifs.log
CMD[aifs]="scripts/download_ecmwf.py --model aifs --start 2024-03-01 --end 2025-12-31"
LOG[aifs]=$W/ecmwf/logs/dl_aifs.log
# MLWP / NWP comparison: AIWP ML models (IFS-initialised) and AROME-Arctic 2.5 km
CMD[aiwp]="scripts/download_aiwp.py --models graphcast pangu fourcastnet --start 2024-02-28 --end 2025-09-16 --workers 12"
LOG[aiwp]=$W/aiwp_logs/dl_aiwp.log
CMD[arome]="scripts/download_meps.py --model arome_arctic --start 2024-02-28 --end 2025-09-16"
LOG[arome]=$W/arome_arctic/logs/dl_all.log
# the same, on the 0.25 deg box of mainland Norway (57.5-74 N, 4-36 E)
CMD[gfs]="scripts/download_gfs.py --start 2024-02-28 --end 2025-09-16"
LOG[gfs]=$W/gfs/logs/dl_gfs.log
CMD[ifsno]="scripts/download_ecmwf.py --model ifs --box norway --start 2024-02-28 --end 2025-09-16"
LOG[ifsno]=$W/ecmwf_no/logs/dl_ifs.log
CMD[aifsno]="scripts/download_ecmwf.py --model aifs --box norway --start 2024-02-28 --end 2025-09-16"
LOG[aifsno]=$W/ecmwf_no/logs/dl_aifs.log

# pressure levels (u, v, t, z at 925 / 850 / 700 / 500 hPa) of MEPS and AROME-Arctic for the
# 2.5 km samples (scripts/build_lam_samples.py), 6-hourly leads; through the aggregations a run
# takes ~4 min (MEPS) / ~7 min (AROME-Arctic), so the period is cut into date chunks:
#   sbatch slurm/download.slurm mepspl1 .. mepspl4 aromepl1 .. aromepl6
PL_LEADS="--leads 0 6 12 18 24 30 36 42 48 54 60 66 --timeout 900"
chunks() {  # chunks N -> N "start end" ranges of 2024-02-28 .. 2025-09-16
  local n=$1 d0 d1 i a b
  d0=$(date -d 2024-02-28 +%s); d1=$(date -d 2025-09-16 +%s)
  for ((i = 0; i < n; i++)); do
    a=$(date -d @$((d0 + (d1 - d0) / 86400 * i / n * 86400)) +%F)
    b=$(date -d @$((d0 + ((d1 - d0) / 86400 * (i + 1) / n - (i + 1 < n ? 1 : 0)) * 86400)) +%F)
    echo "$a $b"
  done
}
i=0; while read -r a b; do i=$((i + 1))
  CMD[mepspl$i]="scripts/download_meps.py --pl --start $a --end $b $PL_LEADS"
  LOG[mepspl$i]=$W/meps_pl/logs/dl_$i.log
done < <(chunks 4)
i=0; while read -r a b; do i=$((i + 1))
  CMD[aromepl$i]="scripts/download_meps.py --pl --model arome_arctic --start $a --end $b $PL_LEADS"
  LOG[aromepl$i]=$W/arome_arctic_pl/logs/dl_$i.log
done < <(chunks 6)

echo "job $SLURM_JOB_ID on $(hostname), streams: ${STREAMS[*]}, resubmit $N/$MAX_RESUBMIT"
declare -A PID
for s in "${STREAMS[@]}"; do
  mkdir -p "$(dirname "${LOG[$s]}")"
  echo "=== job $SLURM_JOB_ID $(date -Is) start" >> "${LOG[$s]}"
  $PY -u ${CMD[$s]} >> "${LOG[$s]}" 2>&1 &
  PID[$s]=$!
done

# stop the streams a few minutes before the time limit, so the resubmission happens here
END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }') - 600 ))
while :; do
  alive=0
  for s in "${STREAMS[@]}"; do kill -0 "${PID[$s]}" 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break
  if [ "$(date +%s)" -ge "$END" ]; then
    echo "$(date -Is) time limit near, stopping streams"
    for s in "${STREAMS[@]}"; do kill "${PID[$s]}" 2>/dev/null; done
    sleep 30
    break
  fi
  sleep 60
done

left=()
for s in "${STREAMS[@]}"; do
  last=$(grep -E "^done:|^=== job" "${LOG[$s]}" | tail -n 1)
  echo "$s: ${last:-no output}"
  [[ "$last" == done:* ]] || left+=("$s")
done
if [ ${#left[@]} -gt 0 ] && [ "$N" -lt "$MAX_RESUBMIT" ]; then
  echo "resubmitting for: ${left[*]}"
  RESUBMIT_COUNT=$((N + 1)) MAX_RESUBMIT=$MAX_RESUBMIT sbatch --export=ALL "$REPO/slurm/download.slurm" "${left[@]}"
elif [ ${#left[@]} -gt 0 ]; then
  echo "not finished after $MAX_RESUBMIT resubmissions: ${left[*]}"
else
  echo "all streams done"
fi
