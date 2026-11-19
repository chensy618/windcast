# windcast

Near-surface weather prediction at 1 km over Tromsø–Senja (wind jointly with t2m, rh2m,
msl and cloud), lead times 0–66 h, by neural operators on top of a forecast driver:
MEPS 2.5 km (track A) or a global ML weather model (track B). Several operator families
are compared under one protocol. Design, data caveats and experiment plan:
[docs/design.md](docs/design.md).

| | |
|---|---|
| `windcast/geo.py` | LAEA working plane (20°E, 68°N), km |
| `windcast/data/dataset.py` | `ForecastDataset`: one (init, lead) per sample; MEPS bilinear to 1 km, terrain, time features; MET Nordic target |
| `windcast/models/` | operator zoo (`unet`, `fno`, `ufno`, more to come); residual on the driver, zero-initialised output |
| `windcast/training/{train,metrics}.py` | training, evaluation per lead and variable (model, driver, persistence) |
| `scripts/download_meps.py` | MEPS det forecasts, OPeNDAP subset |
| `scripts/download_metnordic.py` | MET Nordic v4 analyses on the reference grid |
| `scripts/download_ecmwf.py` | IFS / AIFS open data (GCS mirror), byte-range fetch, cropped to Northern Norway |
| `slurm/download.slurm` | all download streams in one GPU job, self-resubmitting |
| `scripts/meps_static.py`, `scripts/build_grid.py` | MEPS orography; `grid.npz` with the 1 km grid, terrain and interpolation weights |
| `scripts/compute_stats.py` | train normalisation statistics |
| `configs/f1_meps.yaml` | F1: MEPS + terrain → operator → MET Nordic |
| `scripts/build_station_samples.py` | MEPS bilinear at the stations + hourly station wind → `cache/stations_meps_*.npz` |
| `windcast/station_correction.py` | track A station correction: obs-forecast, spatial GP, IDW, metrics |
| `notebooks/a0_station_correction.ipynb` | A0: MEPS correction from station obs-forecast (persistence, ridge regression, GP), 2018-01–07 |
| `scripts/build_multimodel_samples.py` | IFS + AIFS on the common 0.25° grid and at the stations, 2024-03 – 2025-09 → `cache/multimodel_*` |
| `windcast/training/train_station_operator.py` | A1 / A2: neuraloperator models for station correction (`--inputs obs` or `terrain`) |
| `notebooks/a1_multimodel_operators.ipynb` (`make_a1.py`) | A1: all neuraloperator models vs ridge regression and Gaussian process, IFS + AIFS |
| `scripts/build_terrain_descriptors.py` | 28 terrain / surface descriptors on the 1 km grid and at the stations |
| `notebooks/a2_terrain_descriptors.ipynb` (`make_a2.py`) | A2: station correction from terrain descriptors, on held-out station groups |
| `scripts/predict_terrain_grid.py` | A2: applies the terrain models on the 1 km grid (maps + check at withheld stations) |
| `slurm/train_a1.slurm`, `slurm/gpu_script.slurm`, `slurm/run_notebook.slurm` | GPU training runs one after another; one GPU script; notebooks executed on a CPU node |

Station experiments run on the login node with
`/cluster/projects/nn8106k/siyan/envs/windcast-x86` (e2s-notebook + pyarrow, scipy, sklearn).

Data live in `/cluster/work/projects/nn8106k/siyan/windcast/{meps,metnordic,stations,static}`
(on the work disk by choice; it is purged periodically, so the downloads are written to be
resumable and re-runnable). Downloads and training both run on the GH200 nodes with `wfno-arm312`
(downloads as a GPU job, `slurm/download.slurm`).

```bash
# data: one GPU job runs every download stream (accel nodes reach the internet via the
# uan03 proxy); it resubmits itself for streams that are not finished, up to 5 times
sbatch slurm/download.slurm                 # meps metnordic ifs aifs
sbatch slurm/download.slurm meps            # one stream
PY=/cluster/projects/nn8106k/siyan/envs/wfno-arm312/bin/python   # on a GPU node
$PY scripts/meps_static.py && $PY scripts/build_grid.py
$PY scripts/compute_stats.py --config configs/f1_meps.yaml

# baselines, training, evaluation (GPU)
sbatch --time=02:00:00 slurm/train.slurm --config configs/f1_meps.yaml --baseline --split test
sbatch slurm/train.slurm --config configs/f1_meps.yaml --run-name f1_fno_s0
sbatch --time=02:00:00 slurm/train.slurm --config configs/f1_meps.yaml \
    --eval /cluster/work/projects/nn8106k/siyan/windcast/runs/f1_fno_s0/best.pt --split test
```
