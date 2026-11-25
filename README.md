# windcast

10 m wind forecasts at 1 km over Northern Norway, 0–66 h ahead. **WindCast** corrects four
operational forecasts (IFS, AIFS, MEPS, AROME-Arctic) for the local terrain with neural operators,
and runs in real time.

**Live forecast: <https://chensy618.github.io/windcast/>**. It is updated after every 00 / 12 UTC run,
with station observations every hour.

## WindCast

A dual-scale graph neural operator (`windcast/training/train_dual_gno.py`) predicts the
obs-forecast residual of the 10 m wind against the mean of the four forecasts:

```
IFS + AIFS, 0.25°            ── SFNO encoder ──▶ GNO, 60 km radius ─┐
MEPS + AROME-Arctic, 2.5 km  ── FNO encoder  ──▶ GNO, 10 km radius ─┼─▶ MLP ─▶ Δu(x, t)
terrain (28 descriptors), the four forecast winds and time at x ───┘

u(x, t) = mean(IFS, AIFS, MEPS, AROME-Arctic)(x, t) + Δu(x, t)
```

- **Inputs:** 10u, 10v, 2t and msl of each model, plus the terrain of the target point. No
  coordinates and no observations as input, so the same weights apply at any point (station or
  1 km cell).
- **Training:** observations of 99 Norwegian weather stations (MET Norway Frost).
  - Stations are split into 5 spatial groups. Each model is trained without one group and scored
    only on the stations it never saw.
  - 3 seeds per group, so the forecast is the mean of 15 models.
- **Configuration:** width 16, 8 × 16 spectral modes, 4 layers, about 0.15 M parameters. The full
  table is in the report.

### Results

Vector RMSE (m/s) at held-out stations, all lead times, on samples with a MET Nordic analysis.
± is the s.d. of a bootstrap over test runs (`notebooks/06_four_fields.ipynb`).

| Forecast | Original period¹ | Extended period² |
|---|---|---|
| IFS | 3.562 ± 0.033 | 3.772 ± 0.037 |
| AIFS | 3.541 ± 0.035 | 3.746 ± 0.038 |
| MEPS | 3.564 ± 0.036 | 3.760 ± 0.035 |
| AROME-Arctic | 3.552 ± 0.035 | 3.655 ± 0.032 |
| MET Nordic analysis (valid time) | 3.117 ± 0.028 | 3.332 ± 0.032 |
| **WindCast** | **3.153 ± 0.029** | **3.256 ± 0.028** |

¹ train 2024-03 – 2024-12, test 2025-04-01 – 2025-09-12.
² train 2024-03 – 2025-12, test 2026-04-01 – 2026-11-14.

WindCast is a forecast, but it comes close to the MET Nordic analysis, which already knows the
observations at the valid time. In the extended period it beats the analysis.

**Ablation** (`notebooks/07_ablation.ipynb`, original period, 3 seeds each). The terrain features
and the MLP head matter most: removing either raises RMSE by about 0.05 m/s. Dropping the global
branch costs 0.01 m/s. The GNO readout (vs bilinear sampling) and the regional branch change it
by less than the seed spread. Starting from the four-model mean gives no gain.

## Repository

| Path | Contents |
|---|---|
| `windcast/training/train_dual_gno.py` | WindCast: data, dual-scale GNO, training (`--branches`, `--readout`, `--head`, `--no-terrain`, ... for ablations) |
| `windcast/training/train_station_operator.py` | point-wise MLP and SFNO / FNO + MLP station-correction models; sample periods (`WINDCAST_PERIOD`) |
| `windcast/station_correction.py`, `windcast/geo.py` | station groups, upwind terrain shelter; LAEA working plane (20° E, 68° N, km) |
| `scripts/download_*.py` | IFS / AIFS (ECMWF open data), GFS (NOAA), MEPS / AROME-Arctic (MET Norway thredds), MET Nordic analyses, Frost observations |
| `scripts/build_multimodel_samples.py` | IFS + AIFS on the 0.25° grid and at the stations → `cache/multimodel_<period>` |
| `scripts/build_lam_samples.py` | MEPS + AROME-Arctic on a common 2.5 km grid → `cache/multimodel_<period>_lam` |
| `scripts/build_terrain_descriptors.py`, `scripts/build_station_physics.py` | 28 terrain descriptors (1 km grid and stations); θ / mountain-wind features |
| `scripts/eval_metnordic_grid.py` | the trained models on 1 km cells, scored against the MET Nordic analysis |
| `notebooks/06_four_fields.ipynb` | main results: RMSE by lead time and overall, uncertainty, wind-field maps, grid evaluation, encoder size |
| `notebooks/07_ablation.ipynb` | ablation study (figures and LaTeX table in `notebooks/figs/07/`) |
| `notebooks/00_project_overview.ipynb`, `01_training`, `02_experiments`, `a4_model_summary` | project overview, training curves, earlier experiments, scores of every model |
| `notebooks/duallib.py`, `notebooks/runlib.py` | loading and scoring of the held-out predictions |
| `web/` | the website and its real-time pipeline ([web/README.md](web/README.md)) |
| `slurm/` | GPU training (`train_dual.slurm`, `train_a1.slurm`), downloads, grid evaluation, notebooks on a CPU node |
| `docs/` | design notes and project summary |

## Running it

Built for the Olivia cluster (Sigma2). Data and runs live outside the repository in
`/cluster/work/projects/nn8106k/siyan/windcast`. That disk is purged periodically, so every
download and build step can be resumed.

| Environment | Used for |
|---|---|
| `/cluster/projects/nn8106k/siyan/envs/wfno-arm312` | GH200 GPU nodes: downloads, training, website forecasts |
| `/cluster/projects/nn8106k/siyan/envs/windcast-x86` | login nodes: notebooks and scoring (no torch) |

The compute nodes reach the internet only through the `uan03` proxy.

```bash
# 1. data (one GPU job runs every download stream and resubmits itself until done)
sbatch slurm/download.slurm
# then, on a GPU node, build the samples:
#   build_multimodel_samples.py, build_lam_samples.py, build_terrain_descriptors.py

# 2. train WindCast: 5 station folds x 3 seeds; each argument string is one run
A="--width 16 --width-r 16 --modes-g 8 16 --modes-r 8 16 --zero-init --head mlp"
for k in 0 1 2 3 4; do
  sbatch slurm/train_dual.slurm "--fold $k --seed 0 $A" "--fold $k --seed 1 $A" "--fold $k --seed 2 $A"
done
# extended period: prefix with WINDCAST_PERIOD=2024-03-01_2026-11-14

# 3. evaluate
sbatch slurm/eval_metnordic.slurm "--step 2"                       # 1 km grid vs MET Nordic
sbatch slurm/run_notebook.slurm notebooks/06_four_fields.ipynb     # station scores and figures

# 4. website (see web/README.md)
python web/pipeline/dual.py --prepare                              # once after retraining
scrontab web/slurm/scrontab.txt                                    # real-time cycle + GitHub Pages
```

## Data and licences

- Forecasts:
  - ECMWF IFS and AIFS open data (CC BY 4.0).
  - NOAA GFS.
  - MET Norway MEPS and AROME-Arctic (CC BY 4.0).
- Observations and analysis: MET Norway Frost station observations and the MET Nordic analysis
  (CC BY 4.0).
- Terrain: Kartverket DTM 50 m (Geonorge, CC BY 4.0).

## Earlier work

The repository began with a gridded 1 km setup: MEPS + terrain → operator (U-Net, FNO) → MET
Nordic, with several weather variables (`windcast/data/dataset.py`, `windcast/models/`,
`windcast/training/train.py`, `configs/f1_meps.yaml`, `slurm/train.slurm`). The design and data
caveats are in [docs/design.md](docs/design.md). The station-based approach above replaced it,
because MET Nordic is built on MEPS and is not an independent target.
