# Windcast Nord website

A static website with the 1 km 10 m wind forecast for Northern Norway: a table, a graph and a
playable map (2D, or 3D with three.js), for every model configuration. The site is plain
HTML/CSS/JS plus data files, so any static web server can host it. A real-time pipeline on the
cluster refreshes the data twice a day.

```
web/
  site/                 the website (deploy this folder)
    index.html  style.css  app.js
    data/               written by the pipeline (not in git)
      static/           grid.json, bg.webp, elevation.png, verification.json
      runs/index.json   runs on the site, newest first
      runs/<YYYYMMDDTHH>/forecast.json, obs.json, <config>/lead_<LL>.webp
  pipeline/
    run_latest.py       one real-time cycle: downloads -> forecasts -> observations -> index
    forecast.py         forecasts of one run for every configuration (GPU)
    observations.py     Frost station observations of the runs on the site (hourly)
    verification.py     held-out scores of the configurations (once, after training)
  slurm/
    realtime.slurm      GPU job running run_latest.py
    observations.slurm  CPU job running observations.py
    scrontab.txt        the schedule (scrontab web/slurm/scrontab.txt)
```

## Model configurations

| id | input forecasts | network |
|---|---|---|
| `nwp-mlp` | IFS, GFS, MEPS, AROME-Arctic | MLP head on the 1 km terrain descriptors |
| `nwp-sfno` | IFS, GFS, MEPS, AROME-Arctic | SFNO on the 0.25° fields + MLP head |
| `nwp-fno` | IFS, GFS, MEPS, AROME-Arctic | FNO on the 0.25° fields + MLP head |
| `ecmwf-sfno` | IFS, AIFS | SFNO + MLP head |
| `ecmwf-mlp` | IFS, AIFS | MLP head |
| `ecmwf-fno` | IFS, AIFS | FNO + MLP head |

Each is the mean of the 15 trained models (5 held-out station groups x 3 seeds) of
`windcast/training/train_station_operator.py --inputs terrain --lr 1e-5 [--group nwp]`.
The `nwp-*` forecasts exist only inside the MEPS window (about 70 % of the map).

## Real-time pipeline

```bash
sbatch web/slurm/realtime.slurm                     # newest 00 / 12 UTC run at least 7 h old
sbatch web/slurm/realtime.slurm --init 2026-11-18T12
scrontab web/slurm/scrontab.txt                     # install the schedule
python web/pipeline/verification.py                 # after retraining
```

A cycle takes about 6 minutes on one GPU. It downloads the run (ECMWF open data, NOAA GFS on
AWS, MET Norway thredds), runs `forecast.py` for the configurations whose inputs exist (if one
NWP model is missing, the ECMWF configurations are still published), refreshes the
observations, keeps the newest 4 runs and rewrites `runs/index.json`.

Requirements on the cluster: the training caches and runs under
`/cluster/work/projects/nn8106k/siyan/windcast` (the normalisation is rebuilt from the training
data), and a Frost client ID in `FROST_CLIENT_ID` or in the file named by `WINDCAST_ENV_FILE`
(default `~/github/mtno/.env`).

## Running the site yourself

The site is static: no build step and no packages to install. It needs `web/site/` with its
`data/` folder (written by the pipeline on the cluster) served over HTTP; opening `index.html`
as a file does not work, because the page fetches its data.

```bash
cd web && npm start                                  # Node, no dependencies: http://localhost:8000
node web/serve.mjs 8080                              # the same on another port
python3 -m http.server -d web/site 8000              # or Python
```

On a laptop, copy the site with its data from the cluster first, for example
`rsync -a <user>@<cluster>:~/github/windcast/web/site/ windcast-site/` and serve that folder.
On the cluster itself, forward the port: `ssh -L 8000:localhost:8000 <user>@<login node>`,
then run `npm start` there and open http://localhost:8000 locally.

## Deploying

To publish each cycle to a web server, set `WINDCAST_DEPLOY` for the Slurm jobs to a shell
command, for example in `web/slurm/scrontab.txt`:

```bash
export WINDCAST_DEPLOY='rsync -a --delete web/site/ user@server:/var/www/windcast/'
```

The page loads three.js r128 from cdnjs and OrbitControls from jsDelivr when the 3D view is
first opened, and the fonts from Google Fonts. `data/` is about 45 MB per run, mostly the
lossless WebP map images (R = corrected speed / 0.2 m/s, G = uncorrected speed, B = direction).
