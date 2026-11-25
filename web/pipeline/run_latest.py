"""Real-time cycle of the website: newest run -> downloads -> forecasts -> observations -> index.

    python web/pipeline/run_latest.py [--out web/site/data] [--init 2026-11-18T12] [--keep 4]

Run on a GPU node (web/slurm/realtime.slurm, scheduled by web/slurm/scrontab.txt):
1. the init: the newest 00 / 12 UTC run at least 7 h old (ECMWF open data is complete by then),
   or --init; if its forecast.json already has every configuration, only the observations update;
2. downloads into ROOT (skipping existing files): IFS and AIFS (ECMWF open data), GFS (NOAA AWS),
   MEPS and AROME-Arctic (MET Norway thredds, archive or "latest" catalogue), 0-66 h every 6 h;
3. web/pipeline/forecast.py for the configurations whose inputs exist (nwp-* need all four NWP
   models, dual-gno IFS, AIFS, MEPS and AROME-Arctic; if one is missing the ECMWF configurations
   are still published);
4. web/pipeline/observations.py, runs/index.json (newest first) and the deletion of older runs;
5. if WINDCAST_DEPLOY is set, it is run as a shell command afterwards (e.g. an rsync of OUT to
   the web server).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from build_group_samples import path as group_path  # noqa: E402

GROUPS = {"ecmwf": ["ifs", "aifs"], "nwp": ["ifs", "gfs", "meps", "arome_arctic"],
          "dual": ["ifs", "aifs", "meps", "arome_arctic"]}          # dual-gno: WindCast (web/pipeline/dual.py)
N_CONFIGS = 7
LEADS6 = [str(x) for x in range(0, 67, 6)]


def sh(*args):
    print("$", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, *args], cwd=REPO)
    if r.returncode:
        print(f"  exit code {r.returncode}", flush=True)
    return r.returncode == 0


def download(init):
    d, h = f"{init:%Y-%m-%d}", str(init.hour)
    for m in ("ifs", "aifs"):
        sh("scripts/download_ecmwf.py", "--model", m, "--start", d, "--end", d, "--inits", h)
    sh("scripts/download_gfs.py", "--start", d, "--end", d, "--inits", h)
    # the whole 1 km map (--window grid), so that the NWP configurations cover all of it
    sh("scripts/download_meps.py", "--start", d, "--end", d, "--inits", h, "--leads", *LEADS6, "--window", "grid",
       "--max-wait-h", "1")
    sh("scripts/download_meps.py", "--model", "arome_arctic", "--start", d, "--end", d, "--inits", h, "--leads", *LEADS6,
       "--window", "grid", "--max-wait-h", "1")
    return {g: all(group_path(m, init).exists() for m in ms) for g, ms in GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "web" / "site" / "data")
    ap.add_argument("--init", default=None)
    ap.add_argument("--keep", type=int, default=4, help="runs kept on the website")
    a = ap.parse_args()
    now = pd.Timestamp.now("UTC").tz_localize(None)
    init = pd.Timestamp(a.init) if a.init else (now - pd.Timedelta(hours=7)).floor("12h")
    run = f"{init:%Y%m%dT%H}"
    runs_dir = a.out / "runs"
    fc = runs_dir / run / "forecast.json"
    done = fc.exists() and len(json.loads(fc.read_text())["configs"]) == N_CONFIGS
    print(f"init {init} ({'already done' if done else 'to do'})", flush=True)
    if not done:
        have = download(init)
        print("inputs:", have, flush=True)
        configs = [c for g, ok in have.items() if ok
                   for c in (["dual-gno"] if g == "dual" else [f"{g}-sfno", f"{g}-fno", f"{g}-mlp"])]
        # an earlier partial run is redone only if more configurations are possible now
        before = len(json.loads(fc.read_text())["configs"]) if fc.exists() else 0
        if configs and len(configs) > before:
            sh("web/pipeline/forecast.py", "--init", str(init), "--out", str(a.out), "--configs", *configs)
    runs = sorted([p.name for p in runs_dir.glob("*T*") if (p / "forecast.json").exists()], reverse=True)
    for old in runs[a.keep:]:
        shutil.rmtree(runs_dir / old)
    runs = runs[:a.keep]
    if runs:
        (runs_dir / "index.json").write_text(json.dumps(dict(runs=runs, latest=runs[0],
                                                             updated=now.strftime("%Y-%m-%dT%H:%MZ"))))
        sh("web/pipeline/observations.py", "--out", str(a.out))
    if os.environ.get("WINDCAST_DEPLOY"):
        print("$ (WINDCAST_DEPLOY)", flush=True)
        subprocess.run(os.environ["WINDCAST_DEPLOY"], shell=True, cwd=REPO)


if __name__ == "__main__":
    main()
