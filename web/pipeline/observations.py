"""Hourly 10 m wind observations of the website stations from MET Norway's Frost API, for the
forecast runs on the website (shown in the table and graph next to the forecast).

    python web/pipeline/observations.py --out web/site/data      # every run in runs/index.json

Credentials: FROST_CLIENT_ID in the environment, or in the file named by WINDCAST_ENV_FILE
(KEY=value lines; default ~/github/mtno/.env). Never printed.

Writes OUT/runs/<run>/obs.json: {"updated": ..., "obs": {station: [[YYYY-MM-DDTHH, speed, dir], ...]}}
for the valid times of the run up to now (full hours; PT1H values preferred, else the 10-min value
at the full hour).
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
API = "https://frost.met.no/observations/v0.jsonld"
RES_RANK = {"PT1H": 0, "PT10M": 1, "PT30M": 2, "PT1M": 3}


def client_id():
    if os.environ.get("FROST_CLIENT_ID"):
        return os.environ["FROST_CLIENT_ID"]
    f = Path(os.environ.get("WINDCAST_ENV_FILE", Path.home() / "github" / "mtno" / ".env"))
    for line in f.read_text().splitlines():
        if line.startswith("FROST_CLIENT_ID="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("FROST_CLIENT_ID not found")


def fetch(ids, t0, t1, cid):
    rows = []
    for i in range(0, len(ids), 50):
        r = requests.get(API, params=dict(sources=",".join(ids[i:i + 50]), elements="wind_speed,wind_from_direction",
                                          referencetime=f"{t0:%Y-%m-%dT%H:%M}/{t1:%Y-%m-%dT%H:%M}"),
                         auth=(cid, ""), timeout=120)
        if r.status_code == 404:          # no data in the window
            continue
        r.raise_for_status()
        for it in r.json().get("data", []):
            t = pd.Timestamp(it["referenceTime"]).tz_localize(None)
            if t.minute or t.second:
                continue
            for o in it["observations"]:
                rows.append((it["sourceId"].split(":")[0], t, o["elementId"], o["value"],
                             RES_RANK.get(o.get("timeResolution"), 9)))
    df = pd.DataFrame(rows, columns=["st", "t", "el", "v", "rank"])
    if df.empty:
        return {}
    df = df.sort_values("rank").drop_duplicates(["st", "t", "el"]).pivot_table(index=["st", "t"], columns="el", values="v")
    out = {}
    for (st, t), r in df.iterrows():
        sp, d = r.get("wind_speed"), r.get("wind_from_direction")
        if pd.notna(sp):
            out.setdefault(st, []).append([t.strftime("%Y-%m-%dT%H"), round(float(sp), 1),
                                           None if pd.isna(d) else int(round(d))])
    return {k: sorted(v) for k, v in out.items()}


def update(run_dir, cid):
    fc = json.loads((run_dir / "forecast.json").read_text())
    t0 = pd.Timestamp(fc["valid"][0]).tz_localize(None)
    t1 = min(pd.Timestamp(fc["valid"][-1]).tz_localize(None), pd.Timestamp.now("UTC").tz_localize(None).floor("h")) + pd.Timedelta(hours=1)
    if t1 <= t0:
        return
    obs = fetch([s["id"] for s in fc["stations"]], t0, t1, cid)
    tmp = run_dir / "obs.json.tmp"
    tmp.write_text(json.dumps(dict(updated=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"), obs=obs), separators=(",", ":")))
    tmp.rename(run_dir / "obs.json")
    print(run_dir.name, sum(len(v) for v in obs.values()), "observations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "web" / "site" / "data")
    ap.add_argument("--run", default=None, help="one run (YYYYMMDDTHH); default: every run in runs/index.json")
    a = ap.parse_args()
    cid = client_id()
    runs = [a.run] if a.run else json.loads((a.out / "runs" / "index.json").read_text())["runs"]
    for r in runs:
        try:
            update(a.out / "runs" / r, cid)
        except Exception as e:  # one run failing must not stop the others
            print(r, "failed:", str(e).splitlines()[0][:200])


if __name__ == "__main__":
    main()
