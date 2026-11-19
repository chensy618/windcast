"""Hourly observations of the mainland-Norway 10 m wind stations that are missing from the local
Frost copy (ROOT/stations/stations.parquet, from the mtno download), for training on all of
Norway (track A).

    export FROST_CLIENT_ID=...      # https://frost.met.no/auth/requestCredentials.html
    python scripts/download_frost_extra.py --start 2024-02-28 --end 2025-09-17
    python scripts/download_frost_extra.py --all --overlap --out ROOT/stations/frost_no \
        --start 2024-03-01 --end 2026-11-18      # every mainland station (all of Norway)

Stations: every Frost SensorSystem in mainland Norway with a 10 m wind_speed series that covers
the period (--overlap: that overlaps it), minus those already in the local catalogue (--all:
including them). Credentials: FROST_CLIENT_ID, or the file WINDCAST_ENV_FILE (default
~/github/mtno/.env). Existing obs_<year>.parquet files in OUT are merged with, not overwritten. Elements: wind_speed,
wind_from_direction, air_temperature (PT1H).

OUT ROOT/stations/frost_extra/stations.parquet  station, name, lon, lat, masl, country, county,
                                                 municipality (the local catalogue's columns)
    ROOT/stations/frost_extra/obs_<year>.parquet  station, sensor, time (UTC), element, value,
                                                 level, time_offset, quality: the Frost obs layout
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
OUT = ROOT / "stations" / "frost_extra"
API = "https://frost.met.no/"
ELEMENTS = ["wind_speed", "wind_from_direction", "air_temperature"]
BBOX = "POLYGON((4 57.5, 32 57.5, 32 71.5, 4 71.5, 4 57.5))"


def get(path, params, cid, tries=6):
    for k in range(tries):
        r = requests.get(API + path, params=params, auth=(cid, ""), timeout=300)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:                   # no data for this request
            return {"data": []}
        if r.status_code == 401:
            raise SystemExit("Frost 401: check FROST_CLIENT_ID")
        if r.status_code == 403 and "too many" not in r.text.lower():
            raise RuntimeError(f"Frost 403: {r.text[:200]}")
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"Frost {r.status_code} on {path}: {r.text[:200]}")


def client_id():
    if os.environ.get("FROST_CLIENT_ID"):
        return os.environ["FROST_CLIENT_ID"]
    f = Path(os.environ.get("WINDCAST_ENV_FILE", Path.home() / "github" / "mtno" / ".env"))
    for line in f.read_text().splitlines():
        if line.startswith("FROST_CLIENT_ID="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("set FROST_CLIENT_ID")


def candidates(cid, t0, t1, overlap=False, include_local=False):
    src = get("sources/v0.jsonld", {"types": "SensorSystem", "geometry": BBOX}, cid)["data"]
    st = pd.DataFrame([{"station": s["id"], "name": s.get("name"), "lon": s["geometry"]["coordinates"][0],
                        "lat": s["geometry"]["coordinates"][1], "masl": s.get("masl"),
                        "country": s.get("countryCode"), "county": s.get("county"),
                        "municipality": s.get("municipality")}
                       for s in src if s.get("geometry") and s.get("countryCode") == "NO"])
    ts = []
    ids = list(st.station)
    for i in range(0, len(ids), 100):
        js = get("observations/availableTimeSeries/v0.jsonld",
                 {"sources": ",".join(ids[i:i + 100]), "elements": "wind_speed", "timeresolutions": "PT1H"}, cid)
        for d in js["data"]:
            ts.append({"station": d["sourceId"].split(":")[0], "level": (d.get("level") or {}).get("value"),
                       "fr": d.get("validFrom"), "to": d.get("validTo")})
    ts = pd.DataFrame(ts)
    ts["fr"], ts["to"] = pd.to_datetime(ts.fr, utc=True), pd.to_datetime(ts.to, utc=True)
    if overlap:
        sel = (ts.fr <= t1) & (ts.to.isna() | (ts.to >= t0))
    else:
        sel = (ts.fr <= t0) & (ts.to.isna() | (ts.to >= t1))
    ok = ts[(ts.level == 10) & sel].station.unique()
    local = set() if include_local else set(pd.read_parquet(ROOT / "stations" / "stations.parquet").station)
    st = st[st.station.isin(ok) & ~st.station.isin(local)].reset_index(drop=True)
    return st


def observations(stations, a, b, cid):
    js = get("observations/v0.jsonld", {"sources": ",".join(stations), "elements": ",".join(ELEMENTS),
                                        "referencetime": f"{a:%Y-%m-%dT%H:%M}/{b:%Y-%m-%dT%H:%M}",
                                        "timeresolutions": "PT1H"}, cid)
    rows = []
    for rec in js.get("data", []):
        station, _, sensor = rec["sourceId"].partition(":")
        for o in rec.get("observations", []):
            rows.append({"station": station, "sensor": sensor, "time": rec["referenceTime"],
                         "element": o["elementId"], "value": o["value"],
                         "level": (o.get("level") or {}).get("value"), "time_offset": o.get("timeOffset"),
                         "quality": o.get("qualityCode")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--batch", type=int, default=10, help="stations per request")
    ap.add_argument("--all", action="store_true", help="also the stations of the local catalogue")
    ap.add_argument("--overlap", action="store_true", help="series that overlap the period, not only cover it")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cid = client_id()
    out = a.out
    t0, t1 = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    out.mkdir(parents=True, exist_ok=True)
    st = candidates(cid, t0, t1, overlap=a.overlap, include_local=a.all)
    st.to_parquet(out / "stations.parquet", index=False)
    print(f"{len(st)} stations; by county: {st.county.value_counts().to_dict()}", flush=True)
    months = list(pd.date_range(t0.floor("D"), t1, freq="MS", tz="UTC"))
    edges = [t0] + [m for m in months if m > t0] + [t1]
    parts, n = [], 0
    for i in range(0, len(st), a.batch):
        ids = list(st.station[i:i + a.batch])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            rows += observations(ids, lo, hi, cid)
        if rows:
            parts.append(pd.DataFrame(rows))             # compact per batch: dicts would not fit in memory
            n += len(rows)
        print(f"  stations {i + len(ids)} / {len(st)}: {n} records", flush=True)
    o = pd.concat(parts, ignore_index=True)
    o["time"] = pd.to_datetime(o.time, utc=True)
    o["level"] = pd.to_numeric(o.level, errors="coerce")
    o["quality"] = pd.to_numeric(o.quality, errors="coerce")
    key = ["station", "sensor", "time", "element", "level", "time_offset"]
    for y, g in o.groupby(o.time.dt.year):
        f = out / f"obs_{y}.parquet"
        if f.exists():                               # merge: never drop what is already there
            g = pd.concat([pd.read_parquet(f), g]).drop_duplicates(key, keep="last")
        g.sort_values(["station", "time"]).reset_index(drop=True).to_parquet(f, index=False)
    print(f"done: {len(st)} stations, {len(o)} records -> {out}")


if __name__ == "__main__":
    main()
