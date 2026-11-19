"""Hourly 10 m wind at Swedish (SMHI) and Finnish (FMI) stations north of a latitude, to add
inland and mountain stations to the Norwegian Frost set (track A, A2).

    python scripts/download_nordic_obs.py --start 2024-02-28 --end 2025-09-16 [--min-lat 65]

Both archives are open data without a key. Only stations inside the 1 km terrain grid
(ROOT/static/terrain_descriptors_1km.npz, at least 15 cells from its edge) are kept, so the
terrain descriptors can be computed at every station.

  * SMHI metobs: parameter 4 (wind speed) and 3 (wind direction), both the 10-minute mean
    at the full hour, 10 m; the corrected archive (quality-controlled, up to three months
    before today). Only quality G (checked and approved) is kept; Y (suspect or aggregated)
    is stored with quality 3, which load_obs (quality <= 2) drops.
  * FMI open data WFS: ws_10min / wd_10min at timestep 60 (the 10-minute mean ending at the
    full hour), automatic weather stations (network 121), one request per week (the WFS
    limit), all stations of one week in one request. FMI does not publish quality flags here.

Station altitude: SMHI gives it; FMI does not, so it is read from the Copernicus 30 m DEM
(public COG on AWS) at the station position.

OUT ROOT/stations/nordic/stations.parquet  station (SE<id> / FI<fmisid>), name, lon, lat,
                                           masl, country, county (SMHI / FMI)
    ROOT/stations/nordic/obs_{smhi,fmi}.parquet  station, time (UTC), element, value,
                                           level (10.0), quality; the Frost obs layout
"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
OUT = ROOT / "stations" / "nordic"
SMHI = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{p}"
FMI = "https://opendata.fmi.fi/wfs"
DEM = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{a:02d}_00_"
       "{ew}{o:03d}_00_DEM/Copernicus_DSM_COG_10_{ns}{a:02d}_00_{ew}{o:03d}_00_DEM.tif")
UA = {"User-Agent": "windcast-research-client"}


def get(url, params=None, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url} {params}")


def in_grid(lat, lon, margin=15):
    from scipy.spatial import cKDTree
    t = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
    H, W = t["lat"].shape

    def xyz(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    d, i = cKDTree(xyz(t["lat"].ravel(), t["lon"].ravel())).query(xyz(np.asarray(lat), np.asarray(lon)))
    r, c = np.unravel_index(i, (H, W))
    return (d * 6371 < 1.0) & (r >= margin) & (r < H - margin) & (c >= margin) & (c < W - margin)


def dem_height(lat, lon):
    import rasterio
    out = []
    for la, lo in zip(lat, lon):
        f = DEM.format(ns="N", a=int(np.floor(la)), ew="E", o=int(np.floor(lo)))
        with rasterio.open(f) as src:
            out.append(float(next(src.sample([(lo, la)]))[0]))
    return np.array(out)


# --- SMHI ---------------------------------------------------------------------------------
def smhi_stations(t0, t1, min_lat):
    sp = {s["key"]: s for s in get(SMHI.format(p=4) + ".json").json()["station"]}
    sd = {s["key"] for s in get(SMHI.format(p=3) + ".json").json()["station"]}
    st = pd.DataFrame([dict(id=k, name=s["name"], lat=s["latitude"], lon=s["longitude"], masl=s["height"],
                            fr=pd.Timestamp(s["from"], unit="ms", tz="UTC"), to=pd.Timestamp(s["to"], unit="ms", tz="UTC"))
                       for k, s in sp.items() if k in sd])
    st = st[(st.fr <= t0) & (st.to >= t1) & (st.lat >= min_lat)]
    st = st[in_grid(st.lat.values, st.lon.values)].reset_index(drop=True)
    st["station"], st["country"], st["county"] = "SE" + st.id, "SE", "SMHI"
    return st


def smhi_series(sid, p, t0, t1):
    r = get(f"{SMHI.format(p=p)}/station/{sid}/period/corrected-archive/data.csv")
    if r is None:
        return None
    lines = r.content.decode("utf-8-sig").splitlines()
    h = next(i for i, x in enumerate(lines) if x.startswith("Datum;Tid (UTC)"))
    d = pd.read_csv(io.StringIO("\n".join(lines[h + 1:])), sep=";", header=None, usecols=[0, 1, 2, 3],
                    names=["d", "t", "value", "q"], dtype={"d": str, "t": str, "q": str})
    d["time"] = pd.to_datetime(d.d + " " + d.t, utc=True)
    d = d[(d.time >= t0) & (d.time <= t1)]
    return pd.DataFrame({"time": d.time, "value": pd.to_numeric(d.value, errors="coerce"),
                         "quality": np.where(d.q == "G", 0, 3)})


def smhi(t0, t1, min_lat):
    st = smhi_stations(t0, t1, min_lat)
    print(f"SMHI: {len(st)} stations", flush=True)
    parts = []
    for _, s in st.iterrows():
        for p, el in ((4, "wind_speed"), (3, "wind_from_direction")):
            d = smhi_series(s.id, p, t0, t1)
            if d is not None and len(d):
                parts.append(d.assign(station=s.station, element=el))
        print(" ", s.station, s["name"], flush=True)
    return st, pd.concat(parts)


# --- FMI ----------------------------------------------------------------------------------
def fmi_stations(min_lat):
    x = get(FMI, dict(service="WFS", version="2.0.0", request="getFeature",
                      storedquery_id="fmi::ef::stations", networkid="121")).text
    rows = []
    for b in re.findall(r"<ef:EnvironmentalMonitoringFacility.*?</ef:EnvironmentalMonitoringFacility>", x, re.S):
        fm = re.search(r'stationcode/fmisid">(\d+)<', b)
        nm = re.search(r'locationcode/name">([^<]+)<', b)
        pos = re.search(r"<gml:pos>([\d.]+) ([\d.]+)", b)
        if fm and pos:
            rows.append(dict(id=fm.group(1), name=nm.group(1) if nm else "", lat=float(pos.group(1)),
                             lon=float(pos.group(2))))
    st = pd.DataFrame(rows).drop_duplicates("id")
    st = st[st.lat >= min_lat]
    st = st[in_grid(st.lat.values, st.lon.values)].reset_index(drop=True)
    st["station"], st["country"], st["county"] = "FI" + st.id, "FI", "FMI"
    return st


def fmi_week(ids, a, b):
    r = get(FMI, [("service", "WFS"), ("version", "2.0.0"), ("request", "getFeature"),
                  ("storedquery_id", "fmi::observations::weather::multipointcoverage"),
                  ("parameters", "ws_10min,wd_10min"), ("timestep", "60"),
                  ("starttime", f"{a:%Y-%m-%dT%H:%M:%SZ}"), ("endtime", f"{b:%Y-%m-%dT%H:%M:%SZ}")]
            + [("fmisid", i) for i in ids])
    x = r.text
    # stations in the order of the positions; one position line per (station, time)
    locs = re.findall(r'<target:Location gml:id="obsloc-fmisid-(\d+)-pos">', x)
    ptxt = re.search(r"<gmlcov:positions>(.*?)</gmlcov:positions>", x, re.S)
    vtxt = re.search(r"<gml:doubleOrNilReasonTupleList>(.*?)</gml:doubleOrNilReasonTupleList>", x, re.S)
    if not (locs and ptxt and vtxt):
        return None
    ptok = ptxt.group(1).split()
    pos = np.array(ptok, float).reshape(-1, 3)
    val = np.array(vtxt.group(1).split(), float).reshape(-1, 2)
    key = {}
    for i in locs:                       # station -> its coordinates, as printed in positions
        m = re.search(rf'<gml:Point gml:id="point-{i}".*?<gml:pos>\s*(\S+) (\S+)', x, re.S)
        key[(m.group(1), m.group(2))] = i
    sid = [key.get((ptok[3 * k], ptok[3 * k + 1])) for k in range(len(pos))]
    d = pd.DataFrame({"station": ["FI" + s if s else None for s in sid],
                      "time": pd.to_datetime(pos[:, 2].astype(np.int64), unit="s", utc=True),
                      "wind_speed": val[:, 0], "wind_from_direction": val[:, 1]}).dropna(subset=["station"])
    return d.melt(["station", "time"], var_name="element", value_name="value").dropna(subset=["value"])


def fmi(t0, t1, min_lat):
    st = fmi_stations(min_lat)
    st["masl"] = dem_height(st.lat.values, st.lon.values)
    print(f"FMI: {len(st)} stations", flush=True)
    parts = []
    a = t0
    while a < t1:
        b = min(a + pd.Timedelta(days=7) - pd.Timedelta(hours=1), t1)
        for k in range(0, len(st), 20):          # 20 stations per request keeps responses small
            d = fmi_week(st.id.values[k:k + 20], a, b)
            if d is not None:
                parts.append(d)
        print(" ", a.date(), flush=True)
        a = b + pd.Timedelta(hours=1)
    o = pd.concat(parts)
    o["quality"] = 0
    return st, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--min-lat", type=float, default=65.0)
    ap.add_argument("--only", choices=["smhi", "fmi"])
    a = ap.parse_args()
    t0, t1 = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["station", "name", "lon", "lat", "masl", "country", "county"]
    for name, fn in (("smhi", smhi), ("fmi", fmi)):
        if a.only and a.only != name:
            continue
        st, o = fn(t0, t1, a.min_lat)
        o["level"] = 10.0
        o = o[["station", "time", "element", "value", "level", "quality"]].reset_index(drop=True)
        o.to_parquet(OUT / f"obs_{name}.parquet", index=False)
        st[cols].to_parquet(OUT / f"stations_{name}.parquet", index=False)
        n_ok = o[(o.element == "wind_speed") & (o.quality <= 2)].groupby("station").size()
        print(f"{name}: {len(st)} stations, hours with wind per station: median {n_ok.median():.0f}", flush=True)
    st = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("stations_*.parquet"))], ignore_index=True)
    st.to_parquet(OUT / "stations.parquet", index=False)
    print(f"done: {len(st)} stations -> {OUT}")


if __name__ == "__main__":
    main()
