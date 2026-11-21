"""MEPS forecasts at the stations + hourly station wind, for the station-level (track A)
experiments.

    python scripts/build_station_samples.py --start 2018-01-01 --end 2018-07-16

OUT ROOT/cache/stations_meps_<start>_<end>.npz:
    station, lon, lat, masl, xy (S, 2) km, oro (S,) MEPS orography at the station (m)
    inits   (N,) datetime64[h]   MEPS runs found in the period (00 / 12 UTC)
    leads   (L,) hours           0-12 hourly, 18-66 6-hourly
    fc      (N, L, S, 2) float32 MEPS (u10, v10) east / north, bilinear to the station
    time    (T,) datetime64[h]   hourly axis from start - 12 h to end + 66 h
    obs     (T, S, 2) float32    observed (u, v) at 10 m, NaN where missing

Stations: Frost stations in Nordland / Troms / Finnmark inside the MEPS window. Obs:
wind_speed + wind_from_direction at 10 m, quality <= 2, mean over duplicates.
"""
import argparse
from pathlib import Path
import sys

import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from windcast.geo import to_plane  # noqa: E402
from build_grid import MEPS_LCC, bilinear  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
COUNTIES = {"NORDLAND", "TROMS", "FINNMARK"}
LEADS = list(range(13)) + list(range(18, 67, 6))


def load_obs(stations, t0, t1, extra=()):
    """Hourly (u, v) per (time, station); `extra`: more parquet files in the same layout
    (scripts/download_nordic_obs.py)."""
    years = range(t0.year, t1.year + 1)
    files = []
    for y in years:   # years without the county files (2026-): the Norway-wide Frost download
        fy = sorted((ROOT / "stations" / "obs").glob(f"obs_{y}_*.parquet"))
        files += fy or [f for f in [ROOT / "stations" / "frost_no" / f"obs_{y}.parquet"] if f.exists()]
    files += list(extra)
    o = pd.concat([pd.read_parquet(f, filters=[("element", "in", ["wind_speed", "wind_from_direction"]),
                                               ("level", "==", 10.0), ("quality", "<=", 2)],
                                   columns=["station", "time", "element", "value"]) for f in files])
    o = o[o.station.isin(stations)]
    o["time"] = o.time.dt.tz_convert(None).dt.floor("h")
    w = o.pivot_table(index=["time", "station"], columns="element", values="value", aggfunc="mean").dropna()
    w = w[(w.wind_speed >= 0) & (w.wind_speed < 75)]
    rad = np.deg2rad(w.wind_from_direction)
    w["u"], w["v"] = -w.wind_speed * np.sin(rad), -w.wind_speed * np.cos(rad)
    return w[["u", "v"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(hours=12)

    st = pd.read_parquet(ROOT / "stations" / "stations.parquet")
    st = st[st.county.isin(COUNTIES)].reset_index(drop=True)

    # MEPS runs; all files of one grid share the window (checked on the first one per shape)
    inits = [t for t in pd.date_range(t0, t1, freq="12h")
             if (ROOT / "meps" / f"{t:%Y}" / f"meps_{t:%Y%m%dT%H}.nc").exists()]
    print(f"{len(inits)} MEPS runs, {len(st)} candidate stations")
    statics = {}
    for shape in MEPS_LCC:
        m = netCDF4.Dataset(ROOT / "meps" / f"static_{shape}.nc")
        statics[shape] = (m["lon"][:], m["lat"][:], m["orography"][:].reshape(-1))

    def inside(i):
        for shape, (lon, lat, _) in statics.items():
            try:
                bilinear(MEPS_LCC[shape], lon, lat, st.lon.values[i:i + 1], st.lat.values[i:i + 1])
            except ValueError:
                return False
        return True
    st = st[[inside(i) for i in range(len(st))]].reset_index(drop=True)
    weights = {}
    for shape, (lon, lat, o) in statics.items():
        idx, w = bilinear(MEPS_LCC[shape], lon, lat, st.lon.values, st.lat.values)
        weights[lon.shape] = (idx, w, o)

    S, L = len(st), len(LEADS)
    fc = np.full((len(inits), L, S, 2), np.nan, np.float32)
    oro = None
    for n, t in enumerate(inits):
        with netCDF4.Dataset(ROOT / "meps" / f"{t:%Y}" / f"meps_{t:%Y%m%dT%H}.nc") as ds:
            idx, w, o = weights[ds["lon"].shape]
            if oro is None:
                oro = (o[idx] * w).sum(-1)
            leads = [int(x) for x in ds["lead"][:]]
            for k, var in enumerate(["u10", "v10"]):
                a = np.ma.filled(ds[var][:].astype(np.float32), np.nan)
                a = a.reshape(a.shape[0], -1)
                v = (a[:, idx] * w).sum(-1)              # (lead, S)
                for j, ld in enumerate(LEADS):
                    if ld in leads:
                        fc[n, j, :, k] = v[leads.index(ld)]
        if n % 50 == 0:
            print(n, t, flush=True)

    time = pd.date_range(t0 - pd.Timedelta(hours=12), t1 + pd.Timedelta(hours=66), freq="h")
    w = load_obs(set(st.station), time[0], time[-1])
    obs = np.full((len(time), S, 2), np.nan, np.float32)
    ti = pd.Index(time).get_indexer(w.index.get_level_values(0))
    si = pd.Index(st.station).get_indexer(w.index.get_level_values(1))
    ok = (ti >= 0) & (si >= 0)
    obs[ti[ok], si[ok]] = w.values[ok]
    keep = np.isfinite(obs[..., 0]).mean(0) > 0.5          # stations with > 50 % wind obs
    print(f"stations with > 50 % hourly wind: {keep.sum()} / {S}")

    out = ROOT / "cache" / f"stations_meps_{args.start}_{args.end}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, station=st.station.values[keep], lon=st.lon.values[keep],
                        lat=st.lat.values[keep], masl=st.masl.values[keep],
                        xy=to_plane(st.lon.values, st.lat.values)[keep].astype(np.float32),
                        oro=oro[keep].astype(np.float32),
                        inits=np.array(inits, dtype="datetime64[h]"), leads=np.array(LEADS),
                        fc=fc[:, :, keep], time=time.values.astype("datetime64[h]"),
                        obs=obs[:, keep])
    print(out)


if __name__ == "__main__":
    main()
