"""IFS + AIFS forecasts (gridded and at the stations) + hourly station wind, for the
multi-model station correction experiments (track A, A1).

    python scripts/build_multimodel_samples.py --start 2024-03-01 --end 2025-09-12
    python scripts/build_multimodel_samples.py --start 2024-03-01 --end 2025-09-12 --nordic
    python scripts/build_multimodel_samples.py --start 2024-03-01 --end 2025-09-12 --nordic --other-counties

--nordic appends the Swedish / Finnish stations of scripts/download_nordic_obs.py after the
Frost ones (which stay the same, in the same order) and writes <name>_ext.npz;
--other-counties also appends the Frost stations of the other Norwegian counties inside the
0.25 deg box and the 1 km terrain grid (Trondelag) and writes <name>_all.npz. `added` marks
the appended stations. The grid file does not depend on the stations and is reused.

Only runs where both IFS and AIFS exist on the common 0.25 deg grid (49 x 121, 62-74 N,
6-36 E) are kept.

OUT ROOT/cache/multimodel_<start>_<end>.npz:
    models, fields, lat (49,), lon (121,), leads (12,) hours 0-66 every 6 h
    inits    (N,) datetime64[h]
    st_fc    (N, L, M, S, 2) float32   (u10, v10) of each model, bilinear at the stations
    station, lon_st, lat_st, masl, xy (S, 2) km
    time     (T,) datetime64[h]  hourly, start - 12 h .. end + 66 h
    obs      (T, S, 2) float32   observed (u, v) at 10 m, NaN where missing
OUT ROOT/cache/multimodel_<start>_<end>_grid.npy:
    (N, L, M, C, 49, 121) float16, C = fields; K, hPa (msl, sp), m/s
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
from build_station_samples import load_obs, COUNTIES  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
NORDIC = ROOT / "stations" / "nordic"
MODELS = ["ifs", "aifs"]
FIELDS = ["10u", "10v", "2t", "msl", "sp", "t850"]
LEADS = list(range(0, 67, 6))


def path(model, t):
    return ROOT / "ecmwf" / model / f"{t:%Y}" / f"{model}_{t:%Y%m%dT%H}.nc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--nordic", action="store_true")
    ap.add_argument("--other-counties", action="store_true")
    ap.add_argument("--stations-like", help="keep exactly the stations of this sample file (same "
                    "station set and folds as an earlier period) instead of the > 50 %% coverage rule")
    args = ap.parse_args()
    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)

    inits = [t for t in pd.date_range(t0, t1 + pd.Timedelta(hours=12), freq="12h")
             if all(path(m, t).exists() for m in MODELS)]
    with netCDF4.Dataset(path("ifs", inits[0])) as ds:
        lat, lon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]

    st = pd.read_parquet(ROOT / "stations" / "stations.parquet")
    st = st[st.county.isin(COUNTIES) & (st.lat < lat.max() - 0.05) & (st.lat > lat.min() + 0.05)
            & (st.lon > lon.min() + 0.05) & (st.lon < lon.max() - 0.05)].reset_index(drop=True)
    extra = []
    n_base = len(st)
    if args.other_counties:
        assert args.nordic
        from download_nordic_obs import in_grid
        oc = pd.read_parquet(ROOT / "stations" / "stations.parquet")
        oc = oc[~oc.county.isin(COUNTIES | {"SVALBARD"}) & (oc.lat > lat.min() + 0.05)].reset_index(drop=True)
        oc = oc[in_grid(oc.lat.values, oc.lon.values)]
        st = pd.concat([st, oc], ignore_index=True)
    if args.nordic:
        nd = pd.read_parquet(NORDIC / "stations.parquet")
        nd = nd[(nd.lat < lat.max() - 0.05) & (nd.lat > lat.min() + 0.05)
                & (nd.lon > lon.min() + 0.05) & (nd.lon < lon.max() - 0.05)]
        st = pd.concat([st, nd], ignore_index=True)
        extra = sorted(NORDIC.glob("obs_*.parquet")) + sorted((ROOT / "stations" / "nordic_2026").glob("obs_*.parquet"))
    fi, fj = (st.lat.values - lat[0]) / dlat, (st.lon.values - lon[0]) / dlon
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    W = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], -1)        # (S, 4)
    II = np.stack([i0, i0, i0 + 1, i0 + 1], -1)
    JJ = np.stack([j0, j0 + 1, j0, j0 + 1], -1)

    N, L, M, C, S = len(inits), len(LEADS), len(MODELS), len(FIELDS), len(st)
    print(f"{N} runs with both models, {S} candidate stations", flush=True)
    out_grid = ROOT / "cache" / f"multimodel_{args.start}_{args.end}_grid.npy"
    out_grid.parent.mkdir(exist_ok=True)
    write_grid = not (args.nordic and out_grid.exists())
    if write_grid:
        grid = np.lib.format.open_memmap(out_grid, mode="w+", dtype=np.float16,
                                         shape=(N, L, M, C, len(lat), len(lon)))
    else:
        grid = np.load(out_grid, mmap_mode="r")
        assert grid.shape == (N, L, M, C, len(lat), len(lon)), "grid file from other runs"
    st_fc = np.full((N, L, M, S, 2), np.nan, np.float32)
    keep = np.ones(N, bool)
    for n, t in enumerate(inits):
        for k, m in enumerate(MODELS):
            with netCDF4.Dataset(path(m, t)) as ds:
                if ds["lat"].shape != lat.shape or not np.allclose(ds["lat"][:], lat) \
                        or not np.allclose(ds["lon"][:], lon):
                    keep[n] = False
                    continue
                leads = [int(x) for x in ds["lead"][:]]
                if any(ld not in leads for ld in LEADS):
                    keep[n] = False
                    continue
                jl = [leads.index(ld) for ld in LEADS]
                for c, f in enumerate(FIELDS):
                    if not write_grid and f not in ("10u", "10v"):
                        continue
                    v = np.ma.filled(ds[f][:].astype(np.float32), np.nan)[jl]   # (L, lat, lon)
                    if f in ("msl", "sp"):
                        v = v / 100.0                                       # Pa -> hPa (float16 range)
                    if write_grid:
                        grid[n, :, k, c] = v
                    if f in ("10u", "10v"):
                        st_fc[n, :, k, :, FIELDS.index(f)] = (v[:, II, JJ] * W).sum(-1)
        if n % 100 == 0:
            print(n, t, flush=True)
    if write_grid:
        grid.flush()
    finite = np.isfinite(st_fc).all((1, 2, 3, 4))
    keep &= finite
    print(f"runs kept: {keep.sum()} / {N}")

    time = pd.date_range(t0 - pd.Timedelta(hours=12), t1 + pd.Timedelta(hours=12 + 66), freq="h")
    w = load_obs(set(st.station), time[0], time[-1], extra)
    obs = np.full((len(time), S, 2), np.nan, np.float32)
    ti = pd.Index(time).get_indexer(w.index.get_level_values(0))
    si = pd.Index(st.station).get_indexer(w.index.get_level_values(1))
    ok = (ti >= 0) & (si >= 0)
    obs[ti[ok], si[ok]] = w.values[ok]
    good = np.isfinite(obs[..., 0]).mean(0) > 0.5
    print(f"stations with > 50 % hourly wind: {good.sum()} / {S}")
    if args.stations_like:
        ref = list(np.load(args.stations_like, allow_pickle=True)["station"])
        good = st.station.isin(ref).values
        assert list(st.station.values[good]) == ref, "station order differs from --stations-like"
        print(f"stations as in {args.stations_like}: {good.sum()}")

    tag = "_all" if args.other_counties else "_ext" if args.nordic else ""
    added = np.arange(len(st)) >= n_base
    np.savez_compressed(ROOT / "cache" / f"multimodel_{args.start}_{args.end}{tag}.npz",
                        models=MODELS, fields=FIELDS, lat=lat, lon=lon, leads=np.array(LEADS),
                        inits=np.array(inits, dtype="datetime64[h]"), keep=keep,
                        st_fc=st_fc[:, :, :, good], station=st.station.values[good],
                        lon_st=st.lon.values[good], lat_st=st.lat.values[good],
                        masl=st.masl.values[good].astype(float),
                        country=st.country.values[good].astype(str), added=added[good],
                        county=st.county.fillna("").values[good].astype(str),
                        xy=to_plane(st.lon.values, st.lat.values)[good].astype(np.float32),
                        time=time.values.astype("datetime64[h]"), obs=obs[:, good])
    print("done", out_grid)


if __name__ == "__main__":
    main()
