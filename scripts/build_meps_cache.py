"""MEPS 10 m wind of the multimodel sample runs, as dense targets for pretraining the 1 km level
of the two-level operator (windcast/training/train_station_operator.py --pretrain-meps).

    python scripts/build_meps_cache.py          # CPU node, after scripts/download_meps.py

For every init and lead of cache/multimodel_2024-03-01_2025-09-12.npz (00 / 12 UTC, 0-66 h
every 6 h), the MEPS run of the same init (ROOT/meps/YYYY/meps_YYYYMMDDTHH.nc, earth-relative
u10 / v10 on its 2.5 km window), NaN where the run is missing. Plus, for every cell of the 1 km
terrain grid (static/terrain_descriptors_1km.npz), the nearest MEPS cell and whether it lies
inside the MEPS window (nearest MEPS cell within 2 km).

OUT ROOT/cache/meps_uv_2024-03-01_2025-09-12.npy   (N, L, 2, Hm, Wm) float16
    ROOT/cache/meps_uv_2024-03-01_2025-09-12_map.npz  iy, ix (H1, W1) int32; inside (H1, W1);
                                                    have (N,) runs present
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.geo import to_plane  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
TAG = "2024-03-01_2025-09-12"


def main():
    d = np.load(ROOT / "cache" / f"multimodel_{TAG}.npz", allow_pickle=True)
    inits, leads = pd.DatetimeIndex(d["inits"]), [int(x) for x in d["leads"]]
    files = [ROOT / "meps" / f"{t:%Y}" / f"meps_{t:%Y%m%dT%H}.nc" for t in inits]
    ref = next(f for f in files if f.exists())
    with netCDF4.Dataset(ref) as ds:
        lon, lat = ds["lon"][:], ds["lat"][:]
    Hm, Wm = lon.shape
    out = np.lib.format.open_memmap(ROOT / "cache" / f"meps_uv_{TAG}.npy", mode="w+", dtype=np.float16,
                                    shape=(len(inits), len(leads), 2, Hm, Wm))
    have = np.zeros(len(inits), bool)
    for n, f in enumerate(files):
        out[n] = np.nan
        if not f.exists():
            continue
        try:
            with netCDF4.Dataset(f) as ds:
                if ds["lon"].shape != (Hm, Wm):
                    print(f.name, "other grid, skipped")
                    continue
                fl = [int(x) for x in ds["lead"][:]]
                for j, ld in enumerate(leads):
                    if ld in fl:
                        k = fl.index(ld)
                        out[n, j, 0] = np.ma.filled(ds["u10"][k], np.nan)
                        out[n, j, 1] = np.ma.filled(ds["v10"][k], np.nan)
            have[n] = True
        except OSError as e:
            print(f.name, "unreadable", e)
    out.flush()
    t = np.load(ROOT / "static" / "terrain_descriptors_1km.npz")
    xy1 = t["xy"].reshape(-1, 2)
    pm = to_plane(lon.ravel(), lat.ravel())
    dist, idx = cKDTree(pm).query(xy1)
    H1, W1 = t["xy"].shape[:2]
    iy, ix = np.unravel_index(idx, (Hm, Wm))
    np.savez_compressed(ROOT / "cache" / f"meps_uv_{TAG}_map.npz", iy=iy.reshape(H1, W1).astype(np.int32),
                        ix=ix.reshape(H1, W1).astype(np.int32), inside=(dist < 2.0).reshape(H1, W1), have=have)
    tr = (inits >= "2024-03-01") & (inits < "2025-01-01")
    print(f"MEPS runs present: {have.sum()} / {len(inits)} (train period {have[tr].sum()} / {tr.sum()}); "
          f"1 km cells inside the MEPS window {np.mean(dist < 2.0):.2f}")


if __name__ == "__main__":
    main()
