"""Download NOAA GFS 0.25 deg forecasts from the AWS open-data archive (noaa-gfs-bdp-pds), cut
to the 0.25 deg box of mainland Norway, for the MLWP / NWP comparison (global NWP group, track A).

    python scripts/download_gfs.py --start 2024-02-28 --end 2025-09-16

Only the needed GRIB messages are fetched (HTTP byte ranges from each step's .idx): 10 m u / v,
2 m temperature, mean sea-level pressure (PRMSL) and 850 hPa temperature, 0-66 h every 6 h.
Decoded and cropped as in scripts/download_ecmwf.py. One file per run, in the same layout:
OUT/gfs/YYYY/gfs_YYYYMMDDTHH.nc with (lead, lat, lon) fields 10u, 10v, 2t, msl, t850 on
lat 74 .. 57.5 N (descending) x lon 4 .. 36 E. Existing files are skipped.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import eccodes
import netCDF4
import numpy as np
import requests

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/{h:02d}/atmos/gfs.t{h:02d}z.pgrb2.0p25.f{L:03d}"
OUT = Path("/cluster/work/projects/nn8106k/siyan/windcast/gfs")
LEADS = list(range(0, 67, 6))
LAT, LON = (57.5, 74.0), (4.0, 36.0)
# (GRIB short name, level) in the .idx -> output name
WANT = {("UGRD", "10 m above ground"): "10u", ("VGRD", "10 m above ground"): "10v",
        ("TMP", "2 m above ground"): "2t", ("PRMSL", "mean sea level"): "msl", ("TMP", "850 mb"): "t850"}
UNITS = {"10u": "m s**-1", "10v": "m s**-1", "2t": "K", "msl": "Pa", "t850": "K"}
session = requests.Session()


def get(url, **kw):
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code in (429, 503) or b"SlowDown" in r.content[:200]:
                raise IOError(f"throttled {r.status_code}")
            return r
        except (IOError, requests.RequestException):
            time.sleep(10 * (attempt + 1))
    raise IOError(f"giving up on {url}")


def ranges(idx_text):
    """(output name, start, length) of the wanted messages in one .idx"""
    rows = [x.split(":") for x in idx_text.splitlines() if x.strip()]
    out = []
    for i, r in enumerate(rows):
        key = (r[3], r[4])
        if key in WANT:
            a = int(r[1])
            b = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
            out.append((WANT[key], a, None if b is None else b - a + 1))
    return out


def decode(msg):
    gid = eccodes.codes_new_from_message(msg)
    try:
        lats = eccodes.codes_get_array(gid, "distinctLatitudes")
        lons = eccodes.codes_get_array(gid, "distinctLongitudes")
        vals = eccodes.codes_get_values(gid).reshape(len(lats), len(lons))
        miss = eccodes.codes_get(gid, "missingValue")
    finally:
        eccodes.codes_release(gid)
    vals = np.where(vals == miss, np.nan, vals)
    lons = ((lons + 180) % 360) - 180
    order = np.argsort(lons)
    lons, vals = lons[order], vals[:, order]
    iy = (lats >= LAT[0] - 1e-6) & (lats <= LAT[1] + 1e-6)
    ix = (lons >= LON[0] - 1e-6) & (lons <= LON[1] + 1e-6)
    return lats[iy], lons[ix], vals[np.ix_(iy, ix)].astype(np.float32)


def fetch_run(d, h, out, threads=8):
    def one(L):
        url = BASE.format(d=d, h=h, L=L)
        r = get(url + ".idx")
        if r.status_code != 200:
            raise FileNotFoundError(f"no .idx for f{L:03d} (HTTP {r.status_code})")
        res = {}
        for name, a, n in ranges(r.text):
            rng = f"bytes={a}-{a + n - 1}" if n else f"bytes={a}-"
            m = get(url, headers={"Range": rng})
            if m.status_code not in (200, 206):
                raise IOError(f"{url}: HTTP {m.status_code}")
            res[name] = decode(m.content)
        missing = set(WANT.values()) - set(res)
        if missing:
            raise KeyError(f"f{L:03d} lacks {sorted(missing)}")
        return L, res

    with ThreadPoolExecutor(threads) as pool:
        got = dict(pool.map(one, LEADS))
    glat, glon, _ = got[0]["10u"]
    tmp = out.with_suffix(".tmp")
    with netCDF4.Dataset(tmp, "w") as o:
        o.model, o.product, o.init, o.regridded = "gfs", "gfs/pgrb2.0p25", f"{d:%Y-%m-%d}T{h:02d}", 0
        o.createDimension("lead", len(LEADS))
        o.createDimension("lat", len(glat))
        o.createDimension("lon", len(glon))
        o.createVariable("lead", "i2", ("lead",))[:] = LEADS
        o.createVariable("lat", "f8", ("lat",))[:] = glat
        o.createVariable("lon", "f8", ("lon",))[:] = glon
        for k in UNITS:
            v = o.createVariable(k, "f4", ("lead", "lat", "lon"), zlib=True, complevel=3)
            v.units = UNITS[k]
            v[:] = np.stack([got[L][k][2] for L in LEADS])
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 12])
    ap.add_argument("--leads", type=int, nargs="+", default=None, help="default 0-66 h every 6 h")
    ap.add_argument("--out", type=Path, default=None, help="default ROOT/gfs")
    a = ap.parse_args()
    global LEADS, OUT
    LEADS = a.leads or LEADS
    OUT = a.out or OUT
    d, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    n_ok = n_fail = 0
    while d <= d1:
        for h in a.inits:
            out = OUT / f"{d:%Y}" / f"gfs_{d:%Y%m%d}T{h:02d}.nc"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                fetch_run(d, h, out)
                n_ok += 1
                print(f"{d} {h:02d}Z ok {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"{d} {h:02d}Z FAIL {str(e).splitlines()[0][:120]}", flush=True)
        d += timedelta(days=1)
    print(f"done: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
