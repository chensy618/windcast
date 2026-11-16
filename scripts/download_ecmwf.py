"""Download ECMWF open-data forecasts (IFS, AIFS) over Northern Norway from the public
archive, for track B and for multi-model fusion.

    python scripts/download_ecmwf.py --model ifs  --start 2023-07-12 --end 2025-12-31
    python scripts/download_ecmwf.py --model aifs --start 2024-03-01 --end 2025-12-31

Only the needed fields are fetched (HTTP byte ranges from each step's .index), decoded with
eccodes and cropped to the box below. One file per run: OUT/<model>/YYYY/<model>_YYYYMMDDTHH.nc
with (lead, lat, lon) fields named as in the archive ("10u", "2t", ...; pressure levels
"u850", ...), the product path and grid. Fields a product does not have are absent.
Precipitation (tp) is accumulated since init; its unit differs between products (m for
IFS / AIFS, kg m-2 for aifs-single), so every field carries its GRIB "units" attribute.
Existing files are skipped.

Archive layout (checked 2026-11-16; GCS mirror from 2023-07-12, AWS from 2023-01-18 but
throttled for this site):
    IFS   0p4-beta/oper (0.4°, to 2024-01) | 0p25/oper (2024-02) | ifs/0p25/oper (2024-03 on)
    AIFS  aifs/0p25/oper (2024-03 .. 2025-02, experimental) | aifs-single/0p25/oper (2025-02 on)
Leads: the 6-hourly 0-66 h shared with MEPS (IFS also has 3-hourly, AIFS steps are 6 h).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import eccodes
import numpy as np
import requests
import xarray as xr

# GCS only: the AWS bucket (which also has 2023-01-18 .. 2023-07-11) answers this site
# with SlowDown, and its retries stall the run; back-fill that period separately.
MIRRORS = ["https://storage.googleapis.com/ecmwf-open-data"]
PRODUCTS = {"ifs": ["ifs/0p25/oper", "0p25/oper", "0p4-beta/oper"],
            "aifs": ["aifs-single/0p25/oper", "aifs/0p25/oper"]}
OUT = Path("/cluster/work/projects/nn8106k/siyan/windcast/ecmwf")
LAT, LON = (62.0, 74.0), (6.0, 36.0)   # Northern Norway (65.2-71.1 N, 11.9-31.1 E) + margin
SFC = ["10u", "10v", "2t", "2d", "msl", "tp", "tcc", "10fg", "100u", "100v", "sp"]
PL = {"u": [925, 850, 700, 500], "v": [925, 850, 700, 500], "t": [925, 850, 700, 500],
      "q": [925, 850, 700], "gh": [500], "z": [500]}

session = requests.Session()


def get(url: str, **kw) -> requests.Response:
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code in (429, 503) or b"SlowDown" in r.content[:200]:
                raise IOError(f"throttled {r.status_code}")
            return r
        except (IOError, requests.RequestException):
            time.sleep(20 * (attempt + 1))
    raise IOError(f"giving up on {url}")


def stem(d: date, h: int, product: str, lead: int) -> str:
    return f"{d:%Y%m%d}/{h:02d}z/{product}/{d:%Y%m%d}{h:02d}0000-{lead}h-oper-fc"


def find_product(d: date, h: int, model: str, lead: int = 0):
    """(mirror, product path) of the first product that has this run, probed on one step."""
    for m in MIRRORS:
        for p in PRODUCTS[model]:
            r = get(f"{m}/{stem(d, h, p, lead)}.index")
            if r.status_code == 200 and r.content[:1] == b"{":
                return m, p
    return None


def wanted(rec: dict) -> str | None:
    p = rec.get("param")
    if rec.get("levtype") == "sfc" and p in SFC:
        return p
    if rec.get("levtype") == "pl" and p in PL and int(rec.get("levelist", -1)) in PL[p]:
        return f"{p}{int(rec['levelist'])}"
    return None


# target for products not on a regular lat / lon grid (early AIFS runs: reduced Gaussian N320)
REG_LAT = np.arange(LAT[1], LAT[0] - 1e-6, -0.25)
REG_LON = np.arange(LON[0], LON[1] + 1e-6, 0.25)


def decode(msg: bytes):
    gid = eccodes.codes_new_from_message(msg)
    try:
        grid = eccodes.codes_get(gid, "gridType")
        miss = eccodes.codes_get(gid, "missingValue")
        units = eccodes.codes_get(gid, "units")
        vals = eccodes.codes_get_values(gid)
        if grid == "regular_ll":
            lats = eccodes.codes_get_array(gid, "distinctLatitudes")
            lons = eccodes.codes_get_array(gid, "distinctLongitudes")
            vals = vals.reshape(len(lats), len(lons))
        else:
            plat = eccodes.codes_get_array(gid, "latitudes")
            plon = eccodes.codes_get_array(gid, "longitudes")
    finally:
        eccodes.codes_release(gid)
    vals = np.where(vals == miss, np.nan, vals)
    if grid != "regular_ll":  # linear interpolation of the box's points to 0.25°
        from scipy.interpolate import griddata
        plon = ((plon + 180) % 360) - 180
        m = (plat >= LAT[0] - 1) & (plat <= LAT[1] + 1) & (plon >= LON[0] - 1) & (plon <= LON[1] + 1)
        glon, glat = np.meshgrid(REG_LON, REG_LAT)
        out = griddata((plon[m], plat[m]), vals[m], (glon, glat), method="linear")
        return REG_LAT, REG_LON, out.astype(np.float32), units
    lons = ((lons + 180) % 360) - 180
    order = np.argsort(lons)
    lons, vals = lons[order], vals[:, order]
    iy = (lats >= LAT[0]) & (lats <= LAT[1])
    ix = (lons >= LON[0]) & (lons <= LON[1])
    return lats[iy], lons[ix], vals[np.ix_(iy, ix)].astype(np.float32), units


def fetch_field(job):
    url, a, n = job
    r = get(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
    if r.status_code not in (200, 206):
        raise IOError(f"{url}: HTTP {r.status_code}")
    return decode(r.content)


def fetch_run(d: date, h: int, model: str, leads, out: Path, threads: int = 16) -> str:
    found = find_product(d, h, model)
    if found is None:
        raise FileNotFoundError("run not in the archive")
    mirror, product = found
    with ThreadPoolExecutor(threads) as pool:
        idx = dict(zip(leads, pool.map(lambda L: get(f"{mirror}/{stem(d, h, product, L)}.index"), leads)))
        jobs, keys = [], []
        for L, r in idx.items():
            if r.status_code != 200 or r.content[:1] != b"{":
                continue  # step missing in this run
            for rec in (json.loads(x) for x in r.text.splitlines() if x.strip()):
                name = wanted(rec)
                if name is not None:
                    jobs.append((f"{mirror}/{stem(d, h, product, L)}.grib2",
                                 int(rec["_offset"]), int(rec["_length"])))
                    keys.append((name, L))
        if not jobs:
            raise FileNotFoundError("no step of this run in the archive")
        res = list(pool.map(fetch_field, jobs))
    grids = {la.shape + lo.shape for la, lo, _, _ in res}
    if len(grids) > 1:  # mixed grids within a run: put everything on the regular 0.25° grid
        raise IOError(f"mixed grids in one run: {grids}")
    fields, units, lat, lon = {}, {}, res[0][0], res[0][1]
    for (name, L), (la, lo, v, u) in zip(keys, res):
        assert la.shape == lat.shape and lo.shape == lon.shape, "grid changed within a run"
        fields.setdefault(name, {})[L] = v
        units[name] = u
    got = sorted({L for _, L in keys})
    names = sorted(fields)
    data = {}
    for k in names:
        arr = np.full((len(got), len(lat), len(lon)), np.nan, np.float32)
        for i, L in enumerate(got):
            if L in fields[k]:
                arr[i] = fields[k][L]
        data[k] = (("lead", "lat", "lon"), arr, {"units": units[k]})
    tmp = out.with_suffix(".tmp")
    xr.Dataset(data, coords={"lead": ("lead", np.asarray(got, "i2")), "lat": lat, "lon": lon},
               attrs={"model": model, "product": product, "init": f"{d:%Y-%m-%d}T{h:02d}",
                      "regridded": int(lat is REG_LAT)}
               ).to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 3} for k in names})
    tmp.rename(out)
    return product


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(PRODUCTS), required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--inits", type=int, nargs="+", default=[0, 12])
    p.add_argument("--leads", type=int, nargs="+", default=list(range(0, 67, 6)))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    d, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    n_ok = n_fail = 0
    while d <= d1:
        for h in a.inits:
            out = a.out / a.model / f"{d:%Y}" / f"{a.model}_{d:%Y%m%d}T{h:02d}.nc"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                prod = fetch_run(d, h, a.model, a.leads, out)
                n_ok += 1
                print(f"{d} {h:02d}Z ok {prod} {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                n_fail += 1
                msg = str(e).splitlines()[0][:120]
                print(f"{d} {h:02d}Z FAIL {msg}", flush=True)
                with open(a.out / f"missing_{a.model}.txt", "a") as f:
                    f.write(f"{d:%Y%m%d}T{h:02d} {msg}\n")
        d += timedelta(days=1)
    print(f"done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
