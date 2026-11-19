"""Kartverket DTM (Norwegian national height model, bare earth) at 50 m for mainland Norway, from
the open Geonorge WCS, as 100 km tiles in EPSG:25833 (UTM 33N), for the Norway-wide terrain
descriptors (track A).

    python scripts/download_dtm.py

Tiles x -100 .. 1100 km, y 6400 .. 8000 km (UTM 33N): every 100 km square that touches mainland
Norway. OUT ROOT/static/dtm50/dtm50_<x0 km>_<y0 km>.tif (2000 x 2000, float32, metres; outside
Norway the WCS returns 0 / -1). Existing tiles are skipped.
"""
import time
from pathlib import Path

import requests

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
OUT = ROOT / "static" / "dtm50"
WCS = ("https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833?service=WCS&version=1.0.0&request=GetCoverage"
       "&coverage=nhm_dtm_topo_25833&crs=EPSG:25833&bbox={x0},{y0},{x1},{y1}&resx=50&resy=50&format=GeoTIFF")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for x0 in range(-100_000, 1_100_000, 100_000):
        for y0 in range(6_400_000, 8_000_000, 100_000):
            f = OUT / f"dtm50_{x0 // 1000}_{y0 // 1000}.tif"
            if f.exists():
                continue
            url = WCS.format(x0=x0, y0=y0, x1=x0 + 100_000, y1=y0 + 100_000)
            for k in range(5):
                try:
                    r = requests.get(url, timeout=600)
                    if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
                        f.with_suffix(".tmp").write_bytes(r.content)
                        f.with_suffix(".tmp").rename(f)
                        n_ok += 1
                        print(f.name, "ok", flush=True)
                        break
                    raise IOError(f"HTTP {r.status_code} {r.content[:80]!r}")
                except (IOError, requests.RequestException) as e:
                    if k == 4:
                        n_fail += 1
                        print(f.name, "FAIL", e, flush=True)
                    time.sleep(20 * (k + 1))
    print(f"done: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    main()
