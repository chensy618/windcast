"""Potential-temperature and mountain-wind features at the stations, for every (init, lead) of
the multimodel samples (train_dual_gno.py --phys).

    python scripts/build_station_physics.py

Every feature is computed from forecast fields and 1 km terrain only, so the same features exist
at any 1 km cell (no station identity). Wind vectors are earth-relative (east, north).

Sources
  IFS + AIFS (0.25 deg, mean of the two, bilinear at the station): 2t, sp, t850, t700, u850, v850
      (u850 / v850 from IFS alone where AIFS lacks them: the first AIFS runs of March 2024)
  MEPS + AROME-Arctic (2.5 km grid of build_lam_samples.py, mean of the two, nearest cell): 2t,
      msl, tcc, blh; 10u / 10v of all four models from the st_fc of the two sample files
  1 km terrain (static/terrain_descriptors_1km.npz, station_terrain.npz): elevation, Sx; the
      terrain gradient at 1 km (raw) and at ridge scale (Gaussian sigma 5 km); relief within 10 km
  station height (masl) for the station-level pressure

Potential temperature  theta = T (1000 hPa / p)^0.2857
  th2m_g      theta at 2 m on the global models' surface (2t, sp)
  dth_s850    theta850 - th2m_g: low-level stability of the global columns (> 0 stable)
  dth_850_700 theta700 - theta850
  N           bulk Brunt-Vaisala frequency of the surface-850 hPa layer (s-1, from dth_s850 and
              the hypsometric thickness; 850-700 hPa where sp < 870 hPa); 0 where unstable
  th2m_st     theta at 2 m at the station height (regional 2t; p from msl and masl)
  dth_st850   theta850 - th2m_st: stability seen from the station level (valley cold pools)
  log_blh     log10 of the regional boundary-layer height (m)
Mountain winds
  u850, v850  global-model wind at 850 hPa (flow above the ridges)
  log_fr      log10 Froude number U850 / (N H), H = relief within 10 km (>= 10 m); clipped to
              [-2, 2] (2 where N = 0)
  speed10     speed of the four-model mean 10 m wind
  up_sx, up_sx10  upwind shelter (1 km, 10 km) for the four-model mean wind direction
  w10         upslope component U10 . grad z (1 km), m/s; > 0 flow up the slope
  w850        U850 . grad z (ridge scale)
  sin_sun     sine of the solar elevation
  clear       1 - regional total cloud cover
  insol       clear * max(0, cos incidence) on the 1 km slope (anabatic heating)
  ana_u, ana_v  insol * grad z (1 km): up-slope vector, daytime
  kat_u, kat_v  clear * night * (-grad z) (1 km): down-slope vector, clear nights (katabatic);
              night = clip(-10 sin_sun, 0, 1)

OUT ROOT/cache/station_physics_2024-03-01_2025-09-12.npz
    values (N, L, S, F) float32, names (F,), theta (names of the potential-temperature group),
    mountain (names of the mountain-wind group), station (S,)
"""
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windcast.station_correction import SX_DIRS, upwind  # noqa: E402

ROOT = Path("/cluster/work/projects/nn8106k/siyan/windcast")
TAG = os.environ.get("WINDCAST_PERIOD", "2024-03-01_2025-09-12")  # as train_station_operator.py
KAPPA, G, RD = 0.2857, 9.81, 287.0
THETA = ["th2m_g", "dth_s850", "dth_850_700", "N", "th2m_st", "dth_st850", "log_blh"]
MOUNTAIN = ["u850", "v850", "log_fr", "speed10", "up_sx", "up_sx10", "w10", "w850", "sin_sun",
            "clear", "insol", "ana_u", "ana_v", "kat_u", "kat_v"]
GLOBAL_VARS = ["2t", "sp", "t850", "t700", "u850", "v850"]
LAM_FIELDS = ["2t", "msl", "tcc", "blh"]


def theta(t, p_hpa):
    return t * (1000.0 / p_hpa) ** KAPPA


def solar(times, lon, lat):
    """Sine of the solar elevation and solar azimuth (rad, clockwise from north).
    times (N, L) datetime64, lon / lat (S,) deg -> (N, L, S) each."""
    t = pd.DatetimeIndex(times.ravel())
    doy = t.dayofyear.values.reshape(times.shape)[..., None]
    hour = (t.hour + t.minute / 60).values.reshape(times.shape)[..., None]
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))          # minutes
    ha = np.deg2rad((hour * 60 + eqt + 4 * lon) / 4 - 180)                         # hour angle
    phi = np.deg2rad(lat)
    sin_h = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    az = np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(phi) - np.sin(phi) * np.cos(ha)) % (2 * np.pi)
    return sin_h, az


def terrain_at_stations(st_ij):
    """1 km terrain gradient (raw and sigma 5 km, m/m, (S, 2) east / north) and relief within
    10 km (m) at the stations' 1 km cells."""
    t = np.load(ROOT / "static" / "terrain_descriptors_1km.npz", allow_pickle=True)
    names = [str(x) for x in t["names"]]
    z = np.nan_to_num(t["fields"][names.index("elevation")].astype(np.float64))   # (H, W), sea 0
    xy = t["xy"].astype(np.float64)                                              # (H, W, 2) km
    zs = gaussian_filter(z, 5.0)
    i, j = st_ij[:, 0], st_ij[:, 1]

    def grad(f):
        dx = (xy[i, j + 1, 0] - xy[i, j - 1, 0]) * 1000.0
        dy = (xy[i + 1, j, 1] - xy[i - 1, j, 1]) * 1000.0
        return np.stack([(f[i, j + 1] - f[i, j - 1]) / dx, (f[i + 1, j] - f[i - 1, j]) / dy], -1)

    r = 10
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = yy ** 2 + xx ** 2 <= r * r
    relief = np.array([np.ptp(z[a - r:a + r + 1, b - r:b + r + 1][disk]) for a, b in zip(i, j)])
    return grad(z), grad(zs), relief


def global_at(inits, leads, lat, lon, lon_p, lat_p, verbose=False):
    """GLOBAL_VARS of IFS and AIFS, bilinear from the 0.25 deg grid (lat, lon) to the points,
    averaged per variable over the models that have it -> {var: (n, L, P)} (NaN where a run or
    a lead is missing)."""
    fi, fj = (lat_p - lat[0]) / (lat[1] - lat[0]), (lon_p - lon[0]) / (lon[1] - lon[0])
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    wa, wb = fi - i0, fj - j0
    Wb = np.stack([(1 - wa) * (1 - wb), (1 - wa) * wb, wa * (1 - wb), wa * wb], -1)
    II, JJ = np.stack([i0, i0, i0 + 1, i0 + 1], -1), np.stack([j0, j0 + 1, j0, j0 + 1], -1)
    N, L, P = len(inits), len(leads), len(lon_p)
    glob = {v: np.full((N, L, P), np.nan, np.float32) for v in GLOBAL_VARS}
    for n, t in enumerate(inits):
        acc = {v: 0.0 for v in GLOBAL_VARS}
        cnt = {v: 0 for v in GLOBAL_VARS}
        ok = True
        for m in ("ifs", "aifs"):
            f = ROOT / "ecmwf" / m / f"{t:%Y}" / f"{m}_{t:%Y%m%dT%H}.nc"
            if not f.exists():
                ok = False
                break
            with netCDF4.Dataset(f) as ds:
                fl = [int(x) for x in ds["lead"][:]]
                if any(ld not in fl for ld in leads):
                    ok = False
                    break
                jl = [fl.index(ld) for ld in leads]
                for v in GLOBAL_VARS:
                    if v not in ds.variables:       # the first AIFS runs (March 2024) lack u / v 850
                        continue
                    a = np.ma.filled(ds[v][:].astype(np.float32), np.nan)[jl]          # (L, H, W)
                    acc[v] = acc[v] + (a[:, II, JJ] * Wb).sum(-1)
                    cnt[v] += 1
        if ok:
            for v in GLOBAL_VARS:
                if cnt[v]:
                    glob[v][n] = acc[v] / cnt[v]
        if verbose and n % 200 == 0:
            print("global", n, t, flush=True)
    return glob


def regional_at(lam, ns, xy, verbose=False, mm=None):
    """LAM_FIELDS of MEPS and AROME-Arctic at the points xy (P, 2) km: nearest cell of the 2.5 km
    grid, mean of the two models -> (len(ns), L, F, P)."""
    gx, gy = lam["grid_x"].astype(float), lam["grid_y"].astype(float)
    ri = np.rint((xy[:, 1] - gy[0]) / (gy[1] - gy[0])).astype(int)
    rj = np.rint((xy[:, 0] - gx[0]) / (gx[1] - gx[0])).astype(int)
    lf = [str(x) for x in lam["fields"]]
    fsel = [lf.index(f) for f in LAM_FIELDS]
    if mm is None:
        mm = np.load(ROOT / "cache" / f"multimodel_{TAG}_lam_grid.npy", mmap_mode="r")   # (N, L, 2, C, Hr, Wr)
    L = mm.shape[1]
    reg = np.full((len(ns), L, len(LAM_FIELDS), len(xy)), np.nan, np.float32)
    for k, n in enumerate(ns):
        a = np.asarray(mm[n][:, :, fsel][..., ri, rj], np.float32)                     # (L, 2, F, P)
        reg[k] = a.mean(1)
        if verbose and k % 200 == 0:
            print("regional", k, flush=True)
    return reg


def features(glob, sp, t2r, mslr, tcc, blh, u10, v10, times, lon, lat, masl, grad1, grad10, relief, tv, tn):
    """All THETA + MOUNTAIN features at P points (stations or 1 km cells).
    glob: dict of the global variables (..., P) (GLOBAL_VARS, sp in hPa given separately);
    t2r, mslr, tcc (fraction), blh: regional fields (..., P); u10, v10: four-model mean 10 m wind
    (..., P); times (n, L) valid times; lon, lat, masl (P,); grad1, grad10 (P, 2), relief (P,):
    terrain_at_stations(); tv (P, F), tn: the 1 km terrain descriptors and their names.
    The leading dims (...) are (n, L). Returns {name: (n, L, P)}."""
    # --- potential temperature
    th2 = theta(glob["2t"], sp)
    th850, th700 = theta(glob["t850"], 850.0), theta(glob["t700"], 700.0)
    low = sp >= 870.0
    tbar = np.where(low, (glob["2t"] + glob["t850"]) / 2, (glob["t850"] + glob["t700"]) / 2)
    dz = RD * tbar / G * np.where(low, np.log(np.maximum(sp, 870.0) / 850.0), np.log(850.0 / 700.0))
    dth = np.where(low, th850 - th2, th700 - th850)
    n2 = G / np.where(low, (th850 + th2) / 2, (th850 + th700) / 2) * dth / dz
    Nbv = np.sqrt(np.maximum(n2, 0.0))
    p_st = mslr * np.exp(-G * masl / (RD * (t2r + 0.00325 * masl)))
    th_st = theta(t2r, p_st)

    # --- mountain winds
    u8, v8 = glob["u850"], glob["v850"]
    U8 = np.hypot(u8, v8)
    H = np.maximum(relief, 10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(Nbv > 1e-4, np.log10(U8 / (Nbv * H) + 1e-6), 2.0)
    log_fr = np.clip(fr, -2.0, 2.0)
    sin_h, az = solar(times, lon, lat)
    clear = np.clip(1.0 - tcc, 0.0, 1.0)
    gxe, gyn = grad1[:, 0], grad1[:, 1]
    nrm = np.sqrt(1 + gxe ** 2 + gyn ** 2)
    cos_h = np.sqrt(np.clip(1 - sin_h ** 2, 0, 1))
    cos_i = (-gxe * cos_h * np.sin(az) - gyn * cos_h * np.cos(az) + sin_h) / nrm      # slope normal . sun
    insol = clear * np.maximum(cos_i, 0.0) * (sin_h > 0)
    night = np.clip(-10 * sin_h, 0.0, 1.0)
    return {
        "th2m_g": th2, "dth_s850": th850 - th2, "dth_850_700": th700 - th850, "N": Nbv,
        "th2m_st": th_st, "dth_st850": th850 - th_st, "log_blh": np.log10(np.maximum(blh, 10.0)),
        "u850": u8, "v850": v8, "log_fr": log_fr, "speed10": np.hypot(u10, v10),
        "up_sx": upwind(tv[:, [tn.index(f"sx_{k}") for k in SX_DIRS]], u10, v10),
        "up_sx10": upwind(tv[:, [tn.index(f"sx10_{k}") for k in SX_DIRS]], u10, v10),
        "w10": u10 * gxe + v10 * gyn, "w850": u8 * grad10[:, 0] + v8 * grad10[:, 1],
        "sin_sun": sin_h, "clear": clear, "insol": insol,
        "ana_u": insol * gxe, "ana_v": insol * gyn,
        "kat_u": -clear * night * gxe, "kat_v": -clear * night * gyn,
    }


def main():
    # [--stations ext] [output path]; ext: + the Swedish / Finnish stations
    args = sys.argv[1:]
    sfx = ""
    if args[:2] == ["--stations", "ext"]:
        sfx, args = "_ext", args[2:]
    base = np.load(ROOT / "cache" / f"multimodel_{TAG}{sfx}.npz", allow_pickle=True)
    lam = np.load(ROOT / "cache" / f"multimodel_{TAG}{sfx}_lam.npz", allow_pickle=True)
    assert (base["station"] == lam["station"]).all() and (base["inits"] == lam["inits"]).all()
    inits, leads = pd.DatetimeIndex(base["inits"]), [int(x) for x in base["leads"]]
    N, L, S = len(inits), len(leads), len(base["station"])
    lon_st, lat_st, masl = base["lon_st"].astype(float), base["lat_st"].astype(float), base["masl"].astype(float)

    # --- global models: bilinear on the 0.25 deg grid (as build_group_samples.py), mean of IFS and AIFS
    glob = global_at(inits, leads, base["lat"].astype(float), base["lon"].astype(float), lon_st, lat_st, verbose=True)
    sp = glob["sp"] / 100.0 if np.nanmedian(glob["sp"]) > 2000 else glob["sp"]          # Pa -> hPa
    print(f"global: sp median {np.nanmedian(sp):.0f} hPa, t850 median {np.nanmedian(glob['t850']):.1f} K")

    # --- regional models: nearest cell of the 2.5 km grid, mean of MEPS and AROME-Arctic
    reg = regional_at(lam, np.arange(N), lam["xy"], verbose=True)
    t2r, mslr, tcc, blh = (reg[:, :, k] for k in range(4))
    if np.nanmax(tcc) > 1.5:                                                            # percent
        tcc = tcc / 100.0
    print(f"regional: msl median {np.nanmedian(mslr):.0f}, tcc max {np.nanmax(tcc):.2f}, "
          f"blh median {np.nanmedian(blh):.0f} m")

    # --- features (potential temperature, mountain winds)
    tt = np.load(ROOT / "cache" / f"station_terrain{sfx}.npz", allow_pickle=True)
    assert (tt["station"] == base["station"]).all()
    grad1, grad10, relief = terrain_at_stations(tt["ij"])
    fc = np.concatenate([base["st_fc"], lam["st_fc"]], 2)                               # (N, L, 4, S, 2)
    times = np.array([[t + pd.Timedelta(hours=ld) for ld in leads] for t in inits], dtype="datetime64[ns]")
    feats = features(glob, sp, t2r, mslr, tcc, blh, fc[..., 0].mean(2), fc[..., 1].mean(2), times,
                     lon_st, lat_st, masl, grad1, grad10, relief, tt["values"].astype(np.float64),
                     [str(x) for x in tt["names"]])
    names = THETA + MOUNTAIN
    vals = np.stack([np.broadcast_to(feats[k], (N, L, S)) for k in names], -1).astype(np.float32)
    keep = base["keep"] & lam["keep"]
    for k, nm in enumerate(names):
        x = vals[keep][..., k]
        print(f"{nm:12s} median {np.nanmedian(x):9.3f}  p1 {np.nanpercentile(x, 1):9.3f}  "
              f"p99 {np.nanpercentile(x, 99):9.3f}  nan {np.isnan(x).mean():.3f}")
    out = Path(args[0]) if args else ROOT / "cache" / f"station_physics_{TAG}{sfx}.npz"
    np.savez_compressed(out, values=vals, names=np.array(names), theta=np.array(THETA),
                        mountain=np.array(MOUNTAIN), station=base["station"])
    print("done", out)


if __name__ == "__main__":
    main()
