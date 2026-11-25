"""1 km 10 m wind forecasts of one run for every model configuration of the website (GPU).

    python web/pipeline/forecast.py --init 2026-11-18T12 --out web/site/data

Configurations (CONFIGS), trained with the 1 km terrain descriptors
(windcast/training/train_station_operator.py --inputs terrain --lr 1e-5):
    ecmwf-*  IFS + AIFS                         (runs <model>_terrain_g<k>_s<s>_lr1e-05)
The IFS + GFS + MEPS + AROME-Arctic group ("nwp", runs ..._nwp_lr1e-05) is no longer published:
GFS is not used on the website; add "nwp" back to GROUPS to restore it.
    *-sfno / *-fno: operator on the 0.25 deg fields + MLP head; *-mlp: MLP head only.
    dual-gno WindCast: the four-field dual-scale GNO (IFS + AIFS + MEPS + AROME-Arctic), see dual.py
Each configuration is the mean of 15 models (5 station groups x 3 seeds); the spread is the std
of their wind speeds. The normalisation of the training data is rebuilt by Data (dual-gno: the
statistics cached by dual.py --prepare).

Inputs at every 1 km cell (static/terrain_descriptors_1km.npz), as in training:
* global models (IFS, AIFS, GFS): bilinear from the 0.25 deg grid;
* MEPS / AROME-Arctic: the nearest 2.5 km cell (training used it at the stations); cells farther
  than 2 km from any MEPS cell (outside its northern window) get no nwp-* forecast;
* the 28 cell descriptors and the upwind shelter for the base wind.
The operator sees the 0.25 deg fields: MEPS / AROME as the mean of their cells in each 0.25 deg
cell, IFS where MEPS has none (scripts/build_group_samples.py).

Writes OUT/runs/<YYYYMMDDTHH>/:
    <config>/lead_<LL>.webp   R = corrected speed / 0.2 m/s, G = group mean (uncorrected) speed / 0.2,
                              B = corrected direction (from) / 360 * 255; (0, 0, 0) = no data
    forecast.json             run, leads, valid times, configurations, station forecasts
and updates OUT/static (background, elevation, grid geometry) if missing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import windcast.training.train_station_operator as tso  # noqa: E402
from build_group_samples import FIELDS as NWP_FIELDS, LAMBERT, LAMBERT_VARS, Lambert, path as group_path  # noqa: E402
from predict_terrain_grid import DESC, Cells  # noqa: E402
from windcast.geo import to_plane  # noqa: E402
from windcast.station_correction import upwind  # noqa: E402
import dual  # noqa: E402  (web/pipeline/dual.py: WindCast, the four-field dual-scale GNO)

ROOT = tso.ROOT
ECMWF_FIELDS = ["10u", "10v", "2t", "msl", "sp", "t850"]
GROUPS = {"ecmwf": ["ifs", "aifs"]}             # + "nwp": ["ifs", "gfs", "meps", "arome_arctic"] (not published)
GROUP_LABEL = {"ecmwf": "IFS + AIFS", "nwp": "IFS + GFS + MEPS + AROME-Arctic"}
HEADS = {"sfno": "SFNO + MLP", "fno": "FNO + MLP", "point": "MLP"}
CONFIGS = {f"{g}-{'mlp' if h == 'point' else h}": dict(group=g, head=h, label=f"{HEADS[h]} · {GROUP_LABEL[g]}")
           for g in GROUPS for h in HEADS}
GROUP_LABEL["dual"] = "IFS + AIFS + MEPS + AROME-Arctic"
CONFIGS["dual-gno"] = dict(group="dual", head="gno", label=f"WindCast · {GROUP_LABEL['dual']}")
DATA0, GRID0 = tso.DATA, tso.GRID_FILE
STATIC_VERSION = 3   # bump to rewrite OUT/static
CROP_SOUTH = 85      # southernmost image rows (km) left blank: the terrain grid has artefacts there


def speed_dir(uv):
    s = np.hypot(uv[..., 0], uv[..., 1])
    return s, (np.degrees(np.arctan2(-uv[..., 0], -uv[..., 1])) + 360) % 360   # direction the wind blows from


def data_for(group, dev):
    tso.DATA, tso.GRID_FILE = DATA0, GRID0
    tso.use_group(group)
    return tso.Data(dev, terrain=True)


def model_files(group):
    return {g: (ROOT / "ecmwf" / g if g in ("ifs", "aifs") else ROOT / g) for g in GROUPS[group]}


def available(group, init):
    return all(group_path(m, init).exists() for m in GROUPS[group])


def load_inputs(group, D, init, cells):
    """0.25 deg fields (L, M*C, H, W) and the wind of each model at the cells (L, P, M, 2)."""
    L, lat, lon = D.L, D.lat, D.lon
    fields = ECMWF_FIELDS if group == "ecmwf" else NWP_FIELDS
    grid = np.zeros((L, D.M, len(fields), D.H, D.W), np.float32)
    native = {}
    ifs = None
    for k, m in enumerate(GROUPS[group]):
        with netCDF4.Dataset(group_path(m, init)) as ds:
            fl = [int(x) for x in ds["lead"][:]]
            jl = [fl.index(int(ld)) for ld in D.leads]
            if m in LAMBERT:
                lm = Lambert(ds, lat, lon, D_st_lon(D), D_st_lat(D))
                for c, fn in enumerate(fields):
                    v = np.ma.filled(ds[LAMBERT_VARS[fn]][:].astype(np.float32), np.nan)[jl]
                    v = v / 100.0 if fn == "msl" else v
                    grid[:, k, c] = lm.to_grid(v, ifs[:, c])
                    if c < 2:
                        native.setdefault(m, [None, None])[c] = v.reshape(L, -1)
                glon, glat = ds["lon"][:], ds["lat"][:]
                native[m].append(cKDTree(to_plane(glon.ravel(), glat.ravel())))
            else:
                glat, glon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
                ia = [int(np.argmin(abs(glat - x))) for x in lat]
                ja = [int(np.argmin(abs(glon - x))) for x in lon]
                assert np.allclose(glat[ia], lat) and np.allclose(glon[ja], lon), m
                for c, fn in enumerate(fields):
                    v = np.ma.filled(ds[fn][:, ia[0]:ia[-1] + 1, ja[0]:ja[-1] + 1].astype(np.float32), np.nan)[jl]
                    grid[:, k, c] = v / 100.0 if fn in ("msl", "sp") else v
                if m == "ifs":
                    ifs = grid[:, k].copy()
    assert np.isfinite(grid).all(), f"{group} {init}: non-finite input fields"
    dev = D.device
    uv = torch.tensor(grid[:, :, :2].reshape(L, -1, D.H, D.W), device=dev)
    out = {}
    for key, C in cells.items():
        w = F.grid_sample(uv, C.sample[None, None].expand(L, -1, -1, -1), align_corners=True,
                          padding_mode="border")[:, :, 0].permute(0, 2, 1).cpu().numpy().reshape(L, C.P, D.M, 2)
        for k, m in enumerate(GROUPS[group]):
            if m in native:
                u, v, tree = native[m]
                dist, idx = tree.query(C.xy)
                w[:, :, k, 0], w[:, :, k, 1] = u[:, idx], v[:, idx]
                w[:, dist > 2.0, k] = np.nan
        # MEPS and AROME-Arctic (both 2.5 km AROME): where one has no cell (AROME-Arctic ends near
        # 63 N), use the other, so that the nwp-* forecasts cover the whole map
        lam = [k for k, m in enumerate(GROUPS[group]) if m in native]
        if len(lam) == 2:
            for a_, b_ in (lam, lam[::-1]):
                miss = np.isnan(w[0, :, a_, 0])
                w[:, miss, a_] = w[:, miss, b_]
        out[key] = w
    return torch.tensor((grid.reshape(L, -1, D.H, D.W) - D.g_mean) / D.g_std, device=dev), out


def D_st_lon(D):
    return D._st_lon


def D_st_lat(D):
    return D._st_lat


@torch.no_grad()
def run_config(head, group, D, g, tf, cells, wind, chunk=200_000):
    """Ensemble mean correction (L, P, 2), spread of the speed (L, P) and base wind (L, P, 2) per cell set."""
    dev = D.device
    sfx = "_nwp" if group == "nwp" else ""
    models = []
    for s in range(3):
        for k in range(5):
            m = tso.TerrainNet(head, D).to(dev)
            m.load_state_dict(torch.load(tso.RUNS / f"{head}_terrain_g{k}_s{s}{sfx}_lr1e-05" / "best.pt", weights_only=True))
            models.append(m.eval())
    lat = [m.latent(g, tf) for m in models]
    res = {}
    L = D.L
    for key, C in cells.items():
        w = wind[key]                                                       # (L, P, M, 2)
        ok = np.isfinite(w).all((0, 2, 3))
        base = np.nanmean(w, 2)
        corr = np.full((L, C.P, 2), np.nan, np.float32)
        spread = np.full((L, C.P), np.nan, np.float32)
        idx = np.where(ok)[0]
        for a0 in range(0, len(idx), chunk):
            sel = idx[a0:a0 + chunk]
            ww, bb = w[:, sel], base[:, sel]
            u, v = bb[..., 0], bb[..., 1]
            sx = {p: s_[sel] for p, s_ in C.sx.items()}
            dyn = np.concatenate([ww.reshape(L, len(sel), -1), np.hypot(u, v)[..., None],
                                  np.stack([upwind(sx[p], u, v) for p in ("sx", "sx10")], -1)], -1)
            dyn = ((dyn - D.dyn_mu) / D.dyn_sd).astype(np.float32)
            pt = torch.tensor(np.concatenate([dyn, np.broadcast_to(C.static[sel], (L, len(sel), C.static.shape[1]))], -1),
                              device=dev)
            smp = C.sample[torch.as_tensor(sel, device=dev)]
            outs = np.stack([m.head_at(z, tf, pt, smp).float().cpu().numpy() * D.y_std for m, z in zip(models, lat)])
            corr[:, sel] = outs.mean(0)
            spread[:, sel] = np.linalg.norm(bb[None] + outs, axis=-1).std(0)
        res[key] = (base, corr, spread)
    return res


def publish(cfg, res, leads, run_dir, H1, W1, land_img, station_fc, configs):
    """Map images of one configuration and its station forecasts (into station_fc, configs)."""
    base, corr, spread = res["grid"]
    cdir = run_dir / cfg
    cdir.mkdir(exist_ok=True)
    for k, ld in enumerate(leads):
        b = base[k].reshape(H1, W1, 2)[::-1]
        c = corr[k].reshape(H1, W1, 2)[::-1]
        s1, d1 = speed_dir(b + c)
        s0, _ = speed_dir(b)
        ok = np.isfinite(s1)
        px = np.stack([np.clip(np.rint(np.nan_to_num(s1) / 0.2), 1, 255), np.clip(np.rint(np.nan_to_num(s0) / 0.2), 1, 255),
                       np.rint(np.nan_to_num(d1) / 360 * 255) % 256], -1).astype(np.uint8)
        px[~ok | ~land_img] = 0
        Image.fromarray(np.ascontiguousarray(px)).save(cdir / f"lead_{ld:02d}.webp", lossless=True, method=4)
    sb, sc, ss = res["st"]
    s1, d1 = speed_dir(sb + sc)
    s0, d0 = speed_dir(sb)
    station_fc[cfg] = dict(speed=s1.T.round(1), dir=d1.T.round(0), raw_speed=s0.T.round(1),
                           raw_dir=d0.T.round(0), spread=ss.T.round(1))
    group = CONFIGS[cfg]["group"]
    configs[cfg] = dict(label=CONFIGS[cfg]["label"], group=group, head=CONFIGS[cfg]["head"],
                        group_label=GROUP_LABEL[group],
                        # share of the land shown on the map that has a forecast
                        coverage=float(np.isfinite(corr[0, :, 0]).reshape(H1, W1)[::-1][land_img].mean().round(3)))


def write_static(out, t, names, H1, W1):
    """Background hillshade (land), elevation for the 3D view, and the grid geometry; rows north -> south."""
    st = out / "static"
    if (st / "grid.json").exists() and json.loads((st / "grid.json").read_text()).get("version") == STATIC_VERSION:
        return
    st.mkdir(parents=True, exist_ok=True)
    flip = lambda a: np.ascontiguousarray(a[::-1])
    land = flip(t["fields"][names.index("land_fraction")]) > 0
    land[-CROP_SOUTH:] = False
    z = flip(t["fields"][names.index("elevation")]).astype(np.float64)
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(gaussian_filter(z, 1.2), 1000.0)   # smoothed: the southern rows have duplicated columns
    az, alt = np.radians(315), np.radians(40)
    slope, aspect = np.arctan(np.hypot(gx, gy) * 2.0), np.arctan2(-gx, gy)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    g8 = (120 + 135 * shade).astype(np.uint8)
    Image.fromarray(np.dstack([g8, g8, g8, np.where(land, 255, 0).astype(np.uint8)])).save(st / "bg.webp", lossless=True, method=6)
    ze = np.clip(np.rint(np.where(land, z, 0)), 0, 65535).astype(np.uint16)  # m, R = high byte, G = low byte
    Image.fromarray(np.dstack([ze >> 8, ze & 255, land * 255]).astype(np.uint8)).save(st / "elevation.png", optimize=True)
    ii, jj = np.mgrid[0:H1, 0:W1]
    sub = (slice(None, None, 4), slice(None, None, 4))
    A = lambda a, b: np.stack([np.ones_like(a), a, b, a * a, a * b, b * b, a ** 3, a * a * b, a * b * b, b ** 3], -1)
    a, b = ii[sub].ravel() / H1, jj[sub].ravel() / W1
    cl = np.linalg.lstsq(A(a, b), flip(t["lat"])[sub].ravel().astype(np.float64), rcond=None)[0]
    co = np.linalg.lstsq(A(a, b), flip(t["lon"])[sub].ravel().astype(np.float64), rcond=None)[0]
    # inverse: (lat - 67) / 5, (lon - 20) / 10 -> i / H, j / W (place labels on the map)
    la, lo = (flip(t["lat"])[sub].ravel() - 67) / 5, (flip(t["lon"])[sub].ravel() - 20) / 10
    ci = np.linalg.lstsq(A(la, lo), a, rcond=None)[0]
    cj = np.linalg.lstsq(A(la, lo), b, rcond=None)[0]
    (st / "grid.json").write_text(json.dumps(dict(width=W1, height=H1, cell_km=1.0, lat_coef=cl.round(8).tolist(),
                                                  lon_coef=co.round(8).tolist(), i_coef=ci.round(8).tolist(), version=STATIC_VERSION,
                                                  j_coef=cj.round(8).tolist(), max_elevation=float(ze.max()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "web" / "site" / "data")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    init = pd.Timestamp(a.init)
    dev = torch.device("cuda")
    t = np.load(DESC, allow_pickle=True)
    names = [str(x) for x in t["names"]]
    H1, W1 = t["lat"].shape
    write_static(a.out, t, names, H1, W1)
    stt = np.load(ROOT / "cache" / "station_terrain.npz", allow_pickle=True)
    ij = stt["ij"]
    base_npz = np.load(f"{DATA0}.npz", allow_pickle=True)
    assert (base_npz["station"] == stt["station"]).all()
    run_dir = a.out / "runs" / f"{init:%Y%m%dT%H}"
    run_dir.mkdir(parents=True, exist_ok=True)
    station_fc, configs, leads = {}, {}, None
    land_img = t["fields"][names.index("land_fraction")][::-1] > 0   # image rows; no forecast shown over the sea
    land_img[-CROP_SOUTH:] = False

    for group in GROUPS:
        todo = [c for c in a.configs if CONFIGS[c]["group"] == group]
        if not todo:
            continue
        if not available(group, init):
            print(f"{group}: inputs missing for {init}, skipped", flush=True)
            continue
        t0 = time.time()
        D = data_for(group, dev)
        D._st_lon, D._st_lat = base_npz["lon_st"], base_npz["lat_st"]
        leads = [int(x) for x in D.leads]
        cells = {"grid": Cells(D, t["fields"].reshape(len(names), -1).T, names, t["lat"].reshape(-1),
                               t["lon"].reshape(-1), t["xy"].reshape(-1, 2)),
                 "st": Cells(D, t["fields"][:, ij[:, 0], ij[:, 1]].T, names, t["lat"][ij[:, 0], ij[:, 1]],
                             t["lon"][ij[:, 0], ij[:, 1]], t["xy"][ij[:, 0], ij[:, 1]])}
        cells["grid"].xy = t["xy"].reshape(-1, 2)
        cells["st"].xy = t["xy"][ij[:, 0], ij[:, 1]]
        g, wind = load_inputs(group, D, init, cells)
        vt = init + pd.to_timedelta(D.leads, unit="h")
        tf = torch.tensor(np.stack([D.leads / 66.0, np.sin(2 * np.pi * vt.hour / 24), np.cos(2 * np.pi * vt.hour / 24),
                                    np.sin(2 * np.pi * vt.dayofyear / 366), np.cos(2 * np.pi * vt.dayofyear / 366)], -1),
                          dtype=torch.float32, device=dev)
        for cfg in todo:
            res = run_config(CONFIGS[cfg]["head"], group, D, g, tf, cells, wind)
            publish(cfg, res, leads, run_dir, H1, W1, land_img, station_fc, configs)
            print(f"{cfg} done ({time.time() - t0:.0f} s)", flush=True)
        del D
        torch.cuda.empty_cache()

    if "dual-gno" in a.configs:
        if not dual.available(init):
            print("dual-gno: inputs missing for", init, flush=True)
        else:
            t0 = time.time()
            land = t["fields"][names.index("land_fraction")] > 0
            terr = t["fields"].reshape(len(names), -1).T.astype(np.float64)
            terr[~land.reshape(-1)] = np.nan                     # sea: no forecast shown, not computed
            cells = {"grid": (t["lat"].reshape(-1).astype(float), t["lon"].reshape(-1).astype(float),
                              t["xy"].reshape(-1, 2).astype(float), terr),
                     "st": (t["lat"][ij[:, 0], ij[:, 1]].astype(float), t["lon"][ij[:, 0], ij[:, 1]].astype(float),
                            t["xy"][ij[:, 0], ij[:, 1]].astype(float), t["fields"][:, ij[:, 0], ij[:, 1]].T.astype(np.float64))}
            res = dual.forecast(init, cells, dev)
            dl = [int(x) for x in np.load(dual.STATS)["leads"]]
            assert leads is None or dl == leads, (dl, leads)
            leads = dl
            publish("dual-gno", res, leads, run_dir, H1, W1, land_img, station_fc, configs)
            print(f"dual-gno done ({time.time() - t0:.0f} s)", flush=True)

    assert configs, f"no configuration could be run for {init}"
    meta = pd.read_parquet(ROOT / "stations" / "stations.parquet").drop_duplicates("station").set_index("station")
    stations = []
    for n, sid in enumerate(base_npz["station"]):
        sid = str(sid)
        m = meta.loc[sid]
        fc = {cfg: {k: [None if not np.isfinite(x) else round(float(x), 1) for x in v[n]] for k, v in d.items()}
              for cfg, d in station_fc.items()}
        stations.append(dict(id=sid, name=str(m["name"]).title(), municipality=str(m["municipality"]).title(),
                             county=str(m["county"]).title(), lat=round(float(base_npz["lat_st"][n]), 4),
                             lon=round(float(base_npz["lon_st"][n]), 4), masl=float(base_npz["masl"][n]),
                             i=int(H1 - 1 - ij[n, 0]), j=int(ij[n, 1]), fc=fc))
    stations.sort(key=lambda s: s["name"])
    old_f = run_dir / "forecast.json"
    if old_f.exists():                       # keep the configurations of an earlier call for this run
        old = json.loads(old_f.read_text())
        if old["leads"] == leads:
            prev = {s_["id"]: s_["fc"] for s_ in old["stations"]}
            for s_ in stations:
                s_["fc"] = {**prev.get(s_["id"], {}), **s_["fc"]}
            configs = {**old["configs"], **configs}
    out = dict(init=init.strftime("%Y-%m-%dT%H:00Z"), run=f"{init:%Y%m%dT%H}", leads=leads,
               valid=[(init + pd.Timedelta(hours=ld)).strftime("%Y-%m-%dT%H:00Z") for ld in leads],
               configs=configs, stations=stations,
               created=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"))
    (run_dir / "forecast.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print("->", run_dir)


if __name__ == "__main__":
    main()
