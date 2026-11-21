"""Station check for the MET Nordic section: the MET Nordic analysis and the station-corrected
forecasts against the observations at the stations inside the MET Nordic grid, on the common test
samples of make_results.py -> tables/metnordic_stations.json.

    /cluster/projects/nn8106k/siyan/envs/windcast-x86/bin/python docs/report_dual_gno/metnordic_stations.py
"""
import numpy as np, pandas as pd, os
R="/cluster/work/projects/nn8106k/siyan/windcast"; C=R+"/cache/multimodel_2024-03-01_2025-09-12"
G={}
for g,s in [("ecmwf",""),("lam","_lam"),("mix","_mix"),("nwp","_nwp")]:
    d=np.load(C+s+".npz",allow_pickle=True); G[g]=dict(base=d["st_fc"].mean(2),keep=d["keep"])
e=np.load(C+".npz",allow_pickle=True); inits=pd.DatetimeIndex(e["inits"]); ti=pd.Index(e["time"]).get_indexer(inits)
o=np.stack([e["obs"][ti+l] for l in e["leads"]],1)
mask=(inits>="2025-04-01")[:,None,None]&np.isfinite(o).all(-1)
for g in G: mask&=G[g]["keep"][:,None,None]
def pred(name):
    out=None
    for f in range(5):
        p=np.load(f"{R}/runs_a1/{name.format(f=f)}/pred.npz")["test"]
        out=p if out is None else np.where(np.isfinite(p),p,out)
    return out
rm=lambda f: np.sqrt(np.mean(np.sum((f-o)**2,-1)[mask]))
ref=rm(G["ecmwf"]["base"]); print(f"common test samples {mask.sum()}, IFS+AIFS mean {ref:.3f} m/s")
row=lambda lbl,v: print(f"{lbl:44s} "+"  ".join(f"{100*(x/ref-1):+5.1f}" for x in v)+f"   mean {100*(np.mean(v)/ref-1):+5.1f} %")
import xarray as xr, json
from scipy.spatial import cKDTree
import sys; sys.path.insert(0,"/cluster/home/siyan/github/windcast")
from windcast.geo import to_plane
s=xr.open_dataset(R+"/metnordic/static.nc"); mxy=to_plane(s.longitude.values.astype(float), s.latitude.values.astype(float))
st_xy=e["xy"].astype(float); dd,kk=cKDTree(mxy.reshape(-1,2)).query(st_xy); sin_=dd<1.0; sy,sx=np.unravel_index(kk[sin_],mxy.shape[:2])
N,L=len(inits),len(e["leads"]); MNst=np.full((N,L,99,2),np.nan,np.float32)
te=np.where(mask.any((1,2)))[0]
cache={}
for n in te:
    for l,ld in enumerate(e["leads"]):
        t=inits[n]+pd.Timedelta(hours=int(ld)); f=f"{R}/metnordic/{t:%Y}/mn_{t:%Y%m%dT%H}.nc"
        if t not in cache:
            try:
                with xr.open_dataset(f) as d: cache[t]=np.stack([d.u10.values[sy,sx],d.v10.values[sy,sx]],-1)
            except FileNotFoundError: cache[t]=None
        if cache[t] is not None: MNst[n,l][sin_]=cache[t]
m=mask&sin_[None,None]&np.isfinite(MNst).all(-1)
mix=G["mix"]["base"]
def predd(tag,s):
    out=None
    for f in range(5):
        p=np.load(f"{R}/runs_dual/dual_sfno-fno_g{f}_s{s}_{tag}_lr1e-05/pred.npz",allow_pickle=True)["test"]
        out=p if out is None else np.where(np.isfinite(p),p,out)
    return mix+out

r=lambda f: float(np.sqrt(np.mean(np.sum((f-o)**2,-1)[m])))
res={"n_samples":int(m.sum()),"n_stations":int(sin_.sum()),
 "metnordic":r(MNst),"ifs_aifs":r(G["ecmwf"]["base"]),"meps_arome":r(G["lam"]["base"]),"four_mean":r(mix),
 "mlp":float(np.mean([r(mix+pred(f"point_terrain_g{{f}}_s{s}_mix_lr1e-05")) for s in range(3)])),
 "sfno_mlp":float(np.mean([r(mix+pred(f"sfno_terrain_g{{f}}_s{s}_mix_lr1e-05")) for s in range(3)])),
 "dual_gno":float(np.mean([r(predd("rg60_rr10",s)) for s in range(3)])),
 "dual_small_mlp":float(np.mean([r(predd("rg60_rr10_r16m8x16_g16m8x16_z0_head-mlp",s)) for s in range(3)])),
 "dual_small_mlp_phys":float(np.mean([r(predd("rg60_rr10_phys-all_r16m8x16_g16m8x16_z0_head-mlp",s)) for s in range(3)]))}
# by lead: MN vs four mean vs best
res["lead_metnordic"]=[float(np.sqrt(np.mean(np.sum((MNst[:,l]-o[:,l])**2,-1)[m[:,l]]))) for l in range(L)]
print(json.dumps(res,indent=1))
json.dump(res,open("/cluster/work/projects/nn8106k/siyan/windcast/results/report_dual_gno/metnordic_stations.json","w"),indent=1)
