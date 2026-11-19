"""Station-level observation-driven correction of MEPS wind (track A, small experiments).

Notation, per MEPS run at init t and lead tau:
    f_t(tau)       MEPS (u, v) at the station
    y_t(tau)       o(t + tau) - f_t(tau), the residual to predict
    d(t - l)       obs-forecast o(t - l) - f_{t-12}(12 - l): the previous run's 0-12 h
                   forecast, so only information available at issue time is used.

Spatial GP on the obs-forecast at init time (shared by u and v, independent components):
    k(s, s') = sf2 * Matern52(|x_s - x_s'| / ell) [* exp(-dz^2 / (2 ell_z^2))] + sn2 * I
Its posterior mean m_t(s) / variance v_t(s) at every station is then carried to lead tau by
    y_t(tau) ~ rho_tau * m_t(s) + eps,  Var = rho_tau^2 v_t(s) + e_tau
with rho_tau, e_tau fitted per lead on train. "including_target" conditions on every station with an
observation at t, the target station included; "excluding_target" leaves the target station out
(exact leave-one-out from one inverse per time).
"""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    tindex = pd.Index(d["time"])
    inits, leads = d["inits"], d["leads"]
    N, L, S = len(inits), len(leads), len(d["station"])
    obs = d["obs"]
    ti = tindex.get_indexer(inits)
    # targets: residual at the valid time
    y = np.full((N, L, S, 2), np.nan, np.float32)
    o_valid = np.full((N, L, S, 2), np.nan, np.float32)
    for j, ld in enumerate(leads):
        o_valid[:, j] = obs[ti + ld]
    y = o_valid - d["fc"]
    # obs-forecast d(t - l), l = 0..12, from the previous run (t - 12 h)
    prev = pd.Index(inits).get_indexer(inits - np.timedelta64(12, "h"))
    omf = np.full((N, 13, S, 2), np.nan, np.float32)
    ok = prev >= 0
    for l in range(13):
        j = list(leads).index(12 - l)
        omf[ok, l] = obs[ti[ok] - l] - d["fc"][prev[ok], j]
    d.update(y=y, o_valid=o_valid, omf=omf)
    return d


def matern52(r):
    a = np.sqrt(5.0) * r
    return (1 + a + a * a / 3) * np.exp(-a)


class SpatialGP:
    """theta = log(sf2, ell_km, sn2[, ell_z_m])."""

    def __init__(self, xy, z, use_z=True):
        self.r = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        self.dz = np.abs(z[:, None] - z[None])
        self.use_z = use_z
        self.theta = np.log([4.0, 50.0, 2.0] + ([300.0] if use_z else []))

    def kern(self, theta=None):
        p = np.exp(self.theta if theta is None else theta)
        k = p[0] * matern52(self.r / p[1])
        if self.use_z:
            k = k * np.exp(-0.5 * (self.dz / p[3]) ** 2)
        return k, p[2]

    def nll(self, theta, D):
        """D: list of obs-forecast vectors (one per time and component) with their masks."""
        k, sn2 = self.kern(theta)
        tot, n = 0.0, 0
        for a, v in D:
            K = k[np.ix_(a, a)] + sn2 * np.eye(a.sum())
            c, low = cho_factor(K)
            al = cho_solve((c, low), v)
            tot += 0.5 * v @ al + np.log(np.diag(c)).sum()
            n += len(v)
        return tot / n

    def fit(self, omf0):
        """omf0 (T, S, 2): obs-forecast at init time on train."""
        D = []
        for t in range(omf0.shape[0]):
            a = np.isfinite(omf0[t, :, 0])
            if a.sum() >= 5:
                D += [(a, omf0[t, a, 0].astype(float)), (a, omf0[t, a, 1].astype(float))]
        res = minimize(self.nll, self.theta, args=(D,), method="L-BFGS-B")
        self.theta = res.x
        return dict(zip(["sf2", "ell_km", "sn2", "ell_z_m"][:len(res.x)], np.exp(res.x))), res.fun

    def posterior(self, omf0):
        """omf0 (T, S, 2) -> mean / var of the latent obs-forecast field at every station,
        'including_target' (all reporting stations) and 'excluding_target' (target station left out)."""
        k, sn2 = self.kern()
        sf2 = k[0, 0]
        T, S, _ = omf0.shape
        out = {s: (np.full((T, S, 2), np.nan), np.full((T, S), np.nan)) for s in ("including_target", "excluding_target")}
        for t in range(T):
            a = np.isfinite(omf0[t, :, 0])
            if a.sum() < 3:
                continue
            ia = np.where(a)[0]
            K = k[np.ix_(a, a)] + sn2 * np.eye(len(ia))
            Ci = np.linalg.inv(K)
            Y = omf0[t, a].astype(float)                       # (A, 2)
            al = Ci @ Y
            ks = k[:, a]                                          # (S, A)
            m = ks @ al
            v = sf2 - np.einsum("sa,ab,sb->s", ks, Ci, ks)
            out["including_target"][0][t], out["including_target"][1][t] = m, v
            # excluding_target: stations without obs are unchanged; stations in A by exact LOO
            mu, vu = m.copy(), v.copy()
            cd = np.diag(Ci)
            mu[ia] = Y - al / cd[:, None]                         # LOO mean of y_s = of f_s
            vu[ia] = 1.0 / cd - sn2
            out["excluding_target"][0][t], out["excluding_target"][1][t] = mu, vu
        return out


def station_groups(xy, k=5, seed=0, iters=50):
    """Spatially blocked groups by k-means on the station positions (numpy only)."""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(10):
        c = xy[rng.choice(len(xy), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((xy[:, None] - c[None]) ** 2).sum(-1), 1)
            c = np.stack([xy[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        inertia = ((xy - c[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best



def idw(xy, omf0, power=2.0, exclude_self=True, eps_km=1.0):
    """Inverse-distance-weighted mean of other stations' obs-forecast (ridge regression feature)."""
    r = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + eps_km
    w = r ** -power
    if exclude_self:
        np.fill_diagonal(w, 0.0)
    a = np.isfinite(omf0[..., 0])                               # (T, S)
    v = np.nan_to_num(omf0)
    num = np.einsum("sr,trc->tsc", w, v * a[..., None])
    den = np.einsum("sr,tr->ts", w, a.astype(float))
    return num / np.where(den > 0, den, np.nan)[..., None]


def vec_rmse(err):
    """err (..., 2) with NaNs -> vector RMSE over valid entries."""
    e2 = (err ** 2).sum(-1)
    return float(np.sqrt(np.nanmean(e2)))


def crps_gauss(mu, sd, y):
    from scipy.stats import norm
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


SX_DIRS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")   # order of the sx_* / sx10_* descriptors


def upwind(sx, u, v):
    """Shelter angle toward the direction the wind blows from, interpolated between the 8
    sectors. sx (S, 8) in SX_DIRS order; u, v (..., S) -> (..., S)."""
    pos = (np.arctan2(-u, -v) % (2 * np.pi)) / (np.pi / 4)
    i0 = np.floor(pos).astype(int) % 8
    f = pos - np.floor(pos)
    s_ = np.arange(sx.shape[0])
    return (1 - f) * sx[s_, i0] + f * sx[s_, (i0 + 1) % 8]
