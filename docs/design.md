# windcast: design

## Goal

Predict near-surface weather for lead times 0–66 h, using only information available
when the forecast is issued. Targets are 10 m wind (u10, v10) together with the variables
coupled to it: t2m, rh2m, msl, total cloud. They are predicted jointly because stability,
thermal gradients and wind are coupled. Forecasts come from several drivers with
different resolutions and error sources: MEPS 2.5 km, IFS 0.4° / 0.25°, AIFS 0.25°.

The forecast is built in three steps:

1. Fuse the drivers adaptively, depending on lead time.
2. Correct the fused forecast with recent station obs-forecast.
3. Map it with neural operators onto a continuous field over complex terrain.

Several operator families are compared under one protocol.

Research questions (what is new compared with multi-NWP + station post-processing, e.g.
Baggio et al. 2025):

1. **Lead-time-dependent value of each source.** How do recent station observations,
   regional NWP (MEPS) and global physics / ML forecasts (IFS, AIFS) contribute from +0 to
   +66 h?
2. **Station information in complex terrain.** Fjords, ridges and valleys change how far
   a station's obs-forecast is useful. Does explicit terrain beat distance kernels, OI and
   station attention, repeatably and at unseen stations?
3. **Transfer across models and resolutions.** How much skill is lost when a source is
   missing, its resolution changes (IFS 0.4° → 0.25° in 2024-02) or the model changes
   (experimental AIFS → aifs-single in 2025-02)?

Two tracks share targets, splits and evaluation:

* **A, NWP downscaling / correction.** Driver: MEPS.
* **B, data-driven global forecasts.** Driver: AIFS from the operational open-data
  archive, with IFS as its physics counterpart. A pure regional time-stepper is not used.
  On the 1 km Tromsø–Senja grid (≈ 170 km wide), persistence of the MET Nordic field beats
  climatology only up to ≈ +18 h: vector RMSE 3.68 / 5.03 / 5.81 m/s at +6 / 12 / 18 h,
  against 5.95 for climatology (test 2023).

## Domains

* **Station domain: Northern Norway** (Nordland, Troms, Finnmark). There are 113 Frost
  stations with wind and temperature: 96 with wind in 2018, about 110 in 2024–2025. On the
  windcast.geo plane they span x −374 … 414 km and y −299 … 364 km. Every driver is
  downloaded over this box plus 50 km.
* **Grid domain: Tromsø–Senja at 1 km** (314 × 343 cells of the MET Nordic grid), inside
  the station domain. It is used for the gridded output and the auxiliary MET Nordic
  target.

## Data

All data are on the work disk, under `/cluster/work/projects/nn8106k/siyan/windcast`. The
work disk is purged periodically, so every download is resumable.

| | source | grid | period | inits / leads |
|---|---|---|---|---|
| MEPS | thredds `meps25epsarchive`: `meps_det_2_5km`; before 2020-02, member 0 of `meps_subset_2_5km` | 2.5 km Lambert, 332 × 378 window | 2018 – 2025 | 00 / 12; 0–12 h hourly + 18–66 h every 6 h |
| IFS | ECMWF open data (GCS mirror): `0p4-beta` | 0.4° | 2023-07-12 – 2024-01 | 00 / 12; 0–66 h every 6 h |
| IFS | `0p25`, then `ifs/0p25` | 0.25° | 2024-02 – 2025 | as above |
| AIFS | `aifs/0p25`, experimental; some early runs on N320, regridded linearly to 0.25° | 0.25° | 2024-03 – 2025-02 | as above |
| AIFS | `aifs-single/0p25` | 0.25° | 2025-02 – 2025 | as above |
| stations | Frost, hourly (copy of the mtno download) | points | 2016 – 2025 | |
| MET Nordic | v4 analysis, cut to the 1 km grid | 1 km | 2018 – 2025 | 6-hourly |
| static | terrain (16 fields) and MET Nordic altitude / land; MEPS orography | 1 km / 2.5 km | | |

**MEPS.** Fields: u10, v10, gust, t2m, rh2m, msl, tcc, accumulated precipitation, BLH.
u10 / v10 are grid-relative in the files and are rotated to east / north; checked against
`wind_speed` / `wind_direction` to 0.001 m/s. The two MEPS grids have different Lambert
parameters: lat_0 = 63° before 2020-02, 63.3° after.

**ECMWF.** Fields: 10u / 10v, 2t, msl, sp, tp, plus pressure-level u / v / t at 925, 850,
700 and 500 hPa, q at 925–700 hPa and z / gh at 500 hPa. Where a product has them, also
2d, tcc, 10fg and 100u / v; the 0.4° IFS has no 2d. Units differ between products (tp:
m vs kg m⁻²; tcc: % vs 0–1), so every field keeps its GRIB unit. AWS also holds IFS from
2023-01-18, but it throttles this site, so that period is not downloaded yet.

**Downloaders.** `download_meps.py`, `download_metnordic.py` and `download_ecmwf.py` run
on the login node, one stream per archive, because both thredds and S3 rate-limit.

**MET Nordic is not an independent truth.** Measured in Jan 2018, comparing MEPS
interpolated to 1 km with MET Nordic at the valid time:

* At +6 h (valid 06 / 18 UTC of the 00 / 12 UTC runs), msl differs by 0.007 Pa, tcc by
  0.000, wind by 0.57 m/s and t2m by 0.78 K.
* At 0 h and +12 h, msl differs by about 40 Pa.

So the analysis at T is essentially the MEPS run from T − 6 h at +6 h. msl and cloud are
that forecast unchanged, wind is that forecast downscaled, and only t2m / rh2m are
corrected with observations.

Consequences:

* **Stations are the labels and the evaluation.** MET Nordic is used only as an auxiliary
  gridded constraint in M4, never as truth.
* msl and tcc are inputs, not results.

## Availability rule

Every input must be available at the issue time:

* the run's own forecast (MEPS / IFS / AIFS come out about 2–4 h after their init time);
* station observations up to the init time t (conservative);
* past forecasts that were available by then.

Obs − forecast d_s(t − ℓ) = o_s(t − ℓ) − f_s(t − ℓ), ℓ = 0 … 12 h, use the previous run's
0–12 h forecast. Observations at the valid time t + τ never enter the inputs. Observation
windows must not cross the train / val / test boundaries.

## Lead times

Lead times are set by the drivers. Each source keeps its own lead table, and a fused
sample exists only at leads that every source in it provides.

| source | output leads in the archive used | max |
|---|---|---|
| MEPS det | hourly | +66 h |
| IFS open data | 3-hourly to +144, then 6-hourly | +360 h (00 / 12) |
| AIFS | 6-hourly (model step) | +360 h |

* Fusion leads: the intersection, **0–66 h every 6 h**.
* MEPS 0–12 h hourly is used only for obs-forecast.

## Splits

Time periods follow what the archives contain:

* MEPS + stations: 2018 – 2025. Pre-training of the MEPS encoder and correction.
* MEPS + IFS: 2023-07 – 2025.
* MEPS + IFS + AIFS: 2024-03 – 2025. AIFS changes from the experimental model to
  aifs-single in 2025-02.

Proposed split for the fusion experiments:

* train 2023-07 – 2024-12;
* val 2025-01 – 2025-03;
* test 2025-04 – 2025-12 (aifs-single, IFS 0.25°, MEPS).

This makes the AIFS model change part of the test, and it can be reported as such.
MEPS-only stages also report 2018–2021 / 2022 / 2023.

Space: spatially blocked station folds (blocks of neighbouring stations, not random
stations) plus temporal extrapolation. Every score is reported per lead.

## Stages

| stage | model | question |
|---|---|---|
| M0 | raw drivers at the station (bilinear / nearest / best neighbour), best single model, linear MOS, LightGBM on driver + terrain + lead features | strong baselines |
| M1 | source-specific encoders + lead-aware gating over the available sources | are the drivers complementary? |
| M2 | M1 + temporal encoder of the station obs-forecast (0–12 h) | do recent observations help, and up to which lead? |
| M3 | M2 + spatial propagation (GINO / gridded obs-forecast operators / GP, terrain-aware vs distance-only kernels; see Track A) | does it help at unseen stations? |
| M4 | native-grid encoders + neural operator + station correction | a continuous field, with MET Nordic as auxiliary constraint |

Labels for M0–M3 are station (u, v): Huber or vector MSE loss. Observations with speed
but no direction get a speed-only loss. t2m / rh2m are joint targets. Reported: vector
RMSE, speed RMSE / bias, direction MAE (speed > 2 m/s), and speed error for speed > 10 m/s.

mtno result to keep in mind: in analysis mode, propagating station information to unseen
stations gained at most 0.02–0.04 m/s (OI), and GBM was the bar. In forecast mode, error
persistence at short leads should make obs-forecast worth much more; that is the
hypothesis M2 / M3 test.

## Operator comparison (M4)

Operators are taken from an established library, `neuraloperator` 2.0 (`neuralop`,
already installed in `wfno-arm312`), not written by hand. Attention-based operators are
not used. Every operator gets the same inputs, base, loss, budget, a matched parameter
count (≈ 4–8 M) and 3 seeds:

| name | `neuralop` class | role |
|---|---|---|
| `unet` | — (own code) | convolutional reference (not an operator) |
| `fno` | `FNO` | Fourier neural operator, grid → grid |
| `tfno` | `TFNO` (Tucker-factorised FNO) | same at a lower parameter count; less overfitting on 4 years of data |
| `uno` | `UNO` | multi-scale U-shaped FNO; replaces the hand-written `ufno` |
| `localno` | `LocalNO` | FNO + local differential / integral kernels, for sharp terrain-induced gradients |
| `gino` | `GINO` | points → latent grid → points: station obs-forecast in, any query point out |
| `fnogno` | `FNOGNO` | grid in (MEPS + terrain), points out (stations / query points) |

`GINO` and `FNOGNO` are point-queryable and are scored directly at the stations. The
hand-written `fno` / `ufno` stay only as a check against the library versions.

GPs use `GPyTorch` (not yet installed in `wfno-arm312`).

## Track A: observation-driven correction of MEPS

Reference: Peduto et al. 2026, *Observation-Driven Correction of Numerical Weather
Prediction for Marine Winds*, JGR MLC, doi:10.1029/2025JH001174 (ORCA; code on Zenodo
10.5281/zenodo.17793900). We keep its formulation and change the model family: neural
operators and Gaussian processes instead of a transformer.

**What ORCA does** (full text, in [docs/](.), and Zenodo code):

* Formulation (Eq. 2): w(t + lΔt) = f(t + lΔt) + h_θ(w(t − bΔt : t), f(t − bΔt : t + lΔt)).
  The output is a residual on GFS (1°, nearest grid point, most recent available cycle).
  "Lead" here means target time − latest observation time, 1–48 h.
* Data: ICOADS over the North Atlantic, 2015-04 – 2024-09, about 35 M reports from ships,
  buoys, tide gauges and coastal stations. The split is chronological, 80 / 10 / 10.
  There is no spatial hold-out.
* Context: *all* observations from the hour before the target's issue time, each paired
  with GFS there, regardless of distance. Fig. A1 shows that temporal proximity dominates
  spatial distance: correlation drops from 0.92 to 0.50 as the window grows from 5 to
  71 h, but only from 0.92 to 0.84 out to 250 km. A padding mask handles the varying
  count.
* Query: GFS at the target, spherical-harmonic + SIREN location encoding, and sin / cos
  of the hour and day of year.
* Model: self-attention over the context (8 layers, 8 heads, d = 128), then
  cross-attention from the queries. One model per lead.
* Training: loss = mean vector error ‖ŵ − w‖₂ (not squared). AdamW at 1e-4 with cosine
  warm restarts, 100 epochs, patience 25, about 24 h per model on a V100.
* Results, mean vector error (m/s) vs GFS:

  | lead | ORCA | linear MOS | GFS | ERA5 |
  |---|---|---|---|---|
  | +1 h | 1.51 (−45 %) | 1.82 | 2.75 | 2.46 |
  | +8 h | 2.33 (−21 %) | 2.63 | 2.95 | |
  | +24 h | 2.62 (−17 %) | 2.88 | 3.15 | |
  | +48 h | 2.99 (−13 %) | 3.21 | 3.44 | |

  The linear MOS has 12 features, among them inverse-distance-weighted nearby obs-forecast.
  It already gets −34 % at +1 h. The transformer adds another 7–17 %. On a ship subset,
  ViT beats MLP, which beats LSTM.
* Ablations (Table A1): a second hour of observations, extra GFS variables, platform-type
  encoding and a residual GFS connection each change the error by < 0.01 m/s.
* Weaknesses the paper reports, which our design should address:
  1. **High winds.** For the top 5 % of speeds (≥ 12.9 m/s) the gain becomes negative
     after about 18 h, reaching −5 to −7 % at 36–48 h (Fig. A5).
  2. **No spatial generalisation test.** Gains at fixed platforms may be site-specific
     corrections learned on the same sites. Withholding regions is left as future work.
  3. **No locality.** Attention produces spurious corrections far from any observation
     (§6.3), and overfits isolated observations in sparse areas (Fig. A4).
  4. **No UQ.** Probabilistic or conformal output is left as future work.

**What we change.** Three parts of ORCA are what make it work: set input, conditioning
on obs-forecast, and point query. We build each one from an operator or a GP instead of
attention.

| ORCA component | replacement |
|---|---|
| irregular station set + mask | `GINO` input GNO: station obs-forecast (any number, any position) → latent grid. For the grid operators: obs-forecast gridded with a fixed kernel + density channel |
| self-attention over the context | the operator body (`FNO` / `TFNO` / `UNO` / `LocalNO`) on the latent grid. Inputs: MEPS at the valid time, terrain, obs-forecast + density channels for lags ℓ, lead / time features |
| cross-attention to queries | `GINO` / `FNOGNO` output GNO evaluated at the stations or any (x, y) point; for grid-only operators, bilinear sampling of the output grid |
| location embedding | LAEA (x, y) in km + 16 terrain fields; no spherical harmonics needed on a 800 km domain |
| one model per lead | one model for all leads, with lead as an input (FiLM on the operator layers). Per-lead models are kept as an ablation |
| — (no UQ) | GP (below) |

Gaussian processes, in two roles:

1. **GP correction (baseline, the classical counterpart of ORCA).** Exact GP regression of
   the station obs-forecast. The prediction is Δ(s, t + τ) given {d_s′(t − ℓ)}. The kernel
   is a product: k_space (Matérn on horizontal distance) × k_elev (elevation difference) ×
   k_terrain (ARD-RBF on a few terrain descriptors: land/sea, fjord / ridge indices) ×
   k_time. The time kernel covers the gap τ + ℓ, so obs-forecast memory decays with lead.
   Hyperparameters are fitted on train (GPyTorch, marginal likelihood) and conditioned on
   per lead if needed. With about 113 stations × 13 lags, an exact GP is cheap. This
   generalises the OI used in mtno (fixed distance kernel) and answers RQ2 directly:
   compare distance-only vs terrain-aware kernels.
2. **GP on operator residuals (UQ + residual spatial correlation).** The operator gives
   the mean; a GP with the same kind of kernel models what is left at the stations. The
   GP's predictive variance gives calibrated intervals (CRPS, spread–skill, PIT). A
   deep-kernel variant uses operator features as GP inputs.

Wind (u, v) is handled with an independent GP per component first. A coregionalised
(ICM) kernel with t2m / rh2m is a later step.

**Protocol.** These map onto stages M2–M4:

* M2: per-station obs-forecast encoder.
* M3: `GINO` / gridded obs-forecast operators, and the GP correction.
* M4: gridded operator + GP.

All models are trained and evaluated on stations. The loss is Huber on (Δu, Δv), as in
the gridded pipeline. **The metric is RMSE** (vector RMSE for wind, as everywhere else in
this project); ORCA's mean vector error is reported only as a secondary score, to place
our numbers next to its table. Splits, spatially blocked folds and per-lead scores are the
same as elsewhere.

Comparison with ORCA:

* Baselines must include ridge regression on the driver + IDW mean of neighbouring
  stations' obs-forecast (+ own obs-forecast when the station reports). The paper's equivalent
  linear baseline already takes most of the short-lead gain.
* The headline number is the per-lead vector RMSE reduction vs raw MEPS at *held-out*
  stations, plus the same score for speed ≥ p90 / p95.
* Lead must be read as the gap to the latest observation. Ours is init lead + MEPS latency
  (2–4 h), so our +6 h corresponds roughly to ORCA's +8–10 h.

What this adds over ORCA, from its weaknesses 1–4:

* blocked spatial folds (2);
* GP kernels and GNO neighbourhoods with explicit, terrain-aware locality, so corrections relax back
  to MEPS away from stations (3);
* a GP predictive distribution, scored with CRPS / PIT (4);
* a high-wind stratification at every lead (1).

Because ORCA found the last hour most informative and a second hour useless, the
obs-forecast window ℓ = 0 … 12 h is an ablation (ℓ = 0 only vs 0–3 h vs 0–12 h), not an
assumption.

## Code status

The gridded pipeline is implemented and smoke-tested (unet / fno / ufno): MEPS → 1 km
grid → MET Nordic target, per-lead scores against driver and persistence. The station
pipeline (M0–M3) is next.
