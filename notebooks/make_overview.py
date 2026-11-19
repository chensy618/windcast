"""Builds notebooks/00_project_overview.ipynb (markdown only, nothing to execute):
    python notebooks/make_overview.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))

md(r"""# Windcast: project overview

**Goal.** Forecast the 10 m wind on a 1 km grid over Northern Norway, 0–66 h ahead, by correcting
global and regional weather forecasts for the local terrain with neural networks, and publish it as a
real-time website.

**Pipeline.**

1. Collect forecasts, station observations and terrain (section 1).
2. Learn a correction of the forecast wind at stations (sections 2–4).
3. Test it at stations the model never saw (section 5).
4. Apply it at every 1 km cell, every hour, in real time (sections 6–7).

Results of every model: `a4_model_summary.ipynb`. Earlier experiments: `a1`–`a3`.""")

md(r"""## 1. Data

| | source | resolution | use |
|---|---|---|---|
| IFS, AIFS | ECMWF open data | 0.25°, 6 h | input forecasts (ECMWF group) |
| GFS | NOAA (AWS) | 0.25°, 6 h | input forecast (NWP group) |
| MEPS, AROME-Arctic | MET Norway thredds | 2.5 km, 1 h | input forecasts (NWP group) |
| 10 m wind at 99 stations | MET Norway Frost | hourly | target |
| terrain | Kartverket DTM (50 m) → 1 km | 1 km | static descriptors |

* **Domain:** the 0.25° grid 62–74 N, 6–36 E ($49 \times 121$ cells); the 1 km grid covers
  62.6–72.4 N ($1033 \times 964$ cells, Lambert azimuthal equal-area plane).
* **Runs:** initialised at 00 and 12 UTC, lead times $\ell = 0, 6, \dots, 66$ h.
* **Fields:** per model $10u, 10v, 2t, \mathrm{msl}$ (ECMWF group also $\mathrm{sp}, t_{850}$).
* **Split by time:**

| split | period |
|---|---|
| train | 2024-03-01 – 2024-12-31 |
| validation (picks the best epoch) | 2025-01-01 – 2025-03-31 |
| test | 2025-04-01 – 2025-09-12 |

**Input groups.**
* *ECMWF:* IFS + AIFS.
* *NWP:* IFS + GFS + MEPS + AROME-Arctic.

On the 0.25° grid, a 2.5 km model is the mean of its cells inside each 0.25° cell, and IFS where it
has none. At a point (station or 1 km cell) a 2.5 km model gives the value of its nearest cell.""")

md(r"""## 2. What is predicted

For station $s$, valid time $t$ and the $M$ models of a group with forecast wind
$\mathbf f_{m,s}(t) = (u, v)$, the **base forecast** is the model mean
$$\mathbf b_s(t) = \frac{1}{M} \sum_{m=1}^{M} \mathbf f_{m,s}(t).$$
The network learns the **correction** to the observed wind $\mathbf o_s(t)$,
$$\mathbf y_s(t) = \mathbf o_s(t) - \mathbf b_s(t), \qquad \hat{\mathbf o}_s(t) = \mathbf b_s(t) + \hat{\mathbf y}_s(t).$$
Predicting the correction rather than the wind keeps the model close to the forecasts where it has
nothing to add.""")

md(r"""## 3. Model: neural operator + MLP head

$$\hat{\mathbf y}_s = \sigma_y \, h_\psi\!\big(\, \mathbf z(\mathbf x_s),\; \mathbf p_s \,\big), \qquad
\mathbf Z = \mathcal G_\theta(\mathbf X, \boldsymbol\tau).$$

**Operator $\mathcal G_\theta$ (weather pattern).**
* **Input $\mathbf X$:** the forecast fields of all models on the 0.25° grid, each standardised with
  its training mean and standard deviation.
* **Time features $\boldsymbol\tau$:**
  $$\boldsymbol\tau = \Big(\tfrac{\ell}{66},\; \sin\tfrac{2\pi h}{24},\; \cos\tfrac{2\pi h}{24},\;
  \sin\tfrac{2\pi d}{366},\; \cos\tfrac{2\pi d}{366}\Big),$$
  with lead $\ell$, hour of day $h$ and day of year $d$ of the valid time.
* **Output:** $\mathbf Z$ has 32 latent channels on the same grid. It is read at the station
  position $\mathbf x_s$ by bilinear interpolation, which gives $\mathbf z(\mathbf x_s)$.

**Point features $\mathbf p_s$ (local terrain and flow).**
* The forecast wind of every model at the point, and the base wind speed $|\mathbf b_s|$.
* 28 terrain descriptors of the point's 1 km cell, standardised over the stations: elevation,
  slope, aspect, topographic position (elevation minus its 1 / 5 km mean), roughness, land fraction
  around the point, height relative to the 10 / 30 km mean, and shelter angles.
* **Shelter angle $\mathrm{sx}_k$:** the steepest terrain rise within 1 km (and within 10 km) toward
  each of the 8 directions $k$.
* **Upwind shelter:** the shelter angle toward the direction $\phi$ the wind blows from,
  interpolated between the two neighbouring sectors:
  $$\phi = \operatorname{atan2}(-u, -v), \quad i = \Big\lfloor \tfrac{\phi}{\pi/4} \Big\rfloor, \quad
  f = \tfrac{\phi}{\pi/4} - i, \quad \mathrm{sx}^{\uparrow} = (1-f)\,\mathrm{sx}_i + f\,\mathrm{sx}_{i+1}.$$

**Head $h_\psi$.**
* An MLP, $n \to 128 \to 128 \to 2$ with GELU activations.
* $\sigma_y$ is the training standard deviation of $\mathbf y$.
* Every input exists at every 1 km cell, not only at stations, so the same model can be applied on
  the whole grid.

**Operators compared.** Each layer applies a spectral convolution $\mathcal K$ plus a pointwise
linear term, followed by a nonlinearity.

| | spectral weights | parameters (width 64, modes 16 × 32, 4 layers) |
|---|---|---|
| FNO | per Fourier mode: $\widehat{(\mathcal K v)}_{\mathbf k} = W_{\mathbf k}\, \hat v_{\mathbf k}$ | 4.5 M |
| SFNO | per spherical-harmonic degree only: $\widehat{(\mathcal K v)}_{l m} = W_{l}\, \hat v_{l m}$ | 0.6 M |
| TFNO | FNO with Tucker-factorised $W_{\mathbf k}$ | 2.0 M |
| UNO | FNO layers in a U-shape (down- / up-sampling, skips) | 8.9 M |
| MLP only | no operator: $\hat{\mathbf y}_s = \sigma_y\, h_\psi(\boldsymbol\tau, \mathbf p_s)$ | 0.02 M |

SFNO shares one weight matrix over all orders $m$ of a degree $l$. It can only learn isotropic,
smooth kernels, which acts as a strong regulariser for the small training set.""")

md(r"""## 4. Training

The loss is the masked mean squared error of the standardised correction, over the stations $s$
that may be used for training and have an observation ($w_s = 1$):
$$\mathcal L(\theta, \psi) = \frac{\sum_{t,s} w_s(t)\, \big\| \hat{\mathbf y}_s(t) - \mathbf y_s(t) \big\|^2 / \sigma_y^2}{\sum_{t,s} w_s(t)}.$$

* **Optimiser:** AdamW (weight decay $10^{-4}$) with a one-cycle learning-rate schedule, peak
  $10^{-5}$.
* **Training length:** 40 epochs, batches of 16 samples (run, lead), gradient norm clipped at 1.
* **Model selection:** the weights of the epoch with the lowest validation loss are kept.
* **Seeds:** each configuration is trained with 3 seeds.""")

md(r"""## 5. Evaluation

**Spatial cross-validation.** The stations are split into $K = 5$ groups by $k$-means on their
positions, so that neighbouring stations fall in the same group. For each group $g$, a model is
trained on the other 4 groups and predicts group $g$. Every station is therefore scored by a model
that never saw it or its neighbours.

**Metric.** Vector RMSE over the test runs $t$ and stations $s$, per lead $\ell$:
$$\mathrm{RMSE}_\ell = \sqrt{\frac{1}{N_\ell} \sum_{t,s} \big\| \hat{\mathbf o}_s(t) - \mathbf o_s(t) \big\|^2},$$
reported as a change against the IFS + AIFS mean (negative = better):
$$\Delta_\ell = 100 \left( \frac{\mathrm{RMSE}_\ell}{\mathrm{RMSE}_\ell^{\,\mathrm{IFS+AIFS}}} - 1 \right) \%.$$

**Main results** (mean over leads, 3 seeds):

| inputs | model | $\Delta$ |
|---|---|---|
| ECMWF | SFNO + MLP | −4.4 % |
| ECMWF | MLP only | −3.3 % |
| ECMWF | FNO + MLP | −3.0 % |
| NWP | four-model mean, uncorrected | −6.4 % |
| NWP | SFNO + MLP | −9.0 % |
| NWP | **MLP only** | **−9.7 %** |""")

md(r"""## 6. From stations to the 1 km grid

**Ensemble.** A configuration is the mean of its $15$ models ($5$ groups $\times$ $3$ seeds):
$$\hat{\mathbf o}(\mathbf x, t) = \mathbf b(\mathbf x, t) + \frac{1}{15}\sum_{j=1}^{15} \hat{\mathbf y}^{(j)}(\mathbf x, t).$$
The spread shown on the website is the standard deviation of the 15 wind speeds
$\big|\mathbf b + \hat{\mathbf y}^{(j)}\big|$.

**Every 1 km cell $\mathbf x$.** The point features are built at the cell, exactly as at a station:
* global models: bilinear interpolation from the 0.25° grid;
* MEPS / AROME-Arctic: their nearest 2.5 km cell (AROME-Arctic ends near 63 N; south of it, MEPS
  stands in for it);
* the cell's 28 terrain descriptors.

The open sea is masked out: there are no stations to verify it.

**Hourly values.** The models give leads every 6 h. In between, $u$ and $v$ are interpolated
linearly in time:
$$\hat{\mathbf o}(t) = (1-w)\,\hat{\mathbf o}(t_k) + w\,\hat{\mathbf o}(t_{k+1}), \qquad
w = \frac{t - t_k}{t_{k+1} - t_k}.$$
On the test set this keeps almost all of the improvement at the hours between the leads (−9.4 % vs
−9.5 % for the NWP MLP).""")

md(r"""## 7. Real-time system

The cycle runs on a GPU node twice a day (`web/slurm/scrontab.txt`) and takes about 6 minutes:

| step | what | code |
|---|---|---|
| 1 | newest 00 / 12 UTC run at least 7 h old | `web/pipeline/run_latest.py` |
| 2 | download IFS, AIFS, GFS, MEPS, AROME-Arctic | `scripts/download_*.py` |
| 3 | six configurations on the 1 km grid | `web/pipeline/forecast.py` |
| 4 | station observations (also hourly) | `web/pipeline/observations.py` |
| 5 | keep the newest 4 runs, write the index | `web/pipeline/run_latest.py` |

The website (`web/site/`) is static: a table, a graph and a playable 2D / 3D map, with a choice of
model. Run it with `cd web && npm start`. See `web/README.md`.""")

md(r"""## 8. Code map

| file | role |
|---|---|
| `scripts/build_multimodel_samples.py`, `build_group_samples.py` | training samples (ECMWF, NWP groups) |
| `scripts/build_terrain_descriptors.py` | 28 terrain descriptors on the 1 km grid |
| `windcast/training/train_station_operator.py` | models, training, held-out prediction |
| `windcast/station_correction.py` | station groups, upwind shelter |
| `web/pipeline/verification.py` | held-out scores shown on the website |
| `notebooks/a1`–`a4` | experiments and the model summary |""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/00_project_overview.ipynb")
print("wrote notebooks/00_project_overview.ipynb")
