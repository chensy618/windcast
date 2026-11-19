/* Windcast Nord: static front end. Reads data/ written by web/pipeline (see web/README.md).
   The models give the wind every 6 h; the page interpolates it to every hour (u and v linearly in
   time), which keeps the held-out skill (web/README.md). */
(() => {
"use strict";
const $ = id => document.getElementById(id);
const DATA = "data/";
const VIEWS = ["table", "graph", "map"];
const tz = "Europe/Oslo";
const fmtDay = new Intl.DateTimeFormat("en-GB", { timeZone: tz, weekday: "long", day: "numeric", month: "long" });
const fmtHour = new Intl.DateTimeFormat("en-GB", { timeZone: tz, hour: "2-digit", minute: "2-digit" });
const fmtShort = new Intl.DateTimeFormat("en-GB", { timeZone: tz, weekday: "short", hour: "2-digit", minute: "2-digit" });
const fmtDayShort = new Intl.DateTimeFormat("en-GB", { timeZone: tz, weekday: "short", day: "numeric" });
const fmtRun = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const dayKey = d => new Intl.DateTimeFormat("en-CA", { timeZone: tz }).format(d);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
};
const getJSON = async p => { const r = await fetch(DATA + p, { cache: "no-cache" }); if (!r.ok) throw new Error(p + ": " + r.status); return r.json(); };
const fmt = v => (v == null || !isFinite(v)) ? "–" : v.toFixed(1);
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const D2R = Math.PI / 180;

// model order on the page: the best verified first within each input group
const ORDER = ["nwp-mlp", "nwp-sfno", "nwp-fno", "ecmwf-sfno", "ecmwf-mlp", "ecmwf-fno"];
const CFG_COLORS = { "nwp-mlp": "#0b6aa2", "nwp-sfno": "#2a9d8f", "nwp-fno": "#7cb518", "ecmwf-sfno": "#c0392b", "ecmwf-mlp": "#e67e22", "ecmwf-fno": "#8e44ad" };
const PLACES = [["Tromsø", 69.649, 18.956], ["Bodø", 67.280, 14.405], ["Narvik", 68.438, 17.427], ["Harstad", 68.799, 16.541],
  ["Alta", 69.969, 23.272], ["Hammerfest", 70.663, 23.682], ["Vadsø", 70.074, 29.749], ["Kirkenes", 69.727, 30.045],
  ["Mo i Rana", 66.313, 14.142], ["Svolvær", 68.234, 14.568], ["Finnsnes", 69.230, 17.981], ["Mosjøen", 65.837, 13.190],
  ["Kiruna", 67.856, 20.225], ["Luleå", 65.584, 22.155], ["Rovaniemi", 66.503, 25.729], ["Ivalo", 68.658, 27.540],
  ["Karasjok", 69.472, 25.511], ["Andenes", 69.316, 16.119], ["Sortland", 68.697, 15.414], ["Honningsvåg", 70.982, 25.970]];

const S = { grid: null, verif: null, index: null, F: null, obs: {}, hours: [], times: [], cfg: null, pix: {}, loc: null,
            view: "table", dim: "2d", hour: 0, playing: null };

// arrow pointing where the wind blows to (direction "from" + 180)
function arrowSvg(dir, cls = "arrow", color = "currentColor") {
  if (dir == null || !isFinite(dir)) return "";
  return `<svg class="${cls}" viewBox="-10 -10 20 20" role="img" aria-label="Wind from ${Math.round(dir)}°"><g transform="rotate(${dir + 180})"><path d="M0 -8 L5 4 L0 1.5 L-5 4 Z" fill="${color}"/></g></svg>`;
}

// ---------- time: model leads (6-hourly) -> hours ----------
function bracket(h) {                                   // indices k, k+1 of the model leads around hour h, weight
  const L = S.F.leads;
  let k = 0; while (k < L.length - 2 && L[k + 1] <= h) k++;
  return [k, Math.min(1, Math.max(0, (h - L[k]) / (L[k + 1] - L[k])))];
}
const isStep = h => S.F.leads.includes(h);
function hourly(s) {                                    // series at the model leads -> every hour
  if (!s) return s;
  const out = { speed: [], dir: [], spread: s.spread ? [] : null };
  for (const h of S.hours) {
    const [k, w] = bracket(h);
    const uv = i => { const sp = s.speed[i], d = s.dir[i] * D2R; return [-sp * Math.sin(d), -sp * Math.cos(d)]; };
    const a = uv(k), b = uv(k + 1), u = a[0] + (b[0] - a[0]) * w, v = a[1] + (b[1] - a[1]) * w;
    out.speed.push(Math.hypot(u, v));
    out.dir.push((Math.atan2(-u, -v) / D2R + 360) % 360);
    if (s.spread) out.spread.push(s.spread[k] + (s.spread[k + 1] - s.spread[k]) * w);
  }
  return out;
}

// ---------- geometry ----------
const POLY = (a, b) => [1, a, b, a * a, a * b, b * b, a * a * a, a * a * b, a * b * b, b * b * b];
const dot = (A, c) => A.reduce((s, v, k) => s + v * c[k], 0);
function latlon(i, j) { const A = POLY(i / S.grid.height, j / S.grid.width); return [dot(A, S.grid.lat_coef), dot(A, S.grid.lon_coef)]; }
function pixelOf(lat, lon) { const A = POLY((lat - 67) / 5, (lon - 20) / 10); return [dot(A, S.grid.i_coef) * S.grid.height, dot(A, S.grid.j_coef) * S.grid.width]; }
function km(a, b) {
  const dl = (b[0] - a[0]) * D2R, dn = (b[1] - a[1]) * D2R;
  const h = Math.sin(dl / 2) ** 2 + Math.cos(a[0] * D2R) * Math.cos(b[0] * D2R) * Math.sin(dn / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

// ---------- images of a configuration ----------
async function loadImg(src) { const im = new Image(); im.src = src; await im.decode(); return im; }
function readPixels(im) {
  const c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight;
  const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0);
  return x.getImageData(0, 0, c.width, c.height).data;
}
function loadConfig(cfg) {
  if (S.pix[cfg]) return S.pix[cfg].ready;
  const arr = new Array(S.F.leads.length).fill(null);
  const ready = Promise.all(S.F.leads.map(async (ld, k) => {
    const im = await loadImg(`${DATA}runs/${S.F.run}/${cfg}/lead_${String(ld).padStart(2, "0")}.webp`);
    arr[k] = readPixels(im);
  }));
  S.pix[cfg] = { arr, ready };
  return ready;
}
function cellSeries(cfg, i, j) {                        // at the model leads
  const p = S.pix[cfg] && S.pix[cfg].arr, k = (i * S.grid.width + j) * 4;
  if (!p || p.some(x => !x)) return undefined;         // still loading
  if (p[0][k] === 0) return null;                       // no forecast here
  return { speed: p.map(a => a[k] * 0.2), dir: p.map(a => a[k + 2] / 255 * 360), spread: null };
}

// ---------- location ----------
function stationById(id) { return S.F.stations.find(s => s.id === id); }
function setStation(st, remember = true) {
  S.loc = { kind: "station", id: st.id, i: st.i, j: st.j, name: st.name, far: 0,
            sub: `${st.municipality}, ${st.county} · weather station, ${Math.round(st.masl)} m above sea level` };
  if (remember) store.set("wc-loc", st.id);
  renderAll();
}
function setCell(i, j) {
  const ll = latlon(i, j);
  let best = null, bd = 1e9;
  for (const st of S.F.stations) { const d = km(ll, [st.lat, st.lon]); if (d < bd) { bd = d; best = st; } }
  S.loc = { kind: "cell", i, j, far: bd, name: `${ll[0].toFixed(3)}° N, ${ll[1].toFixed(3)}° E`,
            sub: `1 km cell · nearest station ${best.name}, ${bd.toFixed(0)} km away` };
  renderAll();
}
function series(cfg = S.cfg) {
  const raw = S.loc.kind === "station" ? (stationById(S.loc.id).fc[cfg] || null) : cellSeries(cfg, S.loc.i, S.loc.j);
  return raw ? hourly(raw) : raw;
}
function obsFor() { return S.loc.kind === "station" ? (S.obs[S.loc.id] || []) : []; }
function obsAt(t) { const key = t.toISOString().slice(0, 13); return obsFor().find(o => o[0] === key) || null; }

// ---------- rendering ----------
function renderAll() {
  if (!S.loc) return;
  $("loc-name").textContent = S.loc.name; $("loc-sub").textContent = S.loc.sub;
  const warn = $("loc-warn");
  warn.hidden = !(S.loc.far > 50);
  warn.textContent = "This place is far from every station the models learned from, so the correction is less certain here.";
  renderNow(); renderTable();
  if (S.view === "graph") renderGraph();
  if (S.view === "map") drawMarks();
}
function nowIndex() {
  const k = S.times.findIndex(t => t.getTime() > Date.now() - 1800e3);
  return k < 0 ? S.times.length - 1 : k;
}
function renderNow() {
  const s = series(), el = $("now");
  if (!s) { el.innerHTML = ""; return; }
  const k = nowIndex(), mx = Math.max(...s.speed), kmx = s.speed.indexOf(mx);
  const isNow = Math.abs(S.times[k] - Date.now()) < 3600e3;
  el.innerHTML = `
    <div><span class="k">${isNow ? "Now" : "Forecast for"} · ${fmtShort.format(S.times[k])}</span><span class="v">${arrowSvg(s.dir[k], "arrow", css("--accent"))}${fmt(s.speed[k])} m/s</span></div>
    <div><span class="k">Strongest until ${fmtShort.format(S.times[S.times.length - 1])}</span><span class="v">${fmt(mx)} m/s <small>${fmtShort.format(S.times[kmx])}</small></span></div>`;
}
function renderTable() {
  const s = series(), msg = $("tbl-msg");
  if (s === undefined) { $("tbl").innerHTML = ""; msg.hidden = false; msg.textContent = "Loading the forecast for this place…"; return; }
  if (!s) { $("tbl").innerHTML = ""; msg.hidden = false; msg.textContent = "This model has no forecast here (outside the MEPS area). Choose an ECMWF model or another place."; return; }
  msg.hidden = true;
  const hasObs = S.loc.kind === "station", now = Date.now(), k0 = Math.max(0, nowIndex() - 6);
  let h = `<thead><tr><th>Time</th><th class="c">Wind</th><th class="num">Speed</th>${s.spread ? '<th class="num">Spread</th>' : ""}${hasObs ? '<th class="num">Observed</th>' : ""}</tr></thead><tbody>`;
  const cols = 3 + (s.spread ? 1 : 0) + (hasObs ? 1 : 0);
  let day = "";
  for (let k = k0; k < S.times.length; k++) {
    const t = S.times[k];
    if (dayKey(t) !== day) { day = dayKey(t); h += `<tr class="day"><td colspan="${cols}">${fmtDay.format(t)}</td></tr>`; }
    const o = hasObs ? obsAt(t) : null;
    h += `<tr class="${t.getTime() < now - 1800e3 ? "past" : ""}${isStep(S.hours[k]) ? " step" : ""}"><td>${fmtHour.format(t)}</td>
      <td class="c">${arrowSvg(s.dir[k])}</td>
      <td class="num"><span class="spd">${fmt(s.speed[k])}</span> m/s</td>
      ${s.spread ? `<td class="num muted">±${fmt(s.spread[k])}</td>` : ""}
      ${hasObs ? `<td class="num">${o ? `${fmt(o[1])}${arrowSvg(o[2], "arrow sm")}` : '<span class="muted">–</span>'}</td>` : ""}</tr>`;
  }
  $("tbl").innerHTML = h + "</tbody>";
}
function renderGraph() {
  const s = series(), box = $("chart");
  const all = $("all-models").checked && S.loc.kind === "station";
  $("all-wrap").hidden = S.loc.kind !== "station";
  if (!s) { box.innerHTML = `<p class="msg">${s === undefined ? "Loading…" : "This model has no forecast here."}</p>`; $("legend").innerHTML = ""; return; }
  const W = Math.max(300, box.clientWidth), H = 300, m = { l: 40, r: 12, t: 14, b: 64 };
  const t0 = S.times[0].getTime(), t1 = S.times[S.times.length - 1].getTime();
  const obs = obsFor().map(o => [new Date(o[0] + ":00:00Z").getTime(), o[1], o[2]]).filter(o => o[0] >= t0 && o[0] <= t1);
  const others = all ? Object.keys(S.F.configs).filter(c => c !== S.cfg).map(c => [c, series(c)]).filter(x => x[1]) : [];
  const vals = [...s.speed, ...obs.map(o => o[1]), ...(s.spread ? s.speed.map((v, k) => v + s.spread[k]) : []), ...others.flatMap(x => x[1].speed)];
  const ymax = Math.max(4, Math.ceil(Math.max(...vals) / 2) * 2 + 1);
  const x = t => m.l + (t - t0) / (t1 - t0) * (W - m.l - m.r), y = v => H - m.b - v / ymax * (H - m.t - m.b);
  const path = arr => arr.map((v, k) => `${k ? "L" : "M"}${x(S.times[k].getTime()).toFixed(1)},${y(v).toFixed(1)}`).join("");
  let g = "";
  const step = ymax > 14 ? 4 : 2;
  for (let v = 0; v <= ymax; v += step) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  S.times.forEach((d, k) => {
    const hh = fmtHour.format(d), t = d.getTime();
    if (+hh.slice(0, 2) % 6 === 0) g += `<text x="${x(t)}" y="${H - m.b + 16}" text-anchor="middle">${hh.slice(0, 2)}</text>`;
    if (hh.startsWith("00") || k === 0) g += `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--line)"/><text x="${x(t) + 4}" y="${H - 6}" style="font-weight:600">${fmtDayShort.format(d)}</text>`;
  });
  const nowT = Date.now();
  if (nowT > t0 && nowT < t1) g += `<line x1="${x(nowT)}" x2="${x(nowT)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-dasharray="3 3"/><text x="${x(nowT) + 4}" y="${m.t + 10}">Now</text>`;
  if (s.spread) {
    const up = s.speed.map((v, k) => v + s.spread[k]), lo = s.speed.map((v, k) => Math.max(0, v - s.spread[k]));
    g += `<path d="${path(up)}${lo.map((_, k) => { const q = lo.length - 1 - k; return `L${x(S.times[q].getTime())},${y(lo[q])}`; }).join("")}Z" fill="var(--band)"/>`;
  }
  for (const [c, o] of others) g += `<path d="${path(o.speed)}" fill="none" stroke="${CFG_COLORS[c]}" stroke-width="1.5" opacity=".75"/>`;
  g += `<path d="${path(s.speed)}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/>`;
  for (const o of obs) g += `<circle cx="${x(o[0])}" cy="${y(o[1])}" r="3" fill="var(--obs)" stroke="var(--panel)" stroke-width="1.5"/>`;
  S.hours.forEach((hr, k) => { if (isStep(hr)) g += `<circle cx="${x(S.times[k].getTime())}" cy="${y(s.speed[k])}" r="3.5" fill="var(--accent)" stroke="var(--panel)" stroke-width="2"/>`; });
  S.hours.forEach((hr, k) => { if (hr % 3 === 0) g += `<g transform="translate(${x(S.times[k].getTime())},${H - m.b + 34}) rotate(${s.dir[k] + 180})"><path d="M0 -6 L4 3 L0 1.1 L-4 3 Z" fill="var(--accent)"/></g>`; });
  g += `<text x="${m.l - 6}" y="${m.t - 4}" text-anchor="end">m/s</text>`;
  box.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Wind speed forecast">${g}<rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/><line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--fg)" opacity="0"/></svg><div class="tip" id="gtip" hidden></div>`;
  let lg = `<span><span class="sw" style="background:var(--accent)"></span>${S.F.configs[S.cfg].label}</span>`;
  if (s.spread) lg += `<span><span class="sw" style="background:var(--band);height:10px"></span>Spread (±1 sd)</span>`;
  if (obs.length) lg += `<span><span class="sw" style="background:var(--obs);width:7px;height:7px;border-radius:50%"></span>Observed</span>`;
  for (const [c] of others) lg += `<span><span class="sw" style="background:${CFG_COLORS[c]}"></span>${S.F.configs[c].label}</span>`;
  $("legend").innerHTML = lg;
  const hit = box.querySelector("#hit"), xh = box.querySelector("#xh"), tip = $("gtip");
  hit.addEventListener("mousemove", e => {
    const r = hit.getBoundingClientRect(), tx = t0 + (e.clientX - r.left) / r.width * (t1 - t0);
    const k = Math.max(0, Math.min(S.times.length - 1, Math.round((tx - t0) / 3600e3)));
    const X = x(S.times[k].getTime()); xh.setAttribute("x1", X); xh.setAttribute("x2", X); xh.setAttribute("opacity", .35);
    const o = obsAt(S.times[k]);
    tip.innerHTML = `<b>${fmtShort.format(S.times[k])}</b><br>${fmt(s.speed[k])} m/s${s.spread ? ` ±${fmt(s.spread[k])}` : ""} ${arrowSvg(s.dir[k], "arrow sm")}${o ? `<br>Observed ${fmt(o[1])} m/s` : ""}` +
      others.map(([c, q]) => `<br><span style="color:${CFG_COLORS[c]}">${S.F.configs[c].label} ${fmt(q.speed[k])}</span>`).join("");
    tip.hidden = false; tip.style.left = Math.min(X + 12, W - 260) + "px"; tip.style.top = "16px";
  });
  hit.addEventListener("mouseleave", () => { tip.hidden = true; xh.setAttribute("opacity", 0); });
}

// ---------- map ----------
// wind speed colours (m/s), a smooth blue -> teal -> green -> yellow -> red -> purple scale
const SPEED_STOPS = [[0, [74, 104, 170]], [2, [62, 128, 182]], [4, [64, 160, 180]], [6, [76, 170, 140]], [8, [110, 182, 92]],
  [10, [176, 196, 72]], [12, [226, 190, 64]], [14, [234, 150, 56]], [17, [222, 98, 60]], [20, [190, 54, 82]], [25, [140, 42, 120]], [32, [96, 40, 120]]];
const LUT = (() => { const t = new Uint8ClampedArray(256 * 3); for (let q = 0; q < 256; q++) { const c = ramp(SPEED_STOPS, q * 0.15); t.set(c.map(Math.round), q * 3); } return t; })();
function ramp(stops, v) {
  if (v <= stops[0][0]) return stops[0][1];
  for (let k = 1; k < stops.length; k++) if (v <= stops[k][0]) {
    const [a, ca] = stops[k - 1], [b, cb] = stops[k], f = (v - a) / (b - a);
    return ca.map((c, i) => c + (cb[i] - c) * f);
  }
  return stops[stops.length - 1][1];
}
const M = { shade: null, land: null, coast: null, U: null, V: null, pair: null, field: -1 };
function prepareStatic(bg) {                            // hillshade factor, land mask and coastline from bg.webp
  const W = S.grid.width, H = S.grid.height, p = readPixels(bg);
  M.shade = new Float32Array(W * H); M.land = new Uint8Array(W * H);
  // hillshade factor: 1 on flat ground (bg value 0.64), brighter on slopes facing the light, darker away from it
  for (let k = 0; k < W * H; k++) { M.land[k] = p[k * 4 + 3] > 0 ? 1 : 0; M.shade[k] = Math.min(1.2, Math.max(0.6, 1 + 0.6 * ((p[k * 4] - 120) / 135 - 0.64))); }
  const c = $("c-coast"), x = c.getContext("2d"), im = x.createImageData(W, H), d = im.data;
  for (let i = 1; i < H - 1; i++) for (let j = 1; j < W - 1; j++) {
    const k = i * W + j;
    if (M.land[k] && (!M.land[k - 1] || !M.land[k + 1] || !M.land[k - W] || !M.land[k + W])) { d[k * 4 + 3] = 255; }
  }
  x.putImageData(im, 0, 0);
  $("labels").innerHTML = PLACES.map(([nm, la, lo]) => {
    const [i, j] = pixelOf(la, lo);
    return (i < 8 || j < 8 || i > H - 90 || j > W - 8) ? "" : `<span class="place" style="left:${j / W * 100}%;top:${i / H * 100}%">${nm}</span>`;
  }).join("");
}
function drawLabelsTo(lab) {
  const W = S.grid.width, H = S.grid.height;
  lab.font = "600 15px Inter, system-ui, sans-serif"; lab.textBaseline = "middle"; lab.lineJoin = "round";
  for (const [nm, la, lo] of PLACES) {
    const [i, j] = pixelOf(la, lo);
    if (i < 8 || j < 8 || i > H - 90 || j > W - 8) continue;
    lab.beginPath(); lab.arc(j, i, 3, 0, 7); lab.fillStyle = "#1b2a35"; lab.fill();
    lab.lineWidth = 4; lab.strokeStyle = "rgba(255,255,255,.85)"; lab.strokeText(nm, j + 7, i);
    lab.fillStyle = "#1b2a35"; lab.fillText(nm, j + 7, i);
  }
}
function curPixPair() { const p = S.pix[S.cfg]; if (!p) return null; const [k, w] = bracket(S.hour); return p.arr[k] && p.arr[k + 1] ? [p.arr[k], p.arr[k + 1], w, k] : null; }
function frameUV(p) {                                   // packed frame -> u, v (Float32) ; NaN where no data
  const n = p.length / 4, U = new Float32Array(n), V = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    if (p[k * 4] === 0) { U[k] = NaN; V[k] = NaN; continue; }
    const sp = p[k * 4] * 0.2, d = p[k * 4 + 2] / 255 * 2 * Math.PI;
    U[k] = -sp * Math.sin(d); V[k] = -sp * Math.cos(d);
  }
  return [U, V];
}
const uvCache = new Map();
function uvOf(cfg, k, p) { const key = cfg + ":" + k; if (!uvCache.has(key)) { if (uvCache.size > 6) uvCache.delete(uvCache.keys().next().value); uvCache.set(key, frameUV(p)); } return uvCache.get(key); }
function buildField() {                                 // interpolated u, v of the current hour
  const pp = curPixPair(); if (!pp) return false;
  const [pa, pb, w, k] = pp, [Ua, Va] = uvOf(S.cfg, k, pa), [Ub, Vb] = uvOf(S.cfg, k + 1, pb), n = Ua.length;
  if (!M.U || M.U.length !== n) { M.U = new Float32Array(n); M.V = new Float32Array(n); }
  for (let q = 0; q < n; q++) { M.U[q] = Ua[q] + (Ub[q] - Ua[q]) * w; M.V[q] = Va[q] + (Vb[q] - Va[q]) * w; }
  return true;
}
function sizeCanvases() { for (const id of ["c-wind", "c-coast", "c-top"]) { const c = $(id); c.width = S.grid.width; c.height = S.grid.height; } sizeParticles(); }
function drawMap() {
  $("lead-out").textContent = `${fmtShort.format(S.times[S.hour])}`;
  $("lead").value = S.hour;
  const msg = $("map-msg");
  if (!M.shade || !buildField()) { msg.hidden = false; msg.textContent = "Loading the map…"; return; }
  msg.hidden = true;
  const W = S.grid.width, H = S.grid.height, c = $("c-wind").getContext("2d");
  const im = c.createImageData(W, H), d = im.data;
  for (let q = 0; q < W * H; q++) {
    const u = M.U[q];
    if (u !== u) { if (M.land[q]) { const g = 196 * M.shade[q]; d[q * 4] = g; d[q * 4 + 1] = g + 3; d[q * 4 + 2] = g + 6; d[q * 4 + 3] = 255; } continue; }
    const sp = Math.hypot(u, M.V[q]), li = Math.min(255, Math.round(sp / 0.15)) * 3, f = M.shade[q];
    d[q * 4] = LUT[li] * f; d[q * 4 + 1] = LUT[li + 1] * f; d[q * 4 + 2] = LUT[li + 2] * f; d[q * 4 + 3] = 255;
  }
  c.putImageData(im, 0, 0);
  startParticles();
  if (S.dim === "3d") update3d();
  if (Z.full && !$("map-card").hidden) showCard();
}
// moving particles that follow the wind (drawn at screen resolution)
const P = { n: 0, x: null, y: null, age: null, raf: 0, landIdx: null };
function sizeParticles() {
  const c = $("c-particles"), inner = $("map-inner"), dpr = Math.min(2, window.devicePixelRatio || 1) * Math.min(Z.z, 3);
  c.width = Math.max(1, Math.round(inner.offsetWidth * dpr)); c.height = Math.max(1, Math.round(inner.offsetHeight * dpr));
}
function seed(k) {
  const q = P.landIdx[(Math.random() * P.landIdx.length) | 0];
  P.x[k] = q % S.grid.width + Math.random(); P.y[k] = Math.floor(q / S.grid.width) + Math.random(); P.age[k] = Math.random() * 80 | 0;
}
function startParticles() {
  if (!P.landIdx) { const a = []; for (let q = 0; q < M.land.length; q += 3) if (M.land[q]) a.push(q); P.landIdx = Uint32Array.from(a); }
  if (!P.n) { P.n = 3500; P.x = new Float32Array(P.n); P.y = new Float32Array(P.n); P.age = new Uint16Array(P.n); for (let k = 0; k < P.n; k++) seed(k); }
  if (!P.raf && S.view === "map" && S.dim === "2d") P.raf = requestAnimationFrame(stepParticles);
}
function stopParticles() { cancelAnimationFrame(P.raf); P.raf = 0; }
function stepParticles() {
  P.raf = 0;
  if (S.view !== "map" || S.dim !== "2d" || !M.U) return;
  const c = $("c-particles"), x = c.getContext("2d"), W = S.grid.width, sx = c.width / W, sy = c.height / S.grid.height;
  x.globalCompositeOperation = "destination-in"; x.fillStyle = "rgba(0,0,0,.92)"; x.fillRect(0, 0, c.width, c.height);
  x.globalCompositeOperation = "source-over"; x.strokeStyle = "rgba(255,255,255,.8)"; x.lineWidth = 1.1 * (sx > 1.2 ? 1.5 : 1) * Math.min(Z.z, 3) / Z.z;
  x.beginPath();
  const K = 0.35 / Math.sqrt(Z.z);                     // cells per frame per m/s (slower when zoomed in)
  for (let k = 0; k < P.n; k++) {
    const j = P.x[k] | 0, i = P.y[k] | 0, q = i * W + j, u = M.U[q];
    if (u !== u || P.age[k] > 90) { seed(k); continue; }
    const nx = P.x[k] + u * K * 0.25, ny = P.y[k] - M.V[q] * K * 0.25;
    x.moveTo(P.x[k] * sx, P.y[k] * sy); x.lineTo(nx * sx, ny * sy);
    P.x[k] = nx; P.y[k] = ny; P.age[k]++;
    if (nx < 0 || ny < 0 || nx >= W || ny >= S.grid.height) seed(k);
  }
  x.stroke();
  P.raf = requestAnimationFrame(stepParticles);
}
function drawScale() {
  const c = $("scale"), x = c.getContext("2d");
  for (let i = 0; i < c.width; i++) { const col = ramp(SPEED_STOPS, 25 * i / (c.width - 1)); x.fillStyle = `rgb(${col.map(Math.round).join(",")})`; x.fillRect(i, 0, 1, c.height); }
  $("scale-ticks").innerHTML = [0, 5, 10, 15, 20, 25].map(v => `<span style="left:${v / 25 * 100}%">${v}</span>`).join("");
}
function drawMarks() {
  const c = $("c-top").getContext("2d"); c.clearRect(0, 0, S.grid.width, S.grid.height);
  for (const st of S.F.stations) { c.beginPath(); c.arc(st.j, st.i, 3, 0, 7); c.fillStyle = "rgba(255,255,255,.9)"; c.fill(); c.lineWidth = 1.2; c.strokeStyle = "#1b2a35"; c.stroke(); }
  if (S.loc) {
    c.beginPath(); c.arc(S.loc.j, S.loc.i, 10, 0, 7); c.lineWidth = 4; c.strokeStyle = "#fff"; c.stroke();
    c.beginPath(); c.arc(S.loc.j, S.loc.i, 10, 0, 7); c.lineWidth = 2.5; c.strokeStyle = css("--mark"); c.stroke();
  }
  if (S.dim === "3d") update3d();
}
function nearStation(i, j, tol) {
  let best = null, bd = tol;
  for (const st of S.F.stations) { const d = Math.hypot(st.i - i, st.j - j); if (d < bd) { bd = d; best = st; } }
  return best;
}
function pick(i, j, tol) {
  const st = nearStation(i, j, tol);
  if (st) setStation(st); else setCell(i, j);
  if (Z.full) showCard();
}
function showCard() {
  const s = series(), k = S.hour;
  $("card-name").textContent = S.loc.name;
  $("card-val").innerHTML = s ? `${fmtShort.format(S.times[k])}: <b>${fmt(s.speed[k])} m/s</b> ${arrowSvg(s.dir[k], "arrow sm")}` : (s === undefined ? "Loading…" : "No forecast from this model here");
  $("map-card").hidden = false;
}
function hover(i, j, px, py, tol, box) {
  const tip = $("map-tip");
  if (i < 0 || j < 0 || i >= S.grid.height || j >= S.grid.width || !M.U) { tip.hidden = true; return; }
  const q = i * S.grid.width + j, st = nearStation(i, j, tol), u = M.U[q];
  if (u !== u && !st && !M.land[q]) { tip.hidden = true; return; }
  const ll = latlon(i, j);
  tip.innerHTML = (st ? `<b>${st.name}</b><br>` : `<b>${ll[0].toFixed(2)}° N, ${ll[1].toFixed(2)}° E</b><br>`) +
    (u === u ? `${Math.hypot(u, M.V[q]).toFixed(1)} m/s ${arrowSvg((Math.atan2(-u, -M.V[q]) / D2R + 360) % 360, "arrow sm")}` : "No forecast from this model here");
  tip.hidden = false;
  tip.style.left = Math.min(px + 14, box.width - 200) + "px"; tip.style.top = Math.max(py - 54, 4) + "px";
}
function initMap() {
  const top = $("c-top");
  const pos = e => { const r = top.getBoundingClientRect(); return { i: Math.round((e.clientY - r.top) / r.height * S.grid.height), j: Math.round((e.clientX - r.left) / r.width * S.grid.width), px: e.clientX - r.left, py: e.clientY - r.top, tol: 8 * S.grid.width / r.width, r }; };
  top.addEventListener("click", e => { if (Z.dragged) return; const q = pos(e); pick(q.i, q.j, q.tol / Z.z); });
  top.addEventListener("mousemove", e => { if (Z.drag) return; const q = pos(e); const r = $("map").getBoundingClientRect(); hover(q.i, q.j, e.clientX - r.left, e.clientY - r.top, q.tol / Z.z, r); });
  top.addEventListener("mouseleave", () => { $("map-tip").hidden = true; });
  $("lead").addEventListener("input", e => { stop(); S.hour = +e.target.value; drawMap(); });
  $("play").addEventListener("click", () => (S.playing ? stop() : play()));
  document.querySelectorAll("[data-dim]").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll("[data-dim]").forEach(x => x.setAttribute("aria-pressed", x === b)); setDim(b.dataset.dim);
  }));
  new ResizeObserver(() => { fitInner(); sizeParticles(); }).observe($("map"));
  drawScale();
  initZoom();
}

// ---------- full view: zoom and pan ----------
const Z = { full: false, z: 1, tx: 0, ty: 0, drag: null, dragged: false, pts: new Map(), pinch: null, sharpen: 0 };
function fitInner() {
  const inner = $("map-inner"), box = $("map");
  const bw = box.clientWidth, bh = box.clientHeight, ar = S.grid ? S.grid.width / S.grid.height : 964 / 1033;
  if (!Z.full) {                                        // page view: fill the width, crop top and bottom evenly
    const w = bw, h = w / ar;
    inner.style.width = w + "px"; inner.style.height = h + "px";
    inner.style.left = "0px"; inner.style.top = (bh - h) / 2 + "px";
    inner.style.transform = ""; inner.style.setProperty("--z", 1);
    return;
  }
  const w = Math.min(bw, bh * ar), h = w / ar;
  inner.style.width = w + "px"; inner.style.height = h + "px";
  inner.style.left = (bw - w) / 2 + "px"; inner.style.top = (bh - h) / 2 + "px";
  applyZoom();
}
function applyZoom() {
  const inner = $("map-inner"), box = $("map");
  if (!Z.full) return;
  const w = inner.offsetWidth * Z.z, h = inner.offsetHeight * Z.z, bw = box.clientWidth, bh = box.clientHeight;
  const ox = inner.offsetLeft, oy = inner.offsetTop;               // keep at least a quarter of the map in view
  Z.tx = Math.min(bw * 0.75 - ox, Math.max(bw * 0.25 - ox - w, Z.tx));
  Z.ty = Math.min(bh * 0.75 - oy, Math.max(bh * 0.25 - oy - h, Z.ty));
  inner.style.transform = `translate(${Z.tx}px, ${Z.ty}px) scale(${Z.z})`;
  inner.style.setProperty("--z", Z.z);
  box.classList.toggle("zoomed", Z.z > 3);
  clearTimeout(Z.sharpen); Z.sharpen = setTimeout(sizeParticles, 150);
}
function zoomAt(f, px, py) {                             // zoom by factor f around (px, py) in map-box pixels
  const inner = $("map-inner"), z2 = Math.min(16, Math.max(1, Z.z * f));
  const ax = px - inner.offsetLeft, ay = py - inner.offsetTop;
  Z.tx = ax - (ax - Z.tx) * z2 / Z.z; Z.ty = ay - (ay - Z.ty) * z2 / Z.z; Z.z = z2;
  if (z2 === 1) { Z.tx = 0; Z.ty = 0; }
  applyZoom();
}
function setFull(on) {
  Z.full = on; Z.z = 1; Z.tx = 0; Z.ty = 0;
  $("view-map").classList.toggle("full", on); document.body.classList.toggle("map-open", on);
  $("map-zoom").hidden = !on; $("map-card").hidden = true;
  $("map-full").setAttribute("aria-label", on ? "Close the full map" : "Open the full map"); $("map-full").title = $("map-full").getAttribute("aria-label");
  $("map-full-icon").setAttribute("d", on ? "M5 5l10 10M15 5L5 15" : "M3 8V3h5M12 3h5v5M17 12v5h-5M8 17H3v-5");
  $("map").classList.remove("zoomed");
  fitInner(); sizeParticles(); if (M.U) drawMap();
}
function initZoom() {
  const box = $("map"), rel = e => { const r = box.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  $("map-full").addEventListener("click", () => setFull(!Z.full));
  $("zoom-in").addEventListener("click", () => zoomAt(1.6, box.clientWidth / 2, box.clientHeight / 2));
  $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.6, box.clientWidth / 2, box.clientHeight / 2));
  $("zoom-reset").addEventListener("click", () => { Z.z = 1; Z.tx = 0; Z.ty = 0; applyZoom(); });
  $("card-go").addEventListener("click", () => { setFull(false); location.hash = "table"; });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && Z.full) setFull(false); });
  box.addEventListener("wheel", e => {
    if (!Z.full || S.dim !== "2d") return;
    e.preventDefault(); const [px, py] = rel(e); zoomAt(Math.exp(-e.deltaY * 0.0015), px, py);
  }, { passive: false });
  const inner = $("map-inner");
  inner.addEventListener("pointerdown", e => {
    if (!Z.full) return;
    Z.pts.set(e.pointerId, rel(e)); inner.setPointerCapture(e.pointerId);
    if (Z.pts.size === 1) { Z.drag = { p: rel(e), tx: Z.tx, ty: Z.ty }; Z.dragged = false; }
    if (Z.pts.size === 2) { const [a, b] = [...Z.pts.values()]; Z.pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: Z.z }; }
  });
  inner.addEventListener("pointermove", e => {
    if (!Z.full || !Z.pts.has(e.pointerId)) return;
    Z.pts.set(e.pointerId, rel(e));
    if (Z.pts.size === 2 && Z.pinch) {
      const [a, b] = [...Z.pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomAt(Z.pinch.z * d / Z.pinch.d / Z.z, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); Z.dragged = true; return;
    }
    if (Z.drag) {
      const [x, y] = rel(e), dx = x - Z.drag.p[0], dy = y - Z.drag.p[1];
      if (Math.hypot(dx, dy) > 4) { Z.dragged = true; box.classList.add("dragging"); $("map-tip").hidden = true; }
      if (Z.dragged) { Z.tx = Z.drag.tx + dx; Z.ty = Z.drag.ty + dy; applyZoom(); }
    }
  });
  const end = e => {
    Z.pts.delete(e.pointerId); if (Z.pts.size < 2) Z.pinch = null;
    if (Z.pts.size === 0) { Z.drag = null; box.classList.remove("dragging"); setTimeout(() => { Z.dragged = false; }, 0); }
  };
  inner.addEventListener("pointerup", end); inner.addEventListener("pointercancel", end);
}
function play() {
  $("play-icon").setAttribute("d", "M5 4 H8.5 V16 H5 Z M11.5 4 H15 V16 H11.5 Z"); $("play").setAttribute("aria-label", "Pause");
  S.playing = setInterval(() => { S.hour = (S.hour + 1) % S.hours.length; drawMap(); }, reduceMotion() ? 1200 : 350);
}
function stop() {
  if (!S.playing) return;
  clearInterval(S.playing); S.playing = null;
  $("play-icon").setAttribute("d", "M6 4 L16 10 L6 16 Z"); $("play").setAttribute("aria-label", "Play");
}

// ---------- 3D view (three.js, loaded on first use) ----------
const THREE_SRC = ["https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                   "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"];
const T = { ready: null, renderer: null, scene: null, camera: null, controls: null, mesh: null, tex: null, canvas: null, raf: 0 };
function loadScript(src) { return new Promise((ok, err) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => err(new Error(src)); document.head.appendChild(s); }); }
async function init3d() {
  for (const s of THREE_SRC) await loadScript(s);
  const THREE = window.THREE, box = $("map3d");
  const W = S.grid.width, H = S.grid.height;
  const elev = readPixels(await loadImg(DATA + "static/elevation.png"));
  const nx = 321, ny = Math.round(nx * H / W);                       // mesh resolution (about 3 km)
  const geo = new THREE.PlaneGeometry(W, H, nx - 1, ny - 1);
  const pos = geo.attributes.position, EX = 6 / 1000;                  // 1 unit = 1 km; height x6
  for (let v = 0; v < pos.count; v++) {
    const c = v % nx, r = Math.floor(v / nx);
    const j = Math.min(W - 1, Math.round(c / (nx - 1) * (W - 1))), i = Math.min(H - 1, Math.round(r / (ny - 1) * (H - 1)));
    const k = (i * W + j) * 4;
    pos.setZ(v, elev[k + 2] ? (elev[k] * 256 + elev[k + 1]) * EX : 0);
  }
  geo.computeVertexNormals();
  T.canvas = document.createElement("canvas"); T.canvas.width = W; T.canvas.height = H;
  T.tex = new THREE.CanvasTexture(T.canvas); T.tex.anisotropy = 4;
  T.mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ map: T.tex })); T.mesh.rotation.x = -Math.PI / 2;
  T.scene = new THREE.Scene(); T.scene.add(T.mesh);
  T.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 0.45); sun.position.set(-400, 600, -300); T.scene.add(sun);
  T.camera = new THREE.PerspectiveCamera(40, box.clientWidth / box.clientHeight, 1, 8000);
  T.camera.position.set(80, 620, 760);
  T.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  T.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  T.renderer.setSize(box.clientWidth, box.clientHeight);
  box.appendChild(T.renderer.domElement);
  T.controls = new THREE.OrbitControls(T.camera, T.renderer.domElement);
  T.controls.target.set(60, 0, -60); T.controls.maxPolarAngle = Math.PI * 0.47; T.controls.minDistance = 80; T.controls.maxDistance = 2200;
  T.controls.enableDamping = true; T.controls.update();
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const hitAt = e => {
    const r = T.renderer.domElement.getBoundingClientRect();
    ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(ndc, T.camera);
    const h = ray.intersectObject(T.mesh)[0];
    return h ? { i: Math.round((1 - h.uv.y) * (H - 1)), j: Math.round(h.uv.x * (W - 1)), px: e.clientX - r.left, py: e.clientY - r.top, r } : null;
  };
  let down = null;
  T.renderer.domElement.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  T.renderer.domElement.addEventListener("pointerup", e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const q = hitAt(e); if (q) pick(q.i, q.j, 6);
  });
  T.renderer.domElement.addEventListener("mousemove", e => { const q = hitAt(e); if (q) hover(q.i, q.j, q.px, q.py, 6, q.r); else $("map-tip").hidden = true; });
  T.renderer.domElement.addEventListener("mouseleave", () => { $("map-tip").hidden = true; });
  new ResizeObserver(() => { if (!box.clientWidth) return; T.camera.aspect = box.clientWidth / box.clientHeight; T.camera.updateProjectionMatrix(); T.renderer.setSize(box.clientWidth, box.clientHeight); }).observe(box);
}
function update3d() {
  if (!T.tex) return;
  const x = T.canvas.getContext("2d");
  x.fillStyle = css("--sea"); x.fillRect(0, 0, T.canvas.width, T.canvas.height);
  for (const id of ["c-wind", "c-coast", "c-top"]) x.drawImage($(id), 0, 0);
  drawLabelsTo(x);
  T.tex.needsUpdate = true;
}
function loop() { T.raf = requestAnimationFrame(loop); T.controls.update(); T.renderer.render(T.scene, T.camera); }
async function setDim(dim) {
  S.dim = dim;
  const is3 = dim === "3d";
  $("map3d").hidden = !is3;
  $("map-inner").hidden = is3;
  $("map-hint").textContent = is3 ? "Drag to turn, scroll to zoom, click on land to choose a place. Heights are exaggerated six times."
                                  : "Click anywhere on land to get the forecast for that 1 km cell. White dots are the 99 weather stations the models learned from.";
  if (!is3) { cancelAnimationFrame(T.raf); if (M.U) startParticles(); return; }
  stopParticles();
  const msg = $("map-msg");
  try {
    if (!T.ready) { msg.hidden = false; msg.textContent = "Loading the 3D view…"; T.ready = init3d(); }
    await T.ready; msg.hidden = true;
    update3d(); cancelAnimationFrame(T.raf); loop();
  } catch (e) {
    msg.hidden = false; msg.textContent = "The 3D view could not be loaded. The 2D map still works."; console.error(e);
  }
}

// ---------- model, run, verification ----------
function meanChange(c) { const v = S.verif && S.verif.configs[c]; return v ? v.change_pct.reduce((a, b) => a + b, 0) / v.change_pct.length : null; }
function fillModels() {
  const sel = $("model");
  const ids = ORDER.filter(c => S.F.configs[c]).concat(Object.keys(S.F.configs).filter(c => !ORDER.includes(c)));
  sel.innerHTML = ids.map(c => `<option value="${c}">${S.F.configs[c].label}</option>`).join("");
  const saved = store.get("wc-model");
  S.cfg = S.F.configs[saved] ? saved : ids[0];
  sel.value = S.cfg;
  noteModel();
}
function noteModel() {
  const c = S.F.configs[S.cfg], ch = meanChange(S.cfg);
  $("model-note").textContent = (ch != null ? `${Math.abs(ch).toFixed(1)} % lower error than IFS + AIFS. ` : "") +
    (c.coverage < 0.99 ? `Covers ${Math.round(c.coverage * 100)} % of the map (the MEPS area).` : "");
  $("scale-note").hidden = c.coverage >= 0.99;
  $("scale-note").textContent = "No forecast from this model (outside the MEPS area)";
}
async function setModel(cfg) {
  S.cfg = cfg; store.set("wc-model", cfg); noteModel();
  renderVerif(); renderAll();
  if (S.view === "map") drawMap();
  await loadConfig(cfg);
  renderAll(); if (S.view === "map") drawMap();
}
function renderVerif() {
  const v = S.verif; if (!v) return;
  const rows = [["IFS + AIFS mean (reference)", null, v.rmse_reference]];
  for (const c of ORDER) if (v.configs[c]) rows.push([(S.F.configs[c] || { label: c }).label, c, v.configs[c].rmse]);
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length, ref = mean(v.rmse_reference);
  const lo = Math.min(...rows.map(r => mean(r[2]))) * 0.97;
  $("verif").innerHTML = `<div class="tbl"><table class="vt"><thead><tr><th>Model</th><th class="num">RMSE</th><th class="num">Change</th><th></th></tr></thead><tbody>` +
    rows.map(([lab, c, r]) => { const m = mean(r), ch = (m / ref - 1) * 100;
      return `<tr class="${c === S.cfg ? "sel" : ""}"><td>${lab}</td><td class="num">${m.toFixed(2)} m/s</td><td class="num">${c ? ch.toFixed(1) + " %" : "–"}</td><td><span class="bar${c ? "" : " ref"}" style="width:${Math.max(2, (m - lo) / (ref * 1.03 - lo) * 90)}px"></span></td></tr>`; }).join("") +
    `</tbody></table></div><p class="hint">Mean over forecast times 0 to 66 h, ${v.n_stations} stations, ${v.period}. Each station is predicted by models trained without it. The hourly values between the 6-hourly model times are interpolated; they keep almost all of the improvement.</p>`;
}
async function loadRun(run) {
  stop();
  S.F = await getJSON(`runs/${run}/forecast.json`);
  try { S.obs = (await getJSON(`runs/${run}/obs.json`)).obs || {}; } catch (e) { S.obs = {}; }
  const t0 = new Date(S.F.valid[0]).getTime(), last = S.F.leads[S.F.leads.length - 1];
  S.hours = Array.from({ length: last + 1 }, (_, h) => h);
  S.times = S.hours.map(h => new Date(t0 + h * 3600e3));
  S.pix = {}; uvCache.clear();
  $("lead").max = S.hours.length - 1;
  S.hour = Math.min(nowIndex(), S.hours.length - 1);
  $("stations").innerHTML = S.F.stations.map(s => `<option value="${s.name}">${s.municipality}</option>`).join("");
  fillModels();
  renderVerif();
  if (!S.loc || S.loc.kind === "station") {
    const st = stationById(S.loc ? S.loc.id : store.get("wc-loc")) || S.F.stations.find(s => s.name === "Tromsø") || S.F.stations[0];
    setStation(st, false);
  } else renderAll();
  $("foot").textContent = `Run ${fmtRun.format(new Date(S.F.init))} UTC, made ${fmtRun.format(new Date(S.F.created))} UTC. Times are Norwegian time. ` +
    `Observations: MET Norway (Frost). Input forecasts: ECMWF open data (IFS, AIFS), NOAA GFS, MET Norway (MEPS, AROME-Arctic).`;
  if (S.view === "map") drawMap();
  await loadConfig(S.cfg);
  renderAll(); if (S.view === "map") drawMap();
}

// ---------- tabs ----------
function show(v) {
  S.view = v;
  for (const n of VIEWS) { $("view-" + n).hidden = n !== v; $("tab-" + n).setAttribute("aria-selected", n === v); }
  if (v !== "map") { stop(); stopParticles(); }
  if (v === "graph") renderGraph();
  if (v === "map") { fitInner(); sizeParticles(); drawMap(); drawMarks(); }
}
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => { location.hash = b.dataset.view; }));
window.addEventListener("hashchange", () => { const v = location.hash.slice(1); if (VIEWS.includes(v)) show(v); });
window.addEventListener("resize", () => { if (S.view === "graph" && S.loc) renderGraph(); });

// ---------- boot ----------
async function boot() {
  [S.grid, S.index] = await Promise.all([getJSON("static/grid.json"), getJSON("runs/index.json")]);
  try { S.verif = await getJSON("static/verification.json"); } catch (e) { S.verif = null; }
  sizeCanvases(); initMap();
  $("run").innerHTML = S.index.runs.map((r, k) => {
    const d = new Date(`${r.slice(0, 4)}-${r.slice(4, 6)}-${r.slice(6, 8)}T${r.slice(9, 11)}:00Z`);
    return `<option value="${r}">${fmtRun.format(d)} UTC${k === 0 ? " (latest)" : ""}</option>`;
  }).join("");
  $("run").addEventListener("change", e => loadRun(e.target.value));
  $("model").addEventListener("change", e => setModel(e.target.value));
  $("all-models").addEventListener("change", () => { if (S.loc) renderGraph(); });
  $("q").addEventListener("change", e => {
    const st = S.F.stations.find(s => s.name.toLowerCase() === e.target.value.trim().toLowerCase());
    if (st) { setStation(st); e.target.value = ""; }
  });
  loadImg(DATA + "static/bg.webp").then(im => { prepareStatic(im); if (S.view === "map" && S.F) drawMap(); });
  const v0 = location.hash.slice(1);
  await loadRun(S.index.latest);
  show(VIEWS.includes(v0) ? v0 : "table");
}
boot().catch(e => { $("loc-name").textContent = "The forecast could not be loaded"; $("loc-sub").textContent = "Reload the page to try again."; console.error(e); });
})();
