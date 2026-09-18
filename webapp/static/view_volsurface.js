// View 2: Vol Surface. Live shape controls -> synthetic surface; a "Fit
// models" button calibrates Heston/SABR against the current surface and
// shows overlay + residual heatmaps + RMSE.

(function () {
  const SHAPE_FIELDS = [
    ["atm_vol_short", 0.24, 0.05, 0.6, 0.005],
    ["atm_vol_long", 0.18, 0.05, 0.6, 0.005],
    ["term_decay", 1.0, 0.1, 5.0, 0.05],
    ["skew_short", -0.55, -2.0, 0.5, 0.01],
    ["skew_long", -0.10, -2.0, 0.5, 0.01],
    ["skew_decay", 0.75, 0.1, 5.0, 0.05],
    ["smile_short", 0.35, 0.0, 2.0, 0.01],
    ["smile_long", 0.08, 0.0, 2.0, 0.01],
    ["smile_decay", 0.75, 0.1, 5.0, 0.05],
  ];

  const state = { spot: 100, strikes: [], maturities: [], shape: {}, rate: 0.03, divYield: 0.0 };
  let lastCalibration = null;

  function root() { return document.getElementById("view-volsurface"); }

  function shapeSlider(key, def, min, max, step) {
    const valueSpan = el("span", { class: "value" }, [String(state.shape[key])]);
    return el("div", { class: "field" }, [
      el("label", {}, [key]),
      el("input", {
        type: "range", min: String(min), max: String(max), step: String(step), value: String(state.shape[key]),
        oninput: (e) => {
          state.shape[key] = parseFloat(e.target.value);
          valueSpan.textContent = String(state.shape[key]);
          refreshSurfaceOnly();
        },
      }),
      valueSpan,
    ]);
  }

  function render() {
    const r = root();
    r.innerHTML = "";

    const note = el("div", { class: "note" }, [
      "This synthetic surface is NOT arbitrage-checked (no calendar-spread or butterfly no-arbitrage constraints are enforced). ",
      "Everything downstream — Heston/SABR calibration, and the backtest's own 'true market' — ultimately derives from this surface's shape.",
    ]);
    r.appendChild(note);

    const controlsPanel = el("div", { class: "panel" }, [
      el("h2", {}, ["Surface shape"]),
      el("div", { class: "controls" }, [
        el("div", { class: "field" }, [el("label", {}, ["spot"]),
          el("input", { type: "number", value: state.spot, onchange: (e) => { state.spot = parseFloat(e.target.value); refreshSurfaceOnly(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["strikes (min,max,step)"]),
          el("input", { type: "text", value: "60,140,5", onchange: (e) => { state.strikes = parseRange(e.target.value); refreshSurfaceOnly(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["maturities (years, comma-sep)"]),
          el("input", { type: "text", value: "0.083,0.167,0.25,0.5,1,2,3", onchange: (e) => { state.maturities = parseList(e.target.value); refreshSurfaceOnly(); } })]),
      ]),
      el("div", { class: "row", style: "margin-top:8px" }, SHAPE_FIELDS.map(([key, def, min, max, step]) =>
        el("div", { class: "col", style: "min-width:150px;flex:0 0 220px" }, [shapeSlider(key, def, min, max, step)])
      )),
      el("div", { class: "controls", style: "margin-top:10px" }, [
        el("button", { class: "action", id: "fit-btn", onclick: runCalibration }, ["Fit Heston + SABR"]),
        el("span", { class: "note", style: "margin:0" }, ["Calibration is CPU-bound (scipy optimizer over the full grid) — typically 10-25s, not live."]),
      ]),
    ]);
    r.appendChild(controlsPanel);

    const surfacePanel = el("div", { class: "panel" }, [
      el("h2", {}, ["3D surface"]),
      el("div", { class: "controls", id: "surface-toggles" }, [
        toggleCheckbox("Target (synthetic)", "target", true),
        toggleCheckbox("Heston fit", "heston", true),
        toggleCheckbox("SABR fit", "sabr", true),
      ]),
      el("div", { id: "plot-surface3d", class: "plot tall" }),
    ]);
    r.appendChild(surfacePanel);

    const fitPanel = el("div", { class: "panel" }, [
      el("h2", {}, ["Calibration fit quality"]),
      el("div", { id: "rmse-tiles", class: "stat-tiles" }),
      el("div", { class: "row", style: "margin-top:10px" }, [
        el("div", { class: "col" }, [el("h3", {}, ["Heston residual (model − target, vol points)"]), el("div", { id: "plot-residual-heston", class: "plot" })]),
        el("div", { class: "col" }, [el("h3", {}, ["SABR residual (model − target, vol points)"]), el("div", { id: "plot-residual-sabr", class: "plot" })]),
      ]),
    ]);
    r.appendChild(fitPanel);
  }

  function toggleCheckbox(label, traceKey, checked) {
    return el("div", { class: "checkbox-row" }, [
      el("input", {
        type: "checkbox", checked: checked ? "checked" : undefined,
        onchange: (e) => Plotly.restyle("plot-surface3d", { visible: e.target.checked }, [traceIndex(traceKey)]),
      }),
      el("span", {}, [label]),
    ]);
  }

  const TRACE_ORDER = ["target", "heston", "sabr"];
  function traceIndex(key) { return TRACE_ORDER.indexOf(key); }

  function parseRange(text) {
    const [lo, hi, step] = text.split(",").map((s) => parseFloat(s.trim()));
    const out = [];
    for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
    return out;
  }
  function parseList(text) { return text.split(",").map((s) => parseFloat(s.trim())).filter((v) => !Number.isNaN(v)); }

  async function fetchSurface() {
    return API.post("/api/vol-surface/generate", {
      spot: state.spot, strikes: state.strikes, maturities: state.maturities, ...state.shape,
    });
  }

  async function refreshSurfaceOnly() {
    try {
      const body = await fetchSurface();
      plotSurface3D(body, null);
    } catch (e) { toast("Vol surface generate failed: " + e.message, true); }
  }

  function plotSurface3D(target, fits) {
    const traces = [{
      type: "surface", x: target.strikes, y: target.maturities, z: target.iv_grid,
      colorscale: "Viridis", opacity: 1.0, showscale: false, name: "target",
    }];
    if (fits && fits.heston) {
      traces.push({ type: "surface", x: target.strikes, y: target.maturities, z: fits.heston.iv_grid, colorscale: [[0, "#4da3ff"], [1, "#4da3ff"]], opacity: 0.45, showscale: false, name: "heston" });
    } else {
      traces.push({ type: "surface", x: target.strikes, y: target.maturities, z: target.iv_grid, opacity: 0, showscale: false, visible: false, name: "heston" });
    }
    if (fits && fits.sabr) {
      traces.push({ type: "surface", x: target.strikes, y: target.maturities, z: fits.sabr.iv_grid, colorscale: [[0, "#ff9f43"], [1, "#ff9f43"]], opacity: 0.45, showscale: false, name: "sabr" });
    } else {
      traces.push({ type: "surface", x: target.strikes, y: target.maturities, z: target.iv_grid, opacity: 0, showscale: false, visible: false, name: "sabr" });
    }
    const layout = mergedLayout({
      scene: {
        xaxis: { title: "strike", gridcolor: "#2a2f3a" },
        yaxis: { title: "maturity (years)", gridcolor: "#2a2f3a" },
        zaxis: { title: "IV", gridcolor: "#2a2f3a" },
        bgcolor: "#171a21",
      },
      margin: { l: 0, r: 0, t: 10, b: 0 },
    });
    Plotly.newPlot("plot-surface3d", traces, layout, PLOTLY_CONFIG);
  }

  async function runCalibration() {
    const btn = document.getElementById("fit-btn");
    if (btn) { btn.disabled = true; btn.textContent = "Fitting… (10-25s)"; }
    toast("Fitting Heston + SABR — this is a CPU-bound optimizer run, typically 10-25s for the full grid…");
    try {
      const body = await API.post("/api/vol-surface/calibrate", {
        spot: state.spot, strikes: state.strikes, maturities: state.maturities, ...state.shape,
        models: ["heston", "sabr"], rate: state.rate, div_yield: state.divYield,
      });
      lastCalibration = body;
      plotSurface3D(
        { strikes: body.strikes, maturities: body.maturities, iv_grid: body.target_iv_grid },
        body.fits
      );
      renderRmseTiles(body.fits);
      plotResidualHeatmap("plot-residual-heston", body.strikes, body.maturities, body.fits.heston.residual_grid);
      plotResidualHeatmap("plot-residual-sabr", body.strikes, body.maturities, body.fits.sabr.residual_grid);
      toast("Calibration complete.");
    } catch (e) {
      toast("Calibration failed: " + e.message, true);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "Fit Heston + SABR"; }
    }
  }

  function renderRmseTiles(fits) {
    const container = document.getElementById("rmse-tiles");
    container.innerHTML = "";
    container.appendChild(statTile("Heston RMSE (vol pts)", fmt(fits.heston.rmse * 100, 2)));
    container.appendChild(statTile("SABR RMSE (vol pts)", fmt(fits.sabr.rmse * 100, 2)));
  }

  function plotResidualHeatmap(divId, strikes, maturities, residualGrid) {
    const gridVolPts = residualGrid.map((row) => row.map((v) => v * 100));
    const maxAbs = Math.max(1e-6, ...gridVolPts.flat().map((v) => Math.abs(v)));
    const traces = [{
      type: "heatmap", x: strikes, y: maturities, z: gridVolPts,
      colorscale: EDGE_COLORSCALE, zmin: -maxAbs, zmax: maxAbs,
      colorbar: { title: "vol pts" },
    }];
    Plotly.newPlot(divId, traces, mergedLayout({ xaxis: { title: "strike" }, yaxis: { title: "maturity" } }), PLOTLY_CONFIG);
  }

  window.Views.volsurface = {
    async init() {
      const defaults = await API.get("/api/vol-surface/defaults");
      state.spot = defaults.spot;
      state.strikes = defaults.strikes;
      state.maturities = defaults.maturities;
      state.shape = {
        atm_vol_short: defaults.atm_vol_short, atm_vol_long: defaults.atm_vol_long, term_decay: defaults.term_decay,
        skew_short: defaults.skew_short, skew_long: defaults.skew_long, skew_decay: defaults.skew_decay,
        smile_short: defaults.smile_short, smile_long: defaults.smile_long, smile_decay: defaults.smile_decay,
        min_iv: defaults.min_iv,
      };
      render();
      refreshSurfaceOnly();
    },
  };
})();
