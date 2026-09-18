// View 3: Run. Configure/run one backtest (or a same-seed BS-vs-Heston
// pair), time-synced panel stack, fill markers with realized edge, and a
// quote-waterfall panel reconstructed from the cached run.

(function () {
  const COLOR_BS = "#4da3ff";
  const COLOR_HESTON = "#ff9f43";

  const state = {
    mode: "single", // "single" | "pair"
    model: "black_scholes",
    marketSeed: 42, arrivalSeed: 123, nSteps: 63,
    raDelta: "", raVega: "", raGamma: "",
    baseIntensity: "", orderArrivalKappa: "",
    strikes: "", maturities: "",
    runs: { single: null, bs: null, heston: null },
    fillsFilter: { contractId: "all", negativeOnly: false },
    waterfall: { runKey: null, contractId: null, barIndex: 0, raDeltaOverride: null, raVegaOverride: null },
  };

  const PLOT_IDS = ["plot-spot", "plot-variance", "plot-pnl", "plot-greeks-book", "plot-inventory"];

  function root() { return document.getElementById("view-run"); }

  function numOrUndefined(s) { const v = parseFloat(s); return s === "" || Number.isNaN(v) ? undefined : v; }
  function listOrUndefined(s) {
    if (!s || !s.trim()) return undefined;
    return s.split(",").map((x) => parseFloat(x.trim())).filter((x) => !Number.isNaN(x));
  }

  function render() {
    const r = root();
    r.innerHTML = "";

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Configure & run"]),
      el("div", { class: "controls" }, [
        el("div", { class: "field" }, [el("label", {}, ["mode"]),
          el("select", { onchange: (e) => { state.mode = e.target.value; render(); } }, [
            el("option", { value: "single", selected: state.mode === "single" ? "selected" : undefined }, ["single model"]),
            el("option", { value: "pair", selected: state.mode === "pair" ? "selected" : undefined }, ["pair: BS vs Heston (same seed)"]),
          ])]),
        state.mode === "single" ? el("div", { class: "field" }, [el("label", {}, ["model"]),
          el("select", { onchange: (e) => { state.model = e.target.value; } }, [
            el("option", { value: "black_scholes", selected: state.model === "black_scholes" ? "selected" : undefined }, ["black_scholes"]),
            el("option", { value: "heston", selected: state.model === "heston" ? "selected" : undefined }, ["heston"]),
          ])]) : null,
        el("div", { class: "field" }, [el("label", {}, ["market_seed"]),
          el("input", { type: "number", value: state.marketSeed, onchange: (e) => { state.marketSeed = parseInt(e.target.value, 10); } })]),
        el("div", { class: "field" }, [el("label", {}, ["arrival_seed"]),
          el("input", { type: "number", value: state.arrivalSeed, onchange: (e) => { state.arrivalSeed = parseInt(e.target.value, 10); } })]),
        el("div", { class: "field" }, [el("label", {}, ["n_steps"]),
          el("input", { type: "number", value: state.nSteps, onchange: (e) => { state.nSteps = parseInt(e.target.value, 10); } })]),
      ]),
      el("div", { class: "controls", style: "margin-top:8px" }, [
        el("div", { class: "field" }, [el("label", {}, ["risk_aversion.delta"]),
          el("input", { type: "number", step: "any", placeholder: "engine default", value: state.raDelta, onchange: (e) => { state.raDelta = e.target.value; } })]),
        el("div", { class: "field" }, [el("label", {}, ["risk_aversion.vega"]),
          el("input", { type: "number", step: "any", placeholder: "engine default", value: state.raVega, onchange: (e) => { state.raVega = e.target.value; } })]),
        el("div", { class: "field" }, [el("label", {}, ["risk_aversion.gamma"]),
          el("input", { type: "number", step: "any", placeholder: "engine default", value: state.raGamma, onchange: (e) => { state.raGamma = e.target.value; } })]),
        el("div", { class: "field" }, [el("label", {}, ["base_intensity"]),
          el("input", { type: "number", step: "any", placeholder: "engine default", value: state.baseIntensity, onchange: (e) => { state.baseIntensity = e.target.value; } })]),
        el("div", { class: "field" }, [el("label", {}, ["order_arrival_kappa"]),
          el("input", { type: "number", step: "any", placeholder: "engine default", value: state.orderArrivalKappa, onchange: (e) => { state.orderArrivalKappa = e.target.value; } })]),
      ]),
      el("div", { class: "controls", style: "margin-top:8px" }, [
        el("div", { class: "field", style: "min-width:220px" }, [el("label", {}, ["basket strikes (comma-sep, blank = default)"]),
          el("input", { type: "text", value: state.strikes, onchange: (e) => { state.strikes = e.target.value; } })]),
        el("div", { class: "field", style: "min-width:220px" }, [el("label", {}, ["basket maturities, years (comma-sep, blank = default)"]),
          el("input", { type: "text", value: state.maturities, onchange: (e) => { state.maturities = e.target.value; } })]),
      ]),
      el("div", { class: "controls", style: "margin-top:10px" }, [
        el("button", { class: "action", onclick: runNow }, [state.mode === "pair" ? "Run pair (BS vs Heston)" : "Run"]),
      ]),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Summary"]),
      el("div", { id: "run-stats", class: "stat-tiles" }),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Time-synced panels"]),
      el("div", { class: "note" }, ["Zoom/pan on any panel to zoom all others. Click a bar on any panel to open the quote waterfall below."]),
      ...PLOT_IDS.map((id) => el("div", { id, class: "plot short", style: "margin-bottom:6px" })),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Fills"]),
      el("div", { class: "controls" }, [
        el("div", { class: "field" }, [el("label", {}, ["contract filter"]),
          el("select", { id: "fills-contract-filter", onchange: (e) => { state.fillsFilter.contractId = e.target.value; renderFillsAndPnlOverlay(); } },
            [el("option", { value: "all" }, ["all contracts"])])]),
        el("div", { class: "checkbox-row" }, [
          el("input", { type: "checkbox", onchange: (e) => { state.fillsFilter.negativeOnly = e.target.checked; renderFillsAndPnlOverlay(); } }),
          el("span", {}, ["negative-edge fills only"]),
        ]),
      ]),
      el("div", { id: "fills-stats", class: "stat-tiles", style: "margin-top:10px" }),
    ]));

    r.appendChild(el("div", { class: "panel waterfall-panel" }, [
      el("h2", {}, ["Quote waterfall"]),
      el("div", { class: "controls" }, [
        state.mode === "pair" ? el("div", { class: "field" }, [el("label", {}, ["which run"]),
          el("select", { id: "wf-run-select", onchange: (e) => { state.waterfall.runKey = e.target.value; loadWaterfall(); } }, [
            el("option", { value: "bs" }, ["black_scholes"]),
            el("option", { value: "heston" }, ["heston"]),
          ])]) : null,
        el("div", { class: "field" }, [el("label", {}, ["contract"]),
          el("select", { id: "wf-contract-select", onchange: (e) => { state.waterfall.contractId = e.target.value; loadWaterfall(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["bar index"]),
          el("input", { type: "number", id: "wf-bar-input", value: state.waterfall.barIndex, onchange: (e) => { state.waterfall.barIndex = parseInt(e.target.value, 10) || 0; loadWaterfall(); } })]),
      ]),
      el("div", { class: "row", style: "margin-top:8px" }, [
        el("div", { class: "col" }, [
          el("label", {}, ["what-if risk_aversion.delta: ", el("span", { id: "wf-delta-val" }, ["—"])]),
          el("input", { type: "range", id: "wf-delta-slider", min: "0", max: "2", step: "0.01", oninput: onWhatIfChange }),
        ]),
        el("div", { class: "col" }, [
          el("label", {}, ["what-if risk_aversion.vega: ", el("span", { id: "wf-vega-val" }, ["—"])]),
          el("input", { type: "range", id: "wf-vega-slider", min: "0", max: "0.2", step: "0.001", oninput: onWhatIfChange }),
        ]),
        el("button", { class: "action secondary", onclick: resetWhatIf }, ["Reset to run's actual values"]),
      ]),
      el("div", { id: "wf-note", class: "note", style: "display:none" }),
      el("div", { class: "row" }, [
        el("div", { class: "col" }, [el("div", { id: "plot-wf-reservation", class: "plot short" })]),
        el("div", { class: "col" }, [el("div", { id: "plot-wf-ask", class: "plot short" })]),
        el("div", { class: "col" }, [el("div", { id: "plot-wf-bid", class: "plot short" })]),
      ]),
    ]));

    linkXAxes(["plot-spot", "plot-variance", "plot-pnl", "plot-greeks-book", "plot-inventory"]);
  }

  function linkXAxes(ids) {
    ids.forEach((id) => {
      const div = document.getElementById(id);
      if (!div || div._linkedSynced) return;
      div._linkedSynced = true;
      div.on("plotly_relayout", (ev) => {
        const range = ev["xaxis.range[0]"] !== undefined ? [ev["xaxis.range[0]"], ev["xaxis.range[1]"]] : null;
        ids.forEach((otherId) => {
          if (otherId === id) return;
          const other = document.getElementById(otherId);
          if (!other) return;
          if (range) Plotly.relayout(other, { "xaxis.range": range });
          else if (ev["xaxis.autorange"]) Plotly.relayout(other, { "xaxis.autorange": true });
        });
      });
      div.on("plotly_click", (ev) => {
        if (!ev.points || !ev.points.length) return;
        const barIndex = ev.points[0].pointIndex;
        state.waterfall.barIndex = barIndex;
        const barInput = document.getElementById("wf-bar-input");
        if (barInput) barInput.value = barIndex;
        loadWaterfall();
      });
    });
  }

  async function runNow() {
    const common = {
      market_seed: state.marketSeed, arrival_seed: state.arrivalSeed, n_steps: state.nSteps,
      strikes: listOrUndefined(state.strikes), maturities: listOrUndefined(state.maturities),
      risk_aversion_delta: numOrUndefined(state.raDelta), risk_aversion_vega: numOrUndefined(state.raVega),
      risk_aversion_gamma: numOrUndefined(state.raGamma), base_intensity: numOrUndefined(state.baseIntensity),
      order_arrival_kappa: numOrUndefined(state.orderArrivalKappa),
    };
    try {
      if (state.mode === "single") {
        toast("Running backtest…");
        const body = await API.post("/api/backtest/run", { model: state.model, ...common });
        state.runs.single = body;
        state.runs.bs = null; state.runs.heston = null;
      } else {
        toast("Running BS + Heston on the same seed pair…");
        const body = await API.post("/api/backtest/run-pair", common);
        state.runs.bs = body.black_scholes; state.runs.heston = body.heston;
        state.runs.single = null;
        state.waterfall.runKey = "bs";
      }
      toast("Run complete.");
      afterRunsUpdated();
    } catch (e) { toast("Run failed: " + e.message, true); }
  }

  function afterRunsUpdated() {
    renderStats();
    renderPanels();
    populateFillsContractFilter();
    renderFillsAndPnlOverlay();
    populateWaterfallContractSelect();
    loadWaterfall();
  }

  function primaryRun() { return state.mode === "single" ? state.runs.single : state.runs.bs; }

  function renderStats() {
    const container = document.getElementById("run-stats");
    container.innerHTML = "";
    if (state.mode === "single") {
      const r = state.runs.single;
      if (!r) return;
      container.appendChild(statTile("Sharpe", fmt(r.sharpe, 2)));
      container.appendChild(statTile("Max drawdown", fmt(r.max_drawdown, 2), "bad"));
      container.appendChild(statTile("Final P&L", fmt(r.pnl[r.pnl.length - 1], 2), r.pnl[r.pnl.length - 1] >= 0 ? "good" : "bad"));
      container.appendChild(statTile("n_fills", String(r.meta.n_fills)));
    } else {
      const rb = state.runs.bs, rh = state.runs.heston;
      if (!rb || !rh) return;
      [["BS", rb, COLOR_BS], ["Heston", rh, COLOR_HESTON]].forEach(([label, r]) => {
        container.appendChild(statTile(label + " Sharpe", fmt(r.sharpe, 2)));
        container.appendChild(statTile(label + " MaxDD", fmt(r.max_drawdown, 2), "bad"));
        container.appendChild(statTile(label + " P&L", fmt(r.pnl[r.pnl.length - 1], 2), r.pnl[r.pnl.length - 1] >= 0 ? "good" : "bad"));
        container.appendChild(statTile(label + " fills", String(r.meta.n_fills)));
      });
    }
  }

  function renderPanels() {
    if (state.mode === "single") {
      const r = state.runs.single;
      if (!r) return;
      plotLine("plot-spot", [{ x: r.times, y: r.spot_path, color: "#e6e8ec", name: "spot" }], "spot");
      plotLine("plot-variance", [{ x: r.times, y: r.variance_path, color: "#9aa2b1", name: "variance" }], "realized variance");
      plotLine("plot-greeks-book", [
        { x: r.times, y: r.delta_book, color: "#4da3ff", name: "delta_book" },
        { x: r.times, y: r.vega_book, color: "#ff9f43", name: "vega_book" },
      ], "book delta / vega (quoting model's own view)");
      const contractIds = Object.keys(r.inventory);
      plotLine("plot-inventory", contractIds.map((cid, i) => ({
        x: r.times, y: r.inventory[cid], color: colorForIndex(i), name: cid,
      })), "per-contract inventory");
    } else {
      const rb = state.runs.bs, rh = state.runs.heston;
      if (!rb || !rh) return;
      plotLine("plot-spot", [{ x: rb.times, y: rb.spot_path, color: "#e6e8ec", name: "spot (shared)" }], "spot (identical realized market for both runs)");
      plotLine("plot-variance", [{ x: rb.times, y: rb.variance_path, color: "#9aa2b1", name: "variance (shared)" }], "realized variance");
      plotLine("plot-greeks-book", [
        { x: rb.times, y: rb.delta_book, color: COLOR_BS, name: "BS delta_book" },
        { x: rh.times, y: rh.delta_book, color: COLOR_HESTON, name: "Heston delta_book" },
        { x: rb.times, y: rb.vega_book, color: COLOR_BS, name: "BS vega_book", dash: "dot" },
        { x: rh.times, y: rh.vega_book, color: COLOR_HESTON, name: "Heston vega_book", dash: "dot" },
      ], "book delta / vega — solid=delta, dotted=vega");
      plotLine("plot-inventory", [
        { x: rb.times, y: rb.gross_inventory, color: COLOR_BS, name: "BS gross inventory" },
        { x: rh.times, y: rh.gross_inventory, color: COLOR_HESTON, name: "Heston gross inventory" },
      ], "gross inventory (pair mode shows gross, not per-contract, to stay readable)");
    }
  }

  const PALETTE = ["#4da3ff", "#ff9f43", "#3ddc97", "#ff5d5d", "#c792ea", "#f5d76e", "#6ee7ff"];
  function colorForIndex(i) { return PALETTE[i % PALETTE.length]; }

  function plotLine(divId, series, title) {
    const traces = series.map((s) => ({
      x: s.x, y: s.y, mode: "lines", name: s.name,
      line: { color: s.color, width: 1.6, dash: s.dash || "solid" },
    }));
    Plotly.newPlot(divId, traces, mergedLayout({ title: { text: title, font: { size: 11 } } }), PLOTLY_CONFIG);
  }

  function populateFillsContractFilter() {
    const r = primaryRun();
    const select = document.getElementById("fills-contract-filter");
    if (!r || !select) return;
    select.innerHTML = "";
    select.appendChild(el("option", { value: "all" }, ["all contracts"]));
    Object.keys(r.inventory).forEach((cid) => select.appendChild(el("option", { value: cid }, [cid])));
  }

  function filteredFills(r) {
    let fills = r.fills;
    if (state.fillsFilter.contractId !== "all") fills = fills.filter((f) => f.contract_id === state.fillsFilter.contractId);
    if (state.fillsFilter.negativeOnly) fills = fills.filter((f) => f.edge < 0);
    return fills.slice().sort((a, b) => a.t - b.t);
  }

  function renderFillsAndPnlOverlay() {
    const r = primaryRun();
    if (!r) return;
    const fills = filteredFills(r);

    // Fill markers + cumulative edge, overlaid on the PNL panel (secondary y-axis).
    const pnlSeries = state.mode === "single"
      ? [{ x: r.times, y: r.pnl, color: "#e6e8ec", name: "P&L" }]
      : [{ x: state.runs.bs.times, y: state.runs.bs.pnl, color: COLOR_BS, name: "BS P&L" },
         { x: state.runs.heston.times, y: state.runs.heston.pnl, color: COLOR_HESTON, name: "Heston P&L" }];

    const pnlTraces = pnlSeries.map((s) => ({ x: s.x, y: s.y, mode: "lines", name: s.name, line: { color: s.color, width: 1.6 }, yaxis: "y" }));

    const fillX = fills.map((f) => r.times[f.t]);
    const fillY = fills.map((f) => r.pnl[f.t]);
    const fillColor = fills.map((f) => f.edge);
    const fillSymbol = fills.map((f) => (f.side === "buy" ? "triangle-up" : "triangle-down"));
    const fillText = fills.map((f) => `${f.contract_id} ${f.side}<br>price=${fmt(f.price, 3)} true=${fmt(f.true_price, 3)}<br>edge=${fmt(f.edge, 4)}`);
    const maxAbsEdge = Math.max(1e-6, ...fills.map((f) => Math.abs(f.edge)));

    let cumEdge = 0;
    const cumEdgeX = [], cumEdgeY = [];
    fills.forEach((f) => { cumEdge += f.edge; cumEdgeX.push(r.times[f.t]); cumEdgeY.push(cumEdge); });

    const markerTrace = {
      x: fillX, y: fillY, mode: "markers", name: "fills",
      marker: { color: fillColor, colorscale: EDGE_COLORSCALE, cmin: -maxAbsEdge, cmax: maxAbsEdge, symbol: fillSymbol, size: 8, line: { width: 0.5, color: "#0f1115" } },
      text: fillText, hovertemplate: "%{text}<extra></extra>", yaxis: "y",
    };
    const cumEdgeTrace = { x: cumEdgeX, y: cumEdgeY, mode: "lines", name: "cumulative realized edge", line: { color: "#c792ea", width: 1.4, dash: "dot" }, yaxis: "y2" };

    const layout = mergedLayout({
      title: { text: "P&L, fills (colored by realized edge), cumulative edge", font: { size: 11 } },
      yaxis: { title: "P&L ($)" },
      yaxis2: { title: "cumulative edge", overlaying: "y", side: "right", gridcolor: "transparent" },
    });
    Plotly.newPlot("plot-pnl", [...pnlTraces, markerTrace, cumEdgeTrace], layout, PLOTLY_CONFIG);
    linkXAxes(PLOT_IDS);

    renderFillsSummary(fills);
  }

  function renderFillsSummary(fills) {
    const container = document.getElementById("fills-stats");
    container.innerHTML = "";
    const totalEdge = fills.reduce((s, f) => s + f.edge, 0);
    const negCount = fills.filter((f) => f.edge < 0).length;
    const worst = fills.reduce((w, f) => (w === null || f.edge < w.edge ? f : w), null);
    container.appendChild(statTile("Total edge captured", fmt(totalEdge, 3), totalEdge >= 0 ? "good" : "bad"));
    container.appendChild(statTile("Negative-edge fills", `${negCount} / ${fills.length}`));
    container.appendChild(statTile("Worst single fill", worst ? `${fmt(worst.edge, 4)} (${worst.contract_id}, t=${worst.t})` : "—", "bad"));
  }

  // -------------------- Quote waterfall --------------------

  function currentWaterfallRun() {
    if (state.mode === "single") return state.runs.single;
    return state.waterfall.runKey === "heston" ? state.runs.heston : state.runs.bs;
  }

  function populateWaterfallContractSelect() {
    const r = currentWaterfallRun();
    const select = document.getElementById("wf-contract-select");
    if (!r || !select) return;
    select.innerHTML = "";
    Object.keys(r.inventory).forEach((cid) => select.appendChild(el("option", { value: cid }, [cid])));
    if (!state.waterfall.contractId || !(state.waterfall.contractId in r.inventory)) {
      state.waterfall.contractId = Object.keys(r.inventory)[0];
    }
    select.value = state.waterfall.contractId;

    const ra = r.meta.risk_aversion;
    const deltaSlider = document.getElementById("wf-delta-slider");
    const vegaSlider = document.getElementById("wf-vega-slider");
    if (deltaSlider) { deltaSlider.value = ra.delta; document.getElementById("wf-delta-val").textContent = fmt(ra.delta, 3); }
    if (vegaSlider) { vegaSlider.value = ra.vega; document.getElementById("wf-vega-val").textContent = fmt(ra.vega, 3); }
    state.waterfall.raDeltaOverride = null;
    state.waterfall.raVegaOverride = null;
  }

  function onWhatIfChange() {
    const d = parseFloat(document.getElementById("wf-delta-slider").value);
    const v = parseFloat(document.getElementById("wf-vega-slider").value);
    document.getElementById("wf-delta-val").textContent = fmt(d, 3);
    document.getElementById("wf-vega-val").textContent = fmt(v, 3);
    state.waterfall.raDeltaOverride = d;
    state.waterfall.raVegaOverride = v;
    loadWaterfall();
  }

  function resetWhatIf() {
    const r = currentWaterfallRun();
    if (!r) return;
    const ra = r.meta.risk_aversion;
    document.getElementById("wf-delta-slider").value = ra.delta;
    document.getElementById("wf-vega-slider").value = ra.vega;
    document.getElementById("wf-delta-val").textContent = fmt(ra.delta, 3);
    document.getElementById("wf-vega-val").textContent = fmt(ra.vega, 3);
    state.waterfall.raDeltaOverride = null;
    state.waterfall.raVegaOverride = null;
    loadWaterfall();
  }

  async function loadWaterfall() {
    const r = currentWaterfallRun();
    const noteEl = document.getElementById("wf-note");
    if (!r || !r.run_id) return;
    if (!state.waterfall.contractId) populateWaterfallContractSelect();
    const barIndex = Math.max(0, Math.min(r.times.length - 1, state.waterfall.barIndex));
    try {
      const body = await API.post("/api/backtest/waterfall", {
        run_id: r.run_id, contract_id: state.waterfall.contractId, bar_index: barIndex,
        risk_aversion_delta_override: state.waterfall.raDeltaOverride,
        risk_aversion_vega_override: state.waterfall.raVegaOverride,
      });
      if (noteEl) {
        noteEl.style.display = body.is_what_if ? "block" : "none";
        noteEl.textContent = body.is_what_if
          ? "WHAT-IF: showing the quote with overridden risk_aversion — this does not change the completed run, only this decomposition."
          : "";
      }
      plotWaterfall(body);
    } catch (e) { toast("Waterfall failed: " + e.message, true); }
  }

  function waterfallBars(divId, title, steps) {
    // steps: [{label, measure: 'absolute'|'relative'|'total', value}]
    Plotly.newPlot(divId, [{
      type: "waterfall", orientation: "v",
      x: steps.map((s) => s.label), y: steps.map((s) => s.value), measure: steps.map((s) => s.measure),
      connector: { line: { color: "#3a4050" } },
      increasing: { marker: { color: "#3ddc97" } },
      decreasing: { marker: { color: "#ff5d5d" } },
      totals: { marker: { color: "#4da3ff" } },
    }], mergedLayout({ title: { text: title, font: { size: 11 } }, showlegend: false }), PLOTLY_CONFIG);
  }

  function plotWaterfall(body) {
    const wf = body.waterfall;
    waterfallBars("plot-wf-reservation", "fair_value → reservation_price", [
      { label: "fair_value", measure: "absolute", value: wf.fair_value },
      { label: `delta skew (Δ_book=${fmt(wf.delta_book, 2)})`, measure: "relative", value: wf.delta_skew },
      { label: `vega skew (vega_book=${fmt(wf.vega_book, 2)})`, measure: "relative", value: wf.vega_skew },
      { label: "reservation_price", measure: "total", value: wf.reservation_price },
    ]);
    const askSteps = [
      { label: "reservation_price", measure: "absolute", value: wf.reservation_price },
      { label: "+½ risk_spread", measure: "relative", value: (wf.risk_spread_delta + wf.risk_spread_vega) / 2 },
      { label: "+½ liquidity_spread", measure: "relative", value: wf.liquidity_spread / 2 },
    ];
    if (wf.gamma_term > 0) askSteps.push({ label: "+½ gamma_term", measure: "relative", value: wf.gamma_term / 2 });
    askSteps.push({ label: "ask", measure: "total", value: wf.ask });
    waterfallBars("plot-wf-ask", "reservation_price → ask", askSteps);

    const bidSteps = [
      { label: "reservation_price", measure: "absolute", value: wf.reservation_price },
      { label: "−½ risk_spread", measure: "relative", value: -(wf.risk_spread_delta + wf.risk_spread_vega) / 2 },
      { label: "−½ liquidity_spread", measure: "relative", value: -wf.liquidity_spread / 2 },
    ];
    if (wf.gamma_term > 0) bidSteps.push({ label: "−½ gamma_term", measure: "relative", value: -wf.gamma_term / 2 });
    bidSteps.push({ label: "bid", measure: "total", value: wf.bid });
    waterfallBars("plot-wf-bid", "reservation_price → bid", bidSteps);
  }

  // -------------------- Sweep -> Run navigation --------------------

  async function maybeLoadPending() {
    const pending = window.AppState.pendingRunLoad;
    if (!pending) return;
    window.AppState.pendingRunLoad = null;
    state.mode = "pair";
    state.marketSeed = pending.seed;
    state.arrivalSeed = pending.seed + 1000;
    render();
    toast(`Reproducing sweep seed ${pending.seed} (BS vs Heston)…`);
    try {
      const body = await API.post("/api/sweep/reproduce", { seed: pending.seed });
      state.runs.bs = body.black_scholes; state.runs.heston = body.heston; state.runs.single = null;
      state.waterfall.runKey = "bs";
      afterRunsUpdated();
      toast(`Loaded seed ${pending.seed}: BS $${fmt(body.black_scholes.pnl[body.black_scholes.pnl.length - 1], 1)} vs Heston $${fmt(body.heston.pnl[body.heston.pnl.length - 1], 1)}`);
    } catch (e) { toast("Failed to load seed: " + e.message, true); }
  }

  window.Views.run = {
    init() { render(); },
    activate() { maybeLoadPending(); },
  };
})();
