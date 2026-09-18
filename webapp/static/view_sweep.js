// View 4: Sweep / Conditional Edge. Reads the precomputed 40-seed sweep,
// plots pnl_heston - pnl_bs against a regime feature with an OLS fit/CI,
// a histogram of the paired difference, and lets clicking a point re-run
// that seed and jump into View 3 with both paths loaded.

(function () {
  const FEATURES = ["mean_variance", "vol_of_var", "max_spot_drawdown", "terminal_variance", "final_spot"];
  const state = { feature: "mean_variance", rows: null, recomputeJobId: null };

  function root() { return document.getElementById("view-sweep"); }

  function render() {
    const r = root();
    r.innerHTML = "";

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Mechanism (from FINAL_REPORT.md §4)"]),
      el("p", {}, [
        "The difference between the two quoters comes from a genuine, structural difference in deltas: Heston's calibrated " +
        "negative spot-vol correlation gives it higher deltas than flat Black-Scholes on the same book state. This changes each " +
        "quoter's inventory-skew behavior over the course of a backtest — it is not that one model is \"better\", it is that they " +
        "manage risk differently, and which one comes out ahead depends on the realized path. Click any point below to re-run that " +
        "seed for both models and inspect it in the Run view.",
      ]),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Data source"]),
      el("div", { class: "controls" }, [
        el("span", { id: "sweep-source-note", class: "note", style: "margin:0" }, ["loading…"]),
        el("button", { class: "action secondary", onclick: recompute }, ["Recompute sweep (background, ~1-2 min for 40 seeds)"]),
      ]),
      el("div", { id: "recompute-progress", style: "margin-top:8px;display:none" }, [
        el("div", { class: "progress-bar" }, [el("div", { id: "recompute-bar", style: "width:0%" })]),
        el("div", { id: "recompute-label", class: "note", style: "margin-top:6px" }, []),
      ]),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Conditional edge: pnl_heston − pnl_bs vs. regime feature"]),
      el("div", { class: "controls" }, [
        el("div", { class: "field" }, [el("label", {}, ["regime feature"]),
          el("select", { onchange: (e) => { state.feature = e.target.value; refreshRegression(); } },
            FEATURES.map((f) => el("option", { value: f, selected: f === state.feature ? "selected" : undefined }, [f])))]),
      ]),
      el("div", { id: "regression-stats", class: "stat-tiles", style: "margin:10px 0" }),
      el("div", { id: "regression-verdict", class: "note" }, []),
      el("div", { id: "plot-scatter", class: "plot tall" }),
    ]));

    r.appendChild(el("div", { class: "panel" }, [
      el("h2", {}, ["Distribution of the paired difference"]),
      el("div", { id: "diff-stats", class: "stat-tiles", style: "margin-bottom:10px" }),
      el("div", { id: "plot-hist", class: "plot" }),
    ]));
  }

  async function loadRows() {
    try {
      const body = await API.get("/api/sweep/results");
      state.rows = body.rows;
      document.getElementById("sweep-source-note").textContent = `${body.n_seeds} seeds loaded from ${body.csv_path}`;
    } catch (e) {
      document.getElementById("sweep-source-note").textContent = "No sweep data yet — click \"Recompute sweep\".";
    }
  }

  async function refreshRegression() {
    if (!state.rows) return;
    try {
      const reg = await API.get("/api/sweep/regression", { feature: state.feature });
      renderRegressionStats(reg);
      renderScatter(reg);
    } catch (e) { toast("Regression failed: " + e.message, true); }
    renderHistogram();
  }

  function renderRegressionStats(reg) {
    const container = document.getElementById("regression-stats");
    container.innerHTML = "";
    container.appendChild(statTile("slope", fmt(reg.slope, 4)));
    container.appendChild(statTile("R²", fmt(reg.r_squared, 3)));
    container.appendChild(statTile("p-value", fmt(reg.p_value, 3)));
    container.appendChild(statTile("n", String(reg.n)));
    const verdict = document.getElementById("regression-verdict");
    verdict.textContent = reg.verdict;
    verdict.style.borderColor = reg.significant_at_0_05 ? "rgba(61,220,151,0.4)" : "rgba(255,206,84,0.3)";
    verdict.style.color = reg.significant_at_0_05 ? "var(--good)" : "var(--warn)";
  }

  function renderScatter(reg) {
    const rows = state.rows;
    const x = rows.map((r) => r[state.feature]);
    const y = rows.map((r) => r.pnl_diff_heston_minus_bs);
    const totalFills = rows.map((r) => r.nfills_bs + r.nfills_heston);
    const maxFills = Math.max(...totalFills);
    const sizes = totalFills.map((f) => 6 + 18 * (f / maxFills));
    const text = rows.map((r, i) => `seed ${r.seed}<br>${state.feature}=${fmt(x[i], 4)}<br>Δpnl=${fmt(y[i], 2)}<br>fills=${totalFills[i]}`);

    const scatterTrace = {
      x, y, mode: "markers", type: "scatter", name: "seeds",
      marker: { size: sizes, color: y, colorscale: EDGE_COLORSCALE, cmin: -Math.max(...y.map(Math.abs)), cmax: Math.max(...y.map(Math.abs)), line: { width: 1, color: "#0f1115" } },
      text, hovertemplate: "%{text}<extra></extra>",
      customdata: rows.map((r) => r.seed),
    };
    const zeroLine = { x: [Math.min(...x), Math.max(...x)], y: [0, 0], mode: "lines", line: { color: "#555c6b", width: 1.5 }, name: "zero", hoverinfo: "skip" };
    const fitLine = { x: reg.x_grid, y: reg.y_pred, mode: "lines", line: { color: "#4da3ff", width: 2 }, name: "OLS fit" };
    const ciUpper = { x: reg.x_grid, y: reg.y_pred_hi, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" };
    const ciLower = { x: reg.x_grid, y: reg.y_pred_lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(77,163,255,0.15)", showlegend: false, hoverinfo: "skip" };

    const div = document.getElementById("plot-scatter");
    Plotly.newPlot(div, [ciUpper, ciLower, zeroLine, fitLine, scatterTrace], mergedLayout({
      xaxis: { title: state.feature }, yaxis: { title: "pnl_heston − pnl_bs" },
    }), PLOTLY_CONFIG);

    div.removeAllListeners && div.removeAllListeners("plotly_click");
    div.on("plotly_click", (ev) => {
      const point = ev.points.find((p) => p.data === scatterTrace || p.curveNumber === 4);
      if (!point) return;
      const seed = point.customdata !== undefined ? point.customdata : rows[point.pointIndex].seed;
      navigateToRunWithSeed(seed);
    });
  }

  function renderHistogram() {
    const rows = state.rows;
    const diffs = rows.map((r) => r.pnl_diff_heston_minus_bs);
    const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    // Simple bootstrap CI computed client-side (mirrors webapp.derive.bootstrap_mean_ci)
    // so the histogram doesn't need an extra round trip.
    const n = diffs.length, nBoot = 3000;
    const boots = [];
    for (let i = 0; i < nBoot; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += diffs[Math.floor(Math.random() * n)];
      boots.push(s / n);
    }
    boots.sort((a, b) => a - b);
    const lo = boots[Math.floor(0.025 * nBoot)], hi = boots[Math.floor(0.975 * nBoot)];

    const container = document.getElementById("diff-stats");
    container.innerHTML = "";
    container.appendChild(statTile("mean(Δpnl)", fmt(mean, 2), mean >= 0 ? "good" : "bad"));
    container.appendChild(statTile("95% bootstrap CI", `[${fmt(lo, 2)}, ${fmt(hi, 2)}]`));
    container.appendChild(statTile("std(Δpnl)", fmt(Math.sqrt(diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / (n - 1)), 2)));

    Plotly.newPlot("plot-hist", [
      { x: diffs, type: "histogram", marker: { color: "#4da3ff" }, nbinsx: 14 },
    ], mergedLayout({
      xaxis: { title: "pnl_heston − pnl_bs" }, yaxis: { title: "count" },
      shapes: [{ type: "line", x0: mean, x1: mean, y0: 0, y1: 1, yref: "paper", line: { color: "#3ddc97", width: 2 } },
               { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { color: "#555c6b", width: 1, dash: "dot" } }],
    }), PLOTLY_CONFIG);
  }

  async function recompute() {
    try {
      const body = await API.post("/api/sweep/recompute", { n_seeds: 40 });
      state.recomputeJobId = body.job_id;
      document.getElementById("recompute-progress").style.display = "block";
      pollRecompute();
    } catch (e) { toast("Recompute failed to start: " + e.message, true); }
  }

  async function pollRecompute() {
    if (!state.recomputeJobId) return;
    try {
      const status = await API.get(`/api/sweep/recompute/status/${state.recomputeJobId}`);
      const pct = status.total ? Math.round((100 * status.completed) / status.total) : 0;
      document.getElementById("recompute-bar").style.width = pct + "%";
      document.getElementById("recompute-label").textContent = `${status.completed}/${status.total} seeds — ${status.status}`;
      if (status.status === "running") {
        setTimeout(pollRecompute, 1500);
      } else if (status.status === "done") {
        toast("Sweep recompute finished.");
        await loadRows();
        refreshRegression();
      } else if (status.status === "error") {
        toast("Sweep recompute failed: " + status.error, true);
      }
    } catch (e) { toast("Status poll failed: " + e.message, true); }
  }

  window.Views.sweep = {
    async init() {
      render();
      await loadRows();
      if (state.rows) refreshRegression();
    },
  };
})();
