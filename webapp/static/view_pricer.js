// View 1: Pricer. Model dropdown (+ optional ghost overlay), live price/Greeks,
// and Greek profiles across strike.

(function () {
  const MODELS = ["black_scholes", "monte_carlo", "heston", "sabr"];
  const GREEK_KEYS = ["delta", "gamma", "vega", "theta", "rho"];
  const COLORS = { primary: "#4da3ff", ghost: "#ff9f43" };

  const state = {
    spot: 100, strike: 100, maturity: 1.0, optionType: "call",
    model: "black_scholes", ghostModel: "none",
    params: {}, ghostParams: {},
  };

  function root() { return document.getElementById("view-pricer"); }

  function paramField(model, key, value, isGhost) {
    return el("div", { class: "field" }, [
      el("label", {}, [key]),
      el("input", {
        type: "number", step: "any", value: String(value),
        onchange: (e) => {
          const v = parseFloat(e.target.value);
          (isGhost ? state.ghostParams : state.params)[key] = Number.isNaN(v) ? value : v;
          refreshAll();
        },
      }),
    ]);
  }

  async function loadDefaults(model, isGhost) {
    if (model === "none") return;
    const body = await API.get("/api/pricer/defaults", { model });
    if (isGhost) state.ghostParams = Object.assign({}, body.params);
    else state.params = Object.assign({}, body.params);
  }

  function renderParamFields(container, model, params, isGhost) {
    container.innerHTML = "";
    if (model === "none") return;
    Object.keys(params).forEach((key) => {
      container.appendChild(paramField(model, key, params[key], isGhost));
    });
  }

  function render() {
    const r = root();
    r.innerHTML = "";

    const configPanel = el("div", { class: "panel" }, [
      el("h2", {}, ["Configuration"]),
      el("div", { class: "controls" }, [
        el("div", { class: "field" }, [el("label", {}, ["spot"]),
          el("input", { type: "number", step: "any", value: state.spot, onchange: (e) => { state.spot = parseFloat(e.target.value); refreshAll(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["strike"]),
          el("input", { type: "number", step: "any", value: state.strike, onchange: (e) => { state.strike = parseFloat(e.target.value); refreshAll(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["maturity (years)"]),
          el("input", { type: "number", step: "any", value: state.maturity, onchange: (e) => { state.maturity = parseFloat(e.target.value); refreshAll(); } })]),
        el("div", { class: "field" }, [el("label", {}, ["option type"]),
          el("select", { onchange: (e) => { state.optionType = e.target.value; refreshAll(); } }, [
            el("option", { value: "call", selected: "selected" }, ["call"]),
            el("option", { value: "put" }, ["put"]),
          ])]),
      ]),
      el("div", { class: "row", style: "margin-top:12px" }, [
        el("div", { class: "col" }, [
          el("h3", {}, ["Primary model"]),
          el("div", { class: "controls" }, [
            el("div", { class: "field" }, [el("label", {}, ["model"]),
              el("select", { id: "primary-model-select", onchange: async (e) => {
                state.model = e.target.value;
                await loadDefaults(state.model, false);
                render();
                refreshAll();
              } }, MODELS.map((m) => el("option", { value: m, selected: m === state.model ? "selected" : undefined }, [m]))),
            ]),
            state.model === "monte_carlo" ? el("button", { class: "action", onclick: runMonteCarloPoint }, ["Run Monte Carlo"]) : null,
          ]),
          el("div", { class: "controls", id: "primary-params" }),
        ]),
        el("div", { class: "col" }, [
          el("h3", {}, ["Ghost overlay (optional second model)"]),
          el("div", { class: "controls" }, [
            el("div", { class: "field" }, [el("label", {}, ["ghost model"]),
              el("select", { onchange: async (e) => {
                state.ghostModel = e.target.value;
                if (state.ghostModel !== "none") await loadDefaults(state.ghostModel, true);
                render();
                refreshAll();
              } }, ["none"].concat(MODELS).map((m) => el("option", { value: m, selected: m === state.ghostModel ? "selected" : undefined }, [m]))),
            ]),
          ]),
          el("div", { class: "controls", id: "ghost-params" }),
        ]),
      ]),
    ]);
    r.appendChild(configPanel);

    renderParamFields(configPanel.querySelector("#primary-params"), state.model, state.params, false);
    renderParamFields(configPanel.querySelector("#ghost-params"), state.ghostModel, state.ghostParams, true);

    const statsPanel = el("div", { class: "panel" }, [
      el("h2", {}, ["Live price & Greeks"]),
      el("div", { id: "stats-primary", class: "stat-tiles" }),
      el("div", { id: "stats-ghost", class: "stat-tiles", style: "margin-top:8px" }),
      el("div", { id: "mc-note", class: "note", style: "display:none" }),
    ]);
    r.appendChild(statsPanel);

    const profilesPanel = el("div", { class: "panel" }, [
      el("h2", {}, ["Greek profiles across strike"]),
      el("div", { class: "note" }, ["Vertical line marks the current strike. Solid = primary model, dashed = ghost overlay."]),
      el("div", { class: "row", id: "profile-grid" }),
    ]);
    ["price", ...GREEK_KEYS].forEach((k) => {
      profilesPanel.querySelector("#profile-grid").appendChild(
        el("div", { class: "col" }, [el("div", { id: "plot-" + k, class: "plot short" })])
      );
    });
    r.appendChild(profilesPanel);
  }

  function strikeGrid() {
    const lo = state.spot * 0.6, hi = state.spot * 1.4;
    const n = 41;
    const out = [];
    for (let i = 0; i < n; i++) out.push(lo + ((hi - lo) * i) / (n - 1));
    return out;
  }

  async function refreshStats() {
    const statsP = document.getElementById("stats-primary");
    const statsG = document.getElementById("stats-ghost");
    const mcNote = document.getElementById("mc-note");
    statsP.innerHTML = "";
    statsG.innerHTML = "";
    mcNote.style.display = "none";

    if (state.model === "monte_carlo") {
      mcNote.style.display = "block";
      mcNote.textContent = "Monte Carlo is explicit-run only (far slower than the other pricers) — click \"Run Monte Carlo\" above.";
      return;
    }
    try {
      const body = await API.post("/api/pricer/price", {
        model: state.model, params: state.params, spot: state.spot, strike: state.strike,
        maturity: state.maturity, option_type: state.optionType,
      });
      renderPointStats(statsP, "Primary (" + state.model + ")", body);
    } catch (e) { toast("Primary price failed: " + e.message, true); }

    if (state.ghostModel !== "none") {
      try {
        const gbody = await API.post("/api/pricer/price", {
          model: state.ghostModel, params: state.ghostParams, spot: state.spot, strike: state.strike,
          maturity: state.maturity, option_type: state.optionType,
        });
        renderPointStats(statsG, "Ghost (" + state.ghostModel + ")", gbody);
      } catch (e) { toast("Ghost price failed: " + e.message, true); }
    }
  }

  function renderPointStats(container, label, body) {
    container.appendChild(el("div", { class: "stat-tile" }, [
      el("div", { class: "label" }, [label]),
      el("div", { class: "value" }, [fmt(body.price, 4)]),
    ]));
    GREEK_KEYS.forEach((k) => {
      container.appendChild(statTile(k, fmt(body.greeks[k], 4)));
    });
    if (body.stderr !== undefined) {
      container.appendChild(statTile("MC stderr", "±" + fmt(body.stderr, 4)));
    }
    if (body.greeks.vega_raw !== undefined) {
      container.appendChild(statTile("vega_raw (diagnostic)", fmt(body.greeks.vega_raw, 4)));
    }
  }

  async function runMonteCarloPoint() {
    try {
      const body = await API.post("/api/pricer/price", {
        model: "monte_carlo", params: state.params, spot: state.spot, strike: state.strike,
        maturity: state.maturity, option_type: state.optionType,
      });
      const statsP = document.getElementById("stats-primary");
      statsP.innerHTML = "";
      renderPointStats(statsP, "Primary (monte_carlo)", body);
      toast("Monte Carlo run complete (stderr shown).");
    } catch (e) { toast("Monte Carlo run failed: " + e.message, true); }
  }

  async function fetchProfile(model, params) {
    if (model === "none" || model === "monte_carlo") return null;
    return API.post("/api/pricer/profile", {
      model, params, spot: state.spot, strikes: strikeGrid(), maturity: state.maturity, option_type: state.optionType,
    });
  }

  async function refreshProfiles() {
    let primary = null, ghost = null;
    try { primary = await fetchProfile(state.model, state.params); }
    catch (e) { toast("Primary profile failed: " + e.message, true); }
    if (state.ghostModel !== "none") {
      try { ghost = await fetchProfile(state.ghostModel, state.ghostParams); }
      catch (e) { toast("Ghost profile failed: " + e.message, true); }
    }

    const series = [
      { key: "price", primaryY: primary ? primary.prices : null, ghostY: ghost ? ghost.prices : null, title: "Price" },
    ].concat(GREEK_KEYS.map((k) => ({
      key: k,
      primaryY: primary ? primary.greeks[k] : null,
      ghostY: ghost ? ghost.greeks[k] : null,
      title: k,
    })));

    series.forEach((s) => {
      const div = document.getElementById("plot-" + s.key);
      if (!div) return;
      const traces = [];
      if (primary && s.primaryY) {
        traces.push({ x: primary.strikes, y: s.primaryY, mode: "lines", name: "primary", line: { color: COLORS.primary, width: 2 } });
      }
      if (ghost && s.ghostY) {
        traces.push({ x: ghost.strikes, y: s.ghostY, mode: "lines", name: "ghost", line: { color: COLORS.ghost, width: 2, dash: "dash" } });
      }
      const layout = mergedLayout({
        title: { text: s.title, font: { size: 11 } },
        shapes: [{ type: "line", x0: state.strike, x1: state.strike, y0: 0, y1: 1, yref: "paper", line: { color: "#555c6b", dash: "dot" } }],
        showlegend: false,
      });
      Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG);
    });
  }

  const refreshAll = debounce(async () => {
    await refreshStats();
    await refreshProfiles();
  }, 250);

  window.Views.pricer = {
    async init() {
      await loadDefaults(state.model, false);
      render();
      refreshAll();
    },
  };
})();
