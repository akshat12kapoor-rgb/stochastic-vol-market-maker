// Shared helpers used by every view.

// Declared here (loaded first) rather than in app.js (loaded last) --
// each view_*.js file assigns window.Views.<name> at its own load time and
// must find this already present.
window.Views = window.Views || {};

const API = {
  async post(path, body) {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  },
  async get(path, params) {
    const url = new URL(path, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, v);
      });
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  },
};

function toast(message, isError) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ""; }, isError ? 6000 : 3000);
}

function fmt(x, digits) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  digits = digits === undefined ? 2 : digits;
  return Number(x).toFixed(digits);
}

function fmtSigned(x, digits) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const s = fmt(x, digits);
  return x >= 0 ? "+" + s : s;
}

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) {
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    });
  }
  (children || []).forEach((c) => {
    if (c === null || c === undefined) return;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return e;
}

function statTile(label, value, cls) {
  return el("div", { class: "stat-tile" }, [
    el("div", { class: "label" }, [label]),
    el("div", { class: "value" + (cls ? " " + cls : "") }, [value]),
  ]);
}

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "#171a21",
  plot_bgcolor: "#171a21",
  font: { color: "#e6e8ec", size: 11 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: { gridcolor: "#2a2f3a", zerolinecolor: "#3a4050" },
  yaxis: { gridcolor: "#2a2f3a", zerolinecolor: "#3a4050" },
};

function mergedLayout(overrides) {
  return Object.assign({}, PLOTLY_LAYOUT_BASE, overrides || {}, {
    xaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.xaxis, (overrides && overrides.xaxis) || {}),
    yaxis: Object.assign({}, PLOTLY_LAYOUT_BASE.yaxis, (overrides && overrides.yaxis) || {}),
  });
}

const PLOTLY_CONFIG = { displayModeBar: true, displaylogo: false, responsive: true };

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// Diverging red/white/green scale for fill "edge" markers: negative
// (picked off) -> red, zero -> white, positive (spread captured) -> green.
const EDGE_COLORSCALE = [
  [0, "#ff5d5d"],
  [0.5, "#e6e8ec"],
  [1, "#3ddc97"],
];
