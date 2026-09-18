// Tab switching + lazy per-view init. Each view module (view_*.js) defines
// `window.Views.<name> = { init() {...}, activate() {...} }` on `window.Views`.

window.Views = window.Views || {};
const _initialized = {};

// Shared cross-view state: when the Sweep view wants to load a seed into
// the Run view, it stashes the request here and switches tabs.
window.AppState = {
  pendingRunLoad: null, // { seed } or null
};

function activateTab(name) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v.id === "view-" + name);
  });

  const view = window.Views[name];
  if (!view) return;
  if (!_initialized[name]) {
    view.init();
    _initialized[name] = true;
  }
  if (view.activate) view.activate();
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.view));
});

// Called by view_sweep.js when the user clicks a scatter point.
function navigateToRunWithSeed(seed) {
  window.AppState.pendingRunLoad = { seed };
  activateTab("run");
}

// Boot on the default tab.
activateTab("pricer");
