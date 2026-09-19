# CLAUDE.md

Options market maker with stochastic volatility modeling (Black-Scholes, Heston,
SABR pricing; a multi-Greek Avellaneda-Stoikov quoting engine; an event-driven
backtester comparing BS-quoting vs Heston-quoting market makers) plus a FastAPI +
Plotly web UI (`webapp/`) over the same engine.

## Orientation

- **`PROJECT.md`** — charter, package layout, shared rules (test-before-done,
  document-interfaces-in-ARCHITECTURE.md, flag-major-decisions).
- **`ARCHITECTURE.md`** — every package's public interface (signatures, units,
  invariants). Read this before calling into a package you didn't just write;
  don't infer an interface from another module's implementation.
- **`notebooks/FINAL_REPORT.md`** — the capstone analysis: 40-seed BS-vs-Heston
  sweep, regime case studies, honest limitations. Read this for "what did we
  learn," not the code.

## Setup

```bash
source .venv/bin/activate   # already created; do not recreate or reinstall
pytest -q                   # 167 tests, should be green
```

Dependencies are pinned in `requirements.txt` (no poetry — wasn't installed
when this was set up, `requirements.txt` + venv was used instead).

## Common entry points

```bash
python -m pricing.vol_surface        # (import only; see pricing/vol_surface.py for the generator)
python -m vol_models.comparison      # BS vs Heston vs SABR vol-surface fit comparison -> data/vol_model_comparison.png
python -m backtest.comparison        # single-path BS-quoting vs Heston-quoting backtest -> data/backtest_pnl_comparison.png
python -m notebooks.sweep_analysis   # 40-seed sweep + regime case studies -> data/sweep_*.png, notebooks/FINAL_REPORT.md
uvicorn webapp.main:app --reload --port 8001   # web UI -> http://localhost:8001/
# port 8000 conflicts with a Docker container on this machine (also uvicorn,
# also returns {"detail":"Not Found"} for "/") -- browsers prefer its IPv6
# listener over this app's IPv4-only one, so "/" 404s if you use 8000.
```

## Key facts worth knowing before changing anything

- The synthetic vol surface (`pricing.vol_surface.generate_vol_surface`) is
  **not arbitrage-checked**. Everything downstream (Heston/SABR calibration,
  the backtest's "true market") ultimately derives from its shape.
- `pricing.api.price` and `vol_models.api.fair_value` share an **identical
  call signature** `(model, params, spot, strike, maturity, option_type) ->
  (price, greeks)` — this is what lets `market_maker` and `backtest` swap
  pricers with zero code changes. Preserve this contract if you touch either.
- **Vega is BS-equivalent (`dPrice/dIV`) across all models**, not each
  model's raw parameter sensitivity — this was a deliberate fix (see git
  history / `ARCHITECTURE.md`'s `vol_models` section) so risk-aversion
  weights in `market_maker.quoting` are comparable regardless of which
  pricer is active. Don't reintroduce a scale mismatch here.
- `market_maker.quoting.DEFAULT_RISK_AVERSION` and `backtest.engine`'s
  arrival-process parameters were empirically tuned for stability (bounded
  inventory, non-diverging P&L) on this specific basket, not derived from
  theory — see `backtest/engine.py`'s module docstring before assuming
  they generalize to a different basket size or horizon.
- The headline backtest result is a **wash** (BS and Heston roughly tie in
  P&L/Sharpe over 40 seeds) driven by a genuine, structural delta difference
  from Heston's calibrated skew — not a bug. See `notebooks/FINAL_REPORT.md`
  §4 before "fixing" an apparent asymmetry between the two quoting runs.
- `webapp/` imports `pricing`/`vol_models`/`market_maker`/`backtest` as a
  library only — it has never modified any of the four, and shouldn't.
  Its quote-waterfall panel re-derives `market_maker.quoting`'s reservation
  price / spread math and is tested to reconstruct the engine's own output
  exactly (`tests/test_webapp_derive.py`,
  `tests/test_webapp_backtest.py::test_backtest_waterfall_reconstructs_engine_quote_exactly`)
  — if you touch `market_maker/quoting.py`'s formulas, that test is the
  canary; don't hand-patch `webapp/derive.py` to make it pass again without
  checking the two are still saying the same thing for the same reason.
- **Never call `POST /api/sweep/recompute` from a test** — it overwrites
  the committed `data/sweep_results.csv`. `tests/test_webapp_sweep.py`
  deliberately doesn't exercise it.

## Git

GitHub remote `origin` → https://github.com/akshat12kapoor-rgb/stochastic-vol-market-maker
(private). One commit per reviewed phase/agent/webapp-view; see
`git log --oneline` for the build history.
