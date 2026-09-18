# Options Market Maker with Stochastic Volatility — Project Charter

Every subagent reads this file before starting and must not deviate from
it without flagging the deviation back to the orchestrator (the user, via
the orchestrating Claude session).

## Goal

Extend an options pricing toolkit (Black-Scholes, Monte Carlo, Greeks) and
an Avellaneda-Stoikov-style quoting engine into a full options market maker
with stochastic volatility models (Heston, SABR), backtested on synthetic
data. Compare a Black-Scholes-quoting market maker against a
stochastic-vol-quoting market maker on P&L, Sharpe, drawdown, and inventory
risk.

No prior code exists yet in this repo — everything is being built fresh as
part of this project (there is no legacy pricer/AS engine to port in).

## Package layout

```
pricing/       Black-Scholes, Monte Carlo European pricer, synthetic vol surfaces
vol_models/    Heston, SABR, calibration routines
market_maker/  Inventory-aware quoting engine (delta/vega/gamma skew)
backtest/      Event-driven backtest engine, order arrival simulation
data/          Synthetic market data / generated vol surfaces
notebooks/     Analysis and writeups per phase
tests/         pytest suite (mirrors package structure)
```

## Stack

Python 3.14.6, local `.venv`, dependencies pinned in `requirements.txt`:
numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, numba 0.67.0,
pytest 9.1.1. Activate with `source .venv/bin/activate` before running
anything.

## Rules every agent follows

1. **Write tests before marking work done.** Tests live under `tests/`,
   mirroring the module path (e.g. `pricing/black_scholes.py` →
   `tests/test_black_scholes.py`). Run the full suite (`pytest -q`) and
   confirm it's green before reporting completion.
2. **Document your public interface in `ARCHITECTURE.md`.** Function
   signatures, argument shapes/units, return shapes, and any invariants
   the caller can rely on — written so another agent can integrate against
   it *without reading your implementation*.
3. **Do not silently make a major modeling choice.** Calibration optimizer
   choice, variance reduction technique, arrival process model, numerical
   scheme for SDE discretization, etc. — flag these to the orchestrator
   instead of picking silently. Pick a reasonable default, implement it,
   but call it out explicitly as a decision point in your final report.
4. **Stay in your scope.** Only touch your assigned package directory plus
   `tests/<your module>`, `ARCHITECTURE.md` (append-only, your section),
   and read (never write) other agents' already-completed modules.
5. **If you depend on another agent's interface that doesn't exist yet**,
   build against the documented interface in `ARCHITECTURE.md`, not
   against assumptions. If the interface is undocumented or ambiguous,
   stop and flag it rather than guessing.

## Agent roles and dependency order

1. **Pricing Engine Agent** (`/pricing`) — no dependencies. Runs first,
   alone.
2. **Stochastic Vol Agent** (`/vol_models`) — depends on Agent 1's
   synthetic vol surface + pricer API. Runs in parallel with Agent 3.
3. **Market Maker Core Agent** (`/market_maker`) — depends on Agent 1's
   `fair_value`-shaped API (stubs Agent 2's interface from
   `ARCHITECTURE.md` until Agent 2 reports done). Runs in parallel with
   Agent 2.
4. **Backtest Engine Agent** (`/backtest`) — depends on Agent 2 and
   Agent 3 both being done and reviewed.
5. **Integration & Analysis Agent** (`/notebooks`, final report) — depends
   on all others being done. Runs full suite, produces final comparison.

The orchestrator reviews each agent's output (deliverable, test results,
flagged decisions) with the user before the next dependent agent is
dispatched.
