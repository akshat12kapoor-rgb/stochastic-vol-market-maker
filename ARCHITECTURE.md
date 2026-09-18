# Architecture — Public Interfaces

Each agent appends its section when its work is reviewed and accepted.
Read this file before building against another agent's module — do not
read their implementation to infer the interface; if what's written here
is ambiguous or insufficient, stop and flag it rather than guessing.

## `pricing` — Pricing Engine Agent

Status: done. Modules: `pricing/black_scholes.py`, `pricing/monte_carlo.py`,
`pricing/vol_surface.py`, `pricing/api.py` (dispatcher), re-exported from
`pricing/__init__.py`.

Conventions used throughout this package (all functions):
- `maturity` is time-to-expiry in **years** (e.g. 30 calendar days = 30/365).
- `rate`, `vol`, `div_yield` are **annualized, continuously-compounded**
  (e.g. `vol=0.20` means 20% annualized Black-Scholes vol).
- `spot`, `strike` are in the same arbitrary price/currency units.
- `option_type` is the literal string `"call"` or `"put"`.
- Greeks dicts always have exactly the keys `delta, gamma, vega, theta, rho`
  (floats). Units: delta/gamma per 1.0 spot move; vega per 1.0 (100 vol
  points) of vol (divide by 100 for "per vol point"); theta per 1.0 YEAR
  of calendar time decay (divide by 365 for per-day; typically negative
  for long options); rho per 1.0 (100%) of rate (divide by 100 for "per
  1% rate move").

### Model-agnostic entrypoint (preferred for other agents)

```python
from pricing import price
price(model: Literal["black_scholes", "monte_carlo"], params: dict,
      spot: float, strike: float, maturity: float,
      option_type: Literal["call", "put"] = "call") -> tuple[float, dict[str, float]]
```
- `params` required keys: `"rate"` (float), `"vol"` (float).
  Optional: `"div_yield"` (float, default 0.0).
  `monte_carlo`-only optional keys: `"n_paths"` (int, default 100_000),
  `"seed"` (int or None, default 0 — fixed by default for reproducibility).
- Returns `(price, greeks)`: `price` is a float; `greeks` is the dict
  described above.
- Raises `ValueError` for an unrecognized `model` string.
- Never mutates `params`.
- `maturity == 0` is only supported for `black_scholes` (returns intrinsic
  value, greeks zeroed except delta which is 0/±1); `monte_carlo` requires
  `maturity > 0`.

### `pricing.black_scholes`

```python
bs_price(spot, strike, maturity, rate, vol, option_type="call", div_yield=0.0) -> float
bs_greeks(spot, strike, maturity, rate, vol, option_type="call", div_yield=0.0) -> dict[str, float]
```
- Closed-form Black-Scholes-Merton. `bs_price` handles `maturity == 0`
  (intrinsic value) and `vol == 0` (deterministic-forward limit) directly.
- `bs_greeks` **requires** `maturity > 0` and `vol > 0` — raises
  `ValueError` otherwise; callers needing the boundary case should use
  `bs_price` alone or finite-difference near the limit.
- Verified in `tests/test_black_scholes.py`: put-call parity (with and
  without dividends), intrinsic-value limits, and every Greek checked
  against central finite differences.

### `pricing.monte_carlo`

```python
mc_price(spot, strike, maturity, rate, vol, option_type="call", div_yield=0.0,
          n_paths=100_000, antithetic=True, seed=None) -> tuple[float, float]
mc_greeks(spot, strike, maturity, rate, vol, option_type="call", div_yield=0.0,
           n_paths=100_000, seed=0) -> dict[str, float]
```
- `mc_price` returns `(price, stderr)` — `stderr` is the 1-standard-deviation
  Monte Carlo standard error in the same price units, correctly computed
  for the antithetic estimator (paired-average variance, not raw-payoff
  variance — see module docstring). Exact one-step terminal GBM simulation
  is used (no discretization error since only the European terminal payoff
  is needed).
- **Variance reduction: antithetic variates** (default `antithetic=True`).
  For `n_paths` total paths, `n_paths // 2` standard-normal draws `Z` are
  used to build both `Z` and `-Z` paths; each pair's discounted payoffs are
  averaged before computing the mean/stderr across pairs.
- **Cross-validated tolerance**: `tests/test_monte_carlo.py` asserts
  `abs(mc_price - bs_price) < 5 * stderr` (stderr computed from the same
  MC run) across 7 representative configs (ATM/ITM/OTM, short/long
  maturity, call/put, with/without dividends), seed fixed at 42. Empirically
  stderr is ~0.01-0.04 in price units at spot=100 scale for 100k antithetic
  paths; a 5-sigma bound has ~5.7e-7 false-failure probability under the
  CLT normal approximation, so this is a tight but non-flaky bound. See
  the Pricing Engine Agent's final report for the full justification —
  **this tolerance choice needs orchestrator sign-off.**
- `mc_greeks` uses **central-difference bumping with common random numbers**
  (base and bumped simulations reuse identical draws via a fixed seed) to
  keep the finite-difference noise low. This is a heavier/noisier estimate
  than `bs_greeks`; use `bs_greeks` when you don't specifically need
  simulation-based Greeks. Bump sizes: spot ±1% (delta/gamma), vol ±1e-4,
  maturity ±1e-4 (theta = `-dPrice/dMaturity`), rate ±1e-4.
- `price("monte_carlo", params, ...)` (also reachable via `pricing.price`)
  wraps `mc_price` + `mc_greeks` and drops the stderr for API-shape parity
  with `black_scholes.price` — call `mc_price` directly if you need stderr.

### `pricing.vol_surface`

```python
generate_vol_surface(spot, strikes, maturities,
                      atm_vol_short=0.24, atm_vol_long=0.18, term_decay=1.0,
                      skew_short=-0.55, skew_long=-0.10, skew_decay=0.75,
                      smile_short=0.35, smile_long=0.08, smile_decay=0.75,
                      min_iv=0.02) -> pandas.DataFrame
vol_surface_to_long(surface: pd.DataFrame) -> pd.DataFrame   # columns: strike, maturity, iv
vol_surface_to_dict(surface: pd.DataFrame) -> dict[tuple[float, float], float]  # {(strike, maturity): iv}
plot_vol_surface(surface: pd.DataFrame, output_path, title=...) -> pathlib.Path
```
- `generate_vol_surface` returns a DataFrame indexed by `maturity` (years,
  sorted ascending, `index.name == "maturity"`) with `strike` as columns
  (sorted ascending, `columns.name == "strike"`); values are annualized
  IV (e.g. `0.20` == 20%), always `>= min_iv`.
- **Parameterization** (flag for sign-off — see final report): quadratic in
  log-moneyness `m = ln(K / spot)`, with the ATM level, skew coefficient,
  and smile (convexity) coefficient each following an independent
  exponential decay in maturity `T` from a "short" asymptote toward a
  "long" asymptote (e-folding time given by the corresponding `*_decay`
  parameter in years):
  `iv(K,T) = max(atm_iv(T) + skew(T)*m + smile(T)*m^2, min_iv)`.
  This is a hand-tuned qualitative stand-in for an equity-index-like
  surface (steep negative skew short-dated, flattening long-dated,
  convex wings) — it is **not** fit to real market data or guaranteed
  arbitrage-free (no explicit calendar/butterfly no-arb constraints are
  enforced). Downstream calibration agents should treat it as a synthetic
  target, not ground truth.
- `vol_surface_to_long` / `vol_surface_to_dict`: convenience reshapes for
  a calibration routine to consume directly; both are pure functions of
  the DataFrame (no recomputation).
- `plot_vol_surface` uses the `Agg` backend (headless-safe), saves a 3D
  surface PNG, creates parent directories as needed, returns the resolved
  `Path`. Sample output: `data/sample_vol_surface.png` (spot=100, strikes
  60-140 step 5, maturities 1M/2M/3M/6M/1Y/2Y/3Y, default parameters).
- Raises `ValueError` for `spot <= 0`, any non-positive strike, or any
  non-positive maturity.

## `vol_models` — Stochastic Vol Agent

_Not yet started._

## `market_maker` — Market Maker Core Agent

_Not yet started._

## `backtest` — Backtest Engine Agent

_Not yet started._
