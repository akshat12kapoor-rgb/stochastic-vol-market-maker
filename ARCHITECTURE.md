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

Status: done. Modules: `vol_models/heston.py` (COS pricing + full-truncation-
Euler Monte Carlo), `vol_models/sabr.py` (Hagan implied-vol formula),
`vol_models/calibration.py` (least-squares calibration to a vol surface),
`vol_models/api.py` (`fair_value` dispatcher), `vol_models/greeks_utils.py`
(shared finite-difference Greeks + BS implied-vol inversion helpers),
re-exported from `vol_models/__init__.py`. Report-generation script (not a
test): `vol_models/comparison.py`; write-up: `vol_models/CALIBRATION_NOTES.md`.

Conventions match the `pricing` package throughout: `maturity` in years,
`rate`/`vol`-like params annualized continuously-compounded, `option_type`
is `"call"`/`"put"`, greeks dicts always have at least the keys `delta,
gamma, vega, theta, rho` in the exact units `pricing.black_scholes.bs_greeks`
uses, **including `vega`**: it is defined as the Black-Scholes-equivalent
`dPrice/dIV` for every model (`heston`, `sabr`, and `pricing`'s own
`black_scholes`/`monte_carlo`), so it is directly comparable/safe to use
with a single scalar risk weight regardless of which pricer is active —
see "Vega convention" below for how this is computed and why it replaced
an earlier, non-comparable convention. Both `heston_greeks` and
`sabr_greeks` also return one extra, non-required key, `"vega_raw"` — a
diagnostic, model-parameter-bump sensitivity that is NOT on the same scale
as `"vega"` and should not be used by any comparability-sensitive caller
(e.g. `market_maker`'s book-level `vega_book`/`risk_aversion.vega`).

### Model-agnostic entrypoint (preferred for other agents — matches `pricing.api.price`'s shape)

```python
from vol_models import fair_value
fair_value(model: Literal["heston", "sabr"], params: dict,
           spot: float, strike: float, maturity: float,
           option_type: Literal["call", "put"] = "call") -> tuple[float, dict[str, float]]
```
- `"heston"` params — required: `"kappa"` (mean-reversion speed, float >0),
  `"theta"` (long-run variance, float >0), `"sigma"` (vol-of-vol, float >0),
  `"rho"` (spot/vol correlation, float in (-1,1)), `"v0"` (initial
  variance, float >0), `"rate"` (float). Optional: `"div_yield"` (default
  `0.0`), `"n_terms"` (COS series length, default `256`), `"L"` (COS
  truncation width in std devs, default `12.0`).
- `"sabr"` params — required: `"alpha"` (vol-of-forward level, float >0),
  `"rho"` (float in (-1,1)), `"nu"` (vol-of-vol, float >0), `"rate"`
  (float). Optional: `"beta"` (CEV exponent, default `1.0` — see "SABR
  beta convention" below), `"div_yield"` (default `0.0`).
- Returns `(price, greeks)`, same shape as `pricing.api.price`.
- Raises `ValueError` for an unrecognized `model` string, and for
  `maturity <= 0` (unlike `pricing.api.price`, there is **no** `maturity
  == 0` special case here — both the COS method and Hagan's formula
  require `T > 0`). Raises `KeyError` (not `ValueError`) if a required
  `params` key is missing, naming the missing keys.
- Never mutates `params`. Fully deterministic (no RNG in the dispatch
  path — `heston_mc_price` is a separate, non-dispatched function, see
  below).

### `vol_models.heston`

```python
feller_condition(kappa, theta, sigma) -> bool   # True iff 2*kappa*theta >= sigma**2

heston_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
             option_type="call", div_yield=0.0, n_terms=256, L=12.0) -> float

heston_greeks(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
              option_type="call", div_yield=0.0, n_terms=256, L=12.0) -> dict[str, float]

heston_mc_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                 option_type="call", div_yield=0.0, n_paths=50_000, n_steps=200,
                 antithetic=True, seed=0) -> tuple[float, float]
```
- **Pricing method (flagged — needs sign-off):** semi-analytic pricing via
  the **COS method** (Fang & Oosterlee, 2008), not Carr-Madan. Chosen
  because it needs only real-`u` characteristic-function evaluations (no
  numerical integration/quadrature over a damping parameter), converges
  spectrally in `n_terms` for smooth densities, and is a single formula
  that handles call/put via different payoff-coefficient sub-intervals.
  Characteristic function uses the **"little trap" parameterization**
  (Albrecher, Mayer, Schoutens & Tistaert, 2007) rather than the original
  1993 Heston formula, to avoid a branch-cut discontinuity in the complex
  logarithm for long maturities / large vol-of-vol.
- **Truncation range (flagged — needs sign-off):** `[a,b]` is centered on
  the first two cumulants of `ln(S_T/S_0)`, computed **numerically** as
  finite differences of the cumulant-generating function built from the
  same characteristic function used for pricing (not the closed-form
  Heston cumulant expressions, to avoid transcription risk) — step size
  `h=1e-2`, chosen empirically to balance truncation error against
  floating-point cancellation (see module source comment; `h=1e-4` loses
  all precision on the second cumulant for near-deterministic params).
  Default width is `L=12` standard deviations, `n_terms=256` cosine terms;
  `calibration.py` uses cheaper `n_terms=128/96, L=12` during the
  optimization loop for speed. Verified to reproduce `pricing.bs_price` to
  <1e-4 in the `sigma->0, v0=theta=vol**2` deterministic limit, and to
  agree with `heston_mc_price` within the tolerance in
  `tests/test_heston.py` across ATM/ITM/OTM and Feller-satisfied/violated
  configs.
- **Feller condition handling:** `feller_condition` is purely informational
  (returns whether `2*kappa*theta >= sigma**2`, i.e. whether the
  continuous-time CIR variance process is guaranteed to stay positive).
  Calibrated params routinely violate it — `heston_mc_price` handles this
  via **full-truncation Euler discretization** (Lord, Koekkoek & van Dijk,
  2010): at every step, `v_pos = max(v, 0)` is used everywhere `v` would
  feed a square root or the current-level term of the drift, while the
  *state* `v` itself is carried forward unfloored and may go transiently
  negative — this guarantees no `sqrt` of a negative number and no NaN/
  complex values regardless of the Feller condition. See
  `tests/test_heston.py::test_heston_mc_survives_severe_feller_violation`.
  Full-truncation Euler carries a first-order-in-`dt` discretization bias
  in addition to Monte Carlo sampling noise (bias grows with vol-of-vol
  and severity of Feller violation) — empirically ~0.12 in price units at
  `n_steps=100` for a stress config (`kappa=1, theta=0.05, sigma=1.2`),
  shrinking to ~0.02-0.03 by `n_steps=300-600`; default `n_steps=200`
  trades this off against runtime (see `heston_mc_price` docstring for the
  full numbers).
- **Variance reduction:** antithetic variates (default `antithetic=True`),
  same convention/stderr computation as `pricing.monte_carlo.mc_price`
  (paired-average variance across `(Z,-Z)` path pairs, with the sign flip
  applied to both the spot and variance driving noise).
- `heston_mc_price` raises `ValueError` if `maturity <= 0`. Returns
  `(price, stderr)`; `stderr` is the 1-sigma Monte Carlo standard error in
  price units.
- **Vega convention (revised — resolved, no longer needs sign-off):**
  `heston_greeks`'s `"vega"` is the **Black-Scholes-equivalent**
  `dPrice/dIV`: invert `heston_price`'s own output to an implied vol via
  `vol_models.greeks_utils.implied_vol_from_price`, then evaluate
  `pricing.black_scholes.bs_greeks(...,vol=iv,...)["vega"]`
  (`vol_models.greeks_utils.bs_equivalent_vega`). Since `iv` is defined by
  `bs_price(...,iv,...) == heston_price(...)`, the chain rule makes this
  exact — no finite-differencing of Heston's own parameters. This
  replaces an earlier `dPrice/d(sqrt(v0))` convention that was flagged and
  found **not** comparable to SABR's or Black-Scholes' vega on realistic
  params (differed by ~2x for otherwise-matched ATM configs) — `vega` now
  means the same thing ("price move per unit change in the option's own
  BS-equivalent implied vol") across `black_scholes`/`monte_carlo`/
  `heston`/`sabr`, which is required for `market_maker`'s single
  `risk_aversion.vega` weight applied to `vega_book` to mean the same
  thing regardless of which pricer generated the book's Greeks. The old
  `dPrice/d(sqrt(v0))` sensitivity is still available under the key
  `"vega_raw"` (central difference, bump `1e-4`) for anyone who
  specifically wants it — it is not on the same scale as `"vega"` and
  should not be substituted for it. `delta`/`gamma` bump spot ±1%,
  `theta` bumps maturity ±1e-4 (`theta = -dPrice/dMaturity`), `rho` bumps
  rate ±1e-4 — unchanged, all holding every other Heston param fixed
  ("sticky-model-parameter" Greeks, not sticky-strike/sticky-delta).
  Verified in `tests/test_heston.py::test_heston_vega_is_bs_equivalent_by_construction`
  and, cross-model, in
  `tests/test_sabr.py::test_vega_is_comparable_across_black_scholes_heston_sabr_near_atm`
  (BS/Heston/SABR vega now agree to <5% for matched near-ATM implied
  vols, vs. the old convention's ~2x gap in the analogous case).

### `vol_models.sabr`

```python
sabr_implied_vol(forward, strike, maturity, alpha, beta, rho, nu) -> float | np.ndarray
sabr_price(spot, strike, maturity, rate, alpha, beta, rho, nu,
           option_type="call", div_yield=0.0) -> float
sabr_greeks(spot, strike, maturity, rate, alpha, beta, rho, nu,
            option_type="call", div_yield=0.0) -> dict[str, float]
```
- Hagan, Kumar, Lesniewski & Woodward (2002) lognormal SABR asymptotic
  implied-vol formula (general `beta`, not a special-cased `beta=1` or
  `beta=0` formula) — reduces smoothly to the closed-form ATM vol as
  `strike -> forward` (the `z/x(z) -> 1` limit is applied wherever
  `|z| < 1e-8`; no separate ATM branch/division-by-zero). `sabr_price`
  converts that vol to a price via `pricing.black_scholes.bs_price`
  (imported, not reimplemented) using `forward = spot *
  exp((rate-div_yield)*maturity)`; raises `ValueError` if `maturity <= 0`.
  `sabr_implied_vol`'s `forward`/`strike`/`maturity` args are scalar-or-
  broadcastable-array (vectorized, no loop needed for a whole strike/
  maturity grid).
- **SABR beta convention (flagged — needs sign-off):** `beta` defaults to
  `1.0` (lognormal SABR — forward dynamics `dF = alpha*F*dW`) everywhere
  in this module and in `calibrate_sabr`, and is **not calibrated by
  default** (only `alpha, rho, nu` are fit; `beta` is a fixed input).
  Reasoning: beta and rho are close to unidentifiable from a single vol
  smile (both drive skew), so the standard practitioner approach is to
  fix beta from an external view and calibrate the rest; beta=1 is the
  natural match for an equity-index-style, BS-vol-quoted surface (what
  `pricing.vol_surface` produces). `beta` is a free function argument
  throughout (not hardcoded), so callers needing e.g. `beta=0.5` (a
  common rates convention) can pass it directly — `calibrate_sabr` does
  not currently support calibrating `beta` itself.
- **Vega convention (revised — resolved, no longer needs sign-off):**
  `sabr_greeks`'s `"vega"` is the same **Black-Scholes-equivalent**
  `dPrice/dIV` as `heston_greeks` (via `bs_equivalent_vega`) — see the
  Heston section above for the full rationale and the cross-model
  verification test. The old `dPrice/dAlpha` sensitivity (alpha is SABR's
  vol-of-forward level, the closest per-model analog to a BS vol, but not
  identical to it in general) is still available under `"vega_raw"`
  (central difference, bump `1e-4`) — note that for beta=1 near the money
  `vega_raw` and `vega` can sit close together numerically (alpha
  approximates the ATM vol when beta=1), which is expected, not a
  contradiction; they diverge further from the money and for beta<1.
  `delta`/`gamma`/`theta`/`rho` bump conventions are unchanged and
  identical to `heston_greeks` (spot ±1%, maturity ±1e-4 with
  `theta=-dPrice/dMaturity`, rate ±1e-4), holding alpha/beta/rho/nu fixed
  ("sticky-strike-in-alpha", recomputing the forward and Hagan vol at each
  bumped spot).

### `vol_models.calibration`

```python
calibrate_heston(surface, spot, rate=0.0, div_yield=0.0, initial_guess=None,
                  weights="equal", n_terms=128, L=12.0, max_nfev=150, verbose=0) -> dict
calibrate_sabr(surface, spot, rate=0.0, div_yield=0.0, beta=1.0, initial_guess=None,
               weights="equal", max_nfev=200, verbose=0) -> dict
calibrate(model: Literal["heston", "sabr"], surface, spot, **kwargs) -> dict
surface_rmse(model_iv: array-like, target_iv: array-like) -> float
```
- `surface`: a `pricing.vol_surface.generate_vol_surface`-shaped DataFrame
  (maturity-indexed rows, strike columns) — reshaped internally via
  `pricing.vol_surface.vol_surface_to_long`. Both functions calibrate a
  single **global** parameter set across every (strike, maturity) point in
  `surface` — see the "global vs per-maturity SABR" caveat below.
- Returns a dict directly usable as `vol_models.fair_value`'s `params`
  argument for that model (includes `"rate"`/`"div_yield"` echoing the
  inputs, plus `"beta"` for SABR).
- **Optimizer (flagged — needs sign-off):** `scipy.optimize.least_squares`
  with `method="trf"` (bounded trust-region-reflective), minimizing
  per-grid-point residuals `weight_i * (model_iv_i - target_iv_i)` in
  implied-vol space (not price space) — for Heston this requires inverting
  each COS price back to a BS implied vol via
  `vol_models.greeks_utils.implied_vol_from_price` (Brent's method against
  `pricing.bs_price`; grid points where no vol in `[1e-4, 5.0]` attains the
  model price are assigned a fixed penalty residual of `1.0`, not dropped
  or NaN'd, so the optimizer is pushed away from that region rather than
  crashing). SABR needs no inversion (`sabr_implied_vol` gives IV
  directly), so `calibrate_sabr` has no price-inversion step and is
  correspondingly much faster. `trf` (not unconstrained `lm`) was chosen
  specifically so hard parameter-feasibility bounds (kappa/theta/sigma/v0/
  alpha/nu > 0, `|rho| < 1`) can be passed as `bounds` directly.
- **Weighting scheme (flagged — needs sign-off):** `weights="equal"`
  (default) weights every grid point equally in the least-squares
  objective. `weights="vega"` is also implemented — weights by
  `pricing.black_scholes.bs_greeks(..., vol=target_iv)["vega"]` normalized
  to mean 1, which down-weights deep-OTM wings where a given price error
  maps to a large IV error. Equal-weight is the default because the
  target surface is a smooth synthetic construction with no bid/ask
  liquidity signal to weight by; this is a judgment call PROJECT.md rule 3
  asks to flag rather than pick silently.
- **Initial guess / bounds:** if `initial_guess` is omitted, Heston seeds
  `kappa=1.5, sigma=0.6, rho=-0.5`, `theta=v0=mean(target_iv)**2`; SABR
  seeds `alpha=mean(target_iv), rho=-0.3, nu=0.4`. Bounds: Heston
  `kappa∈[0.05,15], theta∈[1e-4,4], sigma∈[0.02,3], rho∈[-0.999,0.999],
  v0∈[1e-4,4]`; SABR `alpha∈[1e-4,5], rho∈[-0.999,0.999], nu∈[1e-4,5]`
  (beta fixed, not bounded/calibrated).
- **Global vs per-maturity SABR (flagged as a modeling limitation, not
  just a decision):** `calibrate_sabr` fits **one** `(alpha, rho, nu)`
  triple across the whole surface, not a separate triple per maturity
  (which is how SABR is normally used in practice — it has no built-in
  term structure). This was chosen to match `fair_value`'s single-flat-
  `params`-dict contract (one dict per model, no maturity-indexed lookup)
  so the Market Maker agent can swap models without restructuring its
  `params` handling. Consequence: SABR's fit quality degrades markedly
  away from whichever maturity the global fit "centers" on — see
  `CALIBRATION_NOTES.md` for the actual RMSE-by-maturity breakdown. Heston
  does not have this limitation (kappa/theta mean-reversion gives it
  genuine, if constrained, term structure).
- `surface_rmse(model_iv, target_iv)`: plain `sqrt(mean((model_iv -
  target_iv)**2))` in implied-vol units (e.g. `0.01` == 1 vol point). Pure
  function, no fitting.

### `vol_models.greeks_utils`

```python
finite_diff_greeks(base_price, price_given_spot, price_given_rate,
                    price_given_maturity, price_given_vol_level,
                    spot, rate, maturity, vol_level) -> dict[str, float]
bs_equivalent_vega(price, spot, strike, maturity, rate,
                    option_type="call", div_yield=0.0) -> float
implied_vol_from_price(price, spot, strike, maturity, rate,
                        option_type="call", div_yield=0.0, lo=1e-4, hi=5.0) -> float
```
- `finite_diff_greeks`: shared central-difference Greeks helper used by
  both `heston_greeks` and `sabr_greeks` — takes four one-argument pricing
  closures (each holding every other parameter fixed) and returns
  `delta,gamma,vega_raw,theta,rho` (**not** `"vega"` — see
  `bs_equivalent_vega` immediately below; this function's own "vega-like"
  output is deliberately named `vega_raw` since it's a raw bump on
  whatever `vol_level` closure the caller passed in, not necessarily
  comparable across models). No common-random-numbers trick needed
  (unlike `pricing.monte_carlo.mc_greeks`) since both `heston_price` (COS)
  and `sabr_price` (Hagan formula) are deterministic.
- `bs_equivalent_vega`: the function both `heston_greeks` and
  `sabr_greeks` call to fill in their actual `"vega"` key — inverts
  `price` to an implied vol via `implied_vol_from_price`, then returns
  `pricing.black_scholes.bs_greeks(...,vol=iv,...)["vega"]`. Exact by the
  chain rule (`dPrice/dIV` where `iv` is defined by `bs_price(...,iv,...)
  == price`), so this is what makes `"vega"` comparable across
  `black_scholes`/`heston`/`sabr` — see the "Vega convention" notes above.
- `implied_vol_from_price`: Brent's method (`scipy.optimize.brentq`)
  inverting `pricing.black_scholes.bs_price` for the vol that reproduces a
  given `price`. Raises `ValueError` if `price` is not attainable by any
  vol in `[lo, hi]` — calibration callers catch this and assign a penalty
  residual (see `calibration.py` above) rather than letting it propagate.

### Comparison report (`vol_models/comparison.py`, `vol_models/CALIBRATION_NOTES.md`)

`python -m vol_models.comparison` (or `run_comparison(...)`) calibrates
Heston and SABR against `pricing.vol_surface.generate_vol_surface`'s
**default-parameter** surface (`spot=100`, `strikes=70..130 step 5` (13
strikes), `maturities=[1M,3M,6M,1Y,2Y,3Y]`, `rate=0.03` and `div_yield=0.0`
assumed — flagged in the module docstring, since the target IV surface
itself carries no rate), computes a single "calibrated" constant BS vol
(closed-form: the mean of all target IVs, the equal-weight-least-squares-
optimal constant), and saves a 2x3 grid of per-maturity smile plots (target
vs BS vs Heston vs SABR) to `data/vol_model_comparison.png`. Not part of
`pytest` — it is a report script, not a correctness test.

**Results on this grid** (full write-up with the strike/maturity error
breakdown in `vol_models/CALIBRATION_NOTES.md`): IV RMSE — BS (flat) =
0.0698, Heston (calibrated) = 0.0147, SABR (calibrated, global) = 0.0339.
BS mis-prices worst at short-dated, far-OTM-put strikes (single worst
point: K=70, T=1M, error 0.235 in IV) where the synthetic surface's
negative skew and positive smile convexity are both steepest, and is
closest to correct near-the-money at longer maturities where the surface's
skew/smile has decayed toward its flatter long-maturity asymptote.

## `market_maker` — Market Maker Core Agent

Status: done. Modules: `market_maker/book.py` (inventory/state), `market_maker/quoting.py`
(multi-Greek Avellaneda-Stoikov-style quoting engine), re-exported from
`market_maker/__init__.py`.

### Pricer contract (integration checkpoint with `vol_models`)

Every function that needs a fair value takes a `pricer` argument that must
be a callable matching **exactly** `pricing.api.price`'s signature:

```python
pricer(model: str, params: dict, spot: float, strike: float,
       maturity: float, option_type: str = "call") -> tuple[float, dict[str, float]]
```

- Returns `(price, greeks)`; `greeks` has keys `delta, gamma, vega, theta, rho`
  with the exact unit conventions documented in the `pricing` section above
  (this module relies on those conventions, e.g. vega = dPrice/dVol per 1.0
  = 100 vol points).
- Must not mutate `params`.

`market_maker/quoting.py` defines this as a structural `Protocol` (`PricerFn`)
— nothing in the package imports `pricing` or hardcodes Black-Scholes.
Today all tests instantiate the pricer as `pricing.api.price` with
`model="black_scholes"`. **Once the Stochastic Vol Agent exposes a
`fair_value` function with this same signature for `model in {"heston",
"sabr"}`, it should be usable as the `pricer` argument here with zero code
changes** — pass `model="heston"`/`"sabr"` and that model's own `params`
dict through `generate_quotes` unchanged. This is the integration
checkpoint the orchestrator should verify once both agents are done.

### `market_maker.book`

```python
OptionType = Literal["call", "put"]

@dataclass(frozen=True)
class OptionContract:
    contract_id: str
    strike: float
    maturity: float       # years, same convention as pricing.api.price
    option_type: OptionType = "call"

@dataclass
class Book:
    contracts: dict[str, OptionContract]   # quotable universe
    inventory: dict[str, float]            # signed qty; +long / -short; missing key == 0.0

    add_contract(contract: OptionContract, quantity: float = 0.0) -> None
    set_inventory(contract_id: str, quantity: float) -> None          # raises KeyError if unregistered
    adjust_inventory(contract_id: str, delta_quantity: float) -> float  # returns new total
    get_inventory(contract_id: str) -> float                          # 0.0 if unset
    contract_ids() -> list[str]
    is_flat() -> bool                                                 # True iff every inventory == 0.0
```

- Inventory units are **option contracts held**, not underlying shares —
  a position of `qty=1.0` in a contract with `delta=0.5` contributes 0.5
  underlying-equivalent units of delta exposure (computed by
  `compute_book_greeks`/`generate_quotes`, not stored directly).
- `Book` is a plain mutable data container; not thread-safe.

### `market_maker.quoting`

```python
class PricerFn(Protocol):
    def __call__(self, model: str, params: dict, spot: float, strike: float,
                 maturity: float, option_type: str = "call") -> tuple[float, dict[str, float]]: ...

ParamsSource = dict | Callable[[OptionContract], dict]   # per-contract params, e.g. vol surface lookup

@dataclass(frozen=True)
class RiskAversion:
    delta: float = 0.1   # >= 0, raises ValueError otherwise
    vega: float = 0.1    # >= 0
    gamma: float = 0.0   # >= 0, optional/off by default — see decisions below

@dataclass(frozen=True)
class Quote:
    contract_id: str
    fair_value: float
    reservation_price: float
    bid: float
    ask: float             # ask - bid == spread, always > 0
    spread: float
    greeks: dict[str, float]

price_contract(pricer, model, params: ParamsSource, spot, contract: OptionContract) -> tuple[float, dict]
price_book(book, pricer, model, params: ParamsSource, spot) -> dict[str, tuple[float, dict]]
compute_book_greeks(book, pricer, model, params: ParamsSource, spot,
                     valuations=None) -> dict[str, float]   # keys: delta,gamma,vega,theta,rho; inventory-weighted sum

reservation_price(fair_value, delta_book, vega_book, sigma_underlying, sigma_vol,
                   horizon, risk_aversion: RiskAversion) -> float

quote_spread(delta_c, vega_c, sigma_underlying, sigma_vol, horizon, risk_aversion: RiskAversion,
             order_arrival_kappa: float = 1.5, gamma_c: float = 0.0, spot: float = 0.0) -> float
             # raises ValueError if order_arrival_kappa <= 0

generate_quotes(book: Book, pricer, model: str, params: ParamsSource, spot: float,
                 sigma_underlying: float, risk_aversion: RiskAversion = RiskAversion(),
                 sigma_vol: float = 0.5, horizon: float | None = None,
                 order_arrival_kappa: float = 1.5) -> dict[str, Quote]
```

**Formulas** (generalizing classic Avellaneda-Stoikov `r = s - q*gamma*sigma^2*(T-t)`,
`spread ~ gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/kappa)` to a multi-Greek
options book — see decision list below):

- `delta_book = sum(qty_i * delta_i)`, `vega_book = sum(qty_i * vega_i)` over
  every contract currently held in `book` (inventory-weighted book-level
  exposure; generalizes AS's single `q`).
- `reservation_price = fair_value - risk_aversion.delta * delta_book * sigma_underlying**2 * horizon
  - risk_aversion.vega * vega_book * sigma_vol**2 * horizon`.
- `spread = risk_spread + liquidity_spread [+ gamma_term]`, where
  `risk_spread = 2*(risk_aversion.delta*|delta_c|*sigma_underlying**2*horizon
  + risk_aversion.vega*|vega_c|*sigma_vol**2*horizon)` uses the **quoted
  contract's own** Greeks (not the book aggregate);
  `liquidity_spread = (2/g)*ln(1+g/order_arrival_kappa)` with
  `g = risk_aversion.delta + risk_aversion.vega` (g→0 limit handled exactly
  as `2/order_arrival_kappa`); `gamma_term` (only if `risk_aversion.gamma > 0`)
  = `risk_aversion.gamma * |gamma_c| * (sigma_underlying*spot)**2 * horizon`,
  spread-widening only, no reservation-price skew.
- `horizon` defaults to each contract's own `maturity` if not passed.
- `sigma_underlying` is a required, explicit argument (not read out of
  `params`) since not every pricer's `params` dict has a scalar `"vol"` key
  (e.g. Heston's params are `v0/kappa/theta/xi/rho`) — this keeps the
  quoting engine pricer-agnostic.
- `sigma_vol` (vol-of-vol) defaults to `0.5` — a placeholder, see decisions.

**Invariants:**
- If `book.is_flat()` (every inventory exactly 0), `reservation_price ==
  fair_value` exactly for every contract, so bid/ask are symmetric around
  fair value (`fair_value - bid == ask - fair_value`). Verified in
  `tests/test_quoting.py::test_zero_inventory_produces_symmetric_quotes_around_fair_value`.
- `bid < ask` always (`quote_spread` is strictly positive given
  `order_arrival_kappa > 0`).
- `generate_quotes` never mutates `book` or `params`.
- Positive `delta_book`/`vega_book` (net long) **lowers** the reservation
  price for every contract in the book, pushing both bid and ask down —
  this makes the market maker's ask more attractive to hit (encourages
  being sold to, i.e. shedding the long position). Symmetric for short
  inventory. Verified directionally in
  `tests/test_quoting.py::test_long_delta_inventory_skews_quotes_down_to_encourage_selling`,
  `test_short_delta_inventory_skews_quotes_up_to_encourage_buying`,
  `test_long_vega_inventory_skews_quotes_down_independent_of_delta`.

### Decisions flagged for orchestrator sign-off

Per PROJECT.md rule 3 (see `market_maker/quoting.py` module docstring for
the full reasoning inline with the code):

1. **Per-Greek risk aversion, not one scalar.** `RiskAversion(delta, vega,
   gamma=0.0)` replaces AS's single `gamma`. Delta risk (directional) and
   vega risk (vol-of-vol) are treated as economically distinct exposures a
   desk would want to limit independently — this is a design choice, not
   a literature result.
2. **Additive, uncorrelated combination of delta-risk and vega-risk terms**
   in both the reservation price and the spread. This ignores any
   spot-vol cross-correlation (e.g. equity skew: spot down often means vol
   up) that would add a cross term to a true portfolio-variance
   calculation. Deliberately dropped for a first pass — flagged as a
   simplification a future revision may want to address (would need a
   spot-vol correlation input, which isn't available from `pricing` today
   either).
3. **`sigma_vol` (vol-of-vol) has no data source yet** — defaults to a
   placeholder `0.5` (50% annualized). Once `vol_models` lands with
   Heston's `xi` or SABR's `alpha`, the natural follow-up is deriving
   `sigma_vol` from that instead of a static placeholder; not done now
   since `vol_models` isn't available yet (rule 5: build against the
   documented interface, don't guess at internals).
4. **Liquidity spread term's `g = risk_aversion.delta + risk_aversion.vega`**
   — a simple sum used only to plug a single scalar into AS's classic
   `(2/gamma)*ln(1+gamma/kappa)` liquidity term. Arbitrary combination,
   called out separately from decision #1.
5. **Order-arrival assumption**: `order_arrival_kappa` assumes fill
   intensity decays as `A*exp(-kappa*quote_distance)` (classic AS/
   Guéant microstructure assumption), exposed as a free parameter
   (default `1.5`) with **no actual order-arrival process implemented
   here** — that belongs to the Backtest Agent. Flagging that the Backtest
   Agent should confirm whether `order_arrival_kappa` should be calibrated
   jointly with its real arrival-process model, or whether this term
   should be replaced once that model exists.
6. **Optional gamma/convexity risk-aversion term** (`RiskAversion.gamma`,
   off by default) is a lower-confidence add-on included only because
   PROJECT.md's package blurb mentions "delta/vega/gamma skew" for this
   module; its `(sigma_underlying * spot)**2` dimensional scaling is a
   judgment call, not derived, and it only widens spread (no skew
   semantics, since long gamma isn't a risk to offload the way long
   delta/vega are). Treat as a documented stub pending review, not a
   validated model.

## `backtest` — Backtest Engine Agent

Status: done. Modules: `backtest/market_sim.py` (Heston realized-market
path generator), `backtest/arrivals.py` (Poisson order-arrival process),
`backtest/quoter_calibration.py` (derives each run's quoting-pricer
params from the true market — implements the central framing decision,
see below), `backtest/engine.py` (`run_backtest`, the event-driven loop),
`backtest/results.py` (`BacktestResults`, Sharpe/drawdown, plotting),
re-exported from `backtest/__init__.py`. Report-generation script (not a
test, mirrors `vol_models/comparison.py`'s convention):
`backtest/comparison.py` (`python -m backtest.comparison`), output:
`data/backtest_pnl_comparison.png`.

### THE CENTRAL FRAMING DECISION (flagged — needs sign-off)

If the Heston-quoting run were handed the exact params that generate the
simulated "true" market, it would have perfect model knowledge no real
market maker has. **Decision: option (b) from the task brief.** The
Heston quoter's params are NOT `backtest.market_sim.TRUE_MARKET_PARAMS`.
Instead (`backtest/quoter_calibration.py`):

1. A synthetic implied-vol surface is built "as if observed" from the
   true market: price a 7-strike x 4-maturity grid under
   `TRUE_MARKET_PARAMS` via `vol_models.api.fair_value`, invert each price
   to a BS implied vol via `vol_models.greeks_utils.implied_vol_from_price`,
   and add `N(0, 0.005²)` IV noise (emulating bid/ask/staleness noise a
   desk would actually see — fixed seed for reproducibility).
2. `vol_models.calibration.calibrate_heston` fits Heston to that noisy
   surface — the same routine Agent 2 built, used the same way a
   downstream desk would use it.
3. The resulting (imperfectly fit) params are used for the **entire**
   backtest — calibrated once at t=0, never re-calibrated/filtered as the
   market evolves (no online re-estimation of `v0` or anything else — a
   real desk periodically recalibrates; that's out of scope here and
   flagged, not solved).

The Heston quoter therefore keeps the advantage of being the **right
model family** (mean-reverting stochastic vol, correct skew/smile
structure) but not the exact numbers — the realistic position a real
quant desk is in. Any P&L/Sharpe/drawdown edge Heston shows in this setup
is attributable to "right model family + reasonable calibration", not
omniscience, and should be read as a **lower bound** on what a
perfectly-informed Heston quoter could achieve — this materially changes
how Phase 6 should narrate "why did Heston do better/worse": it's a
statement about model-family + calibration quality, not about Heston's
theoretical ceiling.

The **BS quoter's flat vol** is, symmetrically, the true market's own ATM
(`strike == spot0`) implied vol at a representative maturity (180d),
computed once at t=0 and held fixed — the most defensible single-number
choice for a flat-vol desk, so BS's performance reflects its structural
inability to price skew/smile/term-structure, not a badly-chosen vol
level.

### Market simulator (`backtest.market_sim`)

```python
@dataclass(frozen=True)
class HestonMarketParams:
    spot0, v0, kappa, theta, sigma, rho, rate: float
    div_yield: float = 0.0
    def as_dict(self, v0_override: float | None = None) -> dict: ...

TRUE_MARKET_PARAMS: HestonMarketParams   # the one "ground truth" market

@dataclass(frozen=True)
class MarketPath:
    times, spot, variance: np.ndarray   # shape (n_steps+1,)
    dt: float
    seed: int

simulate_heston_path(params: HestonMarketParams, n_steps: int, dt: float, seed: int) -> MarketPath
```

- **Discretization: full truncation Euler** (Lord, Koekkoek & van Dijk,
  2010) — same family `vol_models.heston.heston_mc_price` uses, though
  this module is self-contained (no import of `vol_models.heston`): it's
  market-DATA generation, not pricing. `v_pos = max(v_t, 0)` floors the
  variance only where it feeds a `sqrt`/drift term; the *state* itself is
  carried forward unfloored and may go transiently negative (never
  produces NaN/negative spot — verified in
  `tests/test_market_sim.py::test_survives_severe_feller_violation_no_nan_or_negative_spot`).
  Correlated normals: `Z_v = rho*Z_s + sqrt(1-rho**2)*Z_perp`.
- **`TRUE_MARKET_PARAMS` (flagged decision):** `spot0=100, v0=theta=0.045`
  (start at the stationary long-run level, ~21.2% annualized vol),
  `kappa=1.8, sigma=0.55, rho=-0.65, rate=0.03, div_yield=0`. Deliberately
  **violates the Feller condition** (`2*kappa*theta=0.162 <
  sigma**2=0.3025`), matching `vol_models`'s own stress-tested
  Feller-violation case and realistic calibrated-Heston behavior.
- Deterministic given `(params, n_steps, dt, seed)` — verified in
  `tests/test_market_sim.py::test_simulate_heston_path_is_deterministic_given_seed`.

### Order arrivals (`backtest.arrivals`)

```python
fill_intensity(distance: float, base_intensity: float, kappa: float) -> float
    # = base_intensity * exp(-kappa * distance)

poisson_fill_counts(distance_bid, distance_ask, base_intensity, kappa, dt,
                     rng: np.random.Generator,
                     max_expected_arrivals_per_bar: float = 3.0) -> tuple[int, int]
```

- Implements the fill-intensity model `market_maker.quoting`'s
  `order_arrival_kappa` assumes but doesn't simulate:
  `intensity(distance) = A * exp(-kappa * distance)` (classic AS/Gueant).
  `distance_bid = true_price - bid`, `distance_ask = ask - true_price`
  (both normally `>= 0`; smaller distance -> more competitive quote ->
  higher intensity).
- **`kappa` here MUST equal `generate_quotes`'s `order_arrival_kappa`** —
  `backtest.engine.run_backtest` enforces this by construction (single
  `order_arrival_kappa` argument threaded to both), so it cannot silently
  diverge.
- **`base_intensity` (flagged decision):** default `80.0` (arrivals/year
  at zero quote distance) — an order-of-magnitude placeholder for a
  moderately liquid single-name options market; not calibrated to real
  order-flow data (none available synthetically). See "risk_aversion
  retuning" below for why this specific value, paired with the retuned
  risk aversion, was chosen.
- **`max_expected_arrivals_per_bar=3.0` cap (flagged decision):** guards
  `rng.poisson`'s `lam` against blowing up when a quote sits on the
  "wrong" side of the reference price (`distance < 0` ->
  `exp(-kappa*distance) > base_intensity`) — this genuinely happens
  transiently under inventory-skew feedback and, uncapped (an earlier
  default of 25 was tried), produces a **runaway inventory-accumulation
  spiral** (observed empirically: >8000 fills over a 63-bar run, P&L
  diverging to -$140k) rather than the intended mean-reverting behavior.

### `run_backtest` (`backtest.engine`)

```python
def run_backtest(
    model: Literal["black_scholes", "heston"],
    *,
    true_params: HestonMarketParams = TRUE_MARKET_PARAMS,
    basket: list[OptionContract] | None = None,          # default: build_default_basket()
    n_steps: int = 63,                                     # ~1 quarter of daily bars
    dt: float = 1/252,                                     # daily bars
    market_seed: int = 42,
    arrival_seed: int = 123,
    base_intensity: float = 80.0,
    order_arrival_kappa: float = 1.5,                      # matches market_maker.quoting's own default
    risk_aversion: RiskAversion = RiskAversion(delta=0.5, vega=0.02, gamma=0.0),  # retuned, see below
    flat_vol: float | None = None,                         # auto: true ATM vol @180d if None
    heston_quoter_params: dict | None = None,              # auto: calibrated from synthetic surface if None
    sigma_underlying: float | None = None,                 # auto: sqrt(true_params.theta) if None
    sigma_vol: float | None = None,                        # auto: true_params.sigma if None
    variance_floor: float = 1e-6,
) -> BacktestResults
```

`build_default_basket(strikes=(95,100,105), maturities=(120/365,240/365))`
— 6 calls, 3 strikes x 2 maturities. Calls-only and this specific
strike/maturity grid are scope simplifications (puts, and contract
expiry mid-backtest, are out of scope — maturities are chosen so the
shortest-dated contract never expires within the default `n_steps=63`
horizon, min remaining maturity ≈0.079y at the end).

**Loop structure (fixed-bar, not tick-level — flagged decision):** the
realized market advances by fixed `dt` per bar; order arrivals are a
Poisson PROCESS *within* each bar (possibly 0, 1, or several fills per
side per bar). Chosen over a tick-level "market only moves at arrival
instants" design because (a) it matches the standard daily-Sharpe
convention, and (b) it keeps "how often the market diffuses" (continuous,
sampled daily) separate from "how often orders arrive" (genuinely
event-driven), which are different physical processes.

Per bar `t`: age every contract's remaining maturity
(`orig_maturity - t*dt`, floored at `1e-6`) -> reprice the TRUE market's
own fair value at the bar's *realized* (floored) variance state, per
contract, via `vol_models.fair_value("heston", ...)` — this is the
arrival process's reference "true" price -> `generate_quotes` with
whichever pricer is active -> for each contract, draw
`poisson_fill_counts` on both sides and apply fills (`apply_fill`) ->
record cash/inventory/marks/greeks.

**Marking convention (flagged decision):** `pnl` is **always** marked
using the TRUE market's own Heston fair value (never the active quoting
model's own price) — so a systematically wrong model can't show phantom
P&L from marking optimism, and the two runs are comparable on a level
footing. This means `pnl` is not literally "what a desk's own P&L screen
would show" (a real desk marks to its own model or the observed market
mid) — flagged as specific to this synthetic, ground-truth-known setting.
`delta_book`/`vega_book`, by contrast, use the **active quoting model's
own Greeks** (what the market maker itself believes its exposure is).

**Risk-sizing inputs (flagged decision):** `sigma_underlying` and
`sigma_vol` (the AS-formula risk-charge inputs `generate_quotes` needs)
default to `sqrt(true_params.theta)` and `true_params.sigma` — a single
OBJECTIVE, MODEL-INDEPENDENT estimate used **identically in both runs**,
so any P&L/Sharpe/drawdown difference is attributable only to which
fair-value/Greeks engine is quoting, not to one desk having a better risk
estimate. (Still "cheating" a little, in the same documented way the
flat-vol/calibration choices are — these are the *true* theta/sigma, not
estimated from noisy data.)

**`risk_aversion` retuning (flagged decision):** `market_maker.quoting
.RiskAversion()`'s own bare default `(delta=0.1, vega=0.1)` is tuned for
a small single-contract book; on this module's multi-contract basket +
`base_intensity=80`, it's too weak to meaningfully skew quotes against a
growing book (`delta_book`/`vega_book` reach the tens-to-hundreds here),
producing a runaway inventory spiral rather than the intended
mean-reverting behavior (see `backtest.arrivals` note above — this was
diagnosed together with the arrival-cap issue). `RiskAversion(delta=0.5,
vega=0.02)` was chosen empirically (grid search over a handful of
candidates, checking for bounded `max|inventory|` and a non-diverging
P&L path) specifically for this basket/intensity combination — a free
parameter choice, not a literature value.

**Order-arrival-kappa consistency:** `order_arrival_kappa` is a single
argument threaded unchanged to both `generate_quotes` and
`poisson_fill_counts` — cannot silently diverge by construction, resolving
the Market Maker agent's decision #5 flag.

`apply_fill(cash, book, contract_id, qty_signed, price) -> new_cash` —
exposed standalone (not inlined) so the cash/inventory bookkeeping
identity is directly unit-testable; see
`tests/test_engine.py::test_apply_fill_conserves_cash_plus_inventory_value`.

### `BacktestResults` (`backtest.results`)

```python
@dataclass
class BacktestResults:
    model: str
    times, spot_path, variance_path: np.ndarray   # shape (n_steps+1,)
    dt: float
    pnl, cash, delta_book, vega_book: np.ndarray   # shape (n_steps+1,)
    inventory: dict[str, np.ndarray]               # per-contract, shape (n_steps+1,)
    marks: dict[str, np.ndarray]                   # per-contract TRUE-market marks used in pnl
    fills: list[dict]                              # {"t","contract_id","side","price","true_price"}
    sharpe: float
    max_drawdown: float                            # non-positive
    meta: dict                                     # full run config, for reproducibility

    total_inventory() -> np.ndarray   # signed sum across contracts
    gross_inventory() -> np.ndarray   # sum of |inventory| across contracts

compute_sharpe(pnl: np.ndarray, dt: float, ddof: int = 1) -> float
compute_max_drawdown(pnl: np.ndarray) -> float
plot_pnl_comparison(results_bs, results_heston, output_path, title=...) -> Path
```

- **Sharpe convention (flagged decision):** computed on bar-over-bar
  **dollar** P&L increments (`diff(pnl)`), not percentage returns — a
  market-making book has no natural "capital base" to divide by.
  Annualized as `mean(r)/std(r,ddof=1) * sqrt(1/dt)` — the standard daily
  convention generalized to arbitrary `dt` (`dt=1/252` gives the usual
  `sqrt(252)`). Returns `0.0` for <3 bars or zero-variance P&L.
- **Max drawdown:** worst peak-to-trough `pnl - running_max`, returned as
  a **non-positive** float (0.0 if the equity curve never dips below its
  prior peak).
- `plot_pnl_comparison` uses the `Agg` backend (headless-safe, matching
  `pricing.vol_surface.plot_vol_surface`'s convention), saves a 2-panel
  PNG (P&L curves; gross inventory curves), creates parent dirs as needed.

### Comparison report (`backtest/comparison.py`)

`python -m backtest.comparison` runs both models with `market_seed=42,
arrival_seed=123` (identical realized market/arrival-RNG-stream for both —
verified bit-identical via `spot_path`/`variance_path` equality both in
`tests/test_engine.py::test_identical_seeds_give_identical_realized_paths_across_models`
and asserted again at report-generation time), prints a summary, and
saves `data/backtest_pnl_comparison.png`.

**Results on this run** (single market-path/arrival-seed realization —
not a Monte-Carlo average across seeds; see final report for why the
specific winner shouldn't be over-interpreted from one path):

| metric | BS-quoting | Heston-quoting |
|---|---|---|
| final P&L | $70.43 | $7.23 |
| Sharpe (annualized) | 4.37 | 0.33 |
| max drawdown | -$12.82 | -$45.30 |
| n_fills | 113 | 116 |
| max \|inventory\| (any contract) | 22.0 | 17.0 |
| final \|delta_book\| | 1.41 | 3.51 |
| final \|vega_book\| | 123.79 | 27.51 |

Heston quoter's calibrated params:
`{kappa: 1.707, theta: 0.0435, sigma: 0.565, rho: -0.665, v0: 0.0447}`
(true: `kappa=1.8, theta=0.045, sigma=0.55, rho=-0.65, v0=0.045` — close
but not exact, as intended). BS quoter's flat vol: `0.1981` (true ATM vol
at 180d).

## `webapp` — Web UI

Status: done. FastAPI backend (`webapp/main.py`, `webapp/routers/*.py`) +
a static single-page frontend (`webapp/static/`, vanilla JS + Plotly, no
build step). Imports `pricing`, `vol_models`, `market_maker`, `backtest`
as a library only — **none of those four packages were modified.** No
database; an in-memory `webapp/store.py` caches recent backtest runs
(needed because `BacktestResults` doesn't carry the basket that produced
it) and the sweep's deterministic quoter params.

Run with (project root, venv active):
```bash
uvicorn webapp.main:app --reload --port 8000
```
Then open `http://localhost:8000/`.

### The pricer contract, preserved

Every pricer-swapping endpoint routes through the exact shared signature
`(model, params, spot, strike, maturity, option_type) -> (price, greeks)`
(`pricing.api.price` for `black_scholes`/`monte_carlo`,
`vol_models.api.fair_value` for `heston`/`sabr`) — the webapp never
reimplements pricing logic, only dispatches to it. `greeks["vega"]` is
labeled and treated everywhere as the BS-equivalent `dPrice/dIV` (see the
`vol_models` section above); `vega_raw` is surfaced only as an explicitly
labeled diagnostic (e.g. in `POST /api/pricer/price`'s response), never as
the primary vega.

### Quote waterfall — the safety-critical piece

`webapp/derive.py`'s `build_quote_waterfall` decomposes a `Quote` into its
additive terms (`delta_skew`, `vega_skew`, `risk_spread_delta`,
`risk_spread_vega`, `liquidity_spread`, `gamma_term`) by re-deriving each
term from the exact formulas in `market_maker/quoting.py`, but computes
`reservation_price`/`spread`/`bid`/`ask` by **calling**
`market_maker.quoting.reservation_price` / `quote_spread` directly — so
its output can never silently drift from the real engine's. This is
asserted in `tests/test_webapp_derive.py` (unit-level, parametrized) and
`tests/test_webapp_backtest.py::test_backtest_waterfall_reconstructs_engine_quote_exactly`
(API-level, against a real run). If `market_maker/quoting.py`'s formulas
ever change, these tests fail rather than letting the UI "explain" a
mechanism it no longer matches.

`POST /api/backtest/waterfall` reconstructs a bar's quote by rebuilding a
`market_maker.book.Book` from the cached run's basket +
`results.inventory[contract_id][bar_index]` (both already available — no
backtest re-run, no engine changes) and calling
`market_maker.quoting.generate_quotes` fresh, exactly as the build spec
requires. `risk_aversion_delta_override` / `risk_aversion_vega_override`
recompute the decomposition live for the "what-if" sliders without
touching the completed run (`is_what_if: true` in the response flags this).

### Endpoints

- `GET /api/pricer/defaults?model=...`, `POST /api/pricer/price` (point
  price + Greeks; Monte Carlo also returns `stderr` from
  `pricing.monte_carlo.mc_price` directly, since `pricing.api.price` drops
  it for cross-model shape parity), `POST /api/pricer/profile` (Greeks
  across a strike grid — Monte Carlo intentionally excluded, explicit-run
  only per the performance constraints).
- `GET /api/vol-surface/defaults`, `POST /api/vol-surface/generate`,
  `POST /api/vol-surface/calibrate` (Heston + SABR fit, per-model
  `residual_grid` = model IV − target IV, `rmse` via
  `vol_models.calibration.surface_rmse`; response always carries
  `not_arbitrage_checked: true` and an explanatory `note`).
- `POST /api/backtest/run`, `POST /api/backtest/run-pair` (same seed pair,
  both models, one call — the BS-vs-Heston comparison the project is built
  around), `GET /api/backtest/run/{run_id}`, `POST /api/backtest/waterfall`
  (see above). Both model's default `flat_vol`/`heston_quoter_params`, if
  not explicitly passed, come from `webapp.store.get_sweep_quoter_params()`
  — a cached, deterministic function of `TRUE_MARKET_PARAMS` alone (a
  performance judgment call: identical numbers to what `run_backtest` would
  compute unprompted, just not recomputed — mainly saves the ~1s Heston
  calibration — on every request).
- `GET /api/sweep/results` (reads `data/sweep_results.csv`, never
  recomputes — the 40-seed sweep is ~105s and must never block a page
  load), `GET /api/sweep/regression?feature=...` (OLS fit + CI + bootstrap
  CI on the paired difference, via `webapp/derive.py`'s `ols_with_ci` /
  `bootstrap_mean_ci` — states plainly when a feature shows no detectable
  relationship rather than implying one), `POST /api/sweep/reproduce`
  (re-runs one seed for both models with the exact sweep parameterization —
  verified byte-for-byte against `notebooks/FINAL_REPORT.md`'s seed 15 and
  seed 25 case studies), `POST /api/sweep/recompute` +
  `GET /api/sweep/recompute/status/{job_id}` (background thread, pollable
  progress; **do not call `/recompute` in a test** — it overwrites the
  committed `data/sweep_results.csv`).

### Decisions flagged for sign-off

1. **Pair-mode inventory panel shows gross inventory, not per-contract.**
   Showing all 6 contracts x 2 models (12 lines) in View 3's inventory
   panel in pair mode was judged too busy to be readable; per-contract
   detail is only shown in single-run mode. A simplification, not a
   limitation of the data (per-contract inventory for both runs is present
   in the API response either way).
2. **Waterfall rendered as three linked Plotly `waterfall` traces**
   (fair_value → reservation_price, then reservation_price → ask and
   reservation_price → bid as two separate branches), not one connected
   figure — Plotly's native waterfall trace type doesn't support a single
   run forking into two directions from a midpoint. Values are still
   exactly reconstructed (see above); this is a rendering choice only.
3. **Filtering fills by contract/negative-edge also recomputes the
   cumulative-edge line and summary tiles for the filtered subset**, not
   just the markers — judged more useful (a user filtering to one contract
   wants that contract's edge story), though the spec's wording was
   ambiguous on this point.
4. **`GET /api/sweep/regression`'s significance threshold is p < 0.05**,
   stated as a conventional (not sacred) cutoff at whatever `n` the sweep
   currently has — flagged since p-values from n=40 OLS shouldn't be
   over-trusted; the verdict text says so explicitly.
