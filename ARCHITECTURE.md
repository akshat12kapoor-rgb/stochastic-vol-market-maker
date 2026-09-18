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

_Not yet started._
