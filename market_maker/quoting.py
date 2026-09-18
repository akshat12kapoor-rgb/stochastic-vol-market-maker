"""Multi-Greek Avellaneda-Stoikov-style quoting engine.

Generalizes the classic single-asset Avellaneda-Stoikov (2008) inventory-
skew model — built for a market maker in one spot/prediction-market asset —
to an options market maker whose risk lives across an entire book of
contracts (delta AND vega exposure, not just a single inventory count).

--------------------------------------------------------------------------
PRICER CONTRACT (the abstract interface this module is built against)
--------------------------------------------------------------------------
Every function below that needs a fair value takes a `pricer` argument
that must be a callable matching EXACTLY the signature documented for
`pricing.api.price` in ARCHITECTURE.md:

    pricer(model: str, params: dict, spot: float, strike: float,
           maturity: float, option_type: str = "call"
           ) -> tuple[float, dict[str, float]]

    - Returns (price, greeks) where greeks has keys
      "delta", "gamma", "vega", "theta", "rho" (see ARCHITECTURE.md's
      `pricing` section for the exact unit conventions this module relies
      on, e.g. vega is dPrice/dVol per 1.0 = 100 vol points).
    - Must not mutate `params`.

Today we only have `pricing.api.price` with `model="black_scholes"` (or
`"monte_carlo"`) implemented, so all tests in this package instantiate the
pricer as `pricing.api.price` and pass `model="black_scholes"`. The
Stochastic Vol Agent has been asked to expose a `fair_value` function with
this SAME call signature for `model in {"heston", "sabr"}`. Nothing in this
module hardcodes Black-Scholes or imports from `pricing` at all — any
callable matching the signature above works, so once that lands, calling
code should be able to pass `model="heston"` / `model="sabr"` (with that
model's own `params` dict) through unchanged, with zero code changes here.
This is the integration checkpoint the orchestrator should verify once
both agents are done.

--------------------------------------------------------------------------
MODELING DECISION — FLAGGED FOR SIGN-OFF (see final report)
--------------------------------------------------------------------------
The classic AS reservation price / spread are

    r = s - q * gamma * sigma^2 * (T - t)
    spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

where `q` is inventory in the single traded asset, `gamma` is a single
risk-aversion scalar, `sigma^2` is the asset's price variance rate, and
`kappa` is the exponential decay rate of order-arrival intensity in quote
distance. There is no single canonical way to extend this to a book with
both delta and vega risk and no globally-agreed per-Greek risk-aversion
split; the generalization implemented here is a deliberate, documented
choice, not a literature result:

1. **Per-Greek risk aversion, not one scalar.** We replace `gamma` with a
   `RiskAversion(delta, vega, gamma=0.0)` bundle — separate coefficients
   for delta risk and vega risk (and an optional, off-by-default gamma/
   convexity coefficient — see below). Rationale: an option book's delta
   risk (directional price risk) and vega risk (volatility risk) are
   economically different exposures that a real market maker would want
   to price and limit independently (e.g. a desk might be very averse to
   vega risk around an earnings event but comfortable running some delta).
   Collapsing them into one scalar, as the classic model does, throws that
   distinction away.

2. **Portfolio-level (inventory-weighted) exposures, not per-contract.**
   `Delta_book = sum(qty_i * delta_i)` and `Vega_book = sum(qty_i * vega_i)`
   across every contract currently held in the `Book` — this is the
   "inventory-weighted delta exposure" and "vega exposure" the task asks
   for. It generalizes AS's single `q` (shares held) directly: `Delta_book`
   IS a `q` in underlying-equivalent units when there's a single contract,
   so the classic formula is recovered as a special case.

3. **Reservation price = fair value minus a linear risk charge per Greek**,
   treating delta-driven price risk and vega-driven vol risk as two
   independent, additive quadratic-variation-like risk terms, each scaled
   by its own risk-aversion weight:

       r_c = f_c
             - risk_aversion.delta * Delta_book * sigma_s**2   * horizon
             - risk_aversion.vega  * Vega_book  * sigma_vol**2 * horizon

   `sigma_s` is the underlying's annualized price vol and `sigma_vol` is
   an annualized **vol-of-vol** (the standard deviation of the vol process
   itself) — this vol-of-vol term is genuinely new versus the spot AS
   model (a spot asset has no "vol of its own vol"). `horizon` defaults to
   the quoted contract's own time-to-maturity (the natural risk look-ahead
   for that contract), overridable by the caller.

   Additivity across the two risk sources is itself an assumption (it
   ignores delta/vega cross-correlation, e.g. spot-vol correlation/skew —
   a real book's P&L variance also has a delta*vega cross term via
   d(spot)*d(vol) covariance). We deliberately drop the cross term for a
   first pass; flagging that this is a simplification, not an omission.

4. **Spread = risk compensation (this contract's own marginal Greek
   exposure) + a liquidity/order-flow term** carried over from AS almost
   unchanged:

       risk_spread = 2 * ( risk_aversion.delta * |delta_c| * sigma_s**2   * horizon
                          + risk_aversion.vega  * |vega_c|  * sigma_vol**2 * horizon )
       liquidity_spread = (2 / g) * ln(1 + g / order_arrival_kappa),
                           g = risk_aversion.delta + risk_aversion.vega

   The risk_spread term uses the CONTRACT'S OWN delta/vega (the marginal
   risk of trading one more unit of *this* contract), not the book
   aggregate — deliberately different from the reservation-price skew,
   which uses the book aggregate. This mirrors the classic model's
   structure (inventory `q` skews `r`; risk aversion `gamma` widens the
   spread) but is again a choice, not a derivation.

   For the liquidity term we still need *a* single risk-aversion scalar
   (AS's `gamma`) to plug into `(2/gamma)*ln(1+gamma/kappa)` — we use
   `risk_aversion.delta + risk_aversion.vega`. This specific combination
   (a simple sum) is arbitrary and called out separately as its own
   sub-decision.

   **Order-arrival assumption**: `order_arrival_kappa` is the exponential
   decay rate of fill-probability with quote distance from fair value
   (`intensity(delta_price) = A * exp(-kappa * delta_price)`), exactly as
   in the classic AS/Avellaneda-Stoikov-Guéant order-book microstructure
   model. We do NOT implement or calibrate an actual order-arrival
   process here — the Backtest Agent owns that. `order_arrival_kappa` is
   exposed as a free parameter with a placeholder default (1.5) so the
   spread formula is well-defined; the Backtest Agent should confirm
   whether this parameter should be calibrated jointly with their actual
   arrival-process model, or whether this liquidity term should be
   replaced entirely once real order-arrival dynamics exist.

5. **Optional gamma/convexity risk aversion (`RiskAversion.gamma`,
   default 0.0 — inert unless explicitly set).** Included because
   PROJECT.md's package description calls out "delta/vega/gamma skew" for
   this module, but gamma exposure doesn't have the same directional
   "skew" semantics as delta/vega (being long gamma is generally
   desirable, not a risk to offload), so it is wired ONLY as a
   spread-widener (both quotes move out symmetrically), never as a
   reservation-price skew:

       gamma_term = risk_aversion.gamma * |gamma_c| * (sigma_s * spot)**2 * horizon

   This is a much lower-confidence add-on than (1)-(4) above (the
   dimensional scaling by `spot**2` is a judgment call, not derived) and
   is OFF by default. Treat it as a documented stub, not a validated
   model; flagging separately from the main delta/vega generalization.

6. **`sigma_vol` (vol-of-vol) has no data source yet.** Nothing in
   `pricing` or (currently) `vol_models` exposes a vol-of-vol number, so
   this module takes it as a caller-supplied float with a placeholder
   default of 0.5 (50% annualized — a plausible order of magnitude for
   equity-index vol-of-vol, e.g. comparable to Heston's `xi` or SABR's
   `alpha` once those land). Once the Stochastic Vol Agent's Heston/SABR
   params are available, the natural follow-up is to derive `sigma_vol`
   from the model's own vol-of-vol parameter instead of passing a static
   placeholder — flagging this as a follow-up integration task, not
   doing it now since `vol_models` isn't done.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

from market_maker.book import Book, OptionContract, OptionType


class PricerFn(Protocol):
    """Structural type for any pricer matching `pricing.api.price`'s signature."""

    def __call__(
        self,
        model: str,
        params: dict,
        spot: float,
        strike: float,
        maturity: float,
        option_type: OptionType = "call",
    ) -> tuple[float, dict[str, float]]: ...


# A contract's per-contract params can be a single dict shared by every
# contract in the book, or a callable that returns a params dict given the
# contract (e.g. to pull a different implied vol per strike/maturity off a
# vol surface). See `_resolve_params`.
ParamsSource = dict | Callable[[OptionContract], dict]


@dataclass(frozen=True)
class RiskAversion:
    """Per-Greek risk-aversion weights (see module docstring, decision #1).

    Attributes:
        delta: risk-aversion weight applied to book-level delta exposure.
            Must be >= 0.
        vega: risk-aversion weight applied to book-level vega exposure.
            Must be >= 0.
        gamma: OPTIONAL convexity risk-aversion weight, default 0.0
            (inert). Widens spread only, never skews reservation price.
            Must be >= 0.
    """

    delta: float = 0.1
    vega: float = 0.1
    gamma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("delta", "vega", "gamma"):
            if getattr(self, name) < 0:
                raise ValueError(f"RiskAversion.{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class Quote:
    """A single contract's computed quote.

    Attributes:
        contract_id: which contract this quote is for.
        fair_value: the pricer's mid/theoretical price for this contract.
        reservation_price: fair_value adjusted for book-level delta/vega
            inventory risk (see module docstring).
        bid: reservation_price - spread / 2.
        ask: reservation_price + spread / 2.
        spread: ask - bid (always > 0).
        greeks: this contract's own greeks dict, as returned by the pricer
            (keys "delta", "gamma", "vega", "theta", "rho").
    """

    contract_id: str
    fair_value: float
    reservation_price: float
    bid: float
    ask: float
    spread: float
    greeks: dict[str, float]


def _resolve_params(params: ParamsSource, contract: OptionContract) -> dict:
    if callable(params):
        return params(contract)
    return params


def price_contract(
    pricer: PricerFn,
    model: str,
    params: ParamsSource,
    spot: float,
    contract: OptionContract,
) -> tuple[float, dict[str, float]]:
    """Price a single `OptionContract` via `pricer`, resolving per-contract params.

    Thin convenience wrapper so callers (and this module) don't have to
    unpack `OptionContract` fields by hand at every call site.
    """
    resolved = _resolve_params(params, contract)
    return pricer(model, resolved, spot, contract.strike, contract.maturity, contract.option_type)


def compute_book_greeks(
    book: Book,
    pricer: PricerFn,
    model: str,
    params: ParamsSource,
    spot: float,
    valuations: dict[str, tuple[float, dict[str, float]]] | None = None,
) -> dict[str, float]:
    """Aggregate inventory-weighted Greeks across every contract in `book`.

    Returns a dict with keys "delta", "gamma", "vega", "theta", "rho",
    each `sum(inventory[c] * greeks[c][key] for c in book.contracts)`.
    Contracts with zero inventory contribute zero.

    `valuations`, if given, is a cache of `{contract_id: (price, greeks)}`
    (as produced by `price_book`) to avoid re-pricing; if omitted, every
    contract in the book is priced via `pricer`.
    """
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for contract_id, contract in book.contracts.items():
        qty = book.get_inventory(contract_id)
        if qty == 0.0:
            continue
        if valuations is not None and contract_id in valuations:
            _, greeks = valuations[contract_id]
        else:
            _, greeks = price_contract(pricer, model, params, spot, contract)
        for key in totals:
            totals[key] += qty * greeks[key]
    return totals


def price_book(
    book: Book,
    pricer: PricerFn,
    model: str,
    params: ParamsSource,
    spot: float,
) -> dict[str, tuple[float, dict[str, float]]]:
    """Price every contract registered in `book` (regardless of inventory).

    Returns {contract_id: (price, greeks)}. Used internally by
    `generate_quotes`, exposed for callers who want the raw valuations too.
    """
    return {
        contract_id: price_contract(pricer, model, params, spot, contract)
        for contract_id, contract in book.contracts.items()
    }


def reservation_price(
    fair_value: float,
    delta_book: float,
    vega_book: float,
    sigma_underlying: float,
    sigma_vol: float,
    horizon: float,
    risk_aversion: RiskAversion,
) -> float:
    """Fair value adjusted for book-level delta/vega inventory risk.

    r = fair_value
        - risk_aversion.delta * delta_book * sigma_underlying**2 * horizon
        - risk_aversion.vega  * vega_book  * sigma_vol**2        * horizon

    With `delta_book == vega_book == 0` this returns exactly `fair_value`
    (the zero-inventory case: reservation price collapses to fair value,
    which is what makes zero-inventory quotes symmetric around fair value
    — see `tests/test_quoting.py::test_zero_inventory_symmetric_quotes`).

    Positive `delta_book` (net long the underlying via the book) lowers
    the reservation price, which lowers both bid and ask — this makes the
    market maker's ask more competitive (encourages being lifted, i.e.
    selling) and its bid less competitive (discourages buying more),
    pushing inventory back toward flat. Same logic for `vega_book`.
    """
    return (
        fair_value
        - risk_aversion.delta * delta_book * sigma_underlying**2 * horizon
        - risk_aversion.vega * vega_book * sigma_vol**2 * horizon
    )


def quote_spread(
    delta_c: float,
    vega_c: float,
    sigma_underlying: float,
    sigma_vol: float,
    horizon: float,
    risk_aversion: RiskAversion,
    order_arrival_kappa: float = 1.5,
    gamma_c: float = 0.0,
    spot: float = 0.0,
) -> float:
    """Total bid-ask spread width for a contract with own Greeks (delta_c, vega_c[, gamma_c]).

    spread = risk_spread + liquidity_spread (+ optional gamma_term), where

        risk_spread = 2 * ( risk_aversion.delta * |delta_c| * sigma_underlying**2 * horizon
                           + risk_aversion.vega  * |vega_c|  * sigma_vol**2        * horizon )
        liquidity_spread = (2/g) * ln(1 + g/order_arrival_kappa),
                            g = risk_aversion.delta + risk_aversion.vega
                            (g -> 0 limit is exactly 2/order_arrival_kappa)
        gamma_term (only if risk_aversion.gamma > 0) =
            risk_aversion.gamma * |gamma_c| * (sigma_underlying * spot)**2 * horizon

    Always > 0 given `order_arrival_kappa > 0` (liquidity_spread alone is
    already strictly positive). See module docstring decisions #4-#5 for
    the reasoning and the order-arrival-process assumption this bakes in.
    """
    if order_arrival_kappa <= 0:
        raise ValueError(f"order_arrival_kappa must be > 0, got {order_arrival_kappa}")

    risk_spread = 2.0 * (
        risk_aversion.delta * abs(delta_c) * sigma_underlying**2 * horizon
        + risk_aversion.vega * abs(vega_c) * sigma_vol**2 * horizon
    )

    g = risk_aversion.delta + risk_aversion.vega
    if g <= 1e-12:
        liquidity_spread = 2.0 / order_arrival_kappa
    else:
        liquidity_spread = (2.0 / g) * math.log1p(g / order_arrival_kappa)

    gamma_term = 0.0
    if risk_aversion.gamma > 0.0:
        gamma_term = risk_aversion.gamma * abs(gamma_c) * (sigma_underlying * spot) ** 2 * horizon

    return risk_spread + liquidity_spread + gamma_term


def generate_quotes(
    book: Book,
    pricer: PricerFn,
    model: str,
    params: ParamsSource,
    spot: float,
    sigma_underlying: float,
    risk_aversion: RiskAversion = RiskAversion(),
    sigma_vol: float = 0.5,
    horizon: float | None = None,
    order_arrival_kappa: float = 1.5,
) -> dict[str, Quote]:
    """Generate bid/ask quotes for every contract registered in `book`.

    Args:
        book: current market-maker state (see `market_maker.book.Book`).
            Every contract in `book.contracts` gets a quote; inventory
            (`book.inventory`) drives the reservation-price skew via the
            book-level delta/vega aggregates.
        pricer: callable matching `PricerFn` / `pricing.api.price`'s
            signature (see module docstring's PRICER CONTRACT section).
            Pass `pricing.api.price` for Black-Scholes/Monte Carlo today;
            any `fair_value`-shaped callable for `model in {"heston",
            "sabr"}` works unchanged once `vol_models` lands.
        model: model string forwarded to `pricer` unchanged (e.g.
            "black_scholes").
        params: dict of model params forwarded to `pricer`, OR a callable
            `OptionContract -> dict` for per-contract params (e.g. a
            different "vol" per strike/maturity off a vol surface).
        spot: current underlying price.
        sigma_underlying: annualized volatility of the UNDERLYING PRICE
            used for the delta risk term. Passed explicitly (not read out
            of `params`) because not every model's params dict has a
            single scalar "vol" key (e.g. Heston's params are
            v0/kappa/theta/xi/rho) — the quoting engine stays pricer-
            agnostic by requiring the caller state this directly.
        risk_aversion: `RiskAversion(delta, vega, gamma)` weights.
        sigma_vol: annualized vol-of-vol used for the vega risk term.
            Placeholder default 0.5 — see module docstring decision #6.
        horizon: risk look-ahead in years, applied uniformly to the
            reservation-price and spread formulas. Defaults to each
            contract's OWN maturity (`contract.maturity`) when None —
            override to use a fixed desk-level risk horizon instead.
        order_arrival_kappa: liquidity-term decay parameter, see
            `quote_spread`. Must be > 0.

    Returns:
        {contract_id: Quote} for every contract in `book.contracts`.

    Invariants:
        - If `book.is_flat()` (all inventories exactly 0), every quote's
          `reservation_price == fair_value` exactly, so
          `ask - reservation_price == reservation_price - bid` for every
          contract (symmetric quotes around fair value).
        - `bid < ask` always (spread is strictly positive).
        - Never mutates `book` or `params`.
    """
    valuations = price_book(book, pricer, model, params, spot)
    delta_book, vega_book = 0.0, 0.0
    for contract_id in book.contracts:
        qty = book.get_inventory(contract_id)
        if qty == 0.0:
            continue
        _, greeks = valuations[contract_id]
        delta_book += qty * greeks["delta"]
        vega_book += qty * greeks["vega"]

    quotes: dict[str, Quote] = {}
    for contract_id, contract in book.contracts.items():
        fair_value, greeks = valuations[contract_id]
        h = contract.maturity if horizon is None else horizon

        r = reservation_price(
            fair_value, delta_book, vega_book, sigma_underlying, sigma_vol, h, risk_aversion
        )
        spr = quote_spread(
            greeks["delta"],
            greeks["vega"],
            sigma_underlying,
            sigma_vol,
            h,
            risk_aversion,
            order_arrival_kappa=order_arrival_kappa,
            gamma_c=greeks["gamma"],
            spot=spot,
        )
        bid = r - spr / 2.0
        ask = r + spr / 2.0
        quotes[contract_id] = Quote(
            contract_id=contract_id,
            fair_value=fair_value,
            reservation_price=r,
            bid=bid,
            ask=ask,
            spread=spr,
            greeks=greeks,
        )
    return quotes
