"""Event-driven backtest loop for the options market maker.

--------------------------------------------------------------------------
BAR / STEP FREQUENCY (flagged decision -- needs sign-off)
--------------------------------------------------------------------------
FIXED-BAR, not tick-level-Poisson-driven market steps: the realized market
(`backtest.market_sim.simulate_heston_path`) advances by a fixed `dt` per
bar (default `1/252`, i.e. daily bars), and order arrivals are modeled as a
Poisson PROCESS WITHIN each bar (`backtest.arrivals.poisson_fill_counts`,
i.e. possibly zero, one, or several fills per side per bar). This is a
choice, not a limitation of the tooling -- a tick-level loop where the
market itself only ever moves at a Poisson-arrival instant is equally
valid and was considered. Fixed daily bars were chosen because (a) it's the
simplest "one row per day" bookkeeping for P&L/Sharpe/drawdown, matching
the standard "daily Sharpe" convention `backtest.results.compute_sharpe`
implements, and (b) it decouples "how often does the market move"
(continuous Heston diffusion, sampled daily) from "how often do orders
arrive" (Poisson, genuinely event-driven within each bar) -- these are
different physical processes in real markets and conflating them into one
Poisson clock would understate the market's actual (much higher-frequency)
diffusion.

**Horizon (flagged decision):** default `n_steps=63` trading days (~1
calendar quarter) at `dt=1/252`. The basket's shortest maturity (120
calendar days) is chosen so no contract expires mid-backtest (minimum
remaining maturity at the end of the run is `120/365 - 63/252 ~= 0.079`
years, i.e. ~29 days of buffer) -- contract-expiry/settlement handling
(exercise, cash-settlement at intrinsic value, dropping an expired contract
from the quotable book) is explicitly OUT OF SCOPE for this phase.

--------------------------------------------------------------------------
MARKING CONVENTION (flagged decision -- needs sign-off)
--------------------------------------------------------------------------
`pnl` is ALWAYS marked using the TRUE market's own Heston fair value
(`market_params`'s kappa/theta/sigma/rho, with `v0` replaced by that bar's
REALIZED, floored instantaneous variance -- see `_true_mark_params`), never
the active quoting model's own fair value. This is deliberate: if each run
marked its own book with its own (possibly wrong) pricing model, a
systematically overconfident model could show phantom "P&L" just from
marking error, and the two runs' P&L series would not be comparable on a
level footing. Marking both runs to the same, independent, "ground truth"
valuation isolates the comparison to what actually matters here: quoting
and inventory-management SKILL (spread capture, adverse-selection
avoidance, inventory risk management), not marking optimism. This does mean
neither run's `pnl` is literally "what a real trading desk would see on
their PnL screen" (a real desk marks to ITS OWN model, or to the market's
observed mid) -- flagged as a simplification specific to this synthetic,
ground-truth-known backtest setting.

The realized (possibly transiently negative, per full truncation Euler)
variance state is floored at `variance_floor` (default `1e-6`) before being
fed into `vol_models.fair_value` as `v0` -- the same floor-only-at-point-of-
use convention `vol_models.heston`'s own discretization already applies
internally (see `backtest.market_sim` module docstring).

--------------------------------------------------------------------------
RISK-SIZING INPUTS: `sigma_underlying` / `sigma_vol` (flagged decision)
--------------------------------------------------------------------------
`generate_quotes` needs a scalar `sigma_underlying` (spot vol, for the
delta risk term) and `sigma_vol` (vol-of-vol, for the vega risk term) --
see ARCHITECTURE.md's `market_maker` section, decision #6. To keep the two
runs' comparison isolated to "which fair-value/Greeks ENGINE is used", both
default to a SINGLE OBJECTIVE, MODEL-INDEPENDENT estimate derived from the
true market and used IDENTICALLY in both runs:
    sigma_underlying = sqrt(true_params.theta)   (long-run true vol level)
    sigma_vol        = true_params.sigma         (true vol-of-vol)
Rationale: a real desk (either a BS shop or a Heston shop) would estimate
these from historical/observed market data, which is equally available to
both; they are RISK-SIZING inputs to the AS-style spread/skew formula, not
fair-value engines, so using the same objective numbers for both runs means
any P&L/Sharpe/drawdown difference between the two runs is attributable
ONLY to the fair-value/Greeks model each one quotes with -- not to one desk
having a better risk-sizing estimate than the other. Both are still
"cheating" a little (they're the TRUE theta/sigma, not estimated from
noisy data) in the same way `quoter_calibration.compute_flat_vol_for_bs_
quoter`'s ATM-vol choice is -- flagged, not hidden.

--------------------------------------------------------------------------
ORDER-ARRIVAL KAPPA CONSISTENCY
--------------------------------------------------------------------------
`run_backtest` takes a single `order_arrival_kappa` argument, threaded
UNCHANGED to both `generate_quotes` (as its `order_arrival_kappa`) and
`backtest.arrivals.poisson_fill_counts` (as `kappa`) -- see
`backtest.arrivals` module docstring. There is no way to pass a different
value to each by construction.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from market_maker.book import Book, OptionContract
from market_maker.quoting import RiskAversion, compute_book_greeks, generate_quotes
from pricing.api import price as bs_pricer
from vol_models.api import fair_value as heston_pricer

from backtest.arrivals import poisson_fill_counts
from backtest.market_sim import HestonMarketParams, TRUE_MARKET_PARAMS, simulate_heston_path
from backtest.quoter_calibration import calibrate_heston_quoter_params, compute_flat_vol_for_bs_quoter
from backtest.results import BacktestResults, compute_max_drawdown, compute_sharpe

ModelName = Literal["black_scholes", "heston"]

# --- Default basket: a few strikes x a couple maturities, calls only. ---
# Calls-only is a scope simplification (puts would exercise the same
# machinery via put-call-symmetric Greeks; not included to keep the basket
# small and the report readable) -- flagged, not hidden.
DEFAULT_STRIKES: tuple[float, ...] = (95.0, 100.0, 105.0)
DEFAULT_MATURITIES: tuple[float, ...] = (120 / 365, 240 / 365)

DEFAULT_N_STEPS = 63          # ~1 quarter of daily bars
DEFAULT_DT = 1.0 / 252.0      # daily bars
DEFAULT_BASE_INTENSITY = 80.0   # arrivals/year at zero quote distance
DEFAULT_ORDER_ARRIVAL_KAPPA = 1.5  # matches market_maker.quoting's own default
DEFAULT_VARIANCE_FLOOR = 1e-6
DEFAULT_MARKET_SEED = 42
DEFAULT_ARRIVAL_SEED = 123
# NOTE: market_maker.quoting.RiskAversion()'s own bare default (0.1, 0.1) is
# tuned for a single-contract book with a modest vega scale; on THIS
# module's default multi-contract basket + `DEFAULT_BASE_INTENSITY`, it is
# too weak to meaningfully skew quotes against a growing book (delta_book/
# vega_book easily reach the tens-to-hundreds here), which -- combined with
# the arrival process's exponential intensity-vs-distance sensitivity --
# produces an unbounded, runaway inventory-accumulation spiral rather than
# the intended mean-reverting inventory-skew behavior. `DEFAULT_RISK_
# AVERSION` below is retuned (empirically, by inspecting max |inventory|
# and the P&L path for runaway/explosive behavior across a small grid of
# candidates) specifically for this basket/intensity combination so the
# backtest demonstrates genuine risk-managed market making rather than a
# blowup. Flagged per PROJECT.md rule 3 -- this is a free parameter choice,
# not a literature value.
DEFAULT_RISK_AVERSION = RiskAversion(delta=0.5, vega=0.02, gamma=0.0)


def build_default_basket(
    strikes: tuple[float, ...] = DEFAULT_STRIKES,
    maturities: tuple[float, ...] = DEFAULT_MATURITIES,
) -> list[OptionContract]:
    """A small multi-contract basket: `len(strikes) * len(maturities)` calls."""
    contracts = []
    for maturity in maturities:
        for strike in strikes:
            contract_id = f"C_{strike:.0f}_{round(maturity * 365)}D"
            contracts.append(
                OptionContract(contract_id=contract_id, strike=strike, maturity=maturity, option_type="call")
            )
    return contracts


def apply_fill(cash: float, book: Book, contract_id: str, qty_signed: float, price: float) -> float:
    """Apply one fill to `book` (mutated in place) and return the updated cash.

    `qty_signed > 0` is a market-maker BUY (inventory increases, cash
    decreases by `qty_signed * price`); `qty_signed < 0` is a SELL
    (inventory decreases, cash increases by `-qty_signed * price` ==
    `|qty_signed| * price`). Exposed as a standalone function (rather than
    inlined in the loop) so bookkeeping identities can be unit tested in
    isolation -- see `tests/test_engine.py::test_apply_fill_conserves_cash_plus_inventory_value`.
    """
    book.adjust_inventory(contract_id, qty_signed)
    return cash - qty_signed * price


def _true_mark_params(true_params: HestonMarketParams, variance_t: float, variance_floor: float) -> dict:
    return true_params.as_dict(v0_override=max(variance_t, variance_floor))


def run_backtest(
    model: ModelName,
    *,
    true_params: HestonMarketParams = TRUE_MARKET_PARAMS,
    basket: list[OptionContract] | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    dt: float = DEFAULT_DT,
    market_seed: int = DEFAULT_MARKET_SEED,
    arrival_seed: int = DEFAULT_ARRIVAL_SEED,
    base_intensity: float = DEFAULT_BASE_INTENSITY,
    order_arrival_kappa: float = DEFAULT_ORDER_ARRIVAL_KAPPA,
    risk_aversion: RiskAversion = DEFAULT_RISK_AVERSION,
    flat_vol: float | None = None,
    heston_quoter_params: dict | None = None,
    sigma_underlying: float | None = None,
    sigma_vol: float | None = None,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
) -> BacktestResults:
    """Run one event-driven backtest of the options market maker.

    Args:
        model: `"black_scholes"` (flat-vol quoting via `pricing.api.price`)
            or `"heston"` (quoting via `vol_models.api.fair_value`, params
            calibrated from a synthetic surface -- see
            `backtest.quoter_calibration` for the central framing decision).
        true_params: the "ground truth" Heston market to simulate and mark
            against (default `backtest.market_sim.TRUE_MARKET_PARAMS`).
        basket: contracts to quote; defaults to `build_default_basket()`
            (3 strikes x 2 maturities = 6 calls).
        n_steps, dt: bar count and bar length in years (defaults: 63 daily
            bars, ~1 quarter -- see module docstring).
        market_seed: seed for `simulate_heston_path` -- the realized
            (spot, variance) path is generated ONCE per call and is
            IDENTICAL for any two calls sharing `market_seed` (and the same
            `true_params`/`n_steps`/`dt`), REGARDLESS of `model` -- this is
            what makes a `"black_scholes"` run and a `"heston"` run with the
            same `market_seed`/`arrival_seed` an apples-to-apples
            comparison (see `tests/test_engine.py`).
        arrival_seed: seed for the arrival-process RNG (`numpy.random
            .default_rng(arrival_seed)`, freshly constructed per call). Note
            that even with identical `market_seed`/`arrival_seed`, the two
            runs' actual FILL COUNTS will differ, because the Poisson
            intensity at each bar depends on that run's own (model-
            dependent) bid/ask distance from the true price -- matching
            seeds eliminates randomness as a confound, it does not force
            identical outcomes (which would defeat the point of comparing
            two different quoting models).
        base_intensity, order_arrival_kappa: arrival-process parameters,
            see `backtest.arrivals` module docstring.
            `order_arrival_kappa` is threaded to BOTH the arrival process
            and `generate_quotes` -- see module docstring.
        risk_aversion: `market_maker.quoting.RiskAversion` passed through to
            `generate_quotes` unchanged.
        flat_vol: BS quoter's flat vol; if `None`, computed once via
            `quoter_calibration.compute_flat_vol_for_bs_quoter(true_params)`.
            Ignored if `model != "black_scholes"`.
        heston_quoter_params: Heston quoter's params dict; if `None`,
            computed once via `quoter_calibration
            .calibrate_heston_quoter_params(true_params)`. Ignored if
            `model != "heston"`.
        sigma_underlying, sigma_vol: objective risk-sizing inputs to
            `generate_quotes`; default to `sqrt(true_params.theta)` and
            `true_params.sigma` respectively if `None` -- see module
            docstring's "risk-sizing inputs" section.
        variance_floor: floor applied to the realized (possibly transiently
            negative) true variance path before it's used as `v0` for
            true-market marking/reference pricing.

    Returns:
        `BacktestResults` -- see `backtest.results.BacktestResults` for the
        full field-by-field shape.

    Raises:
        ValueError: unrecognized `model`.
    """
    if model not in ("black_scholes", "heston"):
        raise ValueError(f"Unknown model {model!r}; expected 'black_scholes' or 'heston'")

    contracts = basket if basket is not None else build_default_basket()
    original_maturity = {c.contract_id: c.maturity for c in contracts}
    strike_of = {c.contract_id: c.strike for c in contracts}
    option_type_of = {c.contract_id: c.option_type for c in contracts}
    contract_ids = [c.contract_id for c in contracts]

    market_path = simulate_heston_path(true_params, n_steps, dt, market_seed)

    sigma_underlying_eff = sigma_underlying if sigma_underlying is not None else math.sqrt(true_params.theta)
    sigma_vol_eff = sigma_vol if sigma_vol is not None else true_params.sigma

    if model == "black_scholes":
        flat_vol_eff = flat_vol if flat_vol is not None else compute_flat_vol_for_bs_quoter(true_params)
        quoting_pricer = bs_pricer
        quoting_model_name = "black_scholes"
        quoting_params = {"rate": true_params.rate, "vol": flat_vol_eff, "div_yield": true_params.div_yield}
        quoter_meta = {"flat_vol": flat_vol_eff}
    else:
        heston_quoter_params_eff = (
            heston_quoter_params if heston_quoter_params is not None else calibrate_heston_quoter_params(true_params)
        )
        quoting_pricer = heston_pricer
        quoting_model_name = "heston"
        quoting_params = heston_quoter_params_eff
        quoter_meta = {"heston_quoter_params": dict(heston_quoter_params_eff)}

    book = Book()
    for c in contracts:
        book.add_contract(c, quantity=0.0)

    arrival_rng = np.random.default_rng(arrival_seed)

    n_bars = n_steps + 1
    pnl = np.empty(n_bars)
    cash_series = np.empty(n_bars)
    delta_book_series = np.empty(n_bars)
    vega_book_series = np.empty(n_bars)
    inventory_series = {cid: np.empty(n_bars) for cid in contract_ids}
    marks_series = {cid: np.empty(n_bars) for cid in contract_ids}
    fills: list[dict] = []

    cash = 0.0

    def _age_book(t: int) -> dict[str, float]:
        remaining = {}
        for cid in contract_ids:
            tau = max(original_maturity[cid] - t * dt, 1e-6)
            remaining[cid] = tau
            book.contracts[cid] = OptionContract(
                contract_id=cid, strike=strike_of[cid], maturity=tau, option_type=option_type_of[cid]
            )
        return remaining

    def _true_marks(t: int, remaining: dict[str, float]) -> dict[str, float]:
        params_t = _true_mark_params(true_params, market_path.variance[t], variance_floor)
        spot_t = market_path.spot[t]
        marks = {}
        for cid in contract_ids:
            price, _ = heston_pricer("heston", params_t, spot_t, strike_of[cid], remaining[cid], option_type_of[cid])
            marks[cid] = price
        return marks

    for t in range(n_steps):
        spot_t = market_path.spot[t]
        remaining = _age_book(t)
        true_marks_t = _true_marks(t, remaining)

        quotes = generate_quotes(
            book, quoting_pricer, quoting_model_name, quoting_params, spot_t,
            sigma_underlying=sigma_underlying_eff, risk_aversion=risk_aversion,
            sigma_vol=sigma_vol_eff, horizon=None, order_arrival_kappa=order_arrival_kappa,
        )

        for cid in contract_ids:
            quote = quotes[cid]
            true_price = true_marks_t[cid]
            distance_bid = true_price - quote.bid
            distance_ask = quote.ask - true_price
            n_buy, n_sell = poisson_fill_counts(
                distance_bid, distance_ask, base_intensity, order_arrival_kappa, dt, arrival_rng
            )
            for _ in range(n_buy):
                cash = apply_fill(cash, book, cid, +1.0, quote.bid)
                fills.append({"t": t, "contract_id": cid, "side": "buy", "price": quote.bid, "true_price": true_price})
            for _ in range(n_sell):
                cash = apply_fill(cash, book, cid, -1.0, quote.ask)
                fills.append({"t": t, "contract_id": cid, "side": "sell", "price": quote.ask, "true_price": true_price})

        book_greeks = compute_book_greeks(
            book, quoting_pricer, quoting_model_name, quoting_params, spot_t,
            valuations={cid: (quotes[cid].fair_value, quotes[cid].greeks) for cid in contract_ids},
        )

        cash_series[t] = cash
        delta_book_series[t] = book_greeks["delta"]
        vega_book_series[t] = book_greeks["vega"]
        for cid in contract_ids:
            inventory_series[cid][t] = book.get_inventory(cid)
            marks_series[cid][t] = true_marks_t[cid]
        pnl[t] = cash + sum(book.get_inventory(cid) * true_marks_t[cid] for cid in contract_ids)

    # Terminal bar: age the book to its final remaining maturity, mark, but
    # place no new quotes/fills (see module docstring's horizon note).
    t_final = n_steps
    remaining_final = _age_book(t_final)
    true_marks_final = _true_marks(t_final, remaining_final)
    spot_final = market_path.spot[t_final]
    final_greeks = compute_book_greeks(book, quoting_pricer, quoting_model_name, quoting_params, spot_final)

    cash_series[t_final] = cash
    delta_book_series[t_final] = final_greeks["delta"]
    vega_book_series[t_final] = final_greeks["vega"]
    for cid in contract_ids:
        inventory_series[cid][t_final] = book.get_inventory(cid)
        marks_series[cid][t_final] = true_marks_final[cid]
    pnl[t_final] = cash + sum(book.get_inventory(cid) * true_marks_final[cid] for cid in contract_ids)

    sharpe = compute_sharpe(pnl, dt)
    max_dd = compute_max_drawdown(pnl)

    meta = {
        "model": model,
        "market_seed": market_seed,
        "arrival_seed": arrival_seed,
        "n_steps": n_steps,
        "dt": dt,
        "base_intensity": base_intensity,
        "order_arrival_kappa": order_arrival_kappa,
        "sigma_underlying": sigma_underlying_eff,
        "sigma_vol": sigma_vol_eff,
        "risk_aversion": {"delta": risk_aversion.delta, "vega": risk_aversion.vega, "gamma": risk_aversion.gamma},
        "true_params": true_params,
        "n_fills": len(fills),
        **quoter_meta,
    }

    return BacktestResults(
        model=model,
        times=market_path.times,
        dt=dt,
        spot_path=market_path.spot,
        variance_path=market_path.variance,
        pnl=pnl,
        cash=cash_series,
        inventory=inventory_series,
        marks=marks_series,
        delta_book=delta_book_series,
        vega_book=vega_book_series,
        fills=fills,
        sharpe=sharpe,
        max_drawdown=max_dd,
        meta=meta,
    )
