"""Poisson order-arrival simulation.

`market_maker.quoting.generate_quotes`'s `order_arrival_kappa` parameter
assumes (but does not implement) a fill-intensity model of the classic
Avellaneda-Stoikov/Gueant form

    intensity(distance) = A * exp(-kappa * distance)

where `distance` is how far a quote sits from a reference "true" price and
`A` is the intensity at zero distance. This module implements that arrival
PROCESS: given a quote's distance from the reference true price on each
side (bid/ask), it draws the number of Poisson arrivals that hit that quote
within one bar of length `dt`.

**Consistency requirement (explicitly flagged by the Market Maker agent,
see ARCHITECTURE.md's `market_maker` decision #5):** the `kappa` used here
MUST be the same `order_arrival_kappa` passed to `generate_quotes`, or the
quoting engine's spread formula (which assumes this exact intensity model)
is no longer consistent with the simulated market dynamics. `backtest.engine
.run_backtest` enforces this by construction -- it takes a single
`order_arrival_kappa` argument and threads it to both `generate_quotes` and
this module's `poisson_fill_counts`, so the two values cannot silently
diverge.

**Base intensity `A` (flagged decision -- needs sign-off):** exposed as
`base_intensity`, in arrivals-per-year at zero quote distance (i.e. the
rate at which a quote sitting exactly at the reference true price would be
hit). `backtest.engine`'s default is `250.0` (roughly one at-the-money-
priced arrival per trading day per side per contract) -- an order-of-
magnitude placeholder for a moderately liquid single-name options market,
not calibrated to any real order-flow data (none is available in this
synthetic setup). Flagged per PROJECT.md rule 3.
"""
from __future__ import annotations

import math

import numpy as np

# Hard cap on the per-bar expected arrival count on any one side, applied
# AFTER intensity*dt is computed. Guards against a pathological blow-up when
# a quote sits on the "wrong" side of the reference price (distance < 0 ->
# exp(-kappa*distance) > 1 -> intensity > base_intensity), which can happen
# transiently for economically-sensible reasons (e.g. a stale/skewed quote
# right after a large realized market move) but should never make
# `numpy.random.Generator.poisson`'s `lam` argument unreasonably large.
DEFAULT_MAX_EXPECTED_ARRIVALS_PER_BAR = 3.0


def fill_intensity(distance: float, base_intensity: float, kappa: float) -> float:
    """Classic AS/Gueant fill-intensity: `base_intensity * exp(-kappa * distance)`.

    `distance` is typically `>= 0` (a quote posted away from the reference
    true price, e.g. `ask - true_price` or `true_price - bid`), in which
    case this returns a value `<= base_intensity` that decays as the quote
    gets further from the reference price. A negative `distance` (an
    aggressively mispriced quote, e.g. a bid ABOVE the reference price)
    returns a value `> base_intensity` -- not clamped here (see
    `poisson_fill_counts` for where the expected-arrivals cap is applied).
    """
    return base_intensity * math.exp(-kappa * distance)


def poisson_fill_counts(
    distance_bid: float,
    distance_ask: float,
    base_intensity: float,
    kappa: float,
    dt: float,
    rng: np.random.Generator,
    max_expected_arrivals_per_bar: float = DEFAULT_MAX_EXPECTED_ARRIVALS_PER_BAR,
) -> tuple[int, int]:
    """Draw the number of Poisson arrivals hitting the bid and the ask within one bar.

    Args:
        distance_bid: `true_price - bid` -- how far below the reference true
            price the market maker's bid sits (normally `>= 0`; the market
            maker's bid is lifted -- i.e. the market maker BUYS -- by a
            counterparty willing to sell at that price).
        distance_ask: `ask - true_price` -- how far above the reference true
            price the market maker's ask sits (normally `>= 0`; the market
            maker's ask is lifted -- i.e. the market maker SELLS).
        base_intensity: `A` in `A * exp(-kappa * distance)`, arrivals/year at
            zero distance.
        kappa: exponential decay rate of intensity with distance -- MUST
            match the `order_arrival_kappa` passed to
            `market_maker.quoting.generate_quotes` for this same quote (see
            module docstring).
        dt: bar length in years.
        rng: a `numpy.random.Generator` (caller owns seeding/reuse across
            bars, so the arrival stream is reproducible end-to-end -- see
            `backtest.engine.run_backtest`'s `arrival_seed`).
        max_expected_arrivals_per_bar: cap applied to `intensity * dt` on
            each side before sampling, purely a numerical-sanity guard (see
            module docstring).

    Returns:
        `(n_buy, n_sell)`: `n_buy` is the number of times the market maker's
        BID was lifted this bar (market maker buys `n_buy` units); `n_sell`
        is the number of times the ASK was lifted (market maker sells
        `n_sell` units). Both are independent Poisson draws -- a bar can
        have simultaneous buy- and sell-side fills.
    """
    lam_bid = min(fill_intensity(distance_bid, base_intensity, kappa) * dt, max_expected_arrivals_per_bar)
    lam_ask = min(fill_intensity(distance_ask, base_intensity, kappa) * dt, max_expected_arrivals_per_bar)
    n_buy = int(rng.poisson(lam_bid))
    n_sell = int(rng.poisson(lam_ask))
    return n_buy, n_sell
