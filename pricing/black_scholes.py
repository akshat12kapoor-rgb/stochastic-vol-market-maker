"""Black-Scholes-Merton European option pricing with analytic Greeks.

All formulas assume a lognormal underlying under the risk-neutral measure
with continuous dividend yield q. Time is measured in years (ACT/365-style
year fractions), rates and vol are annualized, continuously-compounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(spot: float, strike: float, maturity: float, rate: float,
           vol: float, div_yield: float = 0.0) -> tuple[float, float]:
    """Compute d1, d2 for the Black-Scholes formula.

    maturity, rate, vol, div_yield must be non-negative (maturity, vol
    strictly positive) for a numerically meaningful result; callers with
    maturity == 0 or vol == 0 should use the intrinsic-value limit instead
    (handled by callers of this helper, not here).
    """
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol ** 2) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def bs_price(spot: float, strike: float, maturity: float, rate: float,
             vol: float, option_type: OptionType = "call",
             div_yield: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option.

    Args:
        spot: current underlying price (> 0), in price units.
        strike: strike price (> 0), same price units as spot.
        maturity: time to expiry in years (>= 0). maturity == 0 returns
            intrinsic value.
        rate: continuously-compounded annualized risk-free rate (e.g. 0.05
            for 5%).
        vol: annualized Black-Scholes volatility (e.g. 0.20 for 20%),
            must be > 0 for maturity > 0 (vol == 0 handled as a
            deterministic-forward limit).
        option_type: "call" or "put".
        div_yield: continuously-compounded annualized dividend yield
            (default 0.0).

    Returns:
        The option's present value, in the same currency units as spot/strike.
    """
    if maturity <= 0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    if vol <= 0:
        forward = spot * np.exp((rate - div_yield) * maturity)
        disc = np.exp(-rate * maturity)
        if option_type == "call":
            return disc * max(forward - strike, 0.0)
        return disc * max(strike - forward, 0.0)

    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol, div_yield)
    disc_q = np.exp(-div_yield * maturity)
    disc_r = np.exp(-rate * maturity)

    if option_type == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def bs_greeks(spot: float, strike: float, maturity: float, rate: float,
              vol: float, option_type: OptionType = "call",
              div_yield: float = 0.0) -> dict[str, float]:
    """Analytic Black-Scholes Greeks for a European option.

    Args/units: identical to `bs_price`.

    Returns a dict with keys:
        delta: dPrice/dSpot, unitless (per 1.0 move in spot).
        gamma: d^2Price/dSpot^2, per 1.0 move in spot (same for call/put).
        vega:  dPrice/dVol, per 1.0 (100 vol points) change in vol.
               Divide by 100 for the conventional "per 1 vol point" vega.
        theta: dPrice/dTime, per 1.0 year of time decay (i.e. this is the
               *annualized* theta; divide by 365 for per-calendar-day decay).
               Defined as the derivative w.r.t. calendar time, so it is
               typically negative for long options (price decreasing as
               maturity shrinks holding spot/vol fixed).
        rho:   dPrice/dRate, per 1.0 (100%) change in rate. Divide by 100
               for "per 1% rate move" rho.

    At maturity == 0 all Greeks except delta/gamma degenerate; this
    function requires maturity > 0 and vol > 0 (use finite differences
    near those limits if needed).
    """
    if maturity <= 0 or vol <= 0:
        raise ValueError("bs_greeks requires maturity > 0 and vol > 0; "
                          "use bs_price for the intrinsic-value limit.")

    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol, div_yield)
    sqrt_t = np.sqrt(maturity)
    disc_q = np.exp(-div_yield * maturity)
    disc_r = np.exp(-rate * maturity)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
                  - rate * strike * disc_r * norm.cdf(d2)
                  + div_yield * spot * disc_q * norm.cdf(d1))
        rho = strike * maturity * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
                  + rate * strike * disc_r * norm.cdf(-d2)
                  - div_yield * spot * disc_q * norm.cdf(-d1))
        rho = -strike * maturity * disc_r * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def price(model: Literal["black_scholes"], params: dict, spot: float,
          strike: float, maturity: float, option_type: OptionType = "call"
          ) -> tuple[float, dict[str, float]]:
    """Unified pricing entrypoint for the Black-Scholes model.

    This mirrors the `pricing.api.price` dispatcher signature but is
    scoped to the Black-Scholes model only; prefer importing `price` from
    `pricing.api` for a model-agnostic call. See `pricing.api.price` for
    the full contract (params dict keys, return shape).
    """
    if model != "black_scholes":
        raise ValueError(f"black_scholes.price only supports model='black_scholes', got {model!r}")
    rate = params["rate"]
    vol = params["vol"]
    div_yield = params.get("div_yield", 0.0)
    p = bs_price(spot, strike, maturity, rate, vol, option_type, div_yield)
    if maturity <= 0 or vol <= 0:
        # Greeks undefined/degenerate at these limits; return zeros except delta.
        greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        if maturity <= 0:
            if option_type == "call":
                greeks["delta"] = 1.0 if spot > strike else 0.0
            else:
                greeks["delta"] = -1.0 if spot < strike else 0.0
    else:
        greeks = bs_greeks(spot, strike, maturity, rate, vol, option_type, div_yield)
    return p, greeks
