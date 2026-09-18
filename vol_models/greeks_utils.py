"""Shared numerical helpers for vol_models: finite-difference Greeks,
BS-equivalent vega, and Black-Scholes implied-vol inversion.

Both Heston (semi-analytic COS price) and SABR (Hagan implied vol -> BS
price) are priced through deterministic pricing functions, so plain
central-difference bumping (no common-random-numbers trick needed, unlike
`pricing.monte_carlo.mc_greeks` which bumps a noisy simulation) is enough
to get smooth, low-noise Greeks.

**Vega convention (revised):** "vega" for both Heston and SABR is defined
as the *Black-Scholes-equivalent* `dPrice/dIV`, computed via
`bs_equivalent_vega` below -- NOT a finite difference on the model's own
parameters (Heston's sqrt(v0), SABR's alpha). This makes vega genuinely
comparable across `black_scholes`/`heston`/`sabr`: all three mean "price
move per unit change in the option's own BS-equivalent implied vol," which
is what `market_maker`'s quoting engine needs since it applies a single
`risk_aversion.vega` weight to `vega_book` regardless of which pricer is
active. See ARCHITECTURE.md's "Vega convention" note for the full
rationale (this was flagged and fixed after the initial build -- the
model-parameter-bump version is still available as `finite_diff_greeks`'s
`"vega_raw"` key for anyone who specifically wants dPrice/d(sqrt(v0)) or
dPrice/dAlpha).
"""
from __future__ import annotations

from typing import Callable

from scipy.optimize import brentq

from pricing import bs_greeks, bs_price

# Bump sizes, chosen to mirror pricing.monte_carlo.mc_greeks conventions
# (spot +/-1% relative, rate/maturity 1e-4 absolute). The "vol-level" bump
# (1e-4) is used for the "vega_raw" diagnostic (Heston's sqrt(v0), SABR's
# alpha) -- NOT for the primary "vega", which is BS-equivalent (see module
# docstring).
SPOT_BUMP_REL = 0.01
RATE_BUMP = 1e-4
MATURITY_BUMP = 1e-4
VOL_LEVEL_BUMP = 1e-4


def finite_diff_greeks(
    base_price: float,
    price_given_spot: Callable[[float], float],
    price_given_rate: Callable[[float], float],
    price_given_maturity: Callable[[float], float],
    price_given_vol_level: Callable[[float], float],
    spot: float,
    rate: float,
    maturity: float,
    vol_level: float,
) -> dict:
    """Central-difference Greeks from four one-argument pricing closures.

    Each `price_given_*` closure holds every other parameter fixed and
    returns price as a function of the single bumped input. Returns a
    dict with keys delta, gamma, vega_raw, theta, rho: delta/gamma/theta/
    rho match `pricing.black_scholes.bs_greeks`'s units (theta per 1.0
    year with the sign convention theta = -dPrice/dMaturity, rho per 1.0
    rate move). `vega_raw` is `dPrice/dVolLevel` for whatever "vol-level"
    closure the caller passed in (e.g. sqrt(v0) for Heston, alpha for
    SABR) -- it is NOT the BS-equivalent vega; callers building a
    `fair_value`-facing greeks dict should overwrite/add a proper "vega"
    key via `bs_equivalent_vega` instead of relying on this one.
    """
    h_s = spot * SPOT_BUMP_REL
    p_up = price_given_spot(spot + h_s)
    p_dn = price_given_spot(spot - h_s)
    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * base_price + p_dn) / (h_s ** 2)

    h_v = VOL_LEVEL_BUMP
    v_dn_level = max(vol_level - h_v, 1e-6)
    v_up = price_given_vol_level(vol_level + h_v)
    v_dn = price_given_vol_level(v_dn_level)
    vega_raw = (v_up - v_dn) / ((vol_level + h_v) - v_dn_level)

    h_t = MATURITY_BUMP
    t_dn_level = max(maturity - h_t, 1e-8)
    t_up = price_given_maturity(maturity + h_t)
    t_dn = price_given_maturity(t_dn_level)
    theta = -(t_up - t_dn) / ((maturity + h_t) - t_dn_level)

    h_r = RATE_BUMP
    r_up = price_given_rate(rate + h_r)
    r_dn = price_given_rate(rate - h_r)
    rho = (r_up - r_dn) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega_raw": vega_raw, "theta": theta, "rho": rho}


def bs_equivalent_vega(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    div_yield: float = 0.0,
) -> float:
    """BS-equivalent vega of an arbitrary model price: dPrice/dIV.

    Since implied vol `iv` is defined by
    `bs_price(spot,strike,maturity,rate,iv,option_type,div_yield) == price`,
    the chain rule gives `dPrice/dIV == bs_greeks(...,vol=iv,...)["vega"]`
    exactly -- no finite-differencing of the upstream model's own
    parameters is needed. This is what makes "vega" comparable across
    `black_scholes`/`heston`/`sabr`: all three then mean "price move per
    unit change in the option's own BS-equivalent implied vol."

    Raises ValueError (propagated from `implied_vol_from_price`) if
    `price` is not attainable by any vol in the default inversion bracket.
    """
    iv = implied_vol_from_price(price, spot, strike, maturity, rate, option_type=option_type, div_yield=div_yield)
    return bs_greeks(spot, strike, maturity, rate, iv, option_type=option_type, div_yield=div_yield)["vega"]


def implied_vol_from_price(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    div_yield: float = 0.0,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Invert `pricing.black_scholes.bs_price` for implied vol via Brent's method.

    Used to convert a Heston (or any) model price into a Black-Scholes
    implied vol so it can be compared against a target vol surface. Raises
    ValueError if `price` is not attainable by any vol in [lo, hi] (e.g.
    price outside intrinsic-value/spot bounds due to a bad quote or
    numerical noise in the upstream pricer) -- callers doing calibration
    should catch this and assign a penalty residual rather than propagate.
    """
    def objective(vol: float) -> float:
        return bs_price(spot, strike, maturity, rate, vol, option_type=option_type, div_yield=div_yield) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"price={price} not attainable by any vol in [{lo}, {hi}] "
            f"(bounds give prices [{price - f_lo}, {price - f_hi}])"
        )
    return brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10)
