"""Shared numerical helpers for vol_models: finite-difference Greeks and
Black-Scholes implied-vol inversion.

Both Heston (semi-analytic COS price) and SABR (Hagan implied vol -> BS
price) are priced through deterministic pricing functions, so plain
central-difference bumping (no common-random-numbers trick needed, unlike
`pricing.monte_carlo.mc_greeks` which bumps a noisy simulation) is enough
to get smooth, low-noise Greeks.
"""
from __future__ import annotations

from typing import Callable

from scipy.optimize import brentq

from pricing import bs_price

# Bump sizes, chosen to mirror pricing.monte_carlo.mc_greeks conventions
# (spot +/-1% relative, rate/maturity 1e-4 absolute). The "vol-level" bump
# (1e-4) is used for Heston's sqrt(v0) and SABR's alpha -- see each
# module's fair_value section in ARCHITECTURE.md for what "vega" means
# for that model.
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
    dict with keys delta, gamma, vega, theta, rho -- same shape/units as
    `pricing.black_scholes.bs_greeks` (vega per 1.0 vol-level move, theta
    per 1.0 year with the sign convention theta = -dPrice/dMaturity, rho
    per 1.0 rate move).
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
    vega = (v_up - v_dn) / ((vol_level + h_v) - v_dn_level)

    h_t = MATURITY_BUMP
    t_dn_level = max(maturity - h_t, 1e-8)
    t_up = price_given_maturity(maturity + h_t)
    t_dn = price_given_maturity(t_dn_level)
    theta = -(t_up - t_dn) / ((maturity + h_t) - t_dn_level)

    h_r = RATE_BUMP
    r_up = price_given_rate(rate + h_r)
    r_dn = price_given_rate(rate - h_r)
    rho = (r_up - r_dn) / (2 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


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
