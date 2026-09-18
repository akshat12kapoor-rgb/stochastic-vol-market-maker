"""SABR (Hagan, Kumar, Lesniewski, Woodward, 2002) stochastic-vol model:
Hagan's asymptotic lognormal-vol approximation, converted to price via
`pricing.black_scholes.bs_price`.

SDE (forward measure, forward F_t):
    dF_t = alpha_t * F_t^beta * dW1_t
    dalpha_t = nu * alpha_t * dW2_t
    corr(dW1_t, dW2_t) = rho

Parameters: alpha (>0, instantaneous vol-of-forward level), beta (in
[0, 1], CEV exponent -- see "Beta convention" below), rho (in (-1, 1),
spot/vol correlation), nu (>0, vol-of-vol).

Beta convention: `beta` defaults to 1.0 (lognormal SABR) throughout this
module and in calibration. This is a deliberate, flagged choice (see
ARCHITECTURE.md / final report): beta and rho are close to
unidentifiable from a single vol smile (both control skew), so
practitioners conventionally fix beta from an external view (asset
class / CEV backbone) and calibrate (alpha, rho, nu) to the smile. beta=1
(lognormal dynamics for the forward) is the natural match for an
equity-index-style vol surface quoted in Black-Scholes terms, which is
what `pricing.vol_surface.generate_vol_surface` produces. `beta` is
still a free parameter on every function below (not hardcoded) so a
caller can pass e.g. beta=0.5 (a common rates-market convention).
"""
from __future__ import annotations

import numpy as np

from pricing import bs_price
from vol_models.greeks_utils import bs_equivalent_vega, finite_diff_greeks

__all__ = ["sabr_implied_vol", "sabr_price", "sabr_greeks"]


def sabr_implied_vol(forward, strike, maturity, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied-vol asymptotic formula.

    Args:
        forward: forward price F (> 0). Scalar or array, broadcastable
            with `strike`.
        strike: strike K (> 0). Scalar or array, broadcastable with
            `forward`.
        maturity: time to expiry in years (> 0). Scalar or array,
            broadcastable with `forward`/`strike`.
        alpha, beta, rho, nu: SABR params, see module docstring.

    Returns: Black-Scholes-equivalent implied vol, same shape as the
    broadcast of `forward`/`strike`/`maturity` (float if all scalar).
    Smoothly reduces to the standard ATM Hagan formula as strike ->
    forward (no separate branch needed / no division by zero: the
    z/x(z) -> 1 limit is applied wherever |z| is small).
    """
    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    T = np.asarray(maturity, dtype=float)
    scalar_out = F.ndim == 0 and K.ndim == 0 and T.ndim == 0
    F, K, T = np.broadcast_arrays(F, K, T)
    F = F.astype(float)
    K = K.astype(float)
    T = T.astype(float)

    log_fk = np.log(F / K)
    fk_beta = (F * K) ** ((1.0 - beta) / 2.0)

    z = (nu / alpha) * fk_beta * log_fk
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
        z_over_x = np.where(np.abs(z) < 1e-8, 1.0, np.divide(z, x_z, out=np.ones_like(z), where=np.abs(z) >= 1e-8))

    denom = fk_beta * (1.0 + ((1.0 - beta) ** 2 / 24.0) * log_fk ** 2 + ((1.0 - beta) ** 4 / 1920.0) * log_fk ** 4)
    prefactor = alpha / denom
    correction = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / fk_beta ** 2)
        + (rho * beta * nu * alpha) / (4.0 * fk_beta)
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * T

    vol = prefactor * z_over_x * correction
    return float(vol) if scalar_out else vol


def sabr_price(spot, strike, maturity, rate, alpha, beta, rho, nu, option_type="call", div_yield=0.0):
    """European option price: Hagan SABR implied vol plugged into Black-Scholes.

    forward = spot * exp((rate - div_yield) * maturity); vol =
    sabr_implied_vol(forward, strike, maturity, alpha, beta, rho, nu);
    price = pricing.black_scholes.bs_price(spot, strike, maturity, rate,
    vol, option_type, div_yield). Raises ValueError if maturity <= 0
    (propagated from `bs_greeks`-style conventions elsewhere in this
    project; `bs_price` itself tolerates maturity==0 but SABR's vol
    formula does not, so this module requires maturity > 0 throughout).
    """
    if maturity <= 0:
        raise ValueError("sabr_price requires maturity > 0")
    forward = spot * np.exp((rate - div_yield) * maturity)
    vol = sabr_implied_vol(forward, strike, maturity, alpha, beta, rho, nu)
    return bs_price(spot, strike, maturity, rate, vol, option_type=option_type, div_yield=div_yield)


def sabr_greeks(spot, strike, maturity, rate, alpha, beta, rho, nu, option_type="call", div_yield=0.0) -> dict:
    """Greeks around `sabr_price`.

    Same bump conventions as `vol_models.heston.heston_greeks` for delta/
    gamma/theta/rho: delta/gamma bump spot +/-1% (recomputing the forward
    and hence the Hagan vol at each bumped spot -- a "sticky-strike-in-
    alpha", not sticky-delta, convention since alpha/beta/rho/nu are held
    fixed). theta bumps maturity +/-1e-4 (theta = -dPrice/dMaturity). rho
    bumps rate +/-1e-4.

    **vega** is the *Black-Scholes-equivalent* `dPrice/dIV`
    (`vol_models.greeks_utils.bs_equivalent_vega`), matching
    `pricing.black_scholes.bs_greeks` and `vol_models.heston.heston_greeks`'s
    vega convention exactly -- see `heston_greeks`'s docstring for the
    rationale (market_maker's quoting engine needs vega on one consistent
    scale across pricers). The raw finite-difference sensitivity to
    `alpha` (SABR's own vol-of-forward level, bump +/-1e-4) is also
    returned under `"vega_raw"` for anyone who specifically wants it -- not
    on the same scale as `"vega"`.
    """
    base = sabr_price(spot, strike, maturity, rate, alpha, beta, rho, nu, option_type, div_yield)

    def price_spot(s):
        return sabr_price(s, strike, maturity, rate, alpha, beta, rho, nu, option_type, div_yield)

    def price_rate(r):
        return sabr_price(spot, strike, maturity, r, alpha, beta, rho, nu, option_type, div_yield)

    def price_maturity(t):
        return sabr_price(spot, strike, t, rate, alpha, beta, rho, nu, option_type, div_yield)

    def price_alpha(a):
        return sabr_price(spot, strike, maturity, rate, max(a, 1e-8), beta, rho, nu, option_type, div_yield)

    greeks = finite_diff_greeks(base, price_spot, price_rate, price_maturity, price_alpha,
                                 spot, rate, maturity, alpha)
    greeks["vega"] = bs_equivalent_vega(base, spot, strike, maturity, rate,
                                         option_type=option_type, div_yield=div_yield)
    return greeks
