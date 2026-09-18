"""Heston (1993) stochastic-volatility model: semi-analytic COS pricing plus
a full-truncation-Euler Monte Carlo simulator.

SDE (risk-neutral measure):
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t
    corr(dW1_t, dW2_t) = rho

Parameters (all floats): kappa (mean-reversion speed, >0), theta (long-run
variance level, >0), sigma (vol-of-vol, >0), rho (spot/vol correlation, in
(-1, 1)), v0 (initial variance, >0). "vol" units throughout this module
are *variance* (v0, theta) unless a function name says otherwise; e.g.
theta=0.04 means a long-run annualized vol of sqrt(0.04)=20%.

Feller condition: 2*kappa*theta >= sigma**2 keeps the CIR variance process
away from zero (and mathematically, from a continuous-time diffusion,
it stays strictly positive). Calibrated params routinely violate it in
practice, and the discrete-time Euler scheme can push v negative even
when Feller holds -- `heston_mc_price` handles this via full-truncation
Euler (see its docstring), never taking sqrt of a negative number.
"""
from __future__ import annotations

import numpy as np

from vol_models.greeks_utils import finite_diff_greeks

__all__ = [
    "feller_condition",
    "heston_price",
    "heston_greeks",
    "heston_mc_price",
]


def feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """True iff 2*kappa*theta >= sigma**2 (continuous-time CIR variance stays > 0)."""
    return 2.0 * kappa * theta >= sigma ** 2


# ---------------------------------------------------------------------------
# Characteristic function (Heston "little trap" form, Albrecher et al. 2007)
# ---------------------------------------------------------------------------
def _cf_log_return(u, maturity, rate, div_yield, kappa, theta, sigma, rho, v0):
    """Characteristic function of ln(S_T / S_0) at real argument(s) `u`.

    Uses the numerically-stable "little trap" parameterization: the sign
    convention on `d` in `g` is chosen so |g| < 1 for the whole real-u
    range needed by the COS method, avoiding the branch-cut discontinuity
    of the original 1993 Heston formula (Albrecher, Mayer, Schoutens,
    Tistaert, "The Little Heston Trap", 2007).
    """
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma ** 2 * (iu + u ** 2))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_dT = np.exp(-d * maturity)
    C = iu * (rate - div_yield) * maturity + (kappa * theta / sigma ** 2) * (
        (kappa - rho * sigma * iu - d) * maturity - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = ((kappa - rho * sigma * iu - d) / sigma ** 2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    return np.exp(C + D * v0)


def _cumulants(maturity, rate, div_yield, kappa, theta, sigma, rho, v0):
    """First two cumulants of ln(S_T/S_0), via finite differences on the CGF.

    Computed numerically off the same `_cf_log_return` used for pricing
    (rather than the closed-form Heston cumulant expressions, which are
    long and easy to mistranscribe) so the COS truncation range is always
    self-consistent with the characteristic function actually being
    integrated.
    """
    def logK(t):
        cf = _cf_log_return(-1j * t, maturity, rate, div_yield, kappa, theta, sigma, rho, v0)
        return float(np.log(cf.real))

    # h=1e-2 empirically balances truncation error (too large) against
    # floating-point cancellation in the second difference (too small --
    # e.g. h=1e-4 loses all precision on c2 for near-deterministic params
    # like sigma~0, where K(t) itself is O(1e-6) near t=0).
    h = 1e-2
    k_hi, k_mid, k_lo = logK(h), logK(0.0), logK(-h)
    c1 = (k_hi - k_lo) / (2 * h)
    c2 = (k_hi - 2 * k_mid + k_lo) / (h ** 2)
    return c1, max(c2, 1e-8)


def _chi_psi(w, a, b, c, d):
    """Fang-Oosterlee (2008) chi_k/psi_k coefficients, `w = k*pi/(b-a)`."""
    term1 = np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
    term2 = w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)
    chi = (term1 + term2) / (1.0 + w ** 2)

    psi = np.empty_like(w, dtype=float)
    nz = w != 0
    psi[nz] = (np.sin(w[nz] * (d - a)) - np.sin(w[nz] * (c - a))) / w[nz]
    psi[~nz] = d - c
    return chi, psi


def heston_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                  option_type="call", div_yield=0.0, n_terms=256, L=12.0):
    """European option price under Heston via the COS method (Fang & Oosterlee, 2008).

    Args:
        spot, strike: > 0, same price units.
        maturity: time to expiry in years, > 0 (no maturity==0 special case
            -- use intrinsic value directly if you need that boundary).
        rate, div_yield: annualized continuously-compounded.
        kappa, theta, sigma, rho, v0: Heston params, see module docstring.
        option_type: "call" or "put".
        n_terms: number of COS cosine-series terms (default 256).
        L: truncation width in standard deviations of ln(S_T/S_0), i.e.
            the integration range is [c1-L*sqrt(c2), c1+L*sqrt(c2)] shifted
            by ln(S0/K) (default 12 -- generous; see ARCHITECTURE.md for
            why this default needs sign-off).

    Returns: float price. Raises ValueError if maturity <= 0.
    """
    if maturity <= 0:
        raise ValueError("heston_price requires maturity > 0")
    c1, c2 = _cumulants(maturity, rate, div_yield, kappa, theta, sigma, rho, v0)
    width = 2.0 * L * np.sqrt(c2)
    k = np.arange(n_terms)
    w = k * np.pi / width
    cf = _cf_log_return(w, maturity, rate, div_yield, kappa, theta, sigma, rho, v0)

    x = np.log(spot / strike)
    a = x + c1 - L * np.sqrt(c2)
    b = a + width
    # phi(u; x) = exp(i*u*x) * cf_ratio(u); combined with the standard
    # COS phase exp(-i*u*a) this is exp(i*u*(x-a)) -- note x-a is in fact
    # K-independent (= L*sqrt(c2) - c1) since `a` was shifted by the same
    # x, but writing it this way keeps the K-dependence explicit and
    # matches the Fang & Oosterlee (2008) formula directly.
    phase = np.exp(1j * w * (x - a))

    if option_type == "call":
        c_lo, c_hi = 0.0, b
    elif option_type == "put":
        c_lo, c_hi = a, 0.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    chi, psi = _chi_psi(w, a, b, c_lo, c_hi)
    if option_type == "call":
        Vk = (2.0 / width) * strike * (chi - psi)
    else:
        Vk = (2.0 / width) * strike * (-chi + psi)

    terms = (cf * phase).real * Vk
    terms[0] *= 0.5
    return float(np.exp(-rate * maturity) * np.sum(terms))


def heston_greeks(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                   option_type="call", div_yield=0.0, n_terms=256, L=12.0) -> dict:
    """Central-difference Greeks around `heston_price`.

    delta/gamma: bump spot +/-1%. theta: bump maturity +/-1e-4 (theta =
    -dPrice/dMaturity). rho: bump rate +/-1e-4. vega: bump sqrt(v0) (the
    vol-like, annualized-vol-scale parameter) by +/-1e-4 -- i.e. this is
    "sensitivity to the level of instantaneous vol", not a Black-Scholes
    vol vega; see ARCHITECTURE.md for why this convention was chosen (to
    keep the same units as `pricing.black_scholes.bs_greeks`'s vega) and
    that it needs sign-off. Every other Heston/SABR param (kappa, theta,
    rho, and for SABR beta/rho/nu) is held fixed -- these Greeks are
    "sticky-model-parameter" Greeks, not sticky-strike or sticky-delta.
    """
    base = heston_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                         option_type, div_yield, n_terms, L)

    def price_spot(s):
        return heston_price(s, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                             option_type, div_yield, n_terms, L)

    def price_rate(r):
        return heston_price(spot, strike, maturity, r, kappa, theta, sigma, rho, v0,
                             option_type, div_yield, n_terms, L)

    def price_maturity(t):
        return heston_price(spot, strike, t, rate, kappa, theta, sigma, rho, v0,
                             option_type, div_yield, n_terms, L)

    def price_vol_level(vl):
        return heston_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, max(vl, 1e-8) ** 2,
                             option_type, div_yield, n_terms, L)

    vol_level = np.sqrt(v0)
    return finite_diff_greeks(base, price_spot, price_rate, price_maturity, price_vol_level,
                               spot, rate, maturity, vol_level)


def heston_mc_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0,
                     option_type="call", div_yield=0.0, n_paths=50_000, n_steps=100,
                     antithetic=True, seed=0):
    """Monte Carlo Heston price via full-truncation Euler discretization.

    Full truncation (Lord, Koekkoek & van Dijk, 2010): at every step, the
    variance drift and diffusion use `v_pos = max(v, 0)` wherever `v`
    would otherwise appear under a square root or in the mean-reversion
    drift's current-level term:
        v_{n+1} = v_n + kappa*(theta - v_pos_n)*dt + sigma*sqrt(v_pos_n*dt)*Zv
        logS_{n+1} = logS_n + (r-q-0.5*v_pos_n)*dt + sqrt(v_pos_n*dt)*Zs
    `v_n` itself (not `v_pos_n`) is carried forward as the state and can
    go transiently negative between steps when the Feller condition is
    violated (2*kappa*theta < sigma**2) or discretization noise is large;
    it is floored to 0 only where it feeds a square root or the drift's
    v-dependent term, so the scheme never takes sqrt of a negative number
    or produces NaN/complex values regardless of the Feller condition.

    Variance reduction: antithetic variates (default `antithetic=True`),
    same convention as `pricing.monte_carlo.mc_price` -- `n_paths // 2`
    correlated-normal path pairs (Z, -Z) [same sign flip applied to both
    the spot and variance driving noise], discounted payoffs averaged
    per pair before computing mean/stderr across pairs.

    Returns: (price, stderr) -- stderr is the 1-sigma Monte Carlo standard
    error in price units (paired-average variance for antithetic).
    Raises ValueError if maturity <= 0.
    """
    if maturity <= 0:
        raise ValueError("heston_mc_price requires maturity > 0")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    sqdt = np.sqrt(dt)

    if antithetic:
        n_half = n_paths // 2
        Z1_half = rng.standard_normal((n_half, n_steps))
        Z2_half = rng.standard_normal((n_half, n_steps))
        Z1 = np.vstack([Z1_half, -Z1_half])
        Z2 = np.vstack([Z2_half, -Z2_half])
    else:
        n_half = None
        Z1 = rng.standard_normal((n_paths, n_steps))
        Z2 = rng.standard_normal((n_paths, n_steps))

    Zv = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

    n_sim = Z1.shape[0]
    logS = np.full(n_sim, np.log(spot))
    v = np.full(n_sim, v0, dtype=float)

    for step in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        logS += (rate - div_yield - 0.5 * v_pos) * dt + sqrt_v * sqdt * Z1[:, step]
        v = v + kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqdt * Zv[:, step]

    ST = np.exp(logS)
    if option_type == "call":
        payoff = np.maximum(ST - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    disc_payoff = np.exp(-rate * maturity) * payoff

    if antithetic:
        pair_avg = 0.5 * (disc_payoff[:n_half] + disc_payoff[n_half:])
        price = float(np.mean(pair_avg))
        stderr = float(np.std(pair_avg, ddof=1) / np.sqrt(n_half))
    else:
        price = float(np.mean(disc_payoff))
        stderr = float(np.std(disc_payoff, ddof=1) / np.sqrt(n_paths))

    return price, stderr
