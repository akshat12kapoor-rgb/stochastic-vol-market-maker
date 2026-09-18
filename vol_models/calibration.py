"""Calibrate Heston / SABR params to a `pricing.vol_surface.generate_vol_surface`
target, via `scipy.optimize.least_squares` on implied-vol error.

**Optimizer choice (flagged, needs sign-off):** `scipy.optimize.least_squares`
with `method="trf"` (bounded trust-region-reflective), not the unconstrained
Levenberg-Marquardt (`method="lm"`). Heston/SABR params have hard feasibility
requirements (kappa,theta,sigma,v0,alpha,nu > 0; |rho| < 1) and soft
stability ones (blown-up Feller ratios push the COS truncation/implied-vol
inversion into numerically awkward regions) -- `trf` lets us pass `bounds`
directly rather than penalizing infeasible points ad hoc.

**Weighting scheme (flagged, needs sign-off):** default is `weights="equal"`
(every grid point contributes equally to the IV-RMSE objective). An
optional `weights="vega"` mode is also implemented (weight by
`pricing.black_scholes.bs_greeks(..., vol=target_iv)["vega"]`, normalized
to mean 1) -- this down-weights deep OTM wings where a given price error
translates into a huge IV error (vega -> 0), which is arguably more
representative of P&L impact for a market maker. Equal-weight is used as
the default because it is simpler and the target surface here is a
smooth synthetic construction (no bid/ask liquidity signal to weight by),
but this is exactly the kind of choice PROJECT.md rule 3 asks us to flag
rather than pick silently.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from pricing import bs_greeks, vol_surface_to_long
from vol_models.greeks_utils import implied_vol_from_price
from vol_models.heston import heston_price
from vol_models.sabr import sabr_implied_vol

__all__ = ["calibrate_heston", "calibrate_sabr", "calibrate", "surface_rmse"]

_PENALTY_IV_ERROR = 1.0  # residual (in vol units) assigned when a model price/IV is unattainable


def _grid_and_weights(surface, spot, rate, div_yield, weights, option_type="call"):
    long_df = vol_surface_to_long(surface)
    strikes = long_df["strike"].to_numpy(dtype=float)
    maturities = long_df["maturity"].to_numpy(dtype=float)
    target_iv = long_df["iv"].to_numpy(dtype=float)

    if weights == "equal":
        w = np.ones_like(target_iv)
    elif weights == "vega":
        vegas = np.array([
            bs_greeks(spot, K, T, rate, iv, option_type=option_type, div_yield=div_yield)["vega"]
            for K, T, iv in zip(strikes, maturities, target_iv)
        ])
        w = vegas / np.mean(vegas)
    else:
        raise ValueError(f"weights must be 'equal' or 'vega', got {weights!r}")
    return strikes, maturities, target_iv, w


def surface_rmse(model_iv: np.ndarray, target_iv: np.ndarray) -> float:
    """RMSE in implied-vol units (e.g. 0.02 == 2 vol points) between two IV arrays."""
    return float(np.sqrt(np.mean((np.asarray(model_iv) - np.asarray(target_iv)) ** 2)))


def calibrate_heston(surface, spot, rate=0.0, div_yield=0.0, initial_guess=None,
                      weights="equal", n_terms=128, L=12.0, max_nfev=150, verbose=0) -> dict:
    """Calibrate global Heston params (kappa, theta, sigma, rho, v0) to `surface`.

    Args:
        surface: a `pricing.vol_surface.generate_vol_surface`-shaped
            DataFrame (or any DataFrame with the same maturity-index/
            strike-columns shape) -- reshaped internally via
            `pricing.vol_surface.vol_surface_to_long`.
        spot: underlying spot used to price every grid point.
        rate, div_yield: held fixed during calibration (the target IV
            surface carries no rate information of its own -- see
            ARCHITECTURE.md for why a rate assumption has to be supplied
            and flagged).
        initial_guess: optional [kappa, theta, sigma, rho, v0] starting
            point; defaults to a generic equity-skew-shaped guess seeded
            off the surface's mean IV.
        weights: "equal" (default) or "vega" -- see module docstring.
        n_terms, L: COS method resolution/truncation used *during
            calibration* (kept lower than `heston_price`'s defaults for
            speed; the returned params can be repriced with tighter
            settings via `fair_value`/`heston_price` directly).
        max_nfev: passed to `scipy.optimize.least_squares`.

    Returns: dict with keys "kappa","theta","sigma","rho","v0","rate",
    "div_yield" -- directly usable as `params` for
    `vol_models.api.fair_value("heston", params, ...)`.
    """
    strikes, maturities, target_iv, w = _grid_and_weights(surface, spot, rate, div_yield, weights)
    mean_var = float(np.mean(target_iv) ** 2)

    x0 = np.asarray(initial_guess if initial_guess is not None else [1.5, mean_var, 0.6, -0.5, mean_var])
    lower = np.array([0.05, 1e-4, 0.02, -0.999, 1e-4])
    upper = np.array([15.0, 4.0, 3.0, 0.999, 4.0])
    x0 = np.clip(x0, lower, upper)

    def residuals(x):
        kappa, theta, sigma, rho, v0 = x
        res = np.empty(len(strikes))
        for i, (K, T, iv_t) in enumerate(zip(strikes, maturities, target_iv)):
            try:
                p = heston_price(spot, K, T, rate, kappa, theta, sigma, rho, v0,
                                  option_type="call", div_yield=div_yield, n_terms=n_terms, L=L)
                iv_m = implied_vol_from_price(p, spot, K, T, rate, option_type="call", div_yield=div_yield)
                res[i] = w[i] * (iv_m - iv_t)
            except (ValueError, FloatingPointError):
                res[i] = w[i] * _PENALTY_IV_ERROR
        return res

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf",
                            max_nfev=max_nfev, verbose=verbose)
    kappa, theta, sigma, rho, v0 = result.x
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma),
            "rho": float(rho), "v0": float(v0), "rate": rate, "div_yield": div_yield}


def calibrate_sabr(surface, spot, rate=0.0, div_yield=0.0, beta=1.0, initial_guess=None,
                    weights="equal", max_nfev=200, verbose=0) -> dict:
    """Calibrate global SABR params (alpha, rho, nu; beta fixed) to `surface`.

    `beta` defaults to 1.0 and is NOT calibrated by default -- see the
    "Beta convention" note in `vol_models.sabr` for why (identifiability
    with rho). Pass a different fixed `beta` if desired; this function
    does not currently support calibrating beta itself.

    Unlike `calibrate_heston`, this fits ONE global (alpha, rho, nu)
    triple across every maturity in `surface`, i.e. a single SABR smile
    "slice" stretched over the whole surface. Real SABR desks usually
    calibrate a separate (alpha, rho, nu) per maturity (SABR has no
    built-in term structure); we do a global fit here because
    `vol_models.api.fair_value`'s contract is one flat `params` dict per
    model (no maturity-indexed lookup) so the Market Maker agent can swap
    models with a single dict. This is a modeling simplification, flagged
    in the final report and CALIBRATION_NOTES.md -- expect a materially
    worse fit than Heston away from the maturity the params are "centered"
    on, since global SABR cannot independently flex the term structure of
    ATM level and skew the way Heston's kappa/theta/mean-reversion can.

    Returns: dict with keys "alpha","beta","rho","nu","rate","div_yield"
    -- directly usable as `params` for
    `vol_models.api.fair_value("sabr", params, ...)`.
    """
    strikes, maturities, target_iv, w = _grid_and_weights(surface, spot, rate, div_yield, weights)
    forwards = spot * np.exp((rate - div_yield) * maturities)

    x0 = np.asarray(initial_guess if initial_guess is not None else [float(np.mean(target_iv)), -0.3, 0.4])
    lower = np.array([1e-4, -0.999, 1e-4])
    upper = np.array([5.0, 0.999, 5.0])
    x0 = np.clip(x0, lower, upper)

    def residuals(x):
        alpha, rho, nu = x
        iv_m = sabr_implied_vol(forwards, strikes, maturities, alpha, beta, rho, nu)
        return w * (iv_m - target_iv)

    result = least_squares(residuals, x0, bounds=(lower, upper), method="trf",
                            max_nfev=max_nfev, verbose=verbose)
    alpha, rho, nu = result.x
    return {"alpha": float(alpha), "beta": float(beta), "rho": float(rho), "nu": float(nu),
            "rate": rate, "div_yield": div_yield}


def calibrate(model, surface, spot, **kwargs) -> dict:
    """Dispatch to `calibrate_heston` or `calibrate_sabr` by name (mirrors `fair_value`)."""
    if model == "heston":
        return calibrate_heston(surface, spot, **kwargs)
    if model == "sabr":
        return calibrate_sabr(surface, spot, **kwargs)
    raise ValueError(f"Unknown model {model!r}; expected 'heston' or 'sabr'")
