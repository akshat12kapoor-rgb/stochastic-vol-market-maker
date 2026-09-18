"""Model-agnostic stochastic-vol pricing entrypoint.

Mirrors `pricing.api.price`'s call shape exactly (same positional
argument order and `(price, greeks)` return) so the Market Maker agent
can swap between "black_scholes"/"monte_carlo" (via `pricing.price`) and
"heston"/"sabr" (via this `fair_value`) without any code change beyond
the model name and params dict.
"""
from __future__ import annotations

from typing import Literal

from vol_models.heston import heston_greeks, heston_price
from vol_models.sabr import sabr_greeks, sabr_price

Model = Literal["heston", "sabr"]

_HESTON_REQUIRED = ("kappa", "theta", "sigma", "rho", "v0", "rate")
_SABR_REQUIRED = ("alpha", "rho", "nu", "rate")


def _check_required(params: dict, required: tuple, model: str) -> None:
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"{model} params missing required keys: {missing}")


def fair_value(model: Model, params: dict, spot: float, strike: float, maturity: float,
               option_type: str = "call") -> tuple[float, dict[str, float]]:
    """Price a European option under Heston or SABR and return its Greeks.

    Args:
        model: "heston" (semi-analytic COS pricing) or "sabr" (Hagan
            implied vol -> Black-Scholes price).
        params: dict of model parameters. Never mutated.
            "heston" required: "kappa" (mean-reversion speed, >0),
                "theta" (long-run variance, >0), "sigma" (vol-of-vol,
                >0), "rho" (spot/vol correlation, in (-1,1)), "v0"
                (initial variance, >0), "rate" (annualized
                continuously-compounded risk-free rate).
                Optional: "div_yield" (default 0.0), "n_terms" (COS
                terms, default 256), "L" (COS truncation width in std
                devs, default 12.0).
            "sabr" required: "alpha" (vol-of-forward level, >0), "rho"
                (in (-1,1)), "nu" (vol-of-vol, >0), "rate".
                Optional: "beta" (CEV exponent, default 1.0 -- see
                `vol_models.sabr` module docstring for the convention),
                "div_yield" (default 0.0).
        spot, strike: > 0, same price units.
        maturity: time to expiry in years, > 0 (unlike
            `pricing.api.price`, maturity == 0 is NOT supported here --
            both the COS method and Hagan's formula require T > 0;
            raises ValueError).
        option_type: "call" or "put".

    Returns:
        (price, greeks): `price` is a float. `greeks` is a dict with AT
        LEAST the keys "delta","gamma","vega","theta","rho" (all floats;
        also carries a diagnostic "vega_raw" key, see below), same
        required keys/shape as `pricing.api.price`'s greeks dict. Units
        match `pricing.black_scholes.bs_greeks` for every key, including
        "vega": `vega = dPrice/dIV`, the Black-Scholes-equivalent vega
        (invert this model's own price to an implied vol, then take
        `bs_greeks(...,vol=iv,...)["vega"]` -- see
        `vol_models.greeks_utils.bs_equivalent_vega`), so it is directly
        comparable across "black_scholes"/"heston"/"sabr" and safe to use
        with a single `risk_aversion.vega` weight regardless of which
        pricer is active. A non-comparable diagnostic, the raw finite
        difference sensitivity to the model's own vol-level parameter
        (sqrt(v0) for Heston, alpha for SABR), is also present under
        "vega_raw" for anyone who specifically wants it.

    Raises:
        ValueError: unrecognized `model`, or `maturity <= 0`.
        KeyError: a required `params` key is missing.

    Invariants:
        - Never mutates `params`.
        - Fully deterministic (both pricers are closed-form/semi-analytic,
          no RNG) -- no seed parameter needed, unlike
          `pricing.api.price("monte_carlo", ...)`.
    """
    if model == "heston":
        _check_required(params, _HESTON_REQUIRED, "heston")
        div_yield = params.get("div_yield", 0.0)
        n_terms = params.get("n_terms", 256)
        L = params.get("L", 12.0)
        price = heston_price(spot, strike, maturity, params["rate"], params["kappa"], params["theta"],
                              params["sigma"], params["rho"], params["v0"],
                              option_type=option_type, div_yield=div_yield, n_terms=n_terms, L=L)
        greeks = heston_greeks(spot, strike, maturity, params["rate"], params["kappa"], params["theta"],
                                params["sigma"], params["rho"], params["v0"],
                                option_type=option_type, div_yield=div_yield, n_terms=n_terms, L=L)
    elif model == "sabr":
        _check_required(params, _SABR_REQUIRED, "sabr")
        beta = params.get("beta", 1.0)
        div_yield = params.get("div_yield", 0.0)
        price = sabr_price(spot, strike, maturity, params["rate"], params["alpha"], beta,
                            params["rho"], params["nu"], option_type=option_type, div_yield=div_yield)
        greeks = sabr_greeks(spot, strike, maturity, params["rate"], params["alpha"], beta,
                              params["rho"], params["nu"], option_type=option_type, div_yield=div_yield)
    else:
        raise ValueError(f"Unknown model {model!r}; expected 'heston' or 'sabr'")

    return price, greeks
