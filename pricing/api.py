"""Model-agnostic pricing entrypoint.

Other packages should call `pricing.api.price` (re-exported as
`pricing.price`) rather than reaching into `black_scholes` or
`monte_carlo` directly, so the pricing model can be swapped by changing
one string.
"""
from __future__ import annotations

from typing import Literal

from pricing import black_scholes, monte_carlo
from pricing.black_scholes import OptionType

Model = Literal["black_scholes", "monte_carlo"]

_DISPATCH = {
    "black_scholes": black_scholes.price,
    "monte_carlo": monte_carlo.price,
}


def price(model: Model, params: dict, spot: float, strike: float, maturity: float,
          option_type: OptionType = "call") -> tuple[float, dict[str, float]]:
    """Price a European option and return its Greeks.

    Args:
        model: "black_scholes" (closed-form) or "monte_carlo" (simulation,
            100k antithetic paths by default). Both models price the same
            GBM/Black-Scholes-Merton dynamics; monte_carlo carries sampling
            noise (see `pricing.monte_carlo.mc_price` for the stderr).
        params: dict of model parameters.
            Required: "rate" (float, annualized continuously-compounded
                risk-free rate), "vol" (float, annualized Black-Scholes
                volatility, e.g. 0.20 for 20%).
            Optional: "div_yield" (float, annualized continuous dividend
                yield, default 0.0).
            monte_carlo only, optional: "n_paths" (int, default 100_000,
                total simulated paths), "seed" (int or None, default 0).
        spot: current underlying price (> 0).
        strike: strike price (> 0), same units as spot.
        maturity: time to expiry in YEARS (>= 0; use e.g. 30/365 for 30
            calendar days). maturity == 0 returns intrinsic value (only
            supported by black_scholes; monte_carlo requires maturity > 0).
        option_type: "call" or "put".

    Returns:
        (price, greeks): `price` is a float in the same currency units as
        spot/strike. `greeks` is a dict with keys "delta", "gamma",
        "vega", "theta", "rho" (all floats). Units/conventions:
            delta: dPrice/dSpot (unitless, per 1.0 spot move).
            gamma: d^2Price/dSpot^2 (per 1.0 spot move).
            vega:  dPrice/dVol, per 1.0 (100 vol points) change in vol;
                   divide by 100 for "per 1 vol point".
            theta: dPrice/dCalendarTime, per 1.0 YEAR of time decay
                   (divide by 365 for per-calendar-day); typically
                   negative for long options.
            rho:   dPrice/dRate, per 1.0 (100%) change in rate; divide by
                   100 for "per 1% rate move".

    Invariants:
        - Both models are internally consistent with each other to within
          the tolerance documented/tested in
          tests/test_black_scholes_vs_monte_carlo.py (see ARCHITECTURE.md
          for the exact figure and justification).
        - `price` never mutates `params`.
        - Deterministic for monte_carlo when `params["seed"]` is a fixed
          int (default 0); pass seed=None for nondeterministic draws.
    """
    try:
        fn = _DISPATCH[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model!r}; expected one of {list(_DISPATCH)}") from exc
    return fn(model, params, spot, strike, maturity, option_type)
