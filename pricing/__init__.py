"""Pricing package: Black-Scholes, Monte Carlo European pricer, synthetic vol surfaces.

Public API (see ARCHITECTURE.md for the full contract):

    from pricing import price
    price(model, params, spot, strike, maturity, option_type="call") -> (price, greeks_dict)

    from pricing import bs_price, bs_greeks           # closed-form Black-Scholes
    from pricing import mc_price, mc_greeks           # Monte Carlo
    from pricing import generate_vol_surface          # synthetic vol surface (DataFrame)
    from pricing import vol_surface_to_long, vol_surface_to_dict, plot_vol_surface
"""
from pricing.api import price
from pricing.black_scholes import bs_price, bs_greeks
from pricing.monte_carlo import mc_price, mc_greeks
from pricing.vol_surface import (
    generate_vol_surface,
    vol_surface_to_long,
    vol_surface_to_dict,
    plot_vol_surface,
)

__all__ = [
    "price",
    "bs_price",
    "bs_greeks",
    "mc_price",
    "mc_greeks",
    "generate_vol_surface",
    "vol_surface_to_long",
    "vol_surface_to_dict",
    "plot_vol_surface",
]
