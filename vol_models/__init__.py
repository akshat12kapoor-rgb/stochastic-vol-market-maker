"""vol_models package: Heston, SABR, and calibration to a synthetic vol surface.

Public API (see ARCHITECTURE.md for the full contract):

    from vol_models import fair_value                  # model-agnostic dispatcher
    from vol_models import heston_price, heston_greeks, heston_mc_price, feller_condition
    from vol_models import sabr_implied_vol, sabr_price, sabr_greeks
    from vol_models import calibrate_heston, calibrate_sabr, calibrate, surface_rmse
"""
from vol_models.api import fair_value
from vol_models.calibration import calibrate, calibrate_heston, calibrate_sabr, surface_rmse
from vol_models.heston import feller_condition, heston_greeks, heston_mc_price, heston_price
from vol_models.sabr import sabr_greeks, sabr_implied_vol, sabr_price

__all__ = [
    "fair_value",
    "heston_price",
    "heston_greeks",
    "heston_mc_price",
    "feller_condition",
    "sabr_implied_vol",
    "sabr_price",
    "sabr_greeks",
    "calibrate_heston",
    "calibrate_sabr",
    "calibrate",
    "surface_rmse",
]
