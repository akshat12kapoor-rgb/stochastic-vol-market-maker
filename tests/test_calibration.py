"""Tests for vol_models.calibration: recovers known SABR params from a SABR-generated
surface, and substantially reduces IV-RMSE vs a naive guess when calibrating Heston
against the real `pricing.vol_surface` synthetic target. Grids are kept small (a
handful of strikes/maturities, modest COS resolution) purely for test runtime --
`vol_models/comparison.py` uses a larger grid for the actual comparison report.
"""
import numpy as np
import pandas as pd
import pytest

from pricing import generate_vol_surface, vol_surface_to_long
from vol_models.calibration import calibrate_heston, calibrate_sabr, surface_rmse
from vol_models.greeks_utils import implied_vol_from_price
from vol_models.heston import heston_price
from vol_models.sabr import sabr_implied_vol

SPOT, RATE = 100.0, 0.03
STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
MATURITIES = [0.25, 0.5, 1.0, 2.0]


def _sabr_synthetic_surface(alpha, beta, rho, nu):
    rows = []
    for T in MATURITIES:
        forward = SPOT * np.exp(RATE * T)
        rows.append(sabr_implied_vol(forward, STRIKES, T, alpha, beta, rho, nu))
    return pd.DataFrame(rows, index=pd.Index(MATURITIES, name="maturity"),
                         columns=pd.Index(STRIKES, name="strike"))


def test_calibrate_sabr_recovers_known_params():
    true_params = dict(alpha=0.22, beta=1.0, rho=-0.45, nu=0.5)
    surface = _sabr_synthetic_surface(**true_params)

    fitted = calibrate_sabr(surface, SPOT, rate=RATE, beta=1.0, max_nfev=200)

    assert fitted["alpha"] == pytest.approx(true_params["alpha"], abs=5e-3)
    assert fitted["rho"] == pytest.approx(true_params["rho"], abs=5e-2)
    assert fitted["nu"] == pytest.approx(true_params["nu"], abs=5e-2)

    long_df = vol_surface_to_long(surface)
    forwards = SPOT * np.exp(RATE * long_df["maturity"].to_numpy())
    model_iv = sabr_implied_vol(forwards, long_df["strike"].to_numpy(), long_df["maturity"].to_numpy(),
                                 fitted["alpha"], fitted["beta"], fitted["rho"], fitted["nu"])
    rmse = surface_rmse(model_iv, long_df["iv"].to_numpy())
    assert rmse < 1e-3  # near-exact recovery: SABR calibrating against its own functional form


def test_calibrate_sabr_converges_for_different_true_params():
    """A second, differently-shaped SABR surface should also calibrate to low RMSE."""
    true_params = dict(alpha=0.15, beta=1.0, rho=0.2, nu=0.8)
    surface = _sabr_synthetic_surface(**true_params)
    fitted = calibrate_sabr(surface, SPOT, rate=RATE, beta=1.0, max_nfev=200)

    long_df = vol_surface_to_long(surface)
    forwards = SPOT * np.exp(RATE * long_df["maturity"].to_numpy())
    model_iv = sabr_implied_vol(forwards, long_df["strike"].to_numpy(), long_df["maturity"].to_numpy(),
                                 fitted["alpha"], fitted["beta"], fitted["rho"], fitted["nu"])
    assert surface_rmse(model_iv, long_df["iv"].to_numpy()) < 1e-3


def _heston_iv_surface(surface, params, n_terms=192, L=14.0):
    long_df = vol_surface_to_long(surface)
    ivs = []
    for K, T in zip(long_df["strike"], long_df["maturity"]):
        p = heston_price(SPOT, K, T, RATE, params["kappa"], params["theta"], params["sigma"],
                          params["rho"], params["v0"], n_terms=n_terms, L=L)
        ivs.append(implied_vol_from_price(p, SPOT, K, T, RATE))
    return np.array(ivs), long_df["iv"].to_numpy()


def test_calibrate_heston_reduces_rmse_substantially_vs_naive_guess():
    surface = generate_vol_surface(SPOT, STRIKES, MATURITIES)

    calibrated = calibrate_heston(surface, SPOT, rate=RATE, n_terms=96, L=12.0, max_nfev=100)
    model_iv, target_iv = _heston_iv_surface(surface, calibrated)
    calibrated_rmse = surface_rmse(model_iv, target_iv)

    naive = {"kappa": 1.5, "theta": np.mean(target_iv) ** 2, "sigma": 0.6,
             "rho": -0.5, "v0": np.mean(target_iv) ** 2}
    naive_iv, _ = _heston_iv_surface(surface, naive)
    naive_rmse = surface_rmse(naive_iv, target_iv)

    assert calibrated_rmse < naive_rmse / 2.0
    assert calibrated_rmse < 0.02  # within 2 vol points on this small grid


def test_calibrate_heston_returns_fair_value_ready_params():
    surface = generate_vol_surface(SPOT, STRIKES, MATURITIES)
    params = calibrate_heston(surface, SPOT, rate=RATE, n_terms=96, L=12.0, max_nfev=60)
    for key in ("kappa", "theta", "sigma", "rho", "v0", "rate", "div_yield"):
        assert key in params
    assert params["kappa"] > 0 and params["theta"] > 0 and params["sigma"] > 0 and params["v0"] > 0
    assert -1.0 < params["rho"] < 1.0


def test_calibrate_dispatch_matches_direct_functions():
    from vol_models.calibration import calibrate
    surface = _sabr_synthetic_surface(alpha=0.2, beta=1.0, rho=-0.3, nu=0.4)
    via_dispatch = calibrate("sabr", surface, SPOT, rate=RATE, beta=1.0, max_nfev=50)
    via_direct = calibrate_sabr(surface, SPOT, rate=RATE, beta=1.0, max_nfev=50)
    assert via_dispatch == via_direct

    with pytest.raises(ValueError):
        calibrate("bogus", surface, SPOT)
