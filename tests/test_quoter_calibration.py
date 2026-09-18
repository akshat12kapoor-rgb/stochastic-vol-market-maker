"""Tests for backtest.quoter_calibration -- the central framing-decision code.

Also see `tests/test_engine.py` for the integration-level check that
`run_backtest(model="heston")` actually uses these (not the true params).
"""
import pandas as pd
import pytest

from backtest.market_sim import TRUE_MARKET_PARAMS
from backtest.quoter_calibration import (
    build_true_market_iv_surface,
    calibrate_heston_quoter_params,
    compute_flat_vol_for_bs_quoter,
)
from vol_models.api import fair_value


def test_flat_vol_is_a_plausible_implied_vol():
    vol = compute_flat_vol_for_bs_quoter(TRUE_MARKET_PARAMS)
    assert 0.05 < vol < 1.0


def test_build_true_market_iv_surface_shape_matches_calibration_contract():
    surface = build_true_market_iv_surface(TRUE_MARKET_PARAMS, strikes=(90.0, 100.0, 110.0), maturities=(0.25, 0.5))
    assert isinstance(surface, pd.DataFrame)
    assert surface.index.name == "maturity"
    assert surface.columns.name == "strike"
    assert list(surface.index) == sorted(surface.index)
    assert list(surface.columns) == sorted(surface.columns)
    assert surface.shape == (2, 3)
    assert (surface.to_numpy() > 0).all()


def test_calibrated_heston_params_have_required_keys_and_are_usable(calibrated_heston_params):
    params = calibrated_heston_params
    for key in ("kappa", "theta", "sigma", "rho", "v0", "rate"):
        assert key in params
    assert params["kappa"] > 0
    assert params["theta"] > 0
    assert params["sigma"] > 0
    assert -1 < params["rho"] < 1
    assert params["v0"] > 0

    # round-trips through vol_models.fair_value without error
    price, greeks = fair_value("heston", params, TRUE_MARKET_PARAMS.spot0, 100.0, 0.5, "call")
    assert price > 0
    assert set(("delta", "gamma", "vega", "theta", "rho")).issubset(greeks)


def test_calibrated_params_are_not_simply_the_true_params(calibrated_heston_params):
    # Embodies the central framing decision (see module docstring in
    # backtest/quoter_calibration.py): the Heston quoter is fit from a
    # noisy synthetic surface, not handed TRUE_MARKET_PARAMS verbatim, so
    # its calibrated params should generally differ at least slightly from
    # the true generating params (a bit-exact match would indicate the
    # "calibration" secretly leaked the true params instead of fitting).
    calibrated = calibrated_heston_params
    true = TRUE_MARKET_PARAMS
    diffs = [
        abs(calibrated["kappa"] - true.kappa),
        abs(calibrated["theta"] - true.theta),
        abs(calibrated["sigma"] - true.sigma),
        abs(calibrated["rho"] - true.rho),
        abs(calibrated["v0"] - true.v0),
    ]
    assert any(d > 1e-6 for d in diffs)


def test_calibration_is_deterministic_given_seed():
    p1 = calibrate_heston_quoter_params(TRUE_MARKET_PARAMS, max_nfev=30, seed=11)
    p2 = calibrate_heston_quoter_params(TRUE_MARKET_PARAMS, max_nfev=30, seed=11)
    assert p1 == p2
