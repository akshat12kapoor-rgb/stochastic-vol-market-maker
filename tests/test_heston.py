"""Tests for vol_models.heston: COS pricing, MC cross-validation, Feller handling."""
import numpy as np
import pytest

from pricing import bs_greeks, bs_price
from vol_models.api import fair_value
from vol_models.greeks_utils import implied_vol_from_price
from vol_models.heston import feller_condition, heston_greeks, heston_mc_price, heston_price

SPOT, RATE = 100.0, 0.03


def test_feller_condition():
    assert feller_condition(kappa=2.0, theta=0.04, sigma=0.2) is True  # 2*2*.04=.16 >= .04
    assert feller_condition(kappa=1.0, theta=0.04, sigma=1.0) is False  # 2*1*.04=.08 < 1.0


def test_heston_matches_bs_in_deterministic_limit():
    """sigma -> 0 with v0 == theta == vol**2 collapses Heston to GBM/Black-Scholes."""
    vol = 0.2
    v0 = theta = vol ** 2
    kappa, sigma, rho = 2.0, 1e-4, 0.0
    for K in [70, 85, 100, 115, 130]:
        for T in [0.25, 1.0, 2.0]:
            hp = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0)
            bp = bs_price(SPOT, K, T, RATE, vol)
            assert hp == pytest.approx(bp, abs=1e-4)


def test_heston_put_call_parity():
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.5, -0.6, 0.05
    K, T = 100.0, 1.0
    call = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0, option_type="call")
    put = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0, option_type="put")
    # C - P = S*exp(-qT) - K*exp(-rT), model-independent (both COS-priced under the same
    # risk-neutral dynamics/discounting).
    assert (call - put) == pytest.approx(SPOT - K * np.exp(-RATE * T), abs=1e-5)


# (kappa, theta, sigma, rho, v0, strike, maturity, label)
_CROSS_VAL_CONFIGS = [
    (2.0, 0.04, 0.4, -0.5, 0.045, 100, 0.5, "ATM, Feller satisfied"),
    (2.0, 0.04, 0.4, -0.5, 0.045, 85, 1.0, "ITM call, Feller satisfied"),
    (2.0, 0.04, 0.4, -0.5, 0.045, 120, 0.25, "OTM call, short maturity"),
    (1.0, 0.05, 0.9, -0.6, 0.06, 100, 1.0, "ATM, Feller violated (2kt=0.1 < sigma^2=0.81)"),
    (1.0, 0.05, 0.9, -0.6, 0.06, 80, 2.0, "ITM, Feller violated, long maturity"),
]


@pytest.mark.parametrize("kappa,theta,sigma,rho,v0,K,T,label", _CROSS_VAL_CONFIGS, ids=[c[-1] for c in _CROSS_VAL_CONFIGS])
def test_heston_cos_vs_mc_cross_validation(kappa, theta, sigma, rho, v0, K, T, label):
    """COS price and full-truncation-Euler MC price must agree to within 6 MC stderrs.

    Mirrors the Pricing Engine Agent's `abs(mc_price - bs_price) < 5*stderr` style
    cross-check (see tests/test_monte_carlo.py), widened to 6 sigma here because,
    unlike the exact one-step GBM simulation `pricing.monte_carlo` uses, Heston MC
    also carries a small Euler discretization bias on top of sampling noise (see
    `heston_mc_price` docstring) -- empirically ~0.01-0.03 price units at
    n_steps=300 even for the Feller-violated configs above, well under one stderr
    at the path counts used here.
    """
    cos_price = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0, n_terms=256, L=14)
    mc_price, stderr = heston_mc_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0,
                                        n_paths=100_000, n_steps=300, seed=7)
    assert abs(cos_price - mc_price) < 6 * stderr, (
        f"{label}: cos={cos_price:.4f} mc={mc_price:.4f} stderr={stderr:.4f}"
    )


def test_heston_mc_survives_severe_feller_violation():
    """Full truncation Euler must never produce NaN/inf/complex, however badly Feller is violated."""
    kappa, theta, sigma, rho, v0 = 1.0, 0.05, 1.2, -0.7, 0.05
    assert not feller_condition(kappa, theta, sigma)
    price, stderr = heston_mc_price(SPOT, 100.0, 1.0, RATE, kappa, theta, sigma, rho, v0,
                                     n_paths=50_000, n_steps=200, seed=3)
    assert np.isfinite(price) and np.isfinite(stderr)
    assert price > 0


def test_heston_mc_antithetic_reduces_variance():
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.5, -0.6, 0.05
    _, se_anti = heston_mc_price(SPOT, 100.0, 1.0, RATE, kappa, theta, sigma, rho, v0,
                                  n_paths=40_000, n_steps=100, antithetic=True, seed=1)
    _, se_plain = heston_mc_price(SPOT, 100.0, 1.0, RATE, kappa, theta, sigma, rho, v0,
                                   n_paths=40_000, n_steps=100, antithetic=False, seed=1)
    assert se_anti < se_plain


def test_heston_greeks_sane():
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.5, -0.6, 0.05
    g_call = heston_greeks(SPOT, 100.0, 1.0, RATE, kappa, theta, sigma, rho, v0, option_type="call")
    g_put = heston_greeks(SPOT, 100.0, 1.0, RATE, kappa, theta, sigma, rho, v0, option_type="put")
    assert {"delta", "gamma", "vega", "theta", "rho"} <= set(g_call)  # "vega_raw" also present, see below
    assert 0.0 < g_call["delta"] < 1.0
    assert -1.0 < g_put["delta"] < 0.0
    assert g_call["gamma"] > 0
    assert g_call["gamma"] == pytest.approx(g_put["gamma"], rel=1e-3)
    assert g_call["vega"] > 0
    assert g_call["rho"] > 0  # calls gain value with higher rates
    assert g_put["rho"] < 0


def test_heston_vega_is_bs_equivalent_by_construction():
    """heston_greeks['vega'] must equal bs_greeks(...,vol=implied_vol(price),...)['vega'] exactly."""
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.5, -0.6, 0.045
    K, T = 100.0, 1.0
    price = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0)
    iv = implied_vol_from_price(price, SPOT, K, T, RATE)
    expected_vega = bs_greeks(SPOT, K, T, RATE, iv)["vega"]
    g = heston_greeks(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0)
    assert g["vega"] == pytest.approx(expected_vega, rel=1e-9)
    # "vega_raw" (the old dPrice/d(sqrt(v0)) convention) must still be present but on a
    # different scale -- it should NOT equal the BS-equivalent vega in general.
    assert "vega_raw" in g
    assert g["vega_raw"] != pytest.approx(g["vega"], rel=0.1)


def test_heston_price_raises_on_nonpositive_maturity():
    with pytest.raises(ValueError):
        heston_price(SPOT, 100.0, 0.0, RATE, 2.0, 0.04, 0.5, -0.5, 0.04)


def test_fair_value_heston_matches_direct_call():
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.6, "v0": 0.05, "rate": RATE}
    price, greeks = fair_value("heston", params, SPOT, 100.0, 1.0, "call")
    direct_price = heston_price(SPOT, 100.0, 1.0, RATE, 2.0, 0.04, 0.5, -0.6, 0.05)
    assert price == pytest.approx(direct_price)
    assert {"delta", "gamma", "vega", "theta", "rho"} <= set(greeks)


def test_fair_value_heston_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        fair_value("heston", {"kappa": 2.0}, SPOT, 100.0, 1.0)


def test_fair_value_does_not_mutate_params():
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.6, "v0": 0.05, "rate": RATE}
    snapshot = dict(params)
    fair_value("heston", params, SPOT, 100.0, 1.0)
    assert params == snapshot
