"""Tests for pricing.black_scholes: pricing sanity, put-call parity, Greeks."""
import numpy as np
import pytest

from pricing.black_scholes import bs_price, bs_greeks, price

SPOT, STRIKE, MATURITY, RATE, VOL = 100.0, 100.0, 1.0, 0.02, 0.20


def test_call_put_are_positive_and_bounded():
    c = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "call")
    p = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "put")
    assert c > 0
    assert p > 0
    assert c < SPOT
    assert p < STRIKE


def test_put_call_parity():
    c = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "call")
    p = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "put")
    forward_diff = SPOT - STRIKE * np.exp(-RATE * MATURITY)
    assert c - p == pytest.approx(forward_diff, abs=1e-10)


def test_put_call_parity_with_dividends():
    q = 0.015
    c = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "call", div_yield=q)
    p = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "put", div_yield=q)
    forward_diff = SPOT * np.exp(-q * MATURITY) - STRIKE * np.exp(-RATE * MATURITY)
    assert c - p == pytest.approx(forward_diff, abs=1e-10)


def test_maturity_zero_is_intrinsic():
    assert bs_price(110, 100, 0.0, RATE, VOL, "call") == pytest.approx(10.0)
    assert bs_price(90, 100, 0.0, RATE, VOL, "call") == pytest.approx(0.0)
    assert bs_price(90, 100, 0.0, RATE, VOL, "put") == pytest.approx(10.0)
    assert bs_price(110, 100, 0.0, RATE, VOL, "put") == pytest.approx(0.0)


def test_deep_itm_call_converges_to_discounted_forward_minus_strike():
    c = bs_price(1000, 100, MATURITY, RATE, VOL, "call")
    forward = 1000 * np.exp(RATE * MATURITY)
    expected = np.exp(-RATE * MATURITY) * (forward - 100)
    assert c == pytest.approx(expected, rel=1e-6)


def test_deep_otm_put_near_zero():
    p = bs_price(1000, 100, MATURITY, RATE, VOL, "put")
    assert p == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_delta_matches_finite_difference(option_type, strike):
    h = 1e-4
    analytic = bs_greeks(SPOT, strike, MATURITY, RATE, VOL, option_type)["delta"]
    p_up = bs_price(SPOT + h, strike, MATURITY, RATE, VOL, option_type)
    p_down = bs_price(SPOT - h, strike, MATURITY, RATE, VOL, option_type)
    fd = (p_up - p_down) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(option_type):
    h = 1e-3
    analytic = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, option_type)["gamma"]
    p_up = bs_price(SPOT + h, STRIKE, MATURITY, RATE, VOL, option_type)
    p_mid = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, option_type)
    p_down = bs_price(SPOT - h, STRIKE, MATURITY, RATE, VOL, option_type)
    fd = (p_up - 2 * p_mid + p_down) / (h ** 2)
    assert analytic == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    h = 1e-4
    analytic = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, option_type)["vega"]
    p_up = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL + h, option_type)
    p_down = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL - h, option_type)
    fd = (p_up - p_down) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    h = 1e-5
    analytic = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, option_type)["rho"]
    p_up = bs_price(SPOT, STRIKE, MATURITY, RATE + h, VOL, option_type)
    p_down = bs_price(SPOT, STRIKE, MATURITY, RATE - h, VOL, option_type)
    fd = (p_up - p_down) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    # theta is defined as dPrice/dCalendarTime = -dPrice/dMaturity
    h = 1e-5
    analytic = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, option_type)["theta"]
    p_up = bs_price(SPOT, STRIKE, MATURITY + h, RATE, VOL, option_type)
    p_down = bs_price(SPOT, STRIKE, MATURITY - h, RATE, VOL, option_type)
    fd = -(p_up - p_down) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-3)


def test_call_delta_in_zero_one_put_delta_in_minus_one_zero():
    for strike in [50, 80, 100, 120, 200]:
        dc = bs_greeks(SPOT, strike, MATURITY, RATE, VOL, "call")["delta"]
        dp = bs_greeks(SPOT, strike, MATURITY, RATE, VOL, "put")["delta"]
        assert 0.0 <= dc <= 1.0
        assert -1.0 <= dp <= 0.0


def test_gamma_positive_and_same_for_call_and_put():
    gc = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, "call")["gamma"]
    gp = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, "put")["gamma"]
    assert gc > 0
    assert gc == pytest.approx(gp, rel=1e-10)


def test_vega_positive_and_same_for_call_and_put():
    vc = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, "call")["vega"]
    vp = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, "put")["vega"]
    assert vc > 0
    assert vc == pytest.approx(vp, rel=1e-10)


def test_api_dispatcher_matches_direct_call():
    p_direct = bs_price(SPOT, STRIKE, MATURITY, RATE, VOL, "call")
    g_direct = bs_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, "call")
    p_api, g_api = price("black_scholes", {"rate": RATE, "vol": VOL}, SPOT, STRIKE, MATURITY, "call")
    assert p_api == pytest.approx(p_direct)
    for key in g_direct:
        assert g_api[key] == pytest.approx(g_direct[key])


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        price("not_a_model", {"rate": RATE, "vol": VOL}, SPOT, STRIKE, MATURITY, "call")
