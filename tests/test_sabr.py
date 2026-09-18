"""Tests for vol_models.sabr: Hagan formula sanity checks, pricing, Greeks, dispatch."""
import numpy as np
import pytest

from pricing import bs_greeks, bs_price
from vol_models.api import fair_value
from vol_models.greeks_utils import implied_vol_from_price
from vol_models.heston import heston_greeks, heston_price
from vol_models.sabr import sabr_greeks, sabr_implied_vol, sabr_price

SPOT, RATE = 100.0, 0.03


def test_sabr_atm_formula_matches_general_formula_limit():
    """At F==K the general Hagan formula must reduce to the documented closed-form ATM vol."""
    F, T, alpha, beta, rho, nu = 100.0, 1.0, 0.2, 0.7, -0.4, 0.5
    v_general = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
    v_atm_closed_form = alpha / F ** (1 - beta) * (
        1 + (
            ((1 - beta) ** 2 / 24.0) * alpha ** 2 / F ** (2 - 2 * beta)
            + (rho * beta * nu * alpha) / (4.0 * F ** (1 - beta))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
        ) * T
    )
    assert v_general == pytest.approx(v_atm_closed_form, rel=1e-10)


def test_sabr_nu_zero_beta_one_is_flat_vol_equal_to_alpha():
    """nu=0 kills stochastic vol; beta=1 kills the CEV backbone term -> vol == alpha everywhere."""
    alpha = 0.25
    for K in [60, 80, 100, 120, 160]:
        for rho in [-0.8, 0.0, 0.6]:
            vol = sabr_implied_vol(100.0, K, 1.5, alpha, beta=1.0, rho=rho, nu=0.0)
            assert vol == pytest.approx(alpha, abs=1e-12)


def test_sabr_price_matches_bs_when_nu_zero_beta_one():
    """With flat vol == alpha, sabr_price must exactly match bs_price at that vol."""
    alpha = 0.22
    for K in [85, 100, 115]:
        p_sabr = sabr_price(SPOT, K, 1.0, RATE, alpha, beta=1.0, rho=-0.3, nu=0.0)
        p_bs = bs_price(SPOT, K, 1.0, RATE, alpha)
        assert p_sabr == pytest.approx(p_bs, abs=1e-10)


def test_sabr_monotonic_in_alpha():
    """Implied vol must increase monotonically in alpha (vol-of-forward level), all else fixed."""
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    for K in [90, 100, 110]:
        vols = [sabr_implied_vol(100.0, K, 1.0, a, beta=1.0, rho=-0.3, nu=0.4) for a in alphas]
        assert all(v2 > v1 for v1, v2 in zip(vols, vols[1:]))


def test_sabr_negative_rho_produces_negative_skew():
    """rho<0 (equity-like) must make implied vol strictly decrease as strike rises through the money."""
    strikes = [80, 90, 100, 110, 120]
    vols = [sabr_implied_vol(100.0, K, 1.0, alpha=0.2, beta=1.0, rho=-0.5, nu=0.4) for K in strikes]
    assert all(v1 > v2 for v1, v2 in zip(vols, vols[1:]))


def test_sabr_positive_rho_produces_positive_skew():
    strikes = [80, 90, 100, 110, 120]
    vols = [sabr_implied_vol(100.0, K, 1.0, alpha=0.2, beta=1.0, rho=0.5, nu=0.4) for K in strikes]
    assert all(v1 < v2 for v1, v2 in zip(vols, vols[1:]))


def test_sabr_vectorized_matches_scalar_loop():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vec = sabr_implied_vol(100.0, strikes, 1.0, 0.2, 1.0, -0.4, 0.5)
    scalar = np.array([sabr_implied_vol(100.0, float(K), 1.0, 0.2, 1.0, -0.4, 0.5) for K in strikes])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


def test_sabr_greeks_sane():
    g_call = sabr_greeks(SPOT, 100.0, 1.0, RATE, alpha=0.2, beta=1.0, rho=-0.4, nu=0.4, option_type="call")
    g_put = sabr_greeks(SPOT, 100.0, 1.0, RATE, alpha=0.2, beta=1.0, rho=-0.4, nu=0.4, option_type="put")
    assert {"delta", "gamma", "vega", "theta", "rho"} <= set(g_call)  # "vega_raw" also present, see below
    assert 0.0 < g_call["delta"] < 1.0
    assert -1.0 < g_put["delta"] < 0.0
    assert g_call["gamma"] > 0
    assert g_call["vega"] > 0


def test_sabr_price_raises_on_nonpositive_maturity():
    with pytest.raises(ValueError):
        sabr_price(SPOT, 100.0, 0.0, RATE, 0.2, 1.0, -0.3, 0.4)


def test_sabr_vega_is_bs_equivalent_by_construction():
    """sabr_greeks['vega'] must equal bs_greeks(...,vol=implied_vol(price),...)['vega'] exactly."""
    alpha, beta, rho, nu = 0.21, 1.0, -0.4, 0.4
    K, T = 100.0, 1.0
    price = sabr_price(SPOT, K, T, RATE, alpha, beta, rho, nu)
    iv = implied_vol_from_price(price, SPOT, K, T, RATE)
    expected_vega = bs_greeks(SPOT, K, T, RATE, iv)["vega"]
    g = sabr_greeks(SPOT, K, T, RATE, alpha, beta, rho, nu)
    assert g["vega"] == pytest.approx(expected_vega, rel=1e-9)
    # "vega_raw" (dPrice/dAlpha) is a distinct diagnostic key, present alongside "vega".
    # For beta=1 SABR near the money it happens to sit close to the BS-equivalent vega
    # (alpha *is* approximately the ATM vol for beta=1) -- that's expected, not a bug;
    # the two conventions diverge more away from the money or for beta<1 (see
    # test_heston.py's analogous test, where kappa/theta mean-reversion makes
    # dPrice/d(sqrt(v0)) sit on a visibly different scale from dPrice/dIV).
    assert "vega_raw" in g


def test_vega_is_comparable_across_black_scholes_heston_sabr_near_atm():
    """Coordinator sign-off sanity check: near-ATM options priced by BS, Heston, and
    SABR with matching implied vol levels should now have similar vega magnitudes
    (all mean dPrice/dIV), unlike the old convention where Heston's dPrice/d(sqrt(v0))
    and SABR's dPrice/dAlpha were on unrelated scales (~17 vs ~38 for a comparable case)."""
    K, T = 100.0, 1.0
    target_vol = 0.20

    bs_vega = bs_greeks(SPOT, K, T, RATE, target_vol)["vega"]

    # Heston params chosen so its ATM implied vol comes out close to target_vol.
    kappa, theta, sigma, rho, v0 = 2.0, target_vol ** 2, 0.5, -0.6, target_vol ** 2
    heston_price_val = heston_price(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0)
    heston_iv = implied_vol_from_price(heston_price_val, SPOT, K, T, RATE)
    heston_vega = heston_greeks(SPOT, K, T, RATE, kappa, theta, sigma, rho, v0)["vega"]

    # SABR params chosen so alpha (~ATM vol for beta=1) is close to target_vol.
    alpha, beta, sabr_rho, nu = target_vol, 1.0, -0.4, 0.4
    sabr_price_val = sabr_price(SPOT, K, T, RATE, alpha, beta, sabr_rho, nu)
    sabr_iv = implied_vol_from_price(sabr_price_val, SPOT, K, T, RATE)
    sabr_vega = sabr_greeks(SPOT, K, T, RATE, alpha, beta, sabr_rho, nu)["vega"]

    # Sanity: the implied vols are indeed all close to target_vol (comparable regime).
    assert heston_iv == pytest.approx(target_vol, abs=0.02)
    assert sabr_iv == pytest.approx(target_vol, abs=0.02)

    # The real assertion: vega magnitudes now agree to within a few percent across
    # all three pricers, rather than differing by 2x+ under the old raw-param-bump convention.
    assert heston_vega == pytest.approx(bs_vega, rel=0.05)
    assert sabr_vega == pytest.approx(bs_vega, rel=0.05)


def test_fair_value_sabr_matches_direct_call_and_default_beta():
    params = {"alpha": 0.2, "rho": -0.4, "nu": 0.4, "rate": RATE}
    price, greeks = fair_value("sabr", params, SPOT, 100.0, 1.0, "call")
    direct_price = sabr_price(SPOT, 100.0, 1.0, RATE, 0.2, 1.0, -0.4, 0.4)  # beta defaults to 1.0
    assert price == pytest.approx(direct_price)
    assert {"delta", "gamma", "vega", "theta", "rho"} <= set(greeks)


def test_fair_value_sabr_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        fair_value("sabr", {"alpha": 0.2}, SPOT, 100.0, 1.0)


def test_fair_value_unknown_model_raises_valueerror():
    with pytest.raises(ValueError):
        fair_value("bogus_model", {}, SPOT, 100.0, 1.0)
