"""Safety-critical: proves webapp.derive's quote waterfall reconstructs
market_maker.quoting's own reservation_price/spread/bid/ask exactly.

If this ever fails, the webapp UI is decomposing a formula it no longer
matches -- see webapp/derive.py's module docstring.
"""
from __future__ import annotations

import math

import pytest

from market_maker.quoting import RiskAversion
from market_maker.quoting import quote_spread as engine_quote_spread
from market_maker.quoting import reservation_price as engine_reservation_price
from webapp.derive import build_quote_waterfall


CASES = [
    # (fair_value, delta_book, vega_book, delta_c, vega_c, gamma_c,
    #  sigma_underlying, sigma_vol, horizon, risk_aversion, kappa, spot)
    (5.0, 0.0, 0.0, 0.5, 20.0, 0.02, 0.2, 0.5, 0.5, RiskAversion(delta=0.1, vega=0.1), 1.5, 100.0),
    (5.0, 12.0, 45.0, 0.5, 20.0, 0.02, 0.2, 0.5, 0.5, RiskAversion(delta=0.1, vega=0.1), 1.5, 100.0),
    (7.5, -8.0, -30.0, -0.4, 18.0, 0.015, 0.25, 0.6, 0.25, RiskAversion(delta=0.5, vega=0.02), 1.5, 100.0),
    (10.2, 0.0, 0.0, 0.7, 28.0, 0.03, 0.21, 0.55, 1.0, RiskAversion(delta=0.0, vega=0.0), 1.5, 100.0),
    (3.1, 5.0, -12.0, 0.3, 15.0, 0.01, 0.22, 0.45, 0.3, RiskAversion(delta=0.2, vega=0.05, gamma=0.01), 2.0, 105.0),
]


@pytest.mark.parametrize(
    "fair_value,delta_book,vega_book,delta_c,vega_c,gamma_c,sigma_underlying,sigma_vol,horizon,ra,kappa,spot",
    CASES,
)
def test_waterfall_reconstructs_reservation_price_and_quotes(
    fair_value, delta_book, vega_book, delta_c, vega_c, gamma_c,
    sigma_underlying, sigma_vol, horizon, ra, kappa, spot,
):
    wf = build_quote_waterfall(
        fair_value, delta_book, vega_book, delta_c, vega_c, gamma_c,
        sigma_underlying, sigma_vol, horizon, ra, kappa, spot,
    )

    expected_r = engine_reservation_price(fair_value, delta_book, vega_book, sigma_underlying, sigma_vol, horizon, ra)
    expected_spread = engine_quote_spread(
        delta_c, vega_c, sigma_underlying, sigma_vol, horizon, ra,
        order_arrival_kappa=kappa, gamma_c=gamma_c, spot=spot,
    )

    # The waterfall's own reservation_price/spread must equal the engine's.
    assert wf.reservation_price == pytest.approx(expected_r, abs=1e-12)
    assert wf.spread == pytest.approx(expected_spread, abs=1e-12)

    # Summing the DERIVED terms must reconstruct those same engine values.
    assert fair_value + wf.delta_skew + wf.vega_skew == pytest.approx(expected_r, abs=1e-9)
    summed_spread = wf.risk_spread_delta + wf.risk_spread_vega + wf.liquidity_spread + wf.gamma_term
    assert summed_spread == pytest.approx(expected_spread, abs=1e-9)

    # bid/ask must match reservation_price +/- spread/2 exactly.
    assert wf.bid == pytest.approx(expected_r - expected_spread / 2.0, abs=1e-12)
    assert wf.ask == pytest.approx(expected_r + expected_spread / 2.0, abs=1e-12)


def test_zero_risk_aversion_collapses_skew_terms_to_zero():
    ra = RiskAversion(delta=0.0, vega=0.0, gamma=0.0)
    wf = build_quote_waterfall(5.0, 100.0, 200.0, 0.5, 20.0, 0.02, 0.2, 0.5, 0.5, ra, 1.5, 100.0)
    assert wf.delta_skew == 0.0
    assert wf.vega_skew == 0.0
    assert wf.reservation_price == pytest.approx(5.0, abs=1e-12)
    # liquidity_spread should hit the g->0 limit (2/kappa), per quote_spread's own docstring.
    assert wf.liquidity_spread == pytest.approx(2.0 / 1.5, abs=1e-12)


def test_gamma_term_is_zero_when_gamma_risk_aversion_is_zero():
    ra = RiskAversion(delta=0.1, vega=0.1, gamma=0.0)
    wf = build_quote_waterfall(5.0, 10.0, 20.0, 0.5, 20.0, 0.05, 0.2, 0.5, 0.5, ra, 1.5, 100.0)
    assert wf.gamma_term == 0.0


def test_gamma_term_is_nonzero_and_reconstructs_when_gamma_risk_aversion_positive():
    ra = RiskAversion(delta=0.1, vega=0.1, gamma=0.02)
    wf = build_quote_waterfall(5.0, 10.0, 20.0, 0.5, 20.0, 0.05, 0.2, 0.5, 0.5, ra, 1.5, 100.0)
    assert wf.gamma_term > 0.0
    expected_spread = engine_quote_spread(
        0.5, 20.0, 0.2, 0.5, 0.5, ra, order_arrival_kappa=1.5, gamma_c=0.05, spot=100.0
    )
    assert wf.spread == pytest.approx(expected_spread, abs=1e-12)
    assert (wf.risk_spread_delta + wf.risk_spread_vega + wf.liquidity_spread + wf.gamma_term) == pytest.approx(
        expected_spread, abs=1e-9
    )
