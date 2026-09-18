"""Tests for market_maker.quoting.

Uses `pricing.api.price` with model="black_scholes" as the concrete
pricer, per ARCHITECTURE.md: the market_maker package must work against
ANY callable matching `pricing.api.price`'s signature, and black_scholes
is the one concrete model available today (the Stochastic Vol Agent's
heston/sabr `fair_value` isn't ready yet, so we don't depend on it).
"""
import math

import pytest

from pricing import price as bs_pricer
from market_maker.book import Book, OptionContract
from market_maker.quoting import (
    Quote,
    RiskAversion,
    compute_book_greeks,
    generate_quotes,
    price_book,
    price_contract,
    quote_spread,
    reservation_price,
)

PARAMS = {"rate": 0.02, "vol": 0.20}
SPOT = 100.0
SIGMA_S = 0.20
SIGMA_VOL = 0.5


def _atm_call(cid="ATM_CALL", maturity=0.5):
    return OptionContract(contract_id=cid, strike=100.0, maturity=maturity, option_type="call")


def _otm_put(cid="OTM_PUT", maturity=0.5):
    return OptionContract(contract_id=cid, strike=90.0, maturity=maturity, option_type="put")


# ---------------------------------------------------------------------
# Pure reservation_price / quote_spread unit tests
# ---------------------------------------------------------------------

def test_reservation_price_zero_book_exposure_equals_fair_value():
    r = reservation_price(
        fair_value=12.34,
        delta_book=0.0,
        vega_book=0.0,
        sigma_underlying=SIGMA_S,
        sigma_vol=SIGMA_VOL,
        horizon=0.5,
        risk_aversion=RiskAversion(delta=0.1, vega=0.2),
    )
    assert r == pytest.approx(12.34)


def test_reservation_price_positive_delta_book_lowers_price():
    kwargs = dict(
        fair_value=10.0,
        vega_book=0.0,
        sigma_underlying=SIGMA_S,
        sigma_vol=SIGMA_VOL,
        horizon=0.5,
        risk_aversion=RiskAversion(delta=0.1, vega=0.1),
    )
    r_flat = reservation_price(delta_book=0.0, **kwargs)
    r_long = reservation_price(delta_book=5.0, **kwargs)
    r_short = reservation_price(delta_book=-5.0, **kwargs)
    assert r_long < r_flat < r_short


def test_reservation_price_positive_vega_book_lowers_price():
    kwargs = dict(
        fair_value=10.0,
        delta_book=0.0,
        sigma_underlying=SIGMA_S,
        sigma_vol=SIGMA_VOL,
        horizon=0.5,
        risk_aversion=RiskAversion(delta=0.1, vega=0.1),
    )
    r_flat = reservation_price(vega_book=0.0, **kwargs)
    r_long_vega = reservation_price(vega_book=5.0, **kwargs)
    r_short_vega = reservation_price(vega_book=-5.0, **kwargs)
    assert r_long_vega < r_flat < r_short_vega


def test_quote_spread_positive_and_grows_with_greek_magnitude():
    ra = RiskAversion(delta=0.1, vega=0.1)
    s_small = quote_spread(0.1, 0.5, SIGMA_S, SIGMA_VOL, 0.5, ra)
    s_large = quote_spread(0.9, 5.0, SIGMA_S, SIGMA_VOL, 0.5, ra)
    assert s_small > 0
    assert s_large > s_small


def test_quote_spread_requires_positive_order_arrival_kappa():
    with pytest.raises(ValueError):
        quote_spread(0.5, 1.0, SIGMA_S, SIGMA_VOL, 0.5, RiskAversion(), order_arrival_kappa=0.0)


def test_quote_spread_zero_risk_aversion_limit_matches_liquidity_only():
    # g = delta + vega risk aversion -> 0 limit should be exactly 2/kappa
    ra = RiskAversion(delta=0.0, vega=0.0)
    kappa = 1.5
    spr = quote_spread(0.5, 1.0, SIGMA_S, SIGMA_VOL, 0.5, ra, order_arrival_kappa=kappa)
    assert spr == pytest.approx(2.0 / kappa)


# ---------------------------------------------------------------------
# price_contract / price_book / compute_book_greeks (integration with the
# real pricing.api.price black_scholes pricer)
# ---------------------------------------------------------------------

def test_price_contract_matches_direct_bs_pricer_call():
    contract = _atm_call()
    got_price, got_greeks = price_contract(bs_pricer, "black_scholes", PARAMS, SPOT, contract)
    expected_price, expected_greeks = bs_pricer(
        "black_scholes", PARAMS, SPOT, contract.strike, contract.maturity, contract.option_type
    )
    assert got_price == pytest.approx(expected_price)
    assert got_greeks == expected_greeks


def test_price_book_prices_every_registered_contract():
    book = Book()
    book.add_contract(_atm_call())
    book.add_contract(_otm_put())
    valuations = price_book(book, bs_pricer, "black_scholes", PARAMS, SPOT)
    assert set(valuations.keys()) == {"ATM_CALL", "OTM_PUT"}
    for _, (p, greeks) in valuations.items():
        assert p > 0
        assert set(greeks.keys()) == {"delta", "gamma", "vega", "theta", "rho"}


def test_compute_book_greeks_zero_inventory_is_zero():
    book = Book()
    book.add_contract(_atm_call())
    book.add_contract(_otm_put())
    totals = compute_book_greeks(book, bs_pricer, "black_scholes", PARAMS, SPOT)
    assert totals == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}


def test_compute_book_greeks_weights_by_inventory():
    book = Book()
    book.add_contract(_atm_call(), quantity=2.0)
    _, greeks = bs_pricer("black_scholes", PARAMS, SPOT, 100.0, 0.5, "call")
    totals = compute_book_greeks(book, bs_pricer, "black_scholes", PARAMS, SPOT)
    assert totals["delta"] == pytest.approx(2.0 * greeks["delta"])
    assert totals["vega"] == pytest.approx(2.0 * greeks["vega"])


def test_params_callable_resolves_per_contract():
    # per-contract params (e.g. different implied vol per strike from a
    # vol surface) via a callable instead of a shared dict.
    def params_fn(contract):
        return {"rate": 0.02, "vol": 0.30 if contract.strike < 100 else 0.20}

    book = Book()
    book.add_contract(_atm_call())
    book.add_contract(_otm_put())
    valuations = price_book(book, bs_pricer, "black_scholes", params_fn, SPOT)
    expected_call = bs_pricer("black_scholes", {"rate": 0.02, "vol": 0.20}, SPOT, 100.0, 0.5, "call")
    expected_put = bs_pricer("black_scholes", {"rate": 0.02, "vol": 0.30}, SPOT, 90.0, 0.5, "put")
    assert valuations["ATM_CALL"][0] == pytest.approx(expected_call[0])
    assert valuations["OTM_PUT"][0] == pytest.approx(expected_put[0])


# ---------------------------------------------------------------------
# generate_quotes — the required edge cases
# ---------------------------------------------------------------------

def _flat_book():
    book = Book()
    book.add_contract(_atm_call())
    book.add_contract(_otm_put())
    return book


def test_zero_inventory_produces_symmetric_quotes_around_fair_value():
    """Required edge case: zero inventory across the whole book -> bid/ask
    equidistant from fair value (== reservation price) for every contract."""
    book = _flat_book()
    assert book.is_flat()

    quotes = generate_quotes(
        book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S,
        risk_aversion=RiskAversion(delta=0.1, vega=0.1),
        sigma_vol=SIGMA_VOL,
    )

    assert len(quotes) == 2
    for cid, q in quotes.items():
        assert isinstance(q, Quote)
        assert q.reservation_price == pytest.approx(q.fair_value)
        assert q.bid < q.ask
        dist_bid = q.reservation_price - q.bid
        dist_ask = q.ask - q.reservation_price
        assert dist_bid == pytest.approx(dist_ask), f"{cid}: quotes not symmetric"
        assert dist_bid == pytest.approx(q.spread / 2.0)


def test_long_delta_inventory_skews_quotes_down_to_encourage_selling():
    """Long inventory in a call (positive book delta) should push the
    reservation price (and both bid/ask) DOWN relative to the flat-book
    case: a lower ask makes it more attractive for a counterparty to buy
    from the market maker, which is how the market maker sheds long
    inventory. Spread width itself (contract's own Greeks) is unaffected
    by book-level inventory, so it should be identical in both cases.
    """
    ra = RiskAversion(delta=0.2, vega=0.0)  # isolate delta effect

    flat_book = _flat_book()
    long_book = _flat_book()
    long_book.set_inventory("ATM_CALL", 10.0)  # long calls -> positive book delta

    flat_quotes = generate_quotes(
        flat_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )
    long_quotes = generate_quotes(
        long_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )

    q_flat = flat_quotes["ATM_CALL"]
    q_long = long_quotes["ATM_CALL"]

    assert q_long.reservation_price < q_flat.reservation_price
    assert q_long.bid < q_flat.bid
    assert q_long.ask < q_flat.ask
    # spread depends only on this contract's own greeks, not book inventory
    assert q_long.spread == pytest.approx(q_flat.spread)


def test_short_delta_inventory_skews_quotes_up_to_encourage_buying():
    ra = RiskAversion(delta=0.2, vega=0.0)
    flat_book = _flat_book()
    short_book = _flat_book()
    short_book.set_inventory("ATM_CALL", -10.0)

    flat_quotes = generate_quotes(
        flat_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )
    short_quotes = generate_quotes(
        short_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )

    assert short_quotes["ATM_CALL"].reservation_price > flat_quotes["ATM_CALL"].reservation_price


def test_long_vega_inventory_skews_quotes_down_independent_of_delta():
    """Same as the delta test but isolating vega risk aversion (delta=0),
    demonstrating the per-Greek risk-aversion split is independently
    controllable."""
    ra = RiskAversion(delta=0.0, vega=0.3)  # isolate vega effect

    flat_book = _flat_book()
    long_vega_book = _flat_book()
    long_vega_book.set_inventory("ATM_CALL", 10.0)

    flat_quotes = generate_quotes(
        flat_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )
    long_quotes = generate_quotes(
        long_vega_book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=ra, sigma_vol=SIGMA_VOL,
    )

    assert long_quotes["ATM_CALL"].reservation_price < flat_quotes["ATM_CALL"].reservation_price


def test_generate_quotes_bid_below_ask_for_all_contracts_nonflat_book():
    book = _flat_book()
    book.set_inventory("ATM_CALL", 3.0)
    book.set_inventory("OTM_PUT", -2.0)
    quotes = generate_quotes(
        book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S,
        risk_aversion=RiskAversion(delta=0.15, vega=0.15),
        sigma_vol=SIGMA_VOL,
    )
    for q in quotes.values():
        assert q.bid < q.fair_value < q.ask or q.bid < q.ask  # spread always positive
        assert q.ask - q.bid == pytest.approx(q.spread)
        assert q.spread > 0


def test_generate_quotes_default_horizon_uses_contract_maturity():
    # Two contracts with different maturities but otherwise identical
    # exposure profile should get different risk terms because horizon
    # defaults to each contract's own maturity.
    book = Book()
    book.add_contract(OptionContract("SHORT", strike=100.0, maturity=0.1, option_type="call"), quantity=5.0)
    book.add_contract(OptionContract("LONG", strike=100.0, maturity=1.0, option_type="call"), quantity=5.0)
    quotes = generate_quotes(
        book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S,
        risk_aversion=RiskAversion(delta=0.2, vega=0.0),
        sigma_vol=SIGMA_VOL,
    )
    skew_short = quotes["SHORT"].fair_value - quotes["SHORT"].reservation_price
    skew_long = quotes["LONG"].fair_value - quotes["LONG"].reservation_price
    # longer horizon -> larger risk charge (all else being scaled by horizon linearly,
    # though delta_book itself also differs since deltas differ by maturity; just
    # check both are positive skew-down and not accidentally equal)
    assert skew_short > 0
    assert skew_long > 0
    assert not math.isclose(skew_short, skew_long)


def test_generate_quotes_never_mutates_params_or_book():
    book = _flat_book()
    book.set_inventory("ATM_CALL", 4.0)
    params_copy = dict(PARAMS)
    inventory_before = dict(book.inventory)

    generate_quotes(
        book, bs_pricer, "black_scholes", PARAMS, SPOT,
        sigma_underlying=SIGMA_S, risk_aversion=RiskAversion(), sigma_vol=SIGMA_VOL,
    )

    assert PARAMS == params_copy
    assert book.inventory == inventory_before
