"""Tests for backtest.engine: the event-driven backtest loop.

Covers (per the task's required deliverables):
  - a smoke test that a short backtest runs end-to-end without error for
    both pricers,
  - a test that identical seeds produce identical REALIZED PATHS across the
    two model runs (the "apples to apples" comparison guarantee),
  - a sanity test on P&L bookkeeping (cash + mark-to-market inventory value
    is conserved correctly through a fill).
"""
import numpy as np
import pytest

from market_maker.book import Book, OptionContract

from backtest.engine import apply_fill, build_default_basket, run_backtest

SMALL_BASKET = [
    OptionContract(contract_id="C_100_90D", strike=100.0, maturity=90 / 365, option_type="call"),
    OptionContract(contract_id="C_105_90D", strike=105.0, maturity=90 / 365, option_type="call"),
]


# ---------------------------------------------------------------------
# build_default_basket
# ---------------------------------------------------------------------

def test_build_default_basket_is_a_multi_strike_multi_maturity_grid():
    basket = build_default_basket()
    assert len(basket) == 6  # 3 strikes x 2 maturities
    assert len({c.contract_id for c in basket}) == len(basket)  # unique ids
    maturities = {c.maturity for c in basket}
    strikes = {c.strike for c in basket}
    assert len(maturities) == 2
    assert len(strikes) == 3


# ---------------------------------------------------------------------
# apply_fill: direct bookkeeping unit test
# ---------------------------------------------------------------------

def test_apply_fill_conserves_cash_plus_inventory_value():
    book = Book()
    contract = OptionContract(contract_id="X", strike=100.0, maturity=0.5, option_type="call")
    book.add_contract(contract, quantity=0.0)

    mark = 7.5  # the mark used to value the position, independent of fill price
    cash = 100.0
    total_before = cash + book.get_inventory("X") * mark

    fill_price = 7.0  # market maker buys at a price BELOW the mark (positive edge)
    cash_after = apply_fill(cash, book, "X", +1.0, fill_price)

    assert book.get_inventory("X") == 1.0
    total_after = cash_after + book.get_inventory("X") * mark
    # buying 1 unit at fill_price when marked at `mark` should change total
    # P&L by exactly (mark - fill_price) -- the edge captured by trading
    # away from the mark. This is the core "cash + MTM inventory value is
    # conserved correctly through a fill" identity.
    assert total_after - total_before == pytest.approx(mark - fill_price)

    # Selling back at the mark exactly should be P&L-neutral from here.
    cash_after_2 = apply_fill(cash_after, book, "X", -1.0, mark)
    assert book.get_inventory("X") == 0.0
    total_final = cash_after_2 + book.get_inventory("X") * mark
    assert total_final == pytest.approx(total_after)


def test_apply_fill_sell_side_increases_cash_and_decreases_inventory():
    book = Book()
    contract = OptionContract(contract_id="X", strike=100.0, maturity=0.5, option_type="call")
    book.add_contract(contract, quantity=2.0)
    cash = 0.0
    cash_after = apply_fill(cash, book, "X", -1.0, 5.0)
    assert book.get_inventory("X") == 1.0
    assert cash_after == pytest.approx(5.0)


# ---------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------

def test_run_backtest_smoke_black_scholes():
    results = run_backtest("black_scholes", basket=SMALL_BASKET, n_steps=4, market_seed=1, arrival_seed=1)
    n_bars = 5
    assert results.pnl.shape == (n_bars,)
    assert results.times.shape == (n_bars,)
    assert results.spot_path.shape == (n_bars,)
    assert np.all(np.isfinite(results.pnl))
    for cid in ("C_100_90D", "C_105_90D"):
        assert results.inventory[cid].shape == (n_bars,)
        assert results.marks[cid].shape == (n_bars,)
        assert np.all(np.isfinite(results.marks[cid]))
    assert results.model == "black_scholes"
    assert isinstance(results.sharpe, float)
    assert results.max_drawdown <= 0.0


def test_run_backtest_smoke_heston(calibrated_heston_params):
    results = run_backtest(
        "heston", basket=SMALL_BASKET, n_steps=4, market_seed=1, arrival_seed=1,
        heston_quoter_params=calibrated_heston_params,
    )
    n_bars = 5
    assert results.pnl.shape == (n_bars,)
    assert np.all(np.isfinite(results.pnl))
    assert results.model == "heston"
    assert results.meta["heston_quoter_params"] == calibrated_heston_params


def test_run_backtest_rejects_unknown_model():
    with pytest.raises(ValueError):
        run_backtest("garbage_model", basket=SMALL_BASKET, n_steps=2)


# ---------------------------------------------------------------------
# apples-to-apples comparison guarantee
# ---------------------------------------------------------------------

def test_identical_seeds_give_identical_realized_paths_across_models(calibrated_heston_params):
    res_bs = run_backtest("black_scholes", basket=SMALL_BASKET, n_steps=6, market_seed=77, arrival_seed=55)
    res_h = run_backtest(
        "heston", basket=SMALL_BASKET, n_steps=6, market_seed=77, arrival_seed=55,
        heston_quoter_params=calibrated_heston_params,
    )
    # the underlying REALIZED market (what "actually happened") must be
    # bit-identical between the two runs -- this is what makes comparing
    # their P&L/Sharpe/drawdown meaningful rather than an artifact of
    # different random markets.
    assert np.array_equal(res_bs.spot_path, res_h.spot_path)
    assert np.array_equal(res_bs.variance_path, res_h.variance_path)
    assert np.array_equal(res_bs.times, res_h.times)


def test_different_market_seed_gives_different_realized_path(calibrated_heston_params):
    res_a = run_backtest("black_scholes", basket=SMALL_BASKET, n_steps=6, market_seed=1, arrival_seed=1)
    res_b = run_backtest("black_scholes", basket=SMALL_BASKET, n_steps=6, market_seed=2, arrival_seed=1)
    assert not np.array_equal(res_a.spot_path, res_b.spot_path)


# ---------------------------------------------------------------------
# P&L bookkeeping identity, at the full-engine level
# ---------------------------------------------------------------------

def test_pnl_equals_cash_plus_mark_to_market_inventory_value(calibrated_heston_params):
    results = run_backtest(
        "heston", basket=SMALL_BASKET, n_steps=8, market_seed=3, arrival_seed=9,
        heston_quoter_params=calibrated_heston_params,
    )
    n_bars = len(results.times)
    reconstructed = results.cash.copy()
    for cid in results.inventory:
        reconstructed = reconstructed + results.inventory[cid] * results.marks[cid]
    np.testing.assert_allclose(reconstructed, results.pnl, rtol=1e-10, atol=1e-8)
    assert n_bars == 9


def test_no_fills_means_flat_book_and_zero_pnl():
    # base_intensity=0 -> no arrivals ever -> book stays flat -> pnl stays
    # exactly 0 at every bar (cash never moves, inventory never moves).
    results = run_backtest(
        "black_scholes", basket=SMALL_BASKET, n_steps=5, market_seed=1, arrival_seed=1, base_intensity=0.0
    )
    assert len(results.fills) == 0
    assert np.all(results.pnl == 0.0)
    assert np.all(results.cash == 0.0)
    for arr in results.inventory.values():
        assert np.all(arr == 0.0)
