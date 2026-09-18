"""Shared fixtures for `backtest` tests.

Heston calibration (`backtest.quoter_calibration.calibrate_heston_quoter_params`)
runs a bounded least-squares fit and is comparatively expensive to redo in
every test that needs a `model="heston"` run -- this session-scoped fixture
computes it once (with a reduced `max_nfev` for test-suite speed; the
report-generation script `backtest/comparison.py` uses the full default) and
every test that needs "a" calibrated Heston quoter shares the result.
"""
import pytest

from backtest.market_sim import TRUE_MARKET_PARAMS
from backtest.quoter_calibration import calibrate_heston_quoter_params


@pytest.fixture(scope="session")
def calibrated_heston_params() -> dict:
    return calibrate_heston_quoter_params(TRUE_MARKET_PARAMS, max_nfev=40)
