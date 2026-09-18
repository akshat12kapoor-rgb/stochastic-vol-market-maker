"""Smoke tests: verify the package structure imports cleanly."""
import importlib

import pricing
import vol_models
import market_maker
import backtest


def test_packages_import():
    for pkg in (pricing, vol_models, market_maker, backtest):
        importlib.reload(pkg)


def test_numeric_stack_available():
    import numpy
    import scipy
    import pandas

    assert numpy.__version__
    assert scipy.__version__
    assert pandas.__version__
