"""backtest package: event-driven backtest engine for the options market maker.

Public API (see ARCHITECTURE.md for the full contract):

    from backtest import run_backtest, build_default_basket
    from backtest import HestonMarketParams, TRUE_MARKET_PARAMS, simulate_heston_path, MarketPath
    from backtest import poisson_fill_counts, fill_intensity
    from backtest import BacktestResults, compute_sharpe, compute_max_drawdown, plot_pnl_comparison
    from backtest import calibrate_heston_quoter_params, compute_flat_vol_for_bs_quoter
"""
from backtest.arrivals import fill_intensity, poisson_fill_counts
from backtest.engine import apply_fill, build_default_basket, run_backtest
from backtest.market_sim import HestonMarketParams, MarketPath, TRUE_MARKET_PARAMS, simulate_heston_path
from backtest.quoter_calibration import calibrate_heston_quoter_params, compute_flat_vol_for_bs_quoter
from backtest.results import BacktestResults, compute_max_drawdown, compute_sharpe, plot_pnl_comparison

__all__ = [
    "run_backtest",
    "build_default_basket",
    "apply_fill",
    "HestonMarketParams",
    "MarketPath",
    "TRUE_MARKET_PARAMS",
    "simulate_heston_path",
    "fill_intensity",
    "poisson_fill_counts",
    "BacktestResults",
    "compute_sharpe",
    "compute_max_drawdown",
    "plot_pnl_comparison",
    "calibrate_heston_quoter_params",
    "compute_flat_vol_for_bs_quoter",
]
