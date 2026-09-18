"""Report-generation script: BS-quoting vs Heston-quoting backtest comparison.

Not part of `pytest` (mirrors `vol_models/comparison.py`'s convention -- a
report script, not a correctness test). Run via:

    python -m backtest.comparison

Runs `backtest.engine.run_backtest` TWICE under IDENTICAL market conditions
(same `market_seed` -> bit-identical realized spot/variance path; same
`arrival_seed` -> same RNG stream, though realized fill counts will still
differ because each run's own quotes are model-dependent -- see
`backtest.engine.run_backtest`'s docstring): once with `model="black_scholes"`
(flat vol), once with `model="heston"` (calibrated from a synthetic surface
-- see `backtest.quoter_calibration` for the central framing decision this
whole comparison rests on). Prints a P&L/Sharpe/drawdown/inventory-risk
summary and saves `data/backtest_pnl_comparison.png`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backtest.engine import run_backtest
from backtest.results import plot_pnl_comparison

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "backtest_pnl_comparison.png"


def _summarize(name: str, results) -> None:
    max_abs_inv = max((np.abs(arr).max() for arr in results.inventory.values()), default=0.0)
    gross_inv_mean = float(np.mean(results.gross_inventory()))
    print(f"--- {name} ---")
    print(f"  final P&L:            {results.pnl[-1]:>10.2f}")
    print(f"  Sharpe (annualized):  {results.sharpe:>10.2f}")
    print(f"  max drawdown:         {results.max_drawdown:>10.2f}")
    print(f"  n_fills:              {len(results.fills):>10d}")
    print(f"  max |inventory| (any contract): {max_abs_inv:>6.1f}")
    print(f"  mean gross inventory (sum |qty|): {gross_inv_mean:>6.2f}")
    print(f"  final |delta_book|:   {abs(results.delta_book[-1]):>10.3f}")
    print(f"  final |vega_book|:    {abs(results.vega_book[-1]):>10.3f}")


def run_comparison(output_path: Path = DEFAULT_OUTPUT_PATH, market_seed: int = 42, arrival_seed: int = 123):
    print("Running black_scholes-quoting backtest...")
    res_bs = run_backtest("black_scholes", market_seed=market_seed, arrival_seed=arrival_seed)
    print("Running heston-quoting backtest (calibrating Heston quoter from a synthetic surface)...")
    res_h = run_backtest("heston", market_seed=market_seed, arrival_seed=arrival_seed)

    assert np.array_equal(res_bs.spot_path, res_h.spot_path), "realized market paths diverged -- not apples to apples!"

    _summarize("BLACK-SCHOLES QUOTING", res_bs)
    _summarize("HESTON QUOTING", res_h)

    print("\nHeston quoter calibrated params:", res_h.meta["heston_quoter_params"])
    print("BS quoter flat vol:", res_bs.meta["flat_vol"])

    saved_path = plot_pnl_comparison(res_bs, res_h, output_path)
    print(f"\nSaved comparison plot to {saved_path}")
    return res_bs, res_h


if __name__ == "__main__":
    run_comparison()
