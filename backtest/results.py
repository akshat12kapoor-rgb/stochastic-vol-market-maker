"""Backtest results container, performance-metric computations, and plotting.

`BacktestResults` is what `backtest.engine.run_backtest` returns. All arrays
are indexed by "bar" (0 .. n_steps inclusive, so length `n_steps + 1`),
aligned with `times`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BacktestResults:
    """Full output of one `backtest.engine.run_backtest` run.

    Attributes:
        model: which quoting model produced this run ("black_scholes" or "heston").
        times: shape (n_steps+1,), bar timestamps in years (`times[0] == 0`).
        dt: bar length in years (see `backtest.engine.run_backtest`'s `dt` argument).
        spot_path: shape (n_steps+1,), the REALIZED market spot at each bar
            (shared, bit-identical across a black_scholes run and a heston
            run given the same `market_seed` -- see
            `tests/test_engine.py::test_identical_seeds_give_identical_realized_paths_across_models`).
        variance_path: shape (n_steps+1,), the REALIZED (true, latent)
            market instantaneous variance at each bar.
        pnl: shape (n_steps+1,), mark-to-market total P&L at each bar,
            `pnl[t] = cash[t] + sum_c(inventory[c][t] * marks[c][t])`. ALWAYS
            marked using the TRUE market's own Heston fair value (see
            `backtest.engine` module docstring's "marking convention"
            decision) -- independent of which model is doing the quoting,
            so `pnl` is directly comparable between a black_scholes run and
            a heston run.
        cash: shape (n_steps+1,), realized cash P&L from fills (bid paid /
            ask received), cumulative, no discounting applied.
        inventory: {contract_id: array of shape (n_steps+1,)} signed
            inventory (option contracts held, matching
            `market_maker.book.Book.inventory`'s convention) at each bar.
        marks: {contract_id: array of shape (n_steps+1,)} the TRUE-market
            fair-value mark used for `pnl`'s mark-to-market term at each bar
            (exposed so callers/tests can reconstruct/verify the `pnl`
            identity directly -- see
            `tests/test_engine.py::test_pnl_equals_cash_plus_mark_to_market_inventory_value`).
        delta_book: shape (n_steps+1,), inventory-weighted book delta
            exposure AS SEEN BY THE ACTIVE QUOTING PRICER (i.e. what the
            market maker itself believes its delta exposure is -- computed
            with `model`'s own Greeks, not the true market's) at each bar.
        vega_book: shape (n_steps+1,), same convention as `delta_book`, for vega.
        fills: list of fill-event dicts, one per individual arrival:
            {"t": bar index, "contract_id": str, "side": "buy"|"sell",
             "price": float, "true_price": float}. "buy" means the market
            maker's bid was lifted (market maker buys); "sell" means the
            market maker's ask was lifted (market maker sells).
        sharpe: see `compute_sharpe`.
        max_drawdown: see `compute_max_drawdown` (non-positive number).
        meta: dict of every config value the run was generated with
            (seeds, base_intensity, order_arrival_kappa, quoting params,
            sigma_underlying/sigma_vol, basket spec, etc.) -- for
            reproducibility and for the comparison report.
    """

    model: str
    times: np.ndarray
    dt: float
    spot_path: np.ndarray
    variance_path: np.ndarray
    pnl: np.ndarray
    cash: np.ndarray
    inventory: dict[str, np.ndarray]
    marks: dict[str, np.ndarray]
    delta_book: np.ndarray
    vega_book: np.ndarray
    fills: list[dict] = field(default_factory=list)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    meta: dict = field(default_factory=dict)

    def total_inventory(self) -> np.ndarray:
        """Sum of signed inventory across every contract, at each bar."""
        total = np.zeros_like(self.times)
        for arr in self.inventory.values():
            total = total + arr
        return total

    def gross_inventory(self) -> np.ndarray:
        """Sum of |inventory| across every contract, at each bar -- a simple gross-risk proxy."""
        total = np.zeros_like(self.times)
        for arr in self.inventory.values():
            total = total + np.abs(arr)
        return total


def compute_sharpe(pnl: np.ndarray, dt: float, ddof: int = 1) -> float:
    """Annualized Sharpe ratio of the bar-over-bar dollar P&L increments.

    `r_t = pnl[t] - pnl[t-1]` -- a DOLLAR P&L increment per bar, not a
    percentage return (a market-making book has no natural "capital base"
    to divide by; see `backtest.engine` module docstring). Annualized as

        sharpe = mean(r) / std(r, ddof=ddof) * sqrt(1 / dt)

    which is the standard daily-Sharpe annualization convention generalized
    to an arbitrary bar length `dt` (years) -- e.g. `dt = 1/252` (daily
    bars) gives the usual `sqrt(252)` factor.

    Returns 0.0 if there are fewer than 3 bars, or if the P&L increments
    have (numerically) zero standard deviation (a flat/no-trading run).
    """
    if len(pnl) < 3:
        return 0.0
    r = np.diff(np.asarray(pnl, dtype=float))
    sigma = np.std(r, ddof=ddof)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(r) / sigma * math.sqrt(1.0 / dt))


def compute_max_drawdown(pnl: np.ndarray) -> float:
    """Worst peak-to-trough decline of the P&L equity curve, in dollar units.

    Returns a NON-POSITIVE float: `0.0` if `pnl` never dips below its prior
    running maximum, otherwise the most negative `pnl[t] - running_max[t]`
    over the whole series (e.g. `-12.3` means the equity curve was, at its
    worst point, $12.3 below the best level reached up to that point).
    """
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0:
        return 0.0
    running_max = np.maximum.accumulate(pnl)
    drawdown = pnl - running_max
    return float(drawdown.min())


def plot_pnl_comparison(
    results_bs: BacktestResults,
    results_heston: BacktestResults,
    output_path: str | Path,
    title: str = "BS-quoting vs Heston-quoting market maker",
) -> Path:
    """Save a 2-panel PNG: mark-to-market P&L, and gross inventory, for both runs.

    Uses the `Agg` backend (headless-safe, matching `pricing.vol_surface
    .plot_vol_surface`'s convention). Creates parent directories as needed.
    Returns the resolved output `Path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(
        results_bs.times, results_bs.pnl,
        label=f"BS-quoting (Sharpe={results_bs.sharpe:.2f}, MaxDD={results_bs.max_drawdown:.2f})",
        color="tab:blue",
    )
    axes[0].plot(
        results_heston.times, results_heston.pnl,
        label=f"Heston-quoting (Sharpe={results_heston.sharpe:.2f}, MaxDD={results_heston.max_drawdown:.2f})",
        color="tab:orange",
    )
    axes[0].axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mark-to-market P&L ($)")
    axes[0].set_title(title)
    axes[0].legend(loc="best")

    axes[1].plot(results_bs.times, results_bs.gross_inventory(), label="BS-quoting", color="tab:blue")
    axes[1].plot(results_heston.times, results_heston.gross_inventory(), label="Heston-quoting", color="tab:orange")
    axes[1].set_ylabel("Gross inventory (|contracts| summed)")
    axes[1].set_xlabel("Time (years)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path.resolve()
