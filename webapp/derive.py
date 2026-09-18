"""Derivations the webapp needs that the engine doesn't return directly.

Per PROJECT.md rule 3 / the webapp build's hard constraints: nothing here
adds return fields to `pricing`, `vol_models`, `market_maker`, or `backtest`
-- everything below is computed in `webapp` from values those packages
already expose (or, for the quote waterfall, by calling their own public
functions directly and cross-checking the result).

--------------------------------------------------------------------------
QUOTE WATERFALL -- SAFETY-CRITICAL, READ BEFORE TOUCHING
--------------------------------------------------------------------------
`market_maker.quoting.Quote` gives `fair_value, reservation_price, bid, ask,
spread, greeks` but not the individual additive terms that produce
`reservation_price`/`spread` from `fair_value`. The UI wants to show that
decomposition as a waterfall. `build_quote_waterfall` below re-derives each
term from the EXACT formulas in `market_maker/quoting.py`'s
`reservation_price` and `quote_spread` functions, then calls those two real
functions directly and asserts (in `tests/test_webapp_derive.py`) that
summing the derived terms reproduces their output to floating-point
tolerance. If `market_maker.quoting`'s formulas ever change, that test
fails loudly rather than letting this module silently drift out of sync and
have the UI "explain" a mechanism that isn't what the engine actually did.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from market_maker.quoting import RiskAversion, quote_spread as engine_quote_spread
from market_maker.quoting import reservation_price as engine_reservation_price


@dataclass(frozen=True)
class QuoteWaterfall:
    """Decomposition of one contract's quote into its additive terms.

    Every field here is in price units except `delta_book`/`vega_book`/
    `delta_c`/`vega_c`/`gamma_c`, which are the raw Greek/inventory inputs
    (included so the UI can label bars with them).

    Reconstruction identities (asserted in tests/test_webapp_derive.py):
        fair_value + delta_skew + vega_skew == reservation_price
        risk_spread_delta + risk_spread_vega + liquidity_spread + gamma_term == spread
        reservation_price - spread / 2 == bid
        reservation_price + spread / 2 == ask
    """

    fair_value: float
    delta_book: float
    vega_book: float
    delta_skew: float          # -ra.delta * delta_book * sigma_underlying**2 * horizon
    vega_skew: float           # -ra.vega  * vega_book  * sigma_vol**2        * horizon
    reservation_price: float   # fair_value + delta_skew + vega_skew (engine's own value, not a copy)
    delta_c: float
    vega_c: float
    gamma_c: float
    risk_spread_delta: float   # 2 * ra.delta * |delta_c| * sigma_underlying**2 * horizon
    risk_spread_vega: float    # 2 * ra.vega  * |vega_c|  * sigma_vol**2        * horizon
    liquidity_spread: float    # (2/g) * ln(1 + g/kappa), g = ra.delta + ra.vega
    gamma_term: float          # ra.gamma * |gamma_c| * (sigma_underlying*spot)**2 * horizon, else 0
    spread: float              # engine's own value, not a copy
    bid: float
    ask: float


def build_quote_waterfall(
    fair_value: float,
    delta_book: float,
    vega_book: float,
    delta_c: float,
    vega_c: float,
    gamma_c: float,
    sigma_underlying: float,
    sigma_vol: float,
    horizon: float,
    risk_aversion: RiskAversion,
    order_arrival_kappa: float,
    spot: float,
) -> QuoteWaterfall:
    """Build the waterfall decomposition for one contract's quote.

    `reservation_price` and `spread` (and therefore `bid`/`ask`) are computed
    by CALLING `market_maker.quoting.reservation_price` /
    `market_maker.quoting.quote_spread` directly -- not by summing the
    derived terms below -- so this function's output is always exactly what
    the real engine would produce. The individual terms are a separate,
    parallel computation that the reconstruction test checks against these.
    """
    ra = risk_aversion

    delta_skew = -ra.delta * delta_book * sigma_underlying**2 * horizon
    vega_skew = -ra.vega * vega_book * sigma_vol**2 * horizon
    reservation_price = engine_reservation_price(
        fair_value, delta_book, vega_book, sigma_underlying, sigma_vol, horizon, ra
    )

    risk_spread_delta = 2.0 * ra.delta * abs(delta_c) * sigma_underlying**2 * horizon
    risk_spread_vega = 2.0 * ra.vega * abs(vega_c) * sigma_vol**2 * horizon
    g = ra.delta + ra.vega
    liquidity_spread = (2.0 / order_arrival_kappa) if g <= 1e-12 else (2.0 / g) * math.log1p(g / order_arrival_kappa)
    gamma_term = (ra.gamma * abs(gamma_c) * (sigma_underlying * spot) ** 2 * horizon) if ra.gamma > 0.0 else 0.0

    spread = engine_quote_spread(
        delta_c, vega_c, sigma_underlying, sigma_vol, horizon, ra,
        order_arrival_kappa=order_arrival_kappa, gamma_c=gamma_c, spot=spot,
    )

    return QuoteWaterfall(
        fair_value=fair_value,
        delta_book=delta_book,
        vega_book=vega_book,
        delta_skew=delta_skew,
        vega_skew=vega_skew,
        reservation_price=reservation_price,
        delta_c=delta_c,
        vega_c=vega_c,
        gamma_c=gamma_c,
        risk_spread_delta=risk_spread_delta,
        risk_spread_vega=risk_spread_vega,
        liquidity_spread=liquidity_spread,
        gamma_term=gamma_term,
        spread=spread,
        bid=reservation_price - spread / 2.0,
        ask=reservation_price + spread / 2.0,
    )


# --------------------------------------------------------------------------
# Fill edge (realized spread capture / adverse selection)
# --------------------------------------------------------------------------

def fill_edge(side: str, price: float, true_price: float) -> float:
    """Realized edge of one fill, in price units.

    edge = true_price - price   for a market-maker BUY ("side" == "buy")
    edge = price - true_price   for a market-maker SELL ("side" == "sell")

    Positive edge means the market maker transacted on the favorable side of
    the true (ground-truth) price -- i.e. captured spread. Negative edge
    means the fill happened at a price worse than the true price at that
    instant -- i.e. the market maker was "picked off" (adverse selection).
    """
    if side == "buy":
        return true_price - price
    if side == "sell":
        return price - true_price
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def annotate_fills(fills: list[dict]) -> list[dict]:
    """Return `fills` with an `"edge"` key added to each dict (input untouched)."""
    return [{**f, "edge": fill_edge(f["side"], f["price"], f["true_price"])} for f in fills]


# --------------------------------------------------------------------------
# Simple OLS (for View 4's regime-conditional-edge regression) -- no sklearn
# dependency added just for a one-variable linear fit.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSFit:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int
    x_grid: list[float]
    y_pred: list[float]
    y_pred_lo: list[float]   # 95% CI band, lower
    y_pred_hi: list[float]   # 95% CI band, upper


def ols_with_ci(x: list[float], y: list[float], n_grid: int = 50) -> OLSFit:
    """Simple linear regression y ~ a + b*x with a 95% confidence band and
    a two-sided p-value for the slope (via a t-test on the slope estimate,
    the standard approach for simple OLS -- no distributional assumptions
    beyond the usual Gauss-Markov ones, which is the honest caveat this
    view is built to respect: at n=40ish this is not asymptotic-normal
    territory, so the UI reports the p-value plainly rather than dressing
    it up as more certain than it is).
    """
    import numpy as np
    from scipy import stats

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    if n < 3:
        raise ValueError("ols_with_ci needs at least 3 points")

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
    r_squared = r_value**2

    x_grid = np.linspace(x_arr.min(), x_arr.max(), n_grid)
    y_pred = intercept + slope * x_grid

    # Standard error of the mean prediction at each x_grid point, for the
    # 95% confidence band (not a prediction interval -- this bands the
    # fitted LINE, matching "confidence band" as the spec asks for).
    resid = y_arr - (intercept + slope * x_arr)
    dof = n - 2
    s_err = math.sqrt(float(np.sum(resid**2)) / dof) if dof > 0 else 0.0
    x_mean = float(np.mean(x_arr))
    ssx = float(np.sum((x_arr - x_mean) ** 2))
    t_crit = float(stats.t.ppf(0.975, dof)) if dof > 0 else 0.0

    if ssx > 0 and dof > 0:
        se_line = s_err * np.sqrt(1.0 / n + (x_grid - x_mean) ** 2 / ssx)
    else:
        se_line = np.zeros_like(x_grid)
    band = t_crit * se_line

    return OLSFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        p_value=float(p_value),
        n=n,
        x_grid=x_grid.tolist(),
        y_pred=y_pred.tolist(),
        y_pred_lo=(y_pred - band).tolist(),
        y_pred_hi=(y_pred + band).tolist(),
    )


# --------------------------------------------------------------------------
# Bar-index -> remaining maturity, matching backtest.engine.run_backtest's
# private `_age_book` exactly (that function isn't exported, so this is a
# parallel, tested re-implementation -- not a call into engine internals).
# --------------------------------------------------------------------------

def remaining_maturity(original_maturity: float, bar_index: int, dt: float, floor: float = 1e-6) -> float:
    """Contract's time-to-expiry (years) at `bar_index`, given it started at
    `original_maturity` years and each bar advances `dt` years. Floored at
    `floor` so a pricer is never called with maturity <= 0.
    """
    return max(original_maturity - bar_index * dt, floor)


def quoting_params_for_run(model: str, meta: dict) -> dict:
    """Reconstruct the exact `params` dict passed to the quoting pricer for
    a run, from `BacktestResults.meta` alone (no engine changes needed --
    `meta` already carries everything: `true_params` for rate/div_yield,
    plus either `flat_vol` (black_scholes) or `heston_quoter_params`
    (heston), exactly as `backtest.engine.run_backtest` set them at call
    time).
    """
    true_params = meta["true_params"]
    if model == "black_scholes":
        return {"rate": true_params.rate, "vol": meta["flat_vol"], "div_yield": true_params.div_yield}
    if model == "heston":
        return dict(meta["heston_quoter_params"])
    raise ValueError(f"Unknown model {model!r}; expected 'black_scholes' or 'heston'")


def bootstrap_mean_ci(values: list[float], n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of `values`. Returns (mean, ci_lo, ci_hi)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)
