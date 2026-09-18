"""Realized-market simulator: generates the "ground truth" underlying path.

Simulates Heston spot/variance dynamics via **full truncation Euler**
discretization (Lord, Koekkoek & van Dijk, 2010) -- the same discretization
family `vol_models.heston.heston_mc_price` uses (see ARCHITECTURE.md's
`vol_models` section), though this module is self-contained and does not
import `vol_models.heston` directly: this is market-DATA generation (a
"what actually happened" path we play the backtest against), not pricing,
so it belongs in `backtest`, not `vol_models`.

At every step:
    v_pos          = max(v_t, 0)                                  # floor only where v feeds a sqrt/drift
    v_{t+dt}        = v_t + kappa*(theta - v_pos)*dt + sigma*sqrt(v_pos*dt)*Z_v
    S_{t+dt}        = S_t * exp((rate - div_yield - 0.5*v_pos)*dt + sqrt(v_pos*dt)*Z_s)
    Z_v = rho*Z_s + sqrt(1-rho**2)*Z_perp,  Z_s, Z_perp iid standard normal

The *state* v itself is carried forward unfloored (may go transiently
negative, exactly like `vol_models.heston`'s discretization) -- callers that
need to feed a realized variance value back into a pricer (e.g. the
backtest engine's "true fair value" reference, see `backtest/engine.py`)
must floor it themselves at the point of use.

**TRUE market parameters (flagged decision -- needs sign-off, see final
report):** `TRUE_MARKET_PARAMS` below is the single "ground truth" Heston
market every backtest run is simulated against and compared to.
Deliberately chosen to violate the Feller condition
(`2*kappa*theta = 0.162 < sigma**2 = 0.3025`), consistent with
`vol_models`'s own stress tests of calibrated-Heston-violates-Feller being
the realistic case, and to stress-test full truncation Euler's positivity
handling here too. `v0 == theta` (start at the stationary long-run level)
to avoid an artificial transient drift at the start of every backtest run.
`sqrt(theta) ~= 21.2%` annualized long-run vol and `rho = -0.65` (equity-
index-like negative spot/vol correlation) are order-of-magnitude matches to
`pricing.vol_surface`'s default short-maturity ATM vol (~24%) and skew.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonMarketParams:
    """Heston parameters for a simulated "ground truth" market.

    Attributes:
        spot0: initial underlying price.
        v0: initial instantaneous variance (annualized, e.g. 0.04 == 20% vol).
        kappa: mean-reversion speed of variance (> 0).
        theta: long-run variance level (> 0).
        sigma: vol-of-vol (> 0).
        rho: spot/variance correlation, in (-1, 1).
        rate: annualized continuously-compounded risk-free rate.
        div_yield: annualized continuous dividend yield (default 0.0).
    """

    spot0: float
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    rate: float
    div_yield: float = 0.0

    def as_dict(self, v0_override: float | None = None) -> dict:
        """This market's params as a `vol_models.fair_value("heston", ...)`-shaped dict.

        `v0_override`, if given, replaces `v0` -- used by the backtest engine
        to reprice the TRUE market's own fair value using the CURRENT
        realized (floored) instantaneous variance at each bar, rather than
        the t=0 starting level.
        """
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "v0": self.v0 if v0_override is None else v0_override,
            "rate": self.rate,
            "div_yield": self.div_yield,
        }


# The single "ground truth" market every backtest run in this package is
# simulated against and compared to. See module docstring for the full
# reasoning -- this is a flagged decision, not an arbitrary pick.
TRUE_MARKET_PARAMS = HestonMarketParams(
    spot0=100.0,
    v0=0.045,
    kappa=1.8,
    theta=0.045,
    sigma=0.55,
    rho=-0.65,
    rate=0.03,
    div_yield=0.0,
)


@dataclass(frozen=True)
class MarketPath:
    """A single simulated realization of the underlying's (spot, variance) path.

    Attributes:
        times: shape (n_steps+1,), `times[i] = i*dt` in years, `times[0] == 0`.
        spot: shape (n_steps+1,), realized spot at each time (`spot[0] == params.spot0`).
        variance: shape (n_steps+1,), realized (possibly transiently negative
            under full truncation Euler -- see module docstring) instantaneous
            variance at each time (`variance[0] == params.v0`).
        dt: the fixed per-step time increment (years) used to generate this path.
        seed: the RNG seed used to generate this path (for reproducibility bookkeeping).
    """

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    dt: float
    seed: int


def simulate_heston_path(params: HestonMarketParams, n_steps: int, dt: float, seed: int) -> MarketPath:
    """Simulate one realized (spot, variance) path under `params`'s Heston dynamics.

    Full truncation Euler (see module docstring for the exact update
    equations). Deterministic given `seed` -- same `(params, n_steps, dt,
    seed)` always produces a bit-identical path (verified in
    `tests/test_market_sim.py::test_simulate_heston_path_is_deterministic_given_seed`
    and, at the backtest-engine level, in
    `tests/test_engine.py::test_identical_seeds_give_identical_realized_paths_across_models`
    -- the property the BS-quoting vs Heston-quoting comparison's "apples to
    apples" claim rests on).

    Args:
        params: the market's Heston parameters (see `HestonMarketParams`).
        n_steps: number of time steps to simulate; the returned arrays have
            length `n_steps + 1` (including t=0).
        dt: fixed time increment per step, in years (e.g. `1/252` for daily bars).
        seed: RNG seed (int), consumed by a fresh `numpy.random.default_rng(seed)`.

    Returns:
        `MarketPath` with `spot`/`variance` arrays of length `n_steps + 1`.

    Raises:
        ValueError: `n_steps < 1` or `dt <= 0`.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    rng = np.random.default_rng(seed)
    spot = np.empty(n_steps + 1, dtype=float)
    variance = np.empty(n_steps + 1, dtype=float)
    spot[0] = params.spot0
    variance[0] = params.v0

    sqrt_1m_rho2 = math.sqrt(max(1.0 - params.rho ** 2, 0.0))
    for i in range(n_steps):
        v_pos = max(variance[i], 0.0)
        sqrt_v_dt = math.sqrt(v_pos * dt)
        z_s = rng.standard_normal()
        z_perp = rng.standard_normal()
        z_v = params.rho * z_s + sqrt_1m_rho2 * z_perp

        variance[i + 1] = variance[i] + params.kappa * (params.theta - v_pos) * dt + params.sigma * sqrt_v_dt * z_v
        spot[i + 1] = spot[i] * math.exp(
            (params.rate - params.div_yield - 0.5 * v_pos) * dt + sqrt_v_dt * z_s
        )

    times = np.arange(n_steps + 1, dtype=float) * dt
    return MarketPath(times=times, spot=spot, variance=variance, dt=dt, seed=seed)
