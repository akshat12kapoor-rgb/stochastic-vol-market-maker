"""Monte Carlo European option pricer with antithetic variance reduction.

Simulates terminal lognormal spot prices under the risk-neutral measure
(exact one-step simulation of GBM -- no discretization error since only
the terminal value is needed for a European payoff) and discounts the
expected payoff.

Variance reduction: antithetic variates. For each of n_paths // 2 draws
of a standard normal Z, we simulate both Z and -Z, pair their discounted
payoffs, and average each pair. The reported price is the mean of the
pair-averages; the standard error is std(pair-averages) / sqrt(n_pairs),
which is the statistically correct standard error for the antithetic
estimator (NOT std(all 2*n_pairs raw payoffs) / sqrt(2*n_pairs), which
would overstate the achieved variance reduction).

Greeks are estimated by central-difference bumping of the pricing inputs,
reusing the SAME underlying standard-normal draws (common random numbers,
CRN) for the bumped and base simulations. This cancels most of the Monte
Carlo noise in the *difference*, which is what a finite-difference Greek
needs to be stable.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from pricing.black_scholes import OptionType

DEFAULT_N_PATHS = 100_000

# Bump sizes for finite-difference Greeks (central differences), reusing CRN.
_DS = 1e-2      # relative bump handled via spot * eps below
_DVOL = 1e-4
_DT = 1e-4
_DR = 1e-4


def _simulate_terminal(spot: float, maturity: float, rate: float, vol: float,
                        div_yield: float, z: np.ndarray) -> np.ndarray:
    """Exact terminal GBM spot for standard-normal draws z (any shape)."""
    drift = (rate - div_yield - 0.5 * vol ** 2) * maturity
    diffusion = vol * np.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


def _payoff(s_t: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(s_t - strike, 0.0)
    return np.maximum(strike - s_t, 0.0)


def _pair_averaged_discounted_payoffs(spot: float, strike: float, maturity: float,
                                       rate: float, vol: float, option_type: OptionType,
                                       div_yield: float, z_half: np.ndarray) -> np.ndarray:
    """Return the array of per-pair antithetic-averaged discounted payoffs."""
    s_plus = _simulate_terminal(spot, maturity, rate, vol, div_yield, z_half)
    s_minus = _simulate_terminal(spot, maturity, rate, vol, div_yield, -z_half)
    disc = np.exp(-rate * maturity)
    payoff_plus = disc * _payoff(s_plus, strike, option_type)
    payoff_minus = disc * _payoff(s_minus, strike, option_type)
    return 0.5 * (payoff_plus + payoff_minus)


def mc_price(spot: float, strike: float, maturity: float, rate: float, vol: float,
             option_type: OptionType = "call", div_yield: float = 0.0,
             n_paths: int = DEFAULT_N_PATHS, antithetic: bool = True,
             seed: int | None = None) -> tuple[float, float]:
    """Monte Carlo price of a European option under GBM.

    Args:
        spot, strike, maturity, rate, vol, div_yield: same units/meaning as
            `pricing.black_scholes.bs_price` (maturity in years, rate/vol
            annualized continuously-compounded).
        option_type: "call" or "put".
        n_paths: total number of simulated paths (default 100,000). When
            antithetic=True this must be even; n_paths // 2 antithetic
            pairs are drawn (each pair counts as 2 of the n_paths total).
        antithetic: whether to use antithetic variates (default True).
            Set False to get a plain Monte Carlo estimate (used only for
            comparison/testing -- production callers should leave this True).
        seed: optional RNG seed for reproducibility. None -> nondeterministic.

    Returns:
        (price, stderr): the discounted expected-payoff estimate and its
        Monte Carlo standard error (1 standard deviation), in the same
        price units as spot/strike. For antithetic=True, stderr is
        computed correctly from the paired-average estimator (see module
        docstring), not from the raw per-path payoffs.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        n_pairs = n_paths // 2
        z_half = rng.standard_normal(n_pairs)
        pair_payoffs = _pair_averaged_discounted_payoffs(
            spot, strike, maturity, rate, vol, option_type, div_yield, z_half)
        price_est = float(np.mean(pair_payoffs))
        stderr = float(np.std(pair_payoffs, ddof=1) / np.sqrt(n_pairs))
    else:
        z = rng.standard_normal(n_paths)
        s_t = _simulate_terminal(spot, maturity, rate, vol, div_yield, z)
        disc_payoffs = np.exp(-rate * maturity) * _payoff(s_t, strike, option_type)
        price_est = float(np.mean(disc_payoffs))
        stderr = float(np.std(disc_payoffs, ddof=1) / np.sqrt(n_paths))

    return price_est, stderr


def mc_greeks(spot: float, strike: float, maturity: float, rate: float, vol: float,
              option_type: OptionType = "call", div_yield: float = 0.0,
              n_paths: int = DEFAULT_N_PATHS, seed: int | None = 0
              ) -> dict[str, float]:
    """Monte Carlo Greeks via central-difference bumping with common random numbers.

    Args/units: same as `mc_price`. `seed` defaults to a fixed value (0)
    rather than None so that base and bumped simulations share the exact
    same draws by construction (each bump re-seeds the RNG identically),
    which is what makes the finite differences stable.

    Returns a dict with the same keys/units as `bs_greeks`: delta, gamma,
    vega (per 1.0 vol), theta (per 1.0 year, annualized), rho (per 1.0 rate).
    """
    def price_at(s, k, t, r, v, q):
        p, _ = mc_price(s, k, t, r, v, option_type, q, n_paths, antithetic=True, seed=seed)
        return p

    ds = spot * _DS
    p_up = price_at(spot + ds, strike, maturity, rate, vol, div_yield)
    p_down = price_at(spot - ds, strike, maturity, rate, vol, div_yield)
    p_mid = price_at(spot, strike, maturity, rate, vol, div_yield)
    delta = (p_up - p_down) / (2 * ds)
    gamma = (p_up - 2 * p_mid + p_down) / (ds ** 2)

    p_vol_up = price_at(spot, strike, maturity, rate, vol + _DVOL, div_yield)
    p_vol_down = price_at(spot, strike, maturity, rate, vol - _DVOL, div_yield)
    vega = (p_vol_up - p_vol_down) / (2 * _DVOL)

    # Theta: derivative w.r.t. calendar time = -d(price)/d(maturity).
    p_t_up = price_at(spot, strike, maturity + _DT, rate, vol, div_yield)
    p_t_down = price_at(spot, strike, maturity - _DT, rate, vol, div_yield)
    theta = -(p_t_up - p_t_down) / (2 * _DT)

    p_r_up = price_at(spot, strike, maturity, rate + _DR, vol, div_yield)
    p_r_down = price_at(spot, strike, maturity, rate - _DR, vol, div_yield)
    rho = (p_r_up - p_r_down) / (2 * _DR)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def price(model: Literal["monte_carlo"], params: dict, spot: float, strike: float,
          maturity: float, option_type: OptionType = "call"
          ) -> tuple[float, dict[str, float]]:
    """Unified pricing entrypoint for the Monte Carlo model.

    Prefer importing `price` from `pricing.api` for a model-agnostic call;
    see `pricing.api.price` for the full contract.

    params keys:
        rate (float, required): annualized risk-free rate.
        vol (float, required): annualized volatility.
        div_yield (float, optional, default 0.0).
        n_paths (int, optional, default 100_000): total simulated paths.
        seed (int or None, optional, default 0): RNG seed for reproducibility.

    Returns (price, greeks_dict) -- see `mc_greeks` for the Greeks contract.
    Note the MC price carries sampling noise (see `mc_price` for stderr);
    this dispatcher discards the stderr for API-shape parity with
    `black_scholes.price`. Call `mc_price` directly if you need the stderr.
    """
    if model != "monte_carlo":
        raise ValueError(f"monte_carlo.price only supports model='monte_carlo', got {model!r}")
    rate = params["rate"]
    vol = params["vol"]
    div_yield = params.get("div_yield", 0.0)
    n_paths = params.get("n_paths", DEFAULT_N_PATHS)
    seed = params.get("seed", 0)
    p, _ = mc_price(spot, strike, maturity, rate, vol, option_type, div_yield, n_paths, True, seed)
    greeks = mc_greeks(spot, strike, maturity, rate, vol, option_type, div_yield, n_paths, seed)
    return p, greeks
