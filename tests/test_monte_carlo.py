"""Cross-validation of the Monte Carlo pricer against Black-Scholes, plus
Monte Carlo-specific sanity checks (antithetic variance reduction, API
dispatcher parity).

Tolerance rationale (see ARCHITECTURE.md and the Pricing Engine Agent's
final report for the full justification): at n_paths=100_000 with
antithetic variates, the empirical MC standard error for representative
option configs (ATM/OTM/ITM, short/long maturity, call/put) is on the
order of 0.01-0.04 in price units (spot=100 scale). We assert agreement
with the closed-form Black-Scholes price to within TOLERANCE_SIGMA=5
Monte Carlo standard errors, computed from the same simulation. Under the
CLT this is an extremely low-flakiness bound (a >5-sigma deviation has
~5.7e-7 probability under the normal approximation) while still being
"tight" in absolute terms (typically well under $0.20 for these configs).
A fixed seed is used throughout so results are reproducible.
"""
import numpy as np
import pytest

from pricing.black_scholes import bs_price
from pricing.monte_carlo import mc_price, mc_greeks, price

TOLERANCE_SIGMA = 5
SEED = 42
N_PATHS = 100_000

CONFIGS = [
    # spot, strike, maturity, rate, vol, option_type, div_yield
    (100.0, 100.0, 1.0, 0.02, 0.20, "call", 0.0),
    (100.0, 100.0, 1.0, 0.02, 0.20, "put", 0.0),
    (100.0, 120.0, 0.5, 0.02, 0.25, "call", 0.0),
    (100.0, 80.0, 0.5, 0.02, 0.25, "put", 0.0),
    (100.0, 100.0, 0.05, 0.02, 0.50, "call", 0.0),
    (100.0, 100.0, 2.0, 0.03, 0.15, "call", 0.01),
    (50.0, 45.0, 0.25, 0.01, 0.30, "put", 0.0),
]


@pytest.mark.parametrize("spot,strike,maturity,rate,vol,option_type,div_yield", CONFIGS)
def test_mc_agrees_with_black_scholes(spot, strike, maturity, rate, vol, option_type, div_yield):
    bs = bs_price(spot, strike, maturity, rate, vol, option_type, div_yield)
    mc, stderr = mc_price(spot, strike, maturity, rate, vol, option_type, div_yield,
                           n_paths=N_PATHS, antithetic=True, seed=SEED)
    tolerance = TOLERANCE_SIGMA * stderr
    assert abs(mc - bs) < tolerance, (
        f"MC={mc:.4f} vs BS={bs:.4f}, diff={abs(mc - bs):.4f} "
        f"exceeds {TOLERANCE_SIGMA} x stderr={stderr:.4f}"
    )


def test_antithetic_reduces_standard_error_vs_plain_mc():
    spot, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.02, 0.20
    _, se_antithetic = mc_price(spot, strike, maturity, rate, vol, "call",
                                 n_paths=N_PATHS, antithetic=True, seed=SEED)
    _, se_plain = mc_price(spot, strike, maturity, rate, vol, "call",
                            n_paths=N_PATHS, antithetic=False, seed=SEED)
    assert se_antithetic < se_plain


def test_mc_price_reproducible_with_fixed_seed():
    args = (100.0, 100.0, 1.0, 0.02, 0.20, "call", 0.0)
    p1, se1 = mc_price(*args, n_paths=20_000, seed=7)
    p2, se2 = mc_price(*args, n_paths=20_000, seed=7)
    assert p1 == p2
    assert se1 == se2


def test_mc_price_varies_with_different_seed():
    args = (100.0, 100.0, 1.0, 0.02, 0.20, "call", 0.0)
    p1, _ = mc_price(*args, n_paths=5_000, seed=1)
    p2, _ = mc_price(*args, n_paths=5_000, seed=2)
    assert p1 != p2


def test_mc_greeks_roughly_match_bs_greeks():
    from pricing.black_scholes import bs_greeks
    spot, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.02, 0.20
    bs_g = bs_greeks(spot, strike, maturity, rate, vol, "call")
    mc_g = mc_greeks(spot, strike, maturity, rate, vol, "call", n_paths=100_000, seed=SEED)
    # Finite-difference MC Greeks with CRN are noisier than the price itself;
    # use loose but meaningful relative tolerances.
    assert mc_g["delta"] == pytest.approx(bs_g["delta"], abs=0.03)
    assert mc_g["vega"] == pytest.approx(bs_g["vega"], abs=2.0)
    assert mc_g["gamma"] == pytest.approx(bs_g["gamma"], abs=0.01)


def test_api_dispatcher_matches_direct_call():
    spot, strike, maturity, rate, vol = 100.0, 100.0, 1.0, 0.02, 0.20
    p_direct, _ = mc_price(spot, strike, maturity, rate, vol, "call", n_paths=50_000, seed=1)
    p_api, g_api = price("monte_carlo", {"rate": rate, "vol": vol, "n_paths": 50_000, "seed": 1},
                          spot, strike, maturity, "call")
    assert p_api == pytest.approx(p_direct)
    assert set(g_api.keys()) == {"delta", "gamma", "vega", "theta", "rho"}


def test_n_paths_reduces_standard_error():
    args = (100.0, 100.0, 1.0, 0.02, 0.20, "call", 0.0)
    _, se_small = mc_price(*args, n_paths=2_000, seed=SEED)
    _, se_large = mc_price(*args, n_paths=200_000, seed=SEED)
    assert se_large < se_small
