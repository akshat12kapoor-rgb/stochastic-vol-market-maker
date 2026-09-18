"""Tests for backtest.market_sim."""
import numpy as np
import pytest

from backtest.market_sim import HestonMarketParams, TRUE_MARKET_PARAMS, simulate_heston_path

PARAMS = HestonMarketParams(spot0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.6, rate=0.02)


def test_simulate_heston_path_shapes():
    path = simulate_heston_path(PARAMS, n_steps=20, dt=1 / 252, seed=1)
    assert path.spot.shape == (21,)
    assert path.variance.shape == (21,)
    assert path.times.shape == (21,)
    assert path.spot[0] == PARAMS.spot0
    assert path.variance[0] == PARAMS.v0
    assert path.times[0] == 0.0
    assert path.times[-1] == pytest.approx(20 / 252)
    assert np.all(np.isfinite(path.spot))
    assert np.all(np.isfinite(path.variance))
    assert np.all(path.spot > 0)  # spot is a geometric process, always > 0


def test_simulate_heston_path_is_deterministic_given_seed():
    path_a = simulate_heston_path(PARAMS, n_steps=50, dt=1 / 252, seed=7)
    path_b = simulate_heston_path(PARAMS, n_steps=50, dt=1 / 252, seed=7)
    assert np.array_equal(path_a.spot, path_b.spot)
    assert np.array_equal(path_a.variance, path_b.variance)


def test_different_seeds_give_different_paths():
    path_a = simulate_heston_path(PARAMS, n_steps=50, dt=1 / 252, seed=1)
    path_b = simulate_heston_path(PARAMS, n_steps=50, dt=1 / 252, seed=2)
    assert not np.array_equal(path_a.spot, path_b.spot)


def test_true_market_params_violate_feller_by_design():
    # See backtest.market_sim module docstring: TRUE_MARKET_PARAMS is
    # deliberately chosen to violate the Feller condition, stress-testing
    # full truncation Euler's positivity handling exactly like
    # vol_models.heston's own severe-Feller-violation test.
    assert 2 * TRUE_MARKET_PARAMS.kappa * TRUE_MARKET_PARAMS.theta < TRUE_MARKET_PARAMS.sigma ** 2


def test_survives_severe_feller_violation_no_nan_or_negative_spot():
    path = simulate_heston_path(TRUE_MARKET_PARAMS, n_steps=252, dt=1 / 252, seed=99)
    assert np.all(np.isfinite(path.spot))
    assert np.all(np.isfinite(path.variance))
    assert np.all(path.spot > 0)
    # full truncation Euler explicitly permits the *state* to go transiently
    # negative -- this is not itself a bug, just document that it can happen
    # for this stress config so downstream code (e.g. backtest.engine's
    # variance flooring) isn't surprised by it.


def test_as_dict_matches_vol_models_fair_value_required_keys():
    d = TRUE_MARKET_PARAMS.as_dict()
    for key in ("kappa", "theta", "sigma", "rho", "v0", "rate"):
        assert key in d
    assert d["v0"] == TRUE_MARKET_PARAMS.v0

    overridden = TRUE_MARKET_PARAMS.as_dict(v0_override=0.5)
    assert overridden["v0"] == 0.5
    assert overridden["theta"] == TRUE_MARKET_PARAMS.theta  # only v0 changes


@pytest.mark.parametrize("n_steps,dt", [(0, 1 / 252), (-1, 1 / 252), (10, 0.0), (10, -0.1)])
def test_invalid_args_raise(n_steps, dt):
    with pytest.raises(ValueError):
        simulate_heston_path(PARAMS, n_steps=n_steps, dt=dt, seed=1)
