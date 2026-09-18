"""Tests for backtest.arrivals."""
import math

import numpy as np
import pytest

from backtest.arrivals import fill_intensity, poisson_fill_counts

BASE_INTENSITY = 100.0
KAPPA = 1.5
DT = 1 / 252


def test_fill_intensity_at_zero_distance_equals_base_intensity():
    assert fill_intensity(0.0, BASE_INTENSITY, KAPPA) == pytest.approx(BASE_INTENSITY)


def test_fill_intensity_decays_with_distance():
    near = fill_intensity(0.1, BASE_INTENSITY, KAPPA)
    far = fill_intensity(2.0, BASE_INTENSITY, KAPPA)
    assert near > far > 0
    assert far == pytest.approx(BASE_INTENSITY * math.exp(-KAPPA * 2.0))


def test_fill_intensity_negative_distance_exceeds_base_intensity():
    aggressive = fill_intensity(-0.5, BASE_INTENSITY, KAPPA)
    assert aggressive > BASE_INTENSITY


def test_poisson_fill_counts_are_nonnegative_ints():
    rng = np.random.default_rng(0)
    n_buy, n_sell = poisson_fill_counts(0.3, 0.3, BASE_INTENSITY, KAPPA, DT, rng)
    assert isinstance(n_buy, int) and isinstance(n_sell, int)
    assert n_buy >= 0 and n_sell >= 0


def test_closer_quote_fills_more_often_on_average():
    rng = np.random.default_rng(1)
    close_fills = sum(poisson_fill_counts(0.05, 10.0, BASE_INTENSITY, KAPPA, DT, rng)[0] for _ in range(4000))
    far_fills = sum(poisson_fill_counts(2.0, 10.0, BASE_INTENSITY, KAPPA, DT, rng)[0] for _ in range(4000))
    assert close_fills > far_fills


def test_negative_distance_is_capped_not_unbounded():
    # An extremely negative distance would make exp(-kappa*distance) huge;
    # poisson_fill_counts must clip the expected-arrivals-per-bar so `rng
    # .poisson`'s lambda never explodes into a degenerate/slow regime.
    rng = np.random.default_rng(2)
    counts = [poisson_fill_counts(-50.0, 10.0, BASE_INTENSITY, KAPPA, DT, rng)[0] for _ in range(200)]
    mean_count = np.mean(counts)
    # with the module's default cap (3.0 expected arrivals/bar), the
    # empirical mean should land in the same order of magnitude, not
    # thousands/millions as raw exp(-kappa*-50) would otherwise imply.
    assert mean_count < 20


def test_same_seed_reproduces_same_draws():
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    seq_a = [poisson_fill_counts(0.3, 0.4, BASE_INTENSITY, KAPPA, DT, rng_a) for _ in range(20)]
    seq_b = [poisson_fill_counts(0.3, 0.4, BASE_INTENSITY, KAPPA, DT, rng_b) for _ in range(20)]
    assert seq_a == seq_b
