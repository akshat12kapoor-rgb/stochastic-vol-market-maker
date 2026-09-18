"""Tests for pricing.vol_surface: shape, skew/smile qualitative behavior,
term-structure flattening, and the plot-to-PNG helper.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing.vol_surface import (
    generate_vol_surface,
    vol_surface_to_long,
    vol_surface_to_dict,
    plot_vol_surface,
)

SPOT = 100.0
STRIKES = [70, 80, 90, 100, 110, 120, 130]
MATURITIES = [1 / 12, 0.25, 0.5, 1.0, 2.0]


def _surface():
    return generate_vol_surface(SPOT, STRIKES, MATURITIES)


def test_surface_shape_and_labels():
    surf = _surface()
    assert isinstance(surf, pd.DataFrame)
    assert surf.shape == (len(MATURITIES), len(STRIKES))
    assert surf.index.name == "maturity"
    assert surf.columns.name == "strike"
    assert list(surf.index) == sorted(MATURITIES)
    assert list(surf.columns) == sorted(STRIKES)


def test_all_ivs_positive_and_above_floor():
    min_iv = 0.03
    surf = generate_vol_surface(SPOT, STRIKES, MATURITIES, min_iv=min_iv)
    assert (surf.to_numpy() >= min_iv).all()


def test_negative_skew_low_strikes_higher_iv_than_high_strikes():
    surf = _surface()
    for maturity in surf.index:
        row = surf.loc[maturity]
        assert row[STRIKES[0]] > row[STRIKES[-1]], (
            f"expected low-strike IV > high-strike IV at T={maturity} for negative skew"
        )


def test_skew_flattens_with_maturity():
    surf = _surface()
    # skew magnitude ~ (IV at lowest strike - IV at highest strike) around ATM
    spread_short = surf.loc[MATURITIES[0], STRIKES[0]] - surf.loc[MATURITIES[0], STRIKES[-1]]
    spread_long = surf.loc[MATURITIES[-1], STRIKES[0]] - surf.loc[MATURITIES[-1], STRIKES[-1]]
    assert spread_short > spread_long > 0


def test_atm_vol_between_short_and_long_asymptotes():
    surf = generate_vol_surface(SPOT, [SPOT], MATURITIES,
                                 atm_vol_short=0.24, atm_vol_long=0.18)
    atm_col = surf[SPOT]
    assert atm_col.iloc[0] > atm_col.iloc[-1]  # short maturity closer to atm_vol_short
    lo, hi = min(0.18, 0.24), max(0.18, 0.24)
    assert (atm_col >= lo - 1e-9).all() and (atm_col <= hi + 1e-9).all()


def test_smile_convexity_wings_higher_than_center_when_skew_small():
    # Zero out skew so we isolate the smile term; wings should exceed ATM.
    surf = generate_vol_surface(SPOT, [80, 100, 120], [1.0],
                                 skew_short=0.0, skew_long=0.0,
                                 smile_short=0.3, smile_long=0.3)
    row = surf.loc[1.0]
    assert row[80] > row[100]
    assert row[120] > row[100]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        generate_vol_surface(-1.0, STRIKES, MATURITIES)
    with pytest.raises(ValueError):
        generate_vol_surface(SPOT, [-10, 100], MATURITIES)
    with pytest.raises(ValueError):
        generate_vol_surface(SPOT, STRIKES, [0.0, 1.0])


def test_vol_surface_to_long_roundtrip():
    surf = _surface()
    long_df = vol_surface_to_long(surf)
    assert set(long_df.columns) == {"strike", "maturity", "iv"}
    assert len(long_df) == len(STRIKES) * len(MATURITIES)
    # spot check one point
    row = long_df[(long_df.strike == 100) & (long_df.maturity == MATURITIES[0])]
    assert len(row) == 1
    assert row.iloc[0].iv == pytest.approx(surf.loc[MATURITIES[0], 100])


def test_vol_surface_to_dict():
    surf = _surface()
    d = vol_surface_to_dict(surf)
    assert isinstance(d, dict)
    assert len(d) == len(STRIKES) * len(MATURITIES)
    assert d[(100.0, MATURITIES[0])] == pytest.approx(surf.loc[MATURITIES[0], 100])


def test_plot_vol_surface_writes_png(tmp_path):
    surf = _surface()
    out_path = tmp_path / "vol_surface.png"
    result = plot_vol_surface(surf, out_path)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
