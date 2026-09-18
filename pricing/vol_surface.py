"""Synthetic implied-vol surface generator.

Produces a strike x maturity grid of Black-Scholes implied vols that
qualitatively resembles a real equity-index surface: negative skew
(puts/low strikes trade at higher IV than calls/high strikes) that is
steepest for short maturities and flattens as maturity grows, plus smile
curvature (convexity in log-moneyness) that also flattens with maturity.

Parameterization (quadratic-in-log-moneyness, decaying term structure)
-----------------------------------------------------------------------
For log-moneyness m = ln(K / spot) and maturity T (years):

    atm_iv(T) = atm_vol_long + (atm_vol_short - atm_vol_long) * exp(-T / term_decay)
    skew(T)   = skew_long   + (skew_short   - skew_long)   * exp(-T / skew_decay)
    smile(T)  = smile_long  + (smile_short  - smile_long)  * exp(-T / smile_decay)

    iv(K, T) = max(atm_iv(T) + skew(T) * m + smile(T) * m^2, min_iv)

This is a deliberately simple closed-form choice (not a calibrated
SVI/SABR fit to real market data) — it is a *reasonable stand-in* for a
plausible equity-index-like surface shape, not a claim of realism. See
the "decisions needing sign-off" note in the Pricing Engine Agent's
final report for why this form was chosen over alternatives (raw SVI,
SABR-generated smiles, etc.).

Defaults are calibrated by eye to look like a typical equity index (e.g.
SPX-like): ATM vol ~18-22%, short-dated skew steep and negative, long-dated
skew shallow, smile wings more pronounced short-dated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_vol_surface(
    spot: float,
    strikes,
    maturities,
    atm_vol_short: float = 0.24,
    atm_vol_long: float = 0.18,
    term_decay: float = 1.0,
    skew_short: float = -0.55,
    skew_long: float = -0.10,
    skew_decay: float = 0.75,
    smile_short: float = 0.35,
    smile_long: float = 0.08,
    smile_decay: float = 0.75,
    min_iv: float = 0.02,
) -> pd.DataFrame:
    """Generate a synthetic implied-vol surface.

    Args:
        spot: reference spot price (> 0) used to compute log-moneyness.
        strikes: 1D array-like of strike prices (price units, same as spot).
        maturities: 1D array-like of maturities in YEARS (> 0).
        atm_vol_short / atm_vol_long: at-the-money (K == spot) IV level in
            the T -> 0 and T -> infinity limits respectively (annualized
            vol, e.g. 0.20 == 20%).
        term_decay: e-folding time (years) over which atm_iv(T) relaxes
            from atm_vol_short to atm_vol_long. Larger = slower decay.
        skew_short / skew_long: coefficient on log-moneyness m = ln(K/spot)
            in the T -> 0 / T -> infinity limits. Negative values produce
            the standard equity "higher IV at low strikes" skew.
        skew_decay: e-folding time (years) over which skew(T) relaxes
            from skew_short to skew_long (skew flattens as T grows when
            |skew_short| > |skew_long|).
        smile_short / smile_long: coefficient on m^2 (smile convexity) in
            the T -> 0 / T -> infinity limits. Positive values curve IV
            upward away from ATM on both wings.
        smile_decay: e-folding time (years) for the smile-coefficient term
            structure.
        min_iv: floor applied to every grid point (annualized vol) to keep
            the surface strictly positive at extreme strikes.

    Returns:
        pandas DataFrame indexed by maturity (rows, sorted ascending) with
        strikes as columns (sorted ascending), values = annualized implied
        vol (e.g. 0.20 == 20%). DataFrame.index.name == "maturity",
        DataFrame.columns.name == "strike".

    Invariants:
        - All returned IVs are >= min_iv.
        - For fixed T, IV is minimized near K == spot when skew(T) is
          small relative to smile(T)*m^2 curvature (i.e. the surface has a
          smile), and is monotonically decreasing in K for typical
          negative-skew-dominated equity parameterizations.
        - Increasing T moves both skew(T) and smile(T) toward their
          "_long" asymptotes (exponential decay in T).
    """
    if spot <= 0:
        raise ValueError("spot must be > 0")
    strikes_arr = np.asarray(sorted(strikes), dtype=float)
    maturities_arr = np.asarray(sorted(maturities), dtype=float)
    if np.any(strikes_arr <= 0):
        raise ValueError("all strikes must be > 0")
    if np.any(maturities_arr <= 0):
        raise ValueError("all maturities must be > 0 (years)")

    log_moneyness = np.log(strikes_arr / spot)  # shape (n_strikes,)
    t = maturities_arr[:, None]                  # shape (n_maturities, 1)
    m = log_moneyness[None, :]                   # shape (1, n_strikes)

    atm_iv = atm_vol_long + (atm_vol_short - atm_vol_long) * np.exp(-t / term_decay)
    skew = skew_long + (skew_short - skew_long) * np.exp(-t / skew_decay)
    smile = smile_long + (smile_short - smile_long) * np.exp(-t / smile_decay)

    iv_grid = atm_iv + skew * m + smile * m ** 2
    iv_grid = np.maximum(iv_grid, min_iv)

    return pd.DataFrame(iv_grid, index=pd.Index(maturities_arr, name="maturity"),
                         columns=pd.Index(strikes_arr, name="strike"))


def vol_surface_to_long(surface: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide (maturity x strike) surface DataFrame to long form.

    Args:
        surface: a DataFrame as returned by `generate_vol_surface`
            (index=maturity, columns=strike, values=iv).

    Returns:
        DataFrame with columns ["maturity", "strike", "iv"], one row per
        grid point, suitable for feeding directly into a calibration
        routine (e.g. `vol_models` fitting a Heston/SABR model to points).
    """
    long_df = surface.stack().rename("iv").reset_index()
    long_df.columns = ["maturity", "strike", "iv"]
    return long_df[["strike", "maturity", "iv"]]


def vol_surface_to_dict(surface: pd.DataFrame) -> dict[tuple[float, float], float]:
    """Convert a wide surface DataFrame to a {(strike, maturity): iv} dict."""
    long_df = vol_surface_to_long(surface)
    return {(row.strike, row.maturity): row.iv for row in long_df.itertuples()}


def plot_vol_surface(surface: pd.DataFrame, output_path: str | Path,
                      title: str = "Synthetic Implied Volatility Surface") -> Path:
    """Render a 3D surface plot of `surface` and save it as a PNG.

    Args:
        surface: a DataFrame as returned by `generate_vol_surface`.
        output_path: file path (any parent dirs are created) to write the
            PNG to.
        title: plot title.

    Returns:
        The resolved Path the PNG was written to. No display is used
        (matplotlib "Agg" backend); safe to call in a headless environment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    strikes = surface.columns.to_numpy(dtype=float)
    maturities = surface.index.to_numpy(dtype=float)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    iv_grid = surface.to_numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(strike_grid, maturity_grid, iv_grid * 100.0,
                            cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label="IV (%)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
