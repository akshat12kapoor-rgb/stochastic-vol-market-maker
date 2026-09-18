"""Builds the BS-vs-calibrated-Heston-vs-calibrated-SABR comparison report:
calibrates both models against `pricing.vol_surface.generate_vol_surface`'s
default-parameter target, computes per-model IV RMSE, and saves a
side-by-side smile plot to `data/vol_model_comparison.png`.

Run directly (`python -m vol_models.comparison`) to regenerate the PNG and
print the RMSE table; the Stochastic Vol Agent's final report reproduces
those numbers. Not part of `pytest` (it's a report-generation script, not
a correctness test -- correctness of the underlying pricers/calibration is
covered by tests/test_heston.py, tests/test_sabr.py, tests/test_calibration.py).

**Rate/dividend assumption (flagged, needs sign-off):** `generate_vol_surface`
produces a pure IV surface with no embedded rate -- Heston/SABR calibration
needs a `rate` to convert spot<->forward and to invert Heston prices back to
IV. We assume `rate=0.03`, `div_yield=0.0` throughout this module. Changing
this assumption would shift the calibrated Heston/SABR params (via the
forward) but not materially change the RMSE comparison, since the target
surface itself does not vary with this choice.
"""
from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pricing import generate_vol_surface, vol_surface_to_long  # noqa: E402

from vol_models.calibration import calibrate_heston, calibrate_sabr, surface_rmse  # noqa: E402
from vol_models.greeks_utils import implied_vol_from_price  # noqa: E402
from vol_models.heston import heston_price  # noqa: E402
from vol_models.sabr import sabr_implied_vol  # noqa: E402

SPOT = 100.0
RATE = 0.03
DIV_YIELD = 0.0
# Smaller than the Pricing Engine Agent's sample_vol_surface.png grid
# (60-140 step 5, 7 maturities) to keep Heston calibration (which needs a
# BS implied-vol inversion per grid point per pricing call) fast; still
# wide enough to see the skew/smile/term-structure shape clearly.
STRIKES = np.arange(70, 131, 5)
MATURITIES = [1 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0]


def _bs_calibrated_vol(target_iv: np.ndarray) -> float:
    """Closed-form 'calibration' of a single constant BS vol: the equal-weight
    least-squares minimizer of sum((vol - target_iv_i)^2) is just the mean."""
    return float(np.mean(target_iv))


def _heston_iv_grid(strikes, maturities, params, n_terms=224, L=14.0):
    out = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            p = heston_price(SPOT, K, T, RATE, params["kappa"], params["theta"], params["sigma"],
                              params["rho"], params["v0"], option_type="call",
                              div_yield=DIV_YIELD, n_terms=n_terms, L=L)
            out[i, j] = implied_vol_from_price(p, SPOT, K, T, RATE, div_yield=DIV_YIELD)
    return out


def _sabr_iv_grid(strikes, maturities, params):
    out = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        forward = SPOT * np.exp((RATE - DIV_YIELD) * T)
        out[i, :] = sabr_implied_vol(forward, strikes, T, params["alpha"], params["beta"],
                                      params["rho"], params["nu"])
    return out


def run_comparison(output_dir="data", strikes=None, maturities=None) -> dict:
    strikes = STRIKES if strikes is None else np.asarray(strikes)
    maturities = MATURITIES if maturities is None else list(maturities)

    target_surface = generate_vol_surface(SPOT, strikes, maturities)
    long_df = vol_surface_to_long(target_surface)
    target_iv_flat = long_df["iv"].to_numpy()

    heston_params = calibrate_heston(target_surface, SPOT, rate=RATE, div_yield=DIV_YIELD,
                                      n_terms=128, L=12.0, max_nfev=150)
    sabr_params = calibrate_sabr(target_surface, SPOT, rate=RATE, div_yield=DIV_YIELD,
                                  beta=1.0, max_nfev=200)
    bs_vol = _bs_calibrated_vol(target_iv_flat)

    heston_grid = _heston_iv_grid(strikes, maturities, heston_params)
    sabr_grid = _sabr_iv_grid(strikes, maturities, sabr_params)
    bs_grid = np.full((len(maturities), len(strikes)), bs_vol)
    target_grid = target_surface.to_numpy()

    rmse = {
        "black_scholes": surface_rmse(bs_grid, target_grid),
        "heston": surface_rmse(heston_grid, target_grid),
        "sabr": surface_rmse(sabr_grid, target_grid),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    maturities_to_plot = maturities if len(maturities) <= 6 else maturities[:: len(maturities) // 6 + 1][:6]
    for ax, T in zip(axes.flat, maturities_to_plot):
        i = maturities.index(T)
        ax.plot(strikes, target_grid[i], "ko-", label="Target (synthetic)", linewidth=2, markersize=4)
        ax.plot(strikes, bs_grid[i], "s--", label="BS (flat)", color="tab:red", markersize=3)
        ax.plot(strikes, heston_grid[i], "^--", label="Heston (calibrated)", color="tab:blue", markersize=3)
        ax.plot(strikes, sabr_grid[i], "v--", label="SABR (calibrated)", color="tab:green", markersize=3)
        ax.set_title(f"T = {T:.2f}y")
        ax.set_xlabel("Strike")
        ax.axvline(SPOT, color="gray", linestyle=":", linewidth=0.8)
    axes.flat[0].set_ylabel("Implied vol")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"BS vs calibrated Heston vs calibrated SABR -- IV RMSE: "
        f"BS={rmse['black_scholes']:.4f}  Heston={rmse['heston']:.4f}  SABR={rmse['sabr']:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = pathlib.Path(output_dir) / "vol_model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    return {
        "rmse": rmse,
        "heston_params": heston_params,
        "sabr_params": sabr_params,
        "bs_vol": bs_vol,
        "output_path": str(output_path),
        "strikes": strikes,
        "maturities": maturities,
    }


if __name__ == "__main__":
    result = run_comparison()
    print("RMSE (IV units):", result["rmse"])
    print("Heston params:", result["heston_params"])
    print("SABR params:", result["sabr_params"])
    print("BS calibrated vol:", result["bs_vol"])
    print("Plot saved to:", result["output_path"])
