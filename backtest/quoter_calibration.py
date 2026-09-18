"""Derives each run's quoting-pricer parameters from the TRUE market at t=0.

--------------------------------------------------------------------------
THE CENTRAL FRAMING DECISION OF THIS PHASE (flagged, needs sign-off)
--------------------------------------------------------------------------
If the Heston-quoting run were simply handed `backtest.market_sim
.TRUE_MARKET_PARAMS` verbatim, it would have perfect knowledge of the
exact process generating the realized market -- an advantage no real
market maker ever has (real Heston params are estimated/calibrated from
observed option prices, not known exactly). That would answer a much less
interesting question ("does knowing the true model help" -- trivially
yes) than the one this phase is actually meant to inform Phase 6 with
("does quoting with the RIGHT MODEL FAMILY, imperfectly calibrated the way
a real desk would, do better than a flat-vol Black-Scholes desk").

**Decision: option (b).** The Heston-quoting run's params are produced by:
  1. Building a synthetic implied-vol surface AS IF OBSERVED from the true
     market (`build_true_market_iv_surface`): price a strike/maturity grid
     under `TRUE_MARKET_PARAMS` via `vol_models.api.fair_value`, invert each
     price to a Black-Scholes implied vol via
     `vol_models.greeks_utils.implied_vol_from_price`, and add a small
     amount of Gaussian IV noise (`CALIBRATION_IV_NOISE_STD`, default 0.5
     vol points) to emulate realistic bid/ask/staleness noise a desk would
     actually see rather than a mathematically exact vol surface.
  2. Calibrating Heston to that noisy surface with
     `vol_models.calibration.calibrate_heston` -- the SAME routine Agent 2
     built and the same one a downstream desk would actually run.
  3. Using the resulting (imperfectly fit) params dict as the Heston
     quoter's params for the ENTIRE backtest -- calibrated ONCE, at t=0, and
     held fixed. No re-calibration/filtering of `v0` (or any other param)
     happens as the market evolves during the backtest; this is itself a
     simplification (a real desk periodically recalibrates or runs a
     variance filter) that's flagged here rather than solved, since online
     recalibration is out of scope for this phase.

This means the Heston quoter still benefits from being the RIGHT MODEL
FAMILY (mean-reverting stochastic vol, matching skew/smile structure) but
does NOT get the true kappa/theta/sigma/rho/v0 numbers -- exactly the
"calibrated, not omniscient" position a real quant desk is in. It is a
DELIBERATELY MORE CONSERVATIVE (and more realistic) choice than option (a)
(handing over the exact true params), which the final report flags
explicitly: whatever P&L/Sharpe/drawdown edge the Heston run shows in this
setup is attributable to "having the right model family + a reasonable
calibration", not to omniscience, and is likely a LOWER BOUND on what a
perfectly-informed Heston quoter could achieve.

--------------------------------------------------------------------------
THE BS-QUOTER'S FLAT VOL (flagged decision -- needs sign-off)
--------------------------------------------------------------------------
`compute_flat_vol_for_bs_quoter` uses the TRUE market's own ATM
(`strike == spot0`) implied vol, evaluated at a representative maturity, as
the single flat vol handed to the Black-Scholes quoter for the whole
backtest. Rationale: this is the single most defensible "one number" choice
for a flat-vol desk (an ATM-anchored constant, the classic naive
approximation), and it means the BS quoter is NOT handicapped by a badly
chosen vol level -- its P&L/Sharpe/drawdown shortfall relative to the
Heston quoter (if any) should reflect BS's structural inability to price
skew/smile/term-structure, not a bad vol pick. Like the Heston quoter's
calibration, this is computed ONCE at t=0 from the true market and held
fixed for the whole run -- no re-marking to a rolling ATM vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_models.api import fair_value as vm_fair_value
from vol_models.calibration import calibrate_heston
from vol_models.greeks_utils import implied_vol_from_price

from backtest.market_sim import HestonMarketParams

# Strike/maturity grid used to build the synthetic "as if observed" IV
# surface that `calibrate_heston_quoter_params` fits against. Deliberately
# independent of the basket actually being traded in the backtest (a real
# desk calibrates off the whole liquid surface, not just the contracts it's
# currently quoting).
DEFAULT_CALIBRATION_STRIKES: tuple[float, ...] = (80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0)
DEFAULT_CALIBRATION_MATURITIES: tuple[float, ...] = (30 / 365, 90 / 365, 180 / 365, 365 / 365)

# Synthetic IV noise added to the true-market-implied surface before
# calibration, in IV units (0.005 == half a vol point) -- emulates
# bid/ask/staleness noise a real desk's observed surface would carry, so
# the calibration isn't fitting a mathematically exact target. Flagged per
# PROJECT.md rule 3.
CALIBRATION_IV_NOISE_STD = 0.005
DEFAULT_CALIBRATION_SEED = 7

# Representative maturity for the BS quoter's single flat-vol number.
DEFAULT_FLAT_VOL_MATURITY = 180 / 365


def _true_params_dict(true_params: HestonMarketParams) -> dict:
    return true_params.as_dict()


def build_true_market_iv_surface(
    true_params: HestonMarketParams,
    strikes: tuple[float, ...] = DEFAULT_CALIBRATION_STRIKES,
    maturities: tuple[float, ...] = DEFAULT_CALIBRATION_MATURITIES,
    noise_std: float = CALIBRATION_IV_NOISE_STD,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> pd.DataFrame:
    """Build a `pricing.vol_surface.generate_vol_surface`-shaped IV surface
    "as if observed" from the true market at t=0.

    Prices every (strike, maturity) grid point under `true_params` via
    `vol_models.api.fair_value("heston", ...)`, inverts to a Black-Scholes
    implied vol, and adds i.i.d. Gaussian noise (`noise_std`, IV units) via
    a `numpy.random.default_rng(seed)` for reproducibility. Shape matches
    `pricing.vol_surface.generate_vol_surface`'s output (maturity-indexed
    rows, sorted ascending, `index.name == "maturity"`; strike columns,
    sorted ascending, `columns.name == "strike"`), so it can be passed
    directly to `vol_models.calibration.calibrate_heston`.
    """
    rng = np.random.default_rng(seed)
    strikes_sorted = sorted(strikes)
    maturities_sorted = sorted(maturities)
    params_dict = _true_params_dict(true_params)

    data = np.empty((len(maturities_sorted), len(strikes_sorted)), dtype=float)
    for i, maturity in enumerate(maturities_sorted):
        for j, strike in enumerate(strikes_sorted):
            price, _ = vm_fair_value("heston", params_dict, true_params.spot0, strike, maturity, "call")
            iv = implied_vol_from_price(
                price, true_params.spot0, strike, maturity, true_params.rate,
                option_type="call", div_yield=true_params.div_yield,
            )
            noisy_iv = max(iv + rng.normal(0.0, noise_std), 1e-4)
            data[i, j] = noisy_iv

    return pd.DataFrame(
        data,
        index=pd.Index(maturities_sorted, name="maturity"),
        columns=pd.Index(strikes_sorted, name="strike"),
    )


def calibrate_heston_quoter_params(
    true_params: HestonMarketParams,
    strikes: tuple[float, ...] = DEFAULT_CALIBRATION_STRIKES,
    maturities: tuple[float, ...] = DEFAULT_CALIBRATION_MATURITIES,
    noise_std: float = CALIBRATION_IV_NOISE_STD,
    seed: int = DEFAULT_CALIBRATION_SEED,
    max_nfev: int = 150,
) -> dict:
    """Calibrate the Heston quoter's params to a synthetic "observed" surface.

    See module docstring's "central framing decision" -- this is the
    Heston-quoting run's actual pricing model, NOT `true_params`. Returns a
    dict directly usable as `vol_models.api.fair_value`'s `params` argument.
    """
    surface = build_true_market_iv_surface(true_params, strikes, maturities, noise_std, seed)
    return calibrate_heston(
        surface, true_params.spot0, rate=true_params.rate, div_yield=true_params.div_yield, max_nfev=max_nfev
    )


def compute_flat_vol_for_bs_quoter(
    true_params: HestonMarketParams, maturity: float = DEFAULT_FLAT_VOL_MATURITY
) -> float:
    """ATM (`strike == spot0`) implied vol of the TRUE market at t=0.

    See module docstring's "BS-quoter's flat vol" decision. Used as the
    single flat vol handed to the Black-Scholes quoting run for the whole
    backtest.
    """
    params_dict = _true_params_dict(true_params)
    price, _ = vm_fair_value("heston", params_dict, true_params.spot0, true_params.spot0, maturity, "call")
    return implied_vol_from_price(
        price, true_params.spot0, true_params.spot0, maturity, true_params.rate,
        option_type="call", div_yield=true_params.div_yield,
    )
