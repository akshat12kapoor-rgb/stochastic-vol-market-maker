"""View 2 (Vol Surface) API. Wraps `pricing.vol_surface.generate_vol_surface`
and `vol_models.calibration.{calibrate_heston,calibrate_sabr,surface_rmse}`
directly; residual grids are the only thing derived here (target - model
IV per grid point), which the engine doesn't compute itself.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pricing.vol_surface import generate_vol_surface
from vol_models.api import fair_value
from vol_models.calibration import calibrate_heston, calibrate_sabr, surface_rmse
from vol_models.greeks_utils import implied_vol_from_price

router = APIRouter(prefix="/api/vol-surface", tags=["vol-surface"])

DEFAULT_SPOT = 100.0
DEFAULT_STRIKES = [round(60 + 5 * i, 1) for i in range(17)]  # 60..140 step 5
DEFAULT_MATURITIES = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0]
DEFAULT_SHAPE = {
    "atm_vol_short": 0.24, "atm_vol_long": 0.18, "term_decay": 1.0,
    "skew_short": -0.55, "skew_long": -0.10, "skew_decay": 0.75,
    "smile_short": 0.35, "smile_long": 0.08, "smile_decay": 0.75,
    "min_iv": 0.02,
}


class SurfaceRequest(BaseModel):
    spot: float = DEFAULT_SPOT
    strikes: list[float] = DEFAULT_STRIKES
    maturities: list[float] = DEFAULT_MATURITIES
    atm_vol_short: float = DEFAULT_SHAPE["atm_vol_short"]
    atm_vol_long: float = DEFAULT_SHAPE["atm_vol_long"]
    term_decay: float = DEFAULT_SHAPE["term_decay"]
    skew_short: float = DEFAULT_SHAPE["skew_short"]
    skew_long: float = DEFAULT_SHAPE["skew_long"]
    skew_decay: float = DEFAULT_SHAPE["skew_decay"]
    smile_short: float = DEFAULT_SHAPE["smile_short"]
    smile_long: float = DEFAULT_SHAPE["smile_long"]
    smile_decay: float = DEFAULT_SHAPE["smile_decay"]
    min_iv: float = DEFAULT_SHAPE["min_iv"]


class CalibrateRequest(SurfaceRequest):
    models: list[Literal["heston", "sabr"]] = ["heston", "sabr"]
    rate: float = 0.03
    div_yield: float = 0.0


def _generate(req: SurfaceRequest):
    return generate_vol_surface(
        req.spot, req.strikes, req.maturities,
        atm_vol_short=req.atm_vol_short, atm_vol_long=req.atm_vol_long, term_decay=req.term_decay,
        skew_short=req.skew_short, skew_long=req.skew_long, skew_decay=req.skew_decay,
        smile_short=req.smile_short, smile_long=req.smile_long, smile_decay=req.smile_decay,
        min_iv=req.min_iv,
    )


@router.get("/defaults")
def defaults() -> dict:
    return {"spot": DEFAULT_SPOT, "strikes": DEFAULT_STRIKES, "maturities": DEFAULT_MATURITIES, **DEFAULT_SHAPE}


@router.post("/generate")
def generate(req: SurfaceRequest) -> dict:
    try:
        surface = _generate(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "strikes": surface.columns.tolist(),
        "maturities": surface.index.tolist(),
        "iv_grid": surface.to_numpy().tolist(),  # rows=maturity, cols=strike, matches generate_vol_surface's shape
    }


def _model_iv_grid(model: str, params: dict, spot: float, strikes: list[float], maturities: list[float]) -> list[list[float]]:
    """Price every (maturity, strike) grid point under a calibrated model and
    invert to a BS implied vol, for apples-to-apples overlay against the
    target surface's own IV units. Row-major by maturity, matching
    `generate_vol_surface`'s DataFrame orientation.
    """
    grid = []
    for t in maturities:
        row = []
        for k in strikes:
            price, _ = fair_value(model, params, spot, k, t, "call")
            iv = implied_vol_from_price(price, spot, k, t, params["rate"], option_type="call", div_yield=params.get("div_yield", 0.0))
            row.append(iv)
        grid.append(row)
    return grid


@router.post("/calibrate")
def calibrate(req: CalibrateRequest) -> dict:
    try:
        surface = _generate(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_grid = surface.to_numpy()
    strikes = surface.columns.tolist()
    maturities = surface.index.tolist()

    fits: dict[str, dict] = {}
    for model in req.models:
        if model == "heston":
            params = calibrate_heston(surface, req.spot, rate=req.rate, div_yield=req.div_yield)
        elif model == "sabr":
            params = calibrate_sabr(surface, req.spot, rate=req.rate, div_yield=req.div_yield)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model {model!r}")

        model_grid = np.array(_model_iv_grid(model, params, req.spot, strikes, maturities))
        residual_grid = model_grid - target_grid  # model - target, in IV units (vol points if x100)
        rmse = surface_rmse(model_grid.flatten(), target_grid.flatten())

        fits[model] = {
            "params": params,
            "iv_grid": model_grid.tolist(),
            "residual_grid": residual_grid.tolist(),
            "rmse": rmse,
        }

    return {
        "strikes": strikes,
        "maturities": maturities,
        "target_iv_grid": target_grid.tolist(),
        "fits": fits,
        "not_arbitrage_checked": True,
        "note": (
            "This synthetic surface is a hand-tuned, quadratic-in-log-moneyness stand-in "
            "(pricing.vol_surface.generate_vol_surface) -- it is NOT checked for calendar-spread "
            "or butterfly no-arbitrage violations. Every downstream fit (Heston, SABR, and the "
            "backtest's own 'true market') ultimately derives from this surface's shape."
        ),
    }
