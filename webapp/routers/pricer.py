"""View 1 (Pricer) API. Every endpoint is a thin wrapper: it calls
`pricing.api.price` or `vol_models.api.fair_value` (the shared
`(model, params, spot, strike, maturity, option_type) -> (price, greeks)`
contract) and serializes the result. No pricing logic lives here.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pricing.api import price as bs_or_mc_price
from pricing.monte_carlo import mc_price
from vol_models.api import fair_value as vol_fair_value

router = APIRouter(prefix="/api/pricer", tags=["pricer"])

Model = Literal["black_scholes", "monte_carlo", "heston", "sabr"]

DEFAULT_PARAMS: dict[str, dict] = {
    "black_scholes": {"rate": 0.03, "vol": 0.20, "div_yield": 0.0},
    "monte_carlo": {"rate": 0.03, "vol": 0.20, "div_yield": 0.0, "n_paths": 100_000, "antithetic": True, "seed": 42},
    "heston": {"kappa": 1.8, "theta": 0.045, "sigma": 0.55, "rho": -0.65, "v0": 0.045, "rate": 0.03, "div_yield": 0.0},
    "sabr": {"alpha": 0.3, "beta": 1.0, "rho": -0.4, "nu": 0.5, "rate": 0.03, "div_yield": 0.0},
}


def _dispatch(model: Model, params: dict, spot: float, strike: float, maturity: float, option_type: str):
    """Route to pricing.api.price (BS/MC) or vol_models.api.fair_value (Heston/SABR)."""
    if model in ("black_scholes", "monte_carlo"):
        return bs_or_mc_price(model, params, spot, strike, maturity, option_type)
    if model in ("heston", "sabr"):
        return vol_fair_value(model, params, spot, strike, maturity, option_type)
    raise HTTPException(status_code=400, detail=f"Unknown model {model!r}")


class PricePoint(BaseModel):
    model: Model
    params: dict
    spot: float
    strike: float
    maturity: float
    option_type: Literal["call", "put"] = "call"


class PriceProfileRequest(BaseModel):
    model: Model
    params: dict
    spot: float
    strikes: list[float]
    maturity: float
    option_type: Literal["call", "put"] = "call"


@router.get("/defaults")
def defaults(model: Model) -> dict:
    return {"model": model, "params": DEFAULT_PARAMS[model]}


@router.post("/price")
def price(req: PricePoint) -> dict:
    """Point price + Greeks. For monte_carlo, also returns `stderr` (from
    `pricing.monte_carlo.mc_price` directly, since `pricing.api.price` drops
    it for cross-model API-shape parity -- see pricing/api.py's docstring;
    the spec explicitly asks the UI to always show MC's standard error).
    """
    try:
        price_val, greeks = _dispatch(req.model, req.params, req.spot, req.strike, req.maturity, req.option_type)
    except (ValueError, KeyError, FloatingPointError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result: dict = {"price": price_val, "greeks": greeks}
    if req.model == "monte_carlo":
        p = req.params
        _, stderr = mc_price(
            req.spot, req.strike, req.maturity, p["rate"], p["vol"], req.option_type,
            p.get("div_yield", 0.0), p.get("n_paths", 100_000), p.get("antithetic", True), p.get("seed"),
        )
        result["stderr"] = stderr
    return result


@router.post("/profile")
def profile(req: PriceProfileRequest) -> dict:
    """Price + Greeks across a strike grid, for the 'profile across strike'
    plots (the point spec argues a single-strike number hides where two
    models diverge). Monte Carlo is intentionally excluded here -- it's
    explicit-run-only per the build spec's performance constraints (~1.2ms
    for Heston/SABR/BS vs far slower for 100k-path MC); use POST /price
    per-strike from the frontend if a MC profile is ever needed, with
    explicit debouncing there.
    """
    if req.model == "monte_carlo":
        raise HTTPException(
            status_code=400,
            detail="monte_carlo is explicit-run-only (too slow for a live strike-profile scan); use POST /price per point.",
        )
    strikes = req.strikes
    prices: list[float] = []
    greeks_by_key: dict[str, list[float]] = {"delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
    for k in strikes:
        try:
            p, g = _dispatch(req.model, req.params, req.spot, k, req.maturity, req.option_type)
        except (ValueError, KeyError, FloatingPointError):
            p = float("nan")
            g = {key: float("nan") for key in greeks_by_key}
        prices.append(p)
        for key in greeks_by_key:
            greeks_by_key[key].append(g.get(key, float("nan")))
    return {"strikes": strikes, "prices": prices, "greeks": greeks_by_key}
