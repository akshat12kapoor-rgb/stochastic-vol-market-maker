"""View 3 (Run) API: configure/run a backtest, fetch a cached run, and
reconstruct a single bar's quote waterfall.

`POST /run` and `POST /run-pair` are thin wrappers around
`backtest.engine.run_backtest`. `POST /waterfall` reconstructs a quote by
calling `market_maker.quoting.generate_quotes` with a `Book` rebuilt from
`RunRecord.basket` + `results.inventory[...][bar_index]` (both already
available -- no backtest re-run, no engine changes), exactly as the build
spec requires.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtest.engine import DEFAULT_MATURITIES, DEFAULT_RISK_AVERSION, DEFAULT_STRIKES, build_default_basket, run_backtest
from backtest.results import BacktestResults
from market_maker.book import Book, OptionContract
from market_maker.quoting import RiskAversion, compute_book_greeks, generate_quotes
from pricing.api import price as bs_pricer
from vol_models.api import fair_value as heston_pricer
from webapp.derive import annotate_fills, build_quote_waterfall, quoting_params_for_run, remaining_maturity
from webapp.store import get_sweep_quoter_params, run_store

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

Model = Literal["black_scholes", "heston"]


def _quoting_pricer(model: str):
    return bs_pricer if model == "black_scholes" else heston_pricer


def _build_basket(strikes: list[float] | None, maturities: list[float] | None) -> list[OptionContract]:
    if strikes is None and maturities is None:
        return build_default_basket()
    return build_default_basket(strikes or DEFAULT_STRIKES, maturities or DEFAULT_MATURITIES)


def _build_risk_aversion(delta: float | None, vega: float | None, gamma: float | None) -> RiskAversion | None:
    """None if the caller didn't touch any component -- lets run_backtest use
    its own DEFAULT_RISK_AVERSION rather than us silently re-deriving it.
    """
    if delta is None and vega is None and gamma is None:
        return None
    base = DEFAULT_RISK_AVERSION
    return RiskAversion(
        delta=base.delta if delta is None else delta,
        vega=base.vega if vega is None else vega,
        gamma=base.gamma if gamma is None else gamma,
    )


def serialize_results(results: BacktestResults) -> dict:
    meta = dict(results.meta)
    true_params = meta.get("true_params")
    if true_params is not None:
        meta["true_params"] = dataclasses.asdict(true_params)
    return {
        "model": results.model,
        "times": results.times.tolist(),
        "dt": results.dt,
        "spot_path": results.spot_path.tolist(),
        "variance_path": results.variance_path.tolist(),
        "pnl": results.pnl.tolist(),
        "cash": results.cash.tolist(),
        "inventory": {cid: arr.tolist() for cid, arr in results.inventory.items()},
        "marks": {cid: arr.tolist() for cid, arr in results.marks.items()},
        "delta_book": results.delta_book.tolist(),
        "vega_book": results.vega_book.tolist(),
        "gross_inventory": results.gross_inventory().tolist(),
        "total_inventory": results.total_inventory().tolist(),
        "fills": annotate_fills(results.fills),
        "sharpe": results.sharpe,
        "max_drawdown": results.max_drawdown,
        "meta": meta,
    }


class RunRequest(BaseModel):
    model: Model
    market_seed: int = 42
    arrival_seed: int = 123
    n_steps: int = 63
    strikes: list[float] | None = None
    maturities: list[float] | None = None
    risk_aversion_delta: float | None = None
    risk_aversion_vega: float | None = None
    risk_aversion_gamma: float | None = None
    base_intensity: float | None = None
    order_arrival_kappa: float | None = None
    flat_vol: float | None = None
    heston_quoter_params: dict | None = None


def _run_one(req: RunRequest) -> dict:
    basket = _build_basket(req.strikes, req.maturities)
    kwargs: dict = {"market_seed": req.market_seed, "arrival_seed": req.arrival_seed,
                     "n_steps": req.n_steps, "basket": basket}

    ra = _build_risk_aversion(req.risk_aversion_delta, req.risk_aversion_vega, req.risk_aversion_gamma)
    if ra is not None:
        kwargs["risk_aversion"] = ra
    if req.base_intensity is not None:
        kwargs["base_intensity"] = req.base_intensity
    if req.order_arrival_kappa is not None:
        kwargs["order_arrival_kappa"] = req.order_arrival_kappa

    # Default to the cached, sweep-consistent quoter params (deterministic
    # functions of TRUE_MARKET_PARAMS alone) rather than letting
    # run_backtest recompute a fresh Heston calibration (~1s) per request --
    # a performance judgment call, flagged here: the numbers are identical
    # either way (both are pure functions of the same fixed TRUE_MARKET_PARAMS),
    # this only avoids redundant work.
    cached = get_sweep_quoter_params()
    if req.model == "black_scholes":
        kwargs["flat_vol"] = req.flat_vol if req.flat_vol is not None else cached["flat_vol"]
    else:
        kwargs["heston_quoter_params"] = req.heston_quoter_params or cached["heston_quoter_params"]

    try:
        results = run_backtest(req.model, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = run_store.put(results, basket)
    return {"run_id": record.run_id, **serialize_results(results)}


@router.post("/run")
def run(req: RunRequest) -> dict:
    return _run_one(req)


class RunPairRequest(BaseModel):
    """Same seed pair, both models -- the one-click BS-vs-Heston comparison."""
    market_seed: int = 42
    arrival_seed: int = 123
    n_steps: int = 63
    strikes: list[float] | None = None
    maturities: list[float] | None = None
    risk_aversion_delta: float | None = None
    risk_aversion_vega: float | None = None
    risk_aversion_gamma: float | None = None
    base_intensity: float | None = None
    order_arrival_kappa: float | None = None


@router.post("/run-pair")
def run_pair(req: RunPairRequest) -> dict:
    shared = dict(
        market_seed=req.market_seed, arrival_seed=req.arrival_seed, n_steps=req.n_steps,
        strikes=req.strikes, maturities=req.maturities,
        risk_aversion_delta=req.risk_aversion_delta, risk_aversion_vega=req.risk_aversion_vega,
        risk_aversion_gamma=req.risk_aversion_gamma, base_intensity=req.base_intensity,
        order_arrival_kappa=req.order_arrival_kappa,
    )
    bs_result = _run_one(RunRequest(model="black_scholes", **shared))
    heston_result = _run_one(RunRequest(model="heston", **shared))
    return {"black_scholes": bs_result, "heston": heston_result}


@router.get("/run/{run_id}")
def get_run(run_id: str) -> dict:
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found (server restarted, or never ran)")
    return {"run_id": record.run_id, **serialize_results(record.results)}


class WaterfallRequest(BaseModel):
    run_id: str
    contract_id: str
    bar_index: int
    risk_aversion_delta_override: float | None = None
    risk_aversion_vega_override: float | None = None


@router.post("/waterfall")
def waterfall(req: WaterfallRequest) -> dict:
    record = run_store.get(req.run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run_id {req.run_id!r} not found")
    results = record.results
    n_bars = len(results.times)
    if not (0 <= req.bar_index < n_bars):
        raise HTTPException(status_code=400, detail=f"bar_index must be in [0, {n_bars - 1}]")
    contract_ids = {c.contract_id for c in record.basket}
    if req.contract_id not in contract_ids:
        raise HTTPException(status_code=404, detail=f"contract_id {req.contract_id!r} not in this run's basket")

    dt = results.dt
    model = results.model
    meta = results.meta
    quoting_params = quoting_params_for_run(model, meta)
    pricer = _quoting_pricer(model)

    book = Book()
    for c in record.basket:
        tau = remaining_maturity(c.maturity, req.bar_index, dt)
        aged = OptionContract(contract_id=c.contract_id, strike=c.strike, maturity=tau, option_type=c.option_type)
        inv = float(results.inventory[c.contract_id][req.bar_index])
        book.add_contract(aged, quantity=inv)

    stored_ra = meta["risk_aversion"]
    ra = RiskAversion(
        delta=stored_ra["delta"] if req.risk_aversion_delta_override is None else req.risk_aversion_delta_override,
        vega=stored_ra["vega"] if req.risk_aversion_vega_override is None else req.risk_aversion_vega_override,
        gamma=stored_ra["gamma"],
    )

    spot_t = float(results.spot_path[req.bar_index])
    quotes = generate_quotes(
        book, pricer, model, quoting_params, spot_t,
        sigma_underlying=meta["sigma_underlying"], risk_aversion=ra, sigma_vol=meta["sigma_vol"],
        horizon=None, order_arrival_kappa=meta["order_arrival_kappa"],
    )
    target_quote = quotes[req.contract_id]
    valuations = {cid: (q.fair_value, q.greeks) for cid, q in quotes.items()}
    book_greeks = compute_book_greeks(book, pricer, model, quoting_params, spot_t, valuations=valuations)
    target_contract = book.contracts[req.contract_id]

    wf = build_quote_waterfall(
        target_quote.fair_value, book_greeks["delta"], book_greeks["vega"],
        target_quote.greeks["delta"], target_quote.greeks["vega"], target_quote.greeks["gamma"],
        meta["sigma_underlying"], meta["sigma_vol"], target_contract.maturity, ra,
        meta["order_arrival_kappa"], spot_t,
    )

    return {
        "bar_index": req.bar_index,
        "t": float(results.times[req.bar_index]),
        "spot": spot_t,
        "contract_id": req.contract_id,
        "remaining_maturity": target_contract.maturity,
        "risk_aversion_used": {"delta": ra.delta, "vega": ra.vega, "gamma": ra.gamma},
        "is_what_if": req.risk_aversion_delta_override is not None or req.risk_aversion_vega_override is not None,
        "waterfall": dataclasses.asdict(wf),
        "engine_quote": {
            "fair_value": target_quote.fair_value,
            "reservation_price": target_quote.reservation_price,
            "bid": target_quote.bid,
            "ask": target_quote.ask,
            "spread": target_quote.spread,
        },
    }
