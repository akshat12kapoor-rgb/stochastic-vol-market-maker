"""View 4 (Sweep / Conditional Edge) API.

`GET /results` reads `data/sweep_results.csv` (the 40-seed sweep already
computed by `notebooks/sweep_analysis.py`) rather than recomputing it -- the
build spec is explicit that the ~105s sweep must never block a page load.
`POST /recompute` runs a fresh sweep as a background thread with a pollable
job id. `POST /reproduce` re-runs one seed for both models (a few hundred ms)
so a scatter-point click can load that seed straight into View 3.
"""
from __future__ import annotations

import csv
import pathlib
import threading
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtest.engine import run_backtest
from backtest.market_sim import TRUE_MARKET_PARAMS
from backtest.quoter_calibration import calibrate_heston_quoter_params, compute_flat_vol_for_bs_quoter
from webapp.derive import OLSFit, bootstrap_mean_ci, ols_with_ci
from webapp.store import get_sweep_quoter_params, job_store

router = APIRouter(prefix="/api/sweep", tags=["sweep"])

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SWEEP_CSV = ROOT / "data" / "sweep_results.csv"

REGIME_FEATURES = ["mean_variance", "vol_of_var", "max_spot_drawdown", "terminal_variance", "final_spot"]


def _read_sweep_csv() -> list[dict]:
    if not SWEEP_CSV.exists():
        raise HTTPException(status_code=404, detail=f"{SWEEP_CSV} not found -- run POST /api/sweep/recompute first")
    with open(SWEEP_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {"seed": int(float(row["seed"]))}
            for k, v in row.items():
                if k == "seed":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)
    return rows


@router.get("/results")
def results() -> dict:
    rows = _read_sweep_csv()
    for r in rows:
        r["pnl_diff_heston_minus_bs"] = r["pnl_heston"] - r["pnl_bs"]
    return {"rows": rows, "n_seeds": len(rows), "csv_path": str(SWEEP_CSV)}


@router.get("/regression")
def regression(feature: Literal["mean_variance", "vol_of_var", "max_spot_drawdown", "terminal_variance", "final_spot"]) -> dict:
    """OLS fit of (pnl_heston - pnl_bs) against one regime feature, plus a
    bootstrap CI on the mean paired difference -- see webapp/derive.py.
    """
    rows = _read_sweep_csv()
    x = [r[feature] for r in rows]
    y = [r["pnl_heston"] - r["pnl_bs"] for r in rows]
    fit: OLSFit = ols_with_ci(x, y)
    mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(y)

    # Plain-language significance call, per the build spec: don't imply a
    # trend the data can't support. Using p < 0.05 on the OLS slope as the
    # (conventional, not sacred) threshold, stated explicitly for n=len(rows).
    significant = fit.p_value < 0.05
    return {
        "feature": feature,
        "n": fit.n,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.r_squared,
        "p_value": fit.p_value,
        "significant_at_0_05": significant,
        "verdict": (
            f"Slope is distinguishable from zero at n={fit.n} (p={fit.p_value:.3f})."
            if significant else
            f"No detectable regime dependence at n={fit.n} (p={fit.p_value:.3f}) -- "
            "this is a legitimate 'no relationship found' result, not a failed fit."
        ),
        "x_grid": fit.x_grid,
        "y_pred": fit.y_pred,
        "y_pred_lo": fit.y_pred_lo,
        "y_pred_hi": fit.y_pred_hi,
        "paired_diff_mean": mean_diff,
        "paired_diff_ci_lo": ci_lo,
        "paired_diff_ci_hi": ci_hi,
    }


@router.get("/quoter-params")
def quoter_params() -> dict:
    return get_sweep_quoter_params()


class ReproduceRequest(BaseModel):
    seed: int


@router.post("/reproduce")
def reproduce(req: ReproduceRequest) -> dict:
    """Re-run one sweep seed for both models, with the EXACT parameterization
    notebooks/sweep_analysis.py used (arrival_seed = seed + 1000, cached
    flat_vol / heston_quoter_params, all other params at run_backtest's own
    defaults) -- so clicking seed 15 / seed 25 in the scatter reproduces
    FINAL_REPORT.md's numbers exactly. Delegates to the same /api/backtest
    run path so the response shape (and run_id caching for the waterfall
    panel) matches a normal View 3 run.
    """
    from webapp.routers.backtest import RunRequest, _run_one  # local import avoids a router-to-router cycle

    cached = get_sweep_quoter_params()
    arrival_seed = req.seed + 1000
    bs_result = _run_one(RunRequest(
        model="black_scholes", market_seed=req.seed, arrival_seed=arrival_seed, flat_vol=cached["flat_vol"],
    ))
    heston_result = _run_one(RunRequest(
        model="heston", market_seed=req.seed, arrival_seed=arrival_seed,
        heston_quoter_params=cached["heston_quoter_params"],
    ))
    return {"seed": req.seed, "black_scholes": bs_result, "heston": heston_result}


N_SWEEP_SEEDS_DEFAULT = 40


def _run_sweep_job(job_id: str, n_seeds: int) -> None:
    job = job_store.get(job_id)
    flat_vol = compute_flat_vol_for_bs_quoter(TRUE_MARKET_PARAMS)
    heston_params = calibrate_heston_quoter_params(TRUE_MARKET_PARAMS)
    rows = []
    try:
        for seed in range(1, n_seeds + 1):
            arrival_seed = seed + 1000
            r_bs = run_backtest("black_scholes", market_seed=seed, arrival_seed=arrival_seed, flat_vol=flat_vol)
            r_h = run_backtest("heston", market_seed=seed, arrival_seed=arrival_seed, heston_quoter_params=heston_params)
            rows.append({
                "seed": seed,
                "pnl_bs": r_bs.pnl[-1], "pnl_heston": r_h.pnl[-1],
                "sharpe_bs": r_bs.sharpe, "sharpe_heston": r_h.sharpe,
                "maxdd_bs": r_bs.max_drawdown, "maxdd_heston": r_h.max_drawdown,
                "nfills_bs": r_bs.meta["n_fills"], "nfills_heston": r_h.meta["n_fills"],
                "final_spot": r_bs.spot_path[-1], "spot0": r_bs.spot_path[0],
                "terminal_variance": r_bs.variance_path[-1],
                "mean_variance": float(np.mean(r_bs.variance_path)),
                "vol_of_var": float(np.std(np.diff(r_bs.variance_path))),
                "max_spot_drawdown": float(np.min(r_bs.spot_path / np.maximum.accumulate(r_bs.spot_path) - 1.0)),
            })
            if job is not None:
                job.completed = seed
        SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(SWEEP_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if job is not None:
            job.status = "done"
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the poller rather than dying silently
        if job is not None:
            job.status = "error"
            job.error = str(exc)


class RecomputeRequest(BaseModel):
    n_seeds: int = N_SWEEP_SEEDS_DEFAULT


@router.post("/recompute")
def recompute(req: RecomputeRequest) -> dict:
    job = job_store.create(total=req.n_seeds)
    thread = threading.Thread(target=_run_sweep_job, args=(job.job_id, req.n_seeds), daemon=True)
    thread.start()
    return {"job_id": job.job_id, "total": job.total}


@router.get("/recompute/status/{job_id}")
def recompute_status(job_id: str) -> dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id {job_id!r} not found")
    return {"job_id": job.job_id, "status": job.status, "completed": job.completed, "total": job.total, "error": job.error}
