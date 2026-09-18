"""FastAPI app for the options market-maker webapp.

Run with (from the project root, venv active):
    uvicorn webapp.main:app --reload --port 8000

Then open http://localhost:8000/ .
"""
from __future__ import annotations

import pathlib

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from webapp.routers import backtest, pricer, sweep, vol_surface

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

app = FastAPI(title="Options Market Maker (Stochastic Vol)")

app.include_router(pricer.router)
app.include_router(vol_surface.router)
app.include_router(backtest.router)
app.include_router(sweep.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon() -> Response:
    # Silences the browser's automatic favicon request (was a harmless but
    # noisy 404 against the StaticFiles mount below).
    return Response(status_code=204)


# Serves index.html at "/" and everything else in webapp/static/ (app.js,
# etc.) at its own path -- mounted last so it doesn't shadow the /api routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
