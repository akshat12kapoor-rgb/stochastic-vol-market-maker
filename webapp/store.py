"""In-memory state for the webapp: recent backtest runs and cached
deterministic computations. No database, per the build spec -- this is a
single-process dev server; state is lost on restart, which is fine (a
backtest run is ~0.6-2s, cheap to redo).
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from backtest.results import BacktestResults
from market_maker.book import OptionContract


@dataclass
class RunRecord:
    run_id: str
    results: BacktestResults
    basket: list[OptionContract]


class RunStore:
    def __init__(self, max_runs: int = 200) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._order: list[str] = []
        self._max_runs = max_runs
        self._lock = threading.Lock()

    def put(self, results: BacktestResults, basket: list[OptionContract]) -> RunRecord:
        run_id = uuid.uuid4().hex[:12]
        record = RunRecord(run_id=run_id, results=results, basket=basket)
        with self._lock:
            self._runs[run_id] = record
            self._order.append(run_id)
            while len(self._order) > self._max_runs:
                oldest = self._order.pop(0)
                self._runs.pop(oldest, None)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)


@dataclass
class SweepJob:
    job_id: str
    status: str = "running"  # "running" | "done" | "error"
    completed: int = 0
    total: int = 0
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, SweepJob] = {}
        self._lock = threading.Lock()

    def create(self, total: int) -> SweepJob:
        job = SweepJob(job_id=uuid.uuid4().hex[:12], total=total)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> SweepJob | None:
        return self._jobs.get(job_id)


# Module-level singletons -- one process, one app.
run_store = RunStore()
job_store = JobStore()

# Cache for the sweep's deterministic (TRUE_MARKET_PARAMS-only) quoter
# params, so re-clicking sweep points doesn't recompute a Heston calibration
# (~1s) every time. Cleared only by process restart.
_quoter_params_cache: dict | None = None
_quoter_params_lock = threading.Lock()


def get_sweep_quoter_params() -> dict:
    """flat_vol / heston_quoter_params, computed once, matching exactly what
    notebooks/sweep_analysis.py used for the 40-seed sweep (both are pure
    functions of TRUE_MARKET_PARAMS -- see backtest.quoter_calibration).
    """
    global _quoter_params_cache
    with _quoter_params_lock:
        if _quoter_params_cache is None:
            from backtest.market_sim import TRUE_MARKET_PARAMS
            from backtest.quoter_calibration import (
                calibrate_heston_quoter_params,
                compute_flat_vol_for_bs_quoter,
            )

            _quoter_params_cache = {
                "flat_vol": compute_flat_vol_for_bs_quoter(TRUE_MARKET_PARAMS),
                "heston_quoter_params": calibrate_heston_quoter_params(TRUE_MARKET_PARAMS),
            }
        return _quoter_params_cache
