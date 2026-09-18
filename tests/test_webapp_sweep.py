"""Sweep view tests. Intentionally does NOT exercise POST /api/sweep/recompute
-- that overwrites data/sweep_results.csv (the committed 40-seed sweep
FINAL_REPORT.md references), which is not something a test run should do.
It was verified manually against a running server with n_seeds=2, then the
CSV was restored via `git checkout -- data/sweep_results.csv`.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp.main import app

client = TestClient(app)


def test_sweep_results_reads_existing_csv():
    resp = client.get("/api/sweep/results")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_seeds"] == 40
    assert all("pnl_diff_heston_minus_bs" in r for r in body["rows"])


def test_sweep_regression_reports_feature_and_verdict():
    resp = client.get("/api/sweep/regression", params={"feature": "mean_variance"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n"] == 40
    assert "verdict" in body
    assert len(body["x_grid"]) == len(body["y_pred"])


def test_sweep_reproduce_seed15_matches_final_report():
    """FINAL_REPORT.md §3: seed=15 -> BS $-1.6, Heston $156.8 (Heston wins)."""
    resp = client.post("/api/sweep/reproduce", json={"seed": 15})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    bs_pnl = body["black_scholes"]["pnl"][-1]
    h_pnl = body["heston"]["pnl"][-1]
    assert bs_pnl == pytest.approx(-1.56, abs=0.05)
    assert h_pnl == pytest.approx(156.76, abs=0.5)


def test_sweep_reproduce_seed25_matches_final_report():
    """FINAL_REPORT.md §3: seed=25 -> BS $471.4, Heston $306.99 (BS wins)."""
    resp = client.post("/api/sweep/reproduce", json={"seed": 25})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    bs_pnl = body["black_scholes"]["pnl"][-1]
    h_pnl = body["heston"]["pnl"][-1]
    assert bs_pnl == pytest.approx(471.37, abs=0.5)
    assert h_pnl == pytest.approx(306.99, abs=0.5)
