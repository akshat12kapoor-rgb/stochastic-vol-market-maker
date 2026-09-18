from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp.main import app

client = TestClient(app)


def test_backtest_run_black_scholes_small():
    resp = client.post("/api/backtest/run", json={"model": "black_scholes", "market_seed": 1, "n_steps": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"]
    assert len(body["times"]) == 6
    assert len(body["spot_path"]) == 6
    assert "fills" in body
    for fill in body["fills"]:
        assert "edge" in fill


def test_backtest_run_pair_shares_realized_path():
    resp = client.post("/api/backtest/run-pair", json={"market_seed": 7, "n_steps": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    bs_path = body["black_scholes"]["spot_path"]
    h_path = body["heston"]["spot_path"]
    assert bs_path == h_path


def test_backtest_waterfall_reconstructs_engine_quote_exactly():
    run_resp = client.post("/api/backtest/run", json={"model": "heston", "market_seed": 3, "n_steps": 10})
    assert run_resp.status_code == 200, run_resp.text
    run_body = run_resp.json()
    run_id = run_body["run_id"]
    contract_id = list(run_body["inventory"].keys())[0]

    wf_resp = client.post("/api/backtest/waterfall", json={"run_id": run_id, "contract_id": contract_id, "bar_index": 5})
    assert wf_resp.status_code == 200, wf_resp.text
    body = wf_resp.json()

    wf = body["waterfall"]
    eq = body["engine_quote"]
    assert wf["reservation_price"] == pytest.approx(eq["reservation_price"], abs=1e-9)
    assert wf["bid"] == pytest.approx(eq["bid"], abs=1e-9)
    assert wf["ask"] == pytest.approx(eq["ask"], abs=1e-9)
    assert wf["spread"] == pytest.approx(eq["spread"], abs=1e-9)
    assert wf["fair_value"] + wf["delta_skew"] + wf["vega_skew"] == pytest.approx(eq["reservation_price"], abs=1e-9)
    summed_spread = wf["risk_spread_delta"] + wf["risk_spread_vega"] + wf["liquidity_spread"] + wf["gamma_term"]
    assert summed_spread == pytest.approx(eq["spread"], abs=1e-9)


def test_backtest_waterfall_what_if_override_changes_result():
    run_resp = client.post("/api/backtest/run", json={"model": "heston", "market_seed": 3, "n_steps": 10})
    run_body = run_resp.json()
    run_id = run_body["run_id"]
    contract_id = list(run_body["inventory"].keys())[0]

    base = client.post("/api/backtest/waterfall", json={"run_id": run_id, "contract_id": contract_id, "bar_index": 5}).json()
    whatif = client.post("/api/backtest/waterfall", json={
        "run_id": run_id, "contract_id": contract_id, "bar_index": 5,
        "risk_aversion_delta_override": 2.0, "risk_aversion_vega_override": 2.0,
    }).json()
    assert whatif["is_what_if"] is True
    assert base["is_what_if"] is False
    assert whatif["waterfall"]["spread"] != base["waterfall"]["spread"]


def test_backtest_waterfall_unknown_run_id_404():
    resp = client.post("/api/backtest/waterfall", json={"run_id": "doesnotexist", "contract_id": "x", "bar_index": 0})
    assert resp.status_code == 404
