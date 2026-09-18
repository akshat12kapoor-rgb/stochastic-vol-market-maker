from __future__ import annotations

from fastapi.testclient import TestClient

from webapp.main import app

client = TestClient(app)


def test_pricer_price_black_scholes():
    resp = client.post("/api/pricer/price", json={
        "model": "black_scholes", "params": {"rate": 0.03, "vol": 0.2, "div_yield": 0.0},
        "spot": 100.0, "strike": 100.0, "maturity": 1.0, "option_type": "call",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["price"] > 0
    assert set(body["greeks"]) == {"delta", "gamma", "vega", "theta", "rho"}
    assert "stderr" not in body


def test_pricer_price_monte_carlo_has_stderr():
    resp = client.post("/api/pricer/price", json={
        "model": "monte_carlo", "params": {"rate": 0.03, "vol": 0.2, "div_yield": 0.0, "n_paths": 20000, "seed": 1},
        "spot": 100.0, "strike": 100.0, "maturity": 1.0, "option_type": "call",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["stderr"] > 0


def test_pricer_price_heston_and_sabr():
    for model, params in [
        ("heston", {"kappa": 1.8, "theta": 0.045, "sigma": 0.55, "rho": -0.65, "v0": 0.045, "rate": 0.03}),
        ("sabr", {"alpha": 0.3, "beta": 1.0, "rho": -0.4, "nu": 0.5, "rate": 0.03}),
    ]:
        resp = client.post("/api/pricer/price", json={
            "model": model, "params": params, "spot": 100.0, "strike": 100.0, "maturity": 1.0, "option_type": "call",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["price"] > 0


def test_pricer_profile_across_strikes():
    resp = client.post("/api/pricer/profile", json={
        "model": "black_scholes", "params": {"rate": 0.03, "vol": 0.2},
        "spot": 100.0, "strikes": [80, 90, 100, 110, 120], "maturity": 1.0, "option_type": "call",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["prices"]) == 5
    assert len(body["greeks"]["delta"]) == 5
    assert body["greeks"]["delta"][0] > body["greeks"]["delta"][-1]


def test_pricer_profile_rejects_monte_carlo():
    resp = client.post("/api/pricer/profile", json={
        "model": "monte_carlo", "params": {"rate": 0.03, "vol": 0.2},
        "spot": 100.0, "strikes": [90, 100, 110], "maturity": 1.0,
    })
    assert resp.status_code == 400
