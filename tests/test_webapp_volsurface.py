from __future__ import annotations

from fastapi.testclient import TestClient

from webapp.main import app

client = TestClient(app)


def test_vol_surface_generate():
    resp = client.post("/api/vol-surface/generate", json={
        "spot": 100.0, "strikes": [80, 90, 100, 110, 120], "maturities": [0.25, 1.0],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["iv_grid"]) == 2
    assert len(body["iv_grid"][0]) == 5


def test_vol_surface_calibrate_returns_fits_and_residuals():
    resp = client.post("/api/vol-surface/calibrate", json={
        "spot": 100.0, "strikes": [80, 90, 100, 110, 120], "maturities": [0.25, 1.0],
        "models": ["heston", "sabr"],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["not_arbitrage_checked"] is True
    assert "heston" in body["fits"] and "sabr" in body["fits"]
    for model in ("heston", "sabr"):
        fit = body["fits"][model]
        assert fit["rmse"] >= 0
        assert len(fit["residual_grid"]) == 2
        assert len(fit["residual_grid"][0]) == 5
