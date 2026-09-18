# Options Market Maker with Stochastic Volatility

An options market-making bot: pricing (Black-Scholes, Monte Carlo), stochastic
volatility models (Heston, SABR), an Avellaneda-Stoikov-style quoting engine
extended to a multi-Greek options book, and an event-driven backtester.

## Structure

- `pricing/` — Black-Scholes, Monte Carlo European pricer, synthetic vol surfaces
- `vol_models/` — Heston, SABR, calibration routines
- `market_maker/` — inventory-aware quoting engine (delta/vega/gamma skew)
- `backtest/` — event-driven backtest engine, order arrival simulation
- `data/` — synthetic market data / generated vol surfaces
- `notebooks/` — analysis and writeups per phase
- `tests/` — pytest suite

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
