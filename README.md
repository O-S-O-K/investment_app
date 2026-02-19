# Investment App (Personal Portfolio Intelligence)

[![Live App](https://img.shields.io/badge/Live-App-brightgreen)](https://investment-app-dashboard.onrender.com)

This MVP helps you mirror your real portfolio, track 401(k) allocations, and generate data-driven suggestions using a combined SAA/TAA framework.

## What this includes

- Portfolio + cash + lot-level holding storage (SQLite)
- 401(k) plan options and current allocation storage
- Market data ingestion via Yahoo Finance
- SAA engine:
  - strategic expected returns (historical + shrinkage proxy)
  - covariance estimation
  - risk-aware optimizer with volatility target and weight caps
- TAA engine:
  - trend signal (moving average)
  - momentum signal (12-1 style proxy)
  - tactical overlay to tilt strategic weights
- Forecasting helper:
  - simple machine-learning return forecast (tree ensemble)
- FastAPI endpoints + Streamlit dashboard starter

## Important note

This is a personal research tool, not financial advice. Always validate assumptions and constraints against your IPS, taxes, account restrictions, and fiduciary requirements.

## Quick start

1. Create and activate a virtual environment
2. Install dependencies
3. Create your environment file
4. Run API
5. Run dashboard

### Commands (PowerShell)

```powershell
cd investment_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd investment_app
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/streamlit_app.py
```

Dashboard now includes two tabs:

- `Signals & Allocation`
- `Import & Rebalance` (CSV upload + tax-lot rebalance preview)

## API security

All `/portfolio/*` and `/analytics/*` routes require `X-API-Key`.

- Set `API_KEY` in `.env`
- Use the same key in dashboard sidebar (`X-API-Key` field)

PowerShell example:

```powershell
$headers = @{ "X-API-Key" = "change-me" }
Invoke-RestMethod -Headers $headers -Method Get -Uri "http://127.0.0.1:8000/analytics/signals?tickers=SPY,QQQ,AGG"
```

## API overview

- `GET /health`
- `POST /portfolio/holdings`
- `GET /portfolio/holdings`
- `POST /portfolio/holdings/import-csv`
- `POST /portfolio/rebalance/plan`
- `POST /portfolio/401k/options`
- `GET /portfolio/401k/options`
- `POST /portfolio/401k/allocation`
- `GET /portfolio/401k/allocation`
- `GET /analytics/signals?tickers=SPY,QQQ,AGG`
- `GET /analytics/recommendation?tickers=SPY,QQQ,AGG`

## Broker CSV import

Endpoint: `POST /portfolio/holdings/import-csv`

Required CSV columns:

- `ticker`
- `shares`
- `cost_basis`

Optional CSV column:

- `acquired_at` (format: `YYYY-MM-DD`)

Example:

```csv
ticker,shares,cost_basis,acquired_at
VTI,25,5900,2022-01-15
VXUS,40,2200,2023-06-10
BND,30,2100,2024-04-03
```

## Tax-lot rebalancing plan

Endpoint: `POST /portfolio/rebalance/plan`

Request body example:

```json
{
  "target_weights": {
    "VTI": 0.55,
    "VXUS": 0.25,
    "BND": 0.20
  },
  "short_term_tax_rate": 0.37,
  "long_term_tax_rate": 0.20,
  "min_trade_value": 50
}
```

The planner prefers loss lots first, then lower-tax lots, and returns estimated tax drag per sell trade.

## Deploy for phone access (Render)

This repo includes [render.yaml](render.yaml) for Blueprint deploy.

1. Push your latest changes to GitHub
2. In Render, choose **New +** → **Blueprint**
3. Select this repository
4. Set these environment variables in Render:
  - `investment-app-api`: `API_KEY` (choose a strong key)
  - `investment-app-dashboard`: `API_KEY` (same value as API)
5. Deploy both services
6. Open `https://investment-app-dashboard.onrender.com` on your phone

After deploy, if your dashboard URL differs, update the Live badge link above.

## Suggested next upgrades

- Broker sync adapters (Schwab/Fidelity/IBKR CSV/API)
- Tax-lot aware rebalancing and wash-sale checks
- Brinson attribution and decision journal
- Regime switching model (Markov/HMM)
- Macro feature store (FRED inflation/rates/employment)
