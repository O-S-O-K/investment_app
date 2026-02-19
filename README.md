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

### One-click startup (recommended)

First time only:

```powershell
cd investment_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Every time you want to use the app:

```powershell
cd investment_app
.\start.ps1
```

`start.ps1` automatically:
- Creates `.env` from `.env.example` on first run
- Starts the API on port `8000`
- Starts the Streamlit dashboard on port `8501`
- Prints your LAN IP so you can open on your phone (same Wi-Fi)

### Manual startup (two terminals)

Terminal 1 — API:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Terminal 2 — Dashboard:
```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Dashboard includes two tabs:

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

Required columns: `ticker`, `shares`, `cost_basis`

Optional columns:

- `acquired_at` (format: `YYYY-MM-DD`) — used for short/long-term tax classification
- `account_type` — `taxable`, `401k`, or `ira` (overrides the UI dropdown if present)

A ready-to-edit sample file is at [samples/holdings_example.csv](samples/holdings_example.csv):

```csv
ticker,shares,cost_basis,acquired_at,account_type
VTI,15.000,4125.00,2021-03-10,taxable
VXUS,20.000,1980.00,2021-06-01,taxable
BND,25.000,1875.00,2020-08-12,taxable
GLD,5.000,1050.00,2022-09-05,taxable
```

- `cost_basis` = total cost (not per share)
- Multiple lots for the same ticker = multiple rows
- 401(k) holdings use `account_type=401k`; they will not be included in taxable rebalance plans

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
