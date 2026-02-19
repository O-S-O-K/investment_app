# Investment App — Personal Portfolio Intelligence

A local-first portfolio management tool that mirrors your real holdings, generates data-driven allocation recommendations, and produces a tax-lot-aware rebalance plan — all from a clean two-tab Streamlit dashboard backed by a FastAPI service.

> **This is a personal research tool, not financial advice.** Always validate output against your IPS, tax situation, account restrictions, and fiduciary requirements.

---

## Features

### Portfolio data

- Tax-lot level holdings storage (SQLite via SQLAlchemy)
- Broker CSV import with **auto-detection** of Fidelity, Schwab, and Vanguard column formats
- 401(k) plan options and allocation tracking
- API-key authentication on all portfolio and analytics routes

### Analytics pipeline

| Layer | What it does |
|---|---|
| **Market data** | Parallel per-ticker fetches via `yf.Ticker().history()` (thread-safe), 10-min TTL cache |
| **TAA signals** | Trend (moving average cross) + momentum (12-1 proxy) overlay per ticker |
| **ML forecast** | Per-ticker RandomForest (50 trees, parallel fit) expected-return estimate, 10-min cache |
| **SAA optimizer** | Max-Sharpe portfolio via `scipy.optimize.minimize` with weight caps and volatility target |
| **Recommendation** | Blends SAA weights with TAA tilt and ML forecast; returns final allocations + portfolio metrics |

All analytics run in **background threads** and are polled by the dashboard  no HTTP timeouts regardless of universe size.

### Rebalance planner

- Tax-lot aware: prefers harvesting losses first, then long-term lots
- Returns per-trade estimated tax drag
- **Copy Weights to Rebalance** button auto-populates rebalance targets from the latest recommendation

### Dashboard layout

| Tab | Left panel | Right panel |
|---|---|---|
| **Import & Signals** | Broker CSV upload | Tactical signals table (colour-coded BUY/SELL) |
| **Allocation & Rebalance** | Recommendation + donut chart + portfolio metrics | Tax-lot rebalance planner |

---

## Quick start

### First-time setup

```powershell
cd investment_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Every-day launch (recommended)

```powershell
.\start.ps1
```

`start.ps1`:
- Kills any stale processes on ports 8000 / 8501
- Creates `.env` from `.env.example` on first run
- Launches the API (`localhost:8000`) and dashboard (`localhost:8501`) in separate windows
- Prints your LAN IP so you can open the app on a phone over the same Wi-Fi

### Manual launch (two terminals)

```powershell
# Terminal 1  API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Dashboard
.\.venv\Scripts\python.exe -m streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Typical workflow

1. **Import & Signals tab** → upload your broker CSV → click **Import CSV**
2. **Import & Signals tab** → set your ticker universe in the sidebar → click **Refresh Signals**
3. **Allocation & Rebalance tab** → click **Generate Recommendation** (runs in background, auto-polls every 3 s)
4. **Allocation & Rebalance tab** → review the donut chart and allocation table → click **Copy Weights to Rebalance →**
5. **Allocation & Rebalance tab** → confirm/adjust tax rates → click **Generate Rebalance Plan**
6. Execute the resulting trade list at your broker

---

## Broker CSV import

Supported brokers (auto-detected, no column renaming needed):

| Broker | Detected columns |
|---|---|
| **Fidelity** | `Symbol`, `Quantity`, `Cost Basis Total`  skips preamble rows |
| **Schwab** | `Symbol`, `Quantity`, `Cost Basis Total` |
| **Vanguard** | `Ticker Symbol`, `Shares`, `Average Cost Basis` |
| **Generic** | Falls back to `ticker`, `shares`, `cost_basis` |

A sample file is at [`samples/holdings_example.csv`](samples/holdings_example.csv).

Rules:
- `cost_basis` = **total** cost (not per share)
- Multiple rows for the same ticker = separate tax lots
- Cash sweep tickers (e.g. `SPAXX**`) are skipped automatically
- `account_type` column (if present) overrides the UI dropdown  valid values: `taxable`, `401k`, `ira`
- Dates accepted as `YYYY-MM-DD`, `MM/DD/YYYY`, or `MM/DD/YY`

---

## API reference

### Authentication

All `/portfolio/*` and `/analytics/*` routes require the header `X-API-Key`.
Set `API_KEY` in `.env`; enter the same value in the dashboard sidebar.

```powershell
$h = @{ "X-API-Key" = "change-me" }
Invoke-RestMethod -Headers $h -Uri "http://127.0.0.1:8000/health"
```

### Endpoints

#### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (no auth required) |

#### Portfolio

| Method | Path | Description |
|---|---|---|
| `POST` | `/portfolio/holdings` | Add a single holding / lot |
| `GET` | `/portfolio/holdings` | List all holdings |
| `POST` | `/portfolio/holdings/import-csv` | Bulk import from broker CSV |
| `POST` | `/portfolio/rebalance/plan` | Generate tax-lot rebalance plan |
| `POST` | `/portfolio/401k/options` | Add a 401(k) fund option |
| `GET` | `/portfolio/401k/options` | List 401(k) fund options |
| `POST` | `/portfolio/401k/allocation` | Set current 401(k) allocations |
| `GET` | `/portfolio/401k/allocation` | Get current 401(k) allocations |

#### Analytics (async job pattern)

| Method | Path | Description |
|---|---|---|
| `POST` | `/analytics/signals/start` | Start TAA signal job; returns `{job_id}` immediately |
| `POST` | `/analytics/recommendation/start` | Start recommendation job; returns `{job_id}` immediately |
| `GET` | `/analytics/jobs/{job_id}` | Poll status: `pending` / `running` / `done` / `error` |
| `GET` | `/analytics/signals` | Synchronous signals (small universes / scripting) |
| `GET` | `/analytics/recommendation` | Synchronous recommendation (small universes / scripting) |

Query param for all analytics routes: `?tickers=SPY,QQQ,AGG,GLD`

#### Rebalance plan request body

```json
{
  "target_weights": { "VTI": 0.55, "VXUS": 0.25, "BND": 0.20 },
  "short_term_tax_rate": 0.37,
  "long_term_tax_rate": 0.20,
  "min_trade_value": 50
}
```

---

## Project layout

```
investment_app/
 app/
    main.py               # FastAPI app factory
    models.py             # SQLAlchemy ORM models
    schemas.py            # Pydantic request/response schemas
    config.py             # Settings (reads .env)
    security.py           # API-key dependency
    job_store.py          # In-memory async job store (TTL 5 min)
    routers/
       analytics.py      # Signal + recommendation endpoints
       portfolio.py      # Holdings, CSV import, rebalance, 401k
    services/
        market_data.py    # Parallel yfinance fetcher + TTL cache
        signals.py        # TAA momentum/trend signals
        forecast.py       # RandomForest return forecasts + cache
        recommendation.py # SAA/TAA/ML blended optimizer
        rebalance.py      # Tax-lot rebalance planner
 dashboard/
    streamlit_app.py      # Two-tab Streamlit UI with async polling
 samples/
    holdings_example.csv  # Sample broker CSV (8 lots)
 start.ps1                 # One-click launcher (Windows)
 pyproject.toml
 .env.example
```

---

## Deploying for phone access (Render)

This repo includes [`render.yaml`](render.yaml) for Blueprint deploy.

1. Push your latest changes to GitHub
2. In Render → **New +** → **Blueprint** → select this repo
3. Set environment variables:
   - `investment-app-api` → `API_KEY` (choose a strong secret)
   - `investment-app-dashboard` → `API_KEY` (same value), `API_BASE` (your API service public URL)
4. Deploy both services
5. Open the dashboard URL on your phone

---

## Possible future enhancements

- Wash-sale detection and tracking across lots
- Brinson attribution and decision journal
- Regime-switching model (Markov/HMM) for TAA overlay
- Macro feature store (FRED: inflation, rates, employment)
- Interactive drift chart: current weights vs target over time
