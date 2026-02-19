import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Investment App", layout="wide")
st.title("Personal Portfolio Intelligence")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_target_weights(text: str) -> dict[str, float]:
    target_weights: dict[str, float] = {}
    for entry in text.split(","):
        chunk = entry.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid entry {chunk!r}. Use format TICKER:WEIGHT")
        ticker, value = chunk.split(":", 1)
        target_weights[ticker.strip().upper()] = float(value.strip())
    if not target_weights:
        raise ValueError("At least one ticker:weight pair is required")
    return target_weights


def build_auth_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        return {}
    return {"X-API-Key": api_key.strip()}


def safe_get(url: str, **kwargs):
    try:
        return requests.get(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def safe_post(url: str, **kwargs):
    try:
        return requests.post(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def allocs_to_weight_string(allocations: list) -> str:
    """Convert allocation list to TICKER:weight string for rebalance input."""
    parts = [f"{a['ticker']}:{a['final_weight']:.4f}" for a in allocations]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("API")
    api_base = st.text_input("API Base URL", value=os.getenv("API_BASE", DEFAULT_API_BASE))
    api_key = st.text_input("X-API-Key", value=os.getenv("API_KEY", "change-me"), type="password")

    st.header("Universe")
    tickers_input = st.text_input("Tickers", value="SPY,QQQ,AGG,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.divider()

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------
    with st.expander("About this app", expanded=False):
        st.markdown(
            """
**Personal Portfolio Intelligence** is a local-first portfolio management tool.
It connects to a FastAPI backend that stores your holdings, fetches live market
data, and runs a combined strategic + tactical + machine-learning analytics
pipeline — all on your own machine with no third-party cloud dependency.

---

**What you can do**

| Action | Where |
|---|---|
| Import your holdings from a broker CSV | Import & Signals → left panel |
| View per-ticker tactical signals (BUY / HOLD / SELL) | Import & Signals → right panel |
| Generate an optimised allocation recommendation | Allocation & Rebalance → left panel |
| Copy recommended weights directly into the rebalancer | *Copy Weights to Rebalance →* button |
| Run a tax-lot-aware trade plan | Allocation & Rebalance → right panel |

---

**How to manipulate it**

- **Ticker universe** — add or remove any Yahoo Finance–supported tickers in the
  *Universe* field above. The signals and recommendation will re-run against
  that exact set.
- **API key** — must match the `API_KEY` value in your `.env` file for all
  requests to succeed.
- **Tax rates** — adjust short-term and long-term rates in the rebalance panel
  to model your actual marginal rates.
- **Min trade value** — raise this threshold to suppress small nuisance trades.
- **Target weights** — you can override the auto-filled weights from the
  recommendation with any manual allocation before running the rebalance plan.
- **Account type** — choose *taxable*, *401k*, or *ira* when importing CSV
  lots; 401k and IRA lots are excluded from taxable rebalance plans.

---

*This is a personal research tool, not financial advice.*
"""
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    with st.expander("Architecture & workflow", expanded=False):
        st.markdown(
            """
**System layers**

```
Browser (Streamlit)
    │  HTTP polling every 3 s
    ▼
FastAPI  (localhost:8000)
    │  async job queue (job_store.py)
    │  background thread per request
    ▼
Analytics pipeline
    ├── market_data.py   ← yf.Ticker.history(), parallel threads, 10-min cache
    ├── signals.py       ← trend (MA cross) + momentum (12-1) per ticker
    ├── forecast.py      ← RandomForest 50 trees per ticker, parallel, 10-min cache
    └── recommendation.py
            ├── SAA: max-Sharpe via scipy.optimize (weight caps + vol target)
            ├── TAA: tilt strategic weights by signal score
            └── Blend: final_weight = SAA × TAA overlay × ML forecast nudge
    │
    ▼
SQLite  (investment_app.db)
    ├── holdings        ← ticker, shares, account type
    ├── tax_lots        ← per-lot cost basis, acquired date, long/short flag
    ├── four01k_options
    └── four01k_allocations
```

---

**Request lifecycle — analytics**

1. Dashboard POSTs to `/analytics/recommendation/start`
2. API creates a job record (`pending`) and spawns a background thread
3. API returns `{job_id}` in < 1 s — no timeout risk
4. Background thread runs the full pipeline and writes result to job store
5. Dashboard polls `/analytics/jobs/{job_id}` every 3 s with a spinner
6. When status = `done`, result is rendered; job purged after 5 min TTL

---

**Rebalance planner logic**

1. Fetch latest prices for all held tickers
2. Compute current portfolio value and per-ticker weight
3. Calculate required drift vs target weights
4. Sort lots for each over-weight ticker:
   - Loss lots first (tax-free harvest)
   - Long-term gain lots second (lower tax rate)
   - Short-term gain lots last
5. Emit BUY / SELL trades that close the drift, skipping trades below
   the minimum trade value threshold
6. Estimate tax drag per sell trade and sum total impact

---

**Performance notes**

- Market data: 12 tickers fetched in parallel in ~3–8 s (network dependent)
- RandomForest fit: ~2–5 s per ticker, all tickers run concurrently
- Optimizer: < 1 s for universes up to ~30 tickers
- Total wall-clock for a 12-ticker recommendation: typically 10–20 s
"""
        )

auth_headers = build_auth_headers(api_key)

# Initialise session state keys once
for _k in ("signals_job", "signals_result", "rec_job", "rec_result", "prefill_weights"):
    if _k not in st.session_state:
        st.session_state[_k] = None


def _poll(job_id: str):
    """Poll a job once; return job dict or None on error."""
    r, err = safe_get(f"{api_base}/analytics/jobs/{job_id}", headers=auth_headers, timeout=10)
    if err or not r.ok:
        return None
    return r.json()


# ---------------------------------------------------------------------------
# TABS
# Tab 1 — Import & Signals
# Tab 2 — Allocation & Rebalance
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Import & Signals", "Allocation & Rebalance"])

# ===========================================================================
# TAB 1 — Broker CSV Import  |  Tactical Signals
# ===========================================================================
with tab1:
    imp_col, sig_col = st.columns([1, 1], gap="large")

    # Left — Broker CSV Import
    with imp_col:
        st.subheader("Broker CSV Import")
        uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
        import_account_type = st.selectbox(
            "Account Type", options=["taxable", "401k", "ira"], index=0
        )
        import_source = st.text_input("Broker Source Label", value="csv-import")

        if st.button("Import CSV", use_container_width=True):
            if uploaded is None:
                st.warning("Please upload a CSV file first.")
            else:
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                params = {"account_type": import_account_type, "broker_source": import_source}
                r, err = safe_post(
                    f"{api_base}/portfolio/holdings/import-csv",
                    files=files,
                    params=params,
                    headers=auth_headers,
                    timeout=60,
                )
                if err:
                    st.error(f"Connection error: {err}")
                elif r.ok:
                    payload = r.json()
                    st.success(
                        f"Imported {payload['created_holdings']} holdings and "
                        f"{payload['created_lots']} lots. "
                        f"Skipped {payload['skipped_rows']} rows."
                    )
                    if payload.get("warnings"):
                        st.caption("Warnings: " + " | ".join(payload["warnings"]))
                else:
                    st.error(f"API error: {r.text}")

        st.divider()
        st.caption(
            "Supported formats: Fidelity, Schwab, and Vanguard CSV exports. "
            "See samples/holdings_example.csv for reference."
        )

    # Right — Tactical Signals
    with sig_col:
        st.subheader("Tactical Signals")
        st.caption(f"Universe: {', '.join(tickers)} (edit in sidebar)")

        if st.button("Refresh Signals", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/signals/start",
                params={"tickers": ",".join(tickers)},
                headers=auth_headers,
                timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.signals_job = r.json()["job_id"]
                st.session_state.signals_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        if st.session_state.signals_job:
            job = _poll(st.session_state.signals_job)
            if job is None:
                st.error("Lost contact with job — try again.")
                st.session_state.signals_job = None
            elif job["status"] in ("pending", "running"):
                with st.spinner("Fetching market data... auto-refreshes every 3 s"):
                    time.sleep(3)
                st.rerun()
            elif job["status"] == "done":
                st.session_state.signals_result = job["result"]
                st.session_state.signals_job = None
            elif job["status"] == "error":
                st.error(f"Signals failed: {job['error']}")
                st.session_state.signals_job = None

        if st.session_state.signals_result:
            signals_df = pd.DataFrame(st.session_state.signals_result)

            def _colour_signal(val):
                if str(val).upper() == "BUY":
                    return "background-color: #d4edda; color: #155724"
                if str(val).upper() == "SELL":
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            if "signal" in signals_df.columns:
                st.dataframe(
                    signals_df.style.applymap(_colour_signal, subset=["signal"]),
                    use_container_width=True,
                )
            else:
                st.dataframe(signals_df, use_container_width=True)


# ===========================================================================
# TAB 2 — Allocation Recommendation  |  Rebalance Planner
# ===========================================================================
with tab2:
    rec_col, reb_col = st.columns([1, 1], gap="large")

    # Left — Allocation Recommendation
    with rec_col:
        st.subheader("Allocation Recommendation")
        st.caption(f"Universe: {', '.join(tickers)} (edit in sidebar)")

        if st.button("Generate Recommendation", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/recommendation/start",
                params={"tickers": ",".join(tickers)},
                headers=auth_headers,
                timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.rec_job = r.json()["job_id"]
                st.session_state.rec_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        if st.session_state.rec_job:
            job = _poll(st.session_state.rec_job)
            if job is None:
                st.error("Lost contact with job — try again.")
                st.session_state.rec_job = None
            elif job["status"] in ("pending", "running"):
                with st.spinner("Running optimizer + ML forecast... auto-refreshes every 3 s"):
                    time.sleep(3)
                st.rerun()
            elif job["status"] == "done":
                st.session_state.rec_result = job["result"]
                st.session_state.rec_job = None
            elif job["status"] == "error":
                st.error(f"Recommendation failed: {job['error']}")
                st.session_state.rec_job = None

        if st.session_state.rec_result:
            payload = st.session_state.rec_result
            alloc_df = pd.DataFrame(payload["allocations"])
            ret = payload.get("expected_return")
            vol = payload.get("expected_volatility")

            m1, m2 = st.columns(2)
            m1.metric("Expected Return", f"{ret:.2%}" if ret is not None else "N/A")
            m2.metric("Expected Volatility", f"{vol:.2%}" if vol is not None else "N/A")

            fig = px.pie(
                alloc_df,
                names="ticker",
                values="final_weight",
                title="Recommended Allocation",
                hole=0.35,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(alloc_df, use_container_width=True)

            if payload.get("notes"):
                st.caption(" | ".join(payload["notes"]))

            # --- Copy to Rebalance ---
            if st.button("Copy Weights to Rebalance ->", use_container_width=True, type="primary"):
                st.session_state.prefill_weights = allocs_to_weight_string(payload["allocations"])
                st.success("Weights copied — switch to the Rebalance panel on the right.")

    # Right — Tax-Lot Rebalance Planner
    with reb_col:
        st.subheader("Tax-Lot Rebalance Plan")

        _default_weights = (
            st.session_state.prefill_weights
            if st.session_state.prefill_weights
            else "VTI:0.55,VXUS:0.25,BND:0.20"
        )

        targets_text = st.text_area(
            "Target Weights (TICKER:WEIGHT, comma-separated)",
            value=_default_weights,
            height=120,
            help="Fill manually or click 'Copy Weights to Rebalance' on the left.",
        )

        c1, c2, c3 = st.columns(3)
        short_rate = c1.number_input(
            "Short-Term Tax Rate", min_value=0.0, max_value=1.0, value=0.37, step=0.01
        )
        long_rate = c2.number_input(
            "Long-Term Tax Rate", min_value=0.0, max_value=1.0, value=0.20, step=0.01
        )
        min_trade_value = c3.number_input(
            "Min Trade ($)", min_value=0.0, value=50.0, step=10.0
        )

        if st.button("Generate Rebalance Plan", use_container_width=True):
            try:
                target_weights = parse_target_weights(targets_text)
            except ValueError as exc:
                st.error(str(exc))
            else:
                body = {
                    "target_weights": target_weights,
                    "short_term_tax_rate": short_rate,
                    "long_term_tax_rate": long_rate,
                    "min_trade_value": min_trade_value,
                }
                r, err = safe_post(
                    f"{api_base}/portfolio/rebalance/plan",
                    json=body,
                    headers=auth_headers,
                    timeout=60,
                )
                if err:
                    st.error(f"Connection error: {err}")
                elif r.ok:
                    payload = r.json()
                    rv1, rv2 = st.columns(2)
                    rv1.metric("Portfolio Value", f"${payload['portfolio_value']:,.2f}")
                    rv2.metric(
                        "Est. Tax Impact",
                        f"${payload['estimated_total_tax_impact']:,.2f}",
                    )
                    trades = pd.DataFrame(payload["trades"])
                    if not trades.empty:
                        st.dataframe(trades, use_container_width=True)
                    else:
                        st.info("No trades needed — portfolio is already within tolerance.")
                    if payload.get("notes"):
                        st.caption(" | ".join(payload["notes"]))
                else:
                    st.error(f"API error: {r.text}")

        st.divider()
        st.caption(
            "Import your holdings in the Import & Signals tab first, then generate a "
            "Recommendation and press Copy Weights to Rebalance to pre-fill targets above."
        )

