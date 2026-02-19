"""Portfolio Intelligence Dashboard — 6-tab professional PM UI."""
import os
import time
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Investment App", layout="wide")
st.title("Personal Portfolio Intelligence")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_target_weights(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in text.split(","):
        chunk = entry.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid entry {chunk!r}. Use TICKER:WEIGHT format.")
        ticker, value = chunk.split(":", 1)
        result[ticker.strip().upper()] = float(value.strip())
    if not result:
        raise ValueError("At least one ticker:weight pair is required.")
    return result


def build_auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key.strip()} if api_key.strip() else {}


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


def safe_put(url: str, **kwargs):
    try:
        return requests.put(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def safe_delete(url: str, **kwargs):
    try:
        return requests.delete(url, **kwargs), None
    except requests.RequestException as exc:
        return None, str(exc)


def allocs_to_weight_string(allocations: list) -> str:
    return ", ".join(f"{a['ticker']}:{a['final_weight']:.4f}" for a in allocations)


def _poll(job_id: str, api_base: str, auth_headers: dict):
    r, err = safe_get(
        f"{api_base}/analytics/jobs/{job_id}", headers=auth_headers, timeout=10
    )
    if err or not r.ok:
        return None
    return r.json()


def _run_job_widget(job_key: str, result_key: str, spinner_msg: str, api_base: str, auth_headers: dict):
    """Standard poll-and-display block. Returns True if result is ready."""
    if st.session_state[job_key]:
        job = _poll(st.session_state[job_key], api_base, auth_headers)
        if job is None:
            st.error("Lost contact with job — try again.")
            st.session_state[job_key] = None
        elif job["status"] in ("pending", "running"):
            with st.spinner(f"{spinner_msg}  (auto-refreshes every 3 s)"):
                time.sleep(3)
            st.rerun()
        elif job["status"] == "done":
            st.session_state[result_key] = job["result"]
            st.session_state[job_key] = None
        elif job["status"] == "error":
            st.error(f"Job failed: {job['error']}")
            st.session_state[job_key] = None
    return st.session_state[result_key] is not None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("API")
    api_base = st.text_input(
        "API Base URL", value=os.getenv("API_BASE", DEFAULT_API_BASE)
    )
    api_key = st.text_input(
        "X-API-Key", value=os.getenv("API_KEY", "change-me"), type="password"
    )

    st.header("Universe")
    tickers_input = st.text_input("Tickers", value="SPY,QQQ,AGG,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.divider()

    # ------------------------------------------------------------------
    # Model Parameters  (IPS / optimizer overrides)
    # ------------------------------------------------------------------
    with st.expander("Model Parameters", expanded=False):
        st.caption("Override optimizer defaults — leave blank to use server settings.")

        mp_max_weight = st.slider(
            "Max Weight per Ticker (%)", min_value=5, max_value=100, value=60, step=5
        ) / 100.0

        mp_min_weight = st.slider(
            "Min Weight per Ticker (%)", min_value=0, max_value=20, value=0, step=1
        ) / 100.0

        mp_rfr = st.slider(
            "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.25
        ) / 100.0

        mp_vol_target = st.slider(
            "Target Volatility (%)", min_value=5, max_value=30, value=12, step=1
        ) / 100.0

        mp_mu_blend = st.slider(
            "Hist Return Blend (0=ML, 1=Historical)", min_value=0.0, max_value=1.0,
            value=0.6, step=0.05,
        )

        mp_taa_tilt = st.slider(
            "TAA Tilt Strength (%)", min_value=0, max_value=30, value=10, step=1
        ) / 100.0

        mp_prohibited_raw = st.text_input(
            "Prohibited Tickers (comma-separated)", value="",
            help="IPS exclusion list — these tickers will never appear in the output.",
        )

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

**Feature tabs**

| Tab | Purpose |
|---|---|
| 📥 Import & Signals | Upload broker CSV; view per-ticker BUY/HOLD/SELL signals |
| 📊 Allocation & Rebalance | Optimised weights + risk metrics; tax-lot trade plan |
| 📉 Risk & Drift | Drift monitor against target; drawdown / HWM tracking |
| 🌿 Tax Intelligence | Tax-loss harvest scanner with wash-sale flags |
| 🔬 Analytics | Brinson attribution; scenario stress testing; factor exposure |
| 📓 Journal | Log and review investment thesis entries |

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
    ├── market_data.py   ← yf.Ticker.history(), parallel, 10-min cache
    ├── signals.py       ← MA cross + 12-1 momentum
    ├── forecast.py      ← RandomForest 50 trees (parallel, 10-min cache)
    ├── recommendation.py← SAA (max-Sharpe) + TAA + ML blend
    ├── risk.py          ← MRC, CVaR (historical ES)
    ├── drift.py         ← vs. declared target weights
    ├── harvest.py       ← unrealised loss scan + wash-sale 30-day check
    ├── drawdown.py      ← PortfolioSnapshot HWM series
    ├── attribution.py   ← Brinson-Hood-Beebower decomposition
    ├── scenario.py      ← 7 historical stress scenarios
    └── factors.py       ← OLS vs. SPY/IWM/VTV/MTUM/USMV proxies
    │
    ▼
SQLite  (investment_app.db)
    ├── holdings            ← ticker, shares, account type
    ├── tax_lots            ← per-lot cost basis, long/short flag
    ├── four01k_options
    ├── four01k_allocations
    ├── portfolio_snapshots ← value history for drawdown tracking
    └── journal_entries     ← investment decision log
```
"""
        )

auth_headers = build_auth_headers(api_key)

# Build model params dict for recommendation
_model_params = {
    "max_weight": mp_max_weight,
    "min_weight": mp_min_weight,
    "risk_free_rate": mp_rfr,
    "target_volatility": mp_vol_target,
    "mu_blend_factor": mp_mu_blend,
    "taa_tilt_strength": mp_taa_tilt,
}
if mp_prohibited_raw.strip():
    _model_params["prohibited"] = mp_prohibited_raw.strip()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
_state_keys = [
    "signals_job", "signals_result",
    "rec_job", "rec_result",
    "prefill_weights",
    "drift_job", "drift_result",
    "harvest_job", "harvest_result",
    "drawdown_job", "drawdown_result",
    "attribution_job", "attribution_result",
    "scenarios_job", "scenarios_result",
    "factors_job", "factors_result",
]
for _k in _state_keys:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Import & Signals",
    "📊 Allocation & Rebalance",
    "📉 Risk & Drift",
    "🌿 Tax Intelligence",
    "🔬 Analytics",
    "📓 Journal",
])


# ===========================================================================
# TAB 1 — Broker CSV Import  |  Tactical Signals
# ===========================================================================
with tab1:
    imp_col, sig_col = st.columns([1, 1], gap="large")

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
                params = {
                    "account_type": import_account_type,
                    "broker_source": import_source,
                }
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
            "See samples/holdings_example.csv for the reference format."
        )

    with sig_col:
        st.subheader("Tactical Signals")
        st.caption(f"Universe: {', '.join(tickers)}")

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

        _run_job_widget("signals_job", "signals_result",
                        "Fetching market data…", api_base, auth_headers)

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

    with rec_col:
        st.subheader("Allocation Recommendation")
        st.caption(f"Universe: {', '.join(tickers)}  ·  parameters from Model Parameters sidebar")

        if st.button("Generate Recommendation", use_container_width=True):
            params = {"tickers": ",".join(tickers), **_model_params}
            r, err = safe_post(
                f"{api_base}/analytics/recommendation/start",
                params=params,
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

        _run_job_widget("rec_job", "rec_result",
                        "Running optimizer + ML forecast…", api_base, auth_headers)

        if st.session_state.rec_result:
            payload = st.session_state.rec_result
            alloc_df = pd.DataFrame(payload["allocations"])

            m1, m2, m3, m4 = st.columns(4)
            ret = payload.get("expected_return")
            vol = payload.get("expected_volatility")
            cvar = payload.get("expected_cvar_95")
            sharpe = payload.get("sharpe_ratio")
            m1.metric("Exp Return", f"{ret:.2%}" if ret is not None else "N/A")
            m2.metric("Exp Volatility", f"{vol:.2%}" if vol is not None else "N/A")
            m3.metric("CVaR 95%", f"{cvar:.2%}" if cvar is not None else "N/A")
            m4.metric("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "N/A")

            fig = px.pie(
                alloc_df,
                names="ticker",
                values="final_weight",
                title="Recommended Allocation",
                hole=0.35,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

            display_cols = [c for c in ["ticker", "final_weight", "risk_contribution",
                                        "expected_return", "tactical_score"]
                            if c in alloc_df.columns]
            st.dataframe(alloc_df[display_cols], use_container_width=True)

            if "risk_contribution" in alloc_df.columns:
                fig_mrc = px.bar(
                    alloc_df.sort_values("risk_contribution", ascending=False),
                    x="ticker", y="risk_contribution",
                    title="Marginal Risk Contribution",
                    labels={"risk_contribution": "Fraction of Portfolio Variance"},
                    color="ticker",
                )
                st.plotly_chart(fig_mrc, use_container_width=True)

            if payload.get("notes"):
                st.caption(" | ".join(payload["notes"]))

            if st.button("Copy Weights to Rebalance →", use_container_width=True, type="primary"):
                st.session_state.prefill_weights = allocs_to_weight_string(payload["allocations"])
                st.success("Weights copied — switch to the Rebalance panel.")

    with reb_col:
        st.subheader("Tax-Lot Rebalance Plan")
        _default_weights = (
            st.session_state.prefill_weights
            if st.session_state.prefill_weights
            else "VTI:0.55,VXUS:0.25,BND:0.20"
        )
        targets_text = st.text_area(
            "Target Weights (TICKER:WEIGHT, comma-separated)",
            value=_default_weights, height=120,
            help="Auto-filled from Recommendation or enter manually.",
        )
        c1, c2, c3 = st.columns(3)
        short_rate = c1.number_input("Short-Term Tax Rate", 0.0, 1.0, 0.37, 0.01)
        long_rate = c2.number_input("Long-Term Tax Rate", 0.0, 1.0, 0.20, 0.01)
        min_trade_value = c3.number_input("Min Trade ($)", 0.0, value=50.0, step=10.0)

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
                    json=body, headers=auth_headers, timeout=60,
                )
                if err:
                    st.error(f"Connection error: {err}")
                elif r.ok:
                    p = r.json()
                    rv1, rv2 = st.columns(2)
                    rv1.metric("Portfolio Value", f"${p['portfolio_value']:,.2f}")
                    rv2.metric("Est. Tax Impact", f"${p['estimated_total_tax_impact']:,.2f}")
                    trades = pd.DataFrame(p["trades"])
                    if not trades.empty:
                        st.dataframe(trades, use_container_width=True)
                    else:
                        st.info("No trades needed — portfolio is already within tolerance.")
                    if p.get("notes"):
                        st.caption(" | ".join(p["notes"]))
                else:
                    st.error(f"API error: {r.text}")

        st.divider()
        st.caption(
            "Import holdings first, then generate a Recommendation and press "
            "'Copy Weights to Rebalance' to pre-fill targets above."
        )


# ===========================================================================
# TAB 3 — Risk & Drift
# ===========================================================================
with tab3:
    drift_col, dd_col = st.columns([1, 1], gap="large")

    with drift_col:
        st.subheader("Portfolio Drift Monitor")
        st.caption("Measures how much current holdings have drifted from your target.")

        _default_drift_weights = (
            st.session_state.prefill_weights
            if st.session_state.prefill_weights
            else "VTI:0.55,VXUS:0.25,BND:0.20"
        )
        drift_targets_text = st.text_area(
            "Target Weights (TICKER:WEIGHT)",
            value=_default_drift_weights, height=100, key="drift_targets",
        )
        drift_threshold = st.slider(
            "Alert Threshold (%)", min_value=1, max_value=20, value=5, step=1
        ) / 100.0

        if st.button("Check Drift", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/drift/start",
                params={
                    "target_weights": drift_targets_text.replace(" ", ""),
                    "drift_threshold": drift_threshold,
                },
                headers=auth_headers, timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.drift_job = r.json()["job_id"]
                st.session_state.drift_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        _run_job_widget("drift_job", "drift_result", "Computing drift…", api_base, auth_headers)

        if st.session_state.drift_result:
            dr = st.session_state.drift_result
            dm1, dm2 = st.columns(2)
            dm1.metric("Portfolio Value", f"${dr.get('portfolio_value', 0):,.2f}")
            dm2.metric("Positions Breached", dr.get("n_breached", 0))

            rows = dr.get("rows", [])
            if rows:
                drift_df = pd.DataFrame(rows)

                def _colour_drift(row):
                    if row.get("needs_rebalance"):
                        return ["background-color: #fff3cd"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    drift_df.style.apply(_colour_drift, axis=1),
                    use_container_width=True,
                )
                fig_drift = px.bar(
                    drift_df, x="ticker", y="drift_pct",
                    color="needs_rebalance",
                    title="Drift from Target (%)",
                    labels={"drift_pct": "Drift %", "needs_rebalance": "Needs Rebalance"},
                    color_discrete_map={True: "#dc3545", False: "#28a745"},
                )
                fig_drift.add_hline(y=drift_threshold, line_dash="dash", line_color="orange",
                                    annotation_text="Threshold")
                fig_drift.add_hline(y=-drift_threshold, line_dash="dash", line_color="orange")
                st.plotly_chart(fig_drift, use_container_width=True)
            else:
                st.info("No holdings found. Import holdings first.")

    with dd_col:
        st.subheader("Drawdown & High-Water Mark")

        with st.expander("Record Snapshot", expanded=False):
            snap_val = st.number_input(
                "Current Portfolio Value ($)", min_value=0.01, value=100000.0, step=1000.0
            )
            if st.button("Save Snapshot", use_container_width=True):
                r, err = safe_post(
                    f"{api_base}/analytics/snapshot",
                    params={"portfolio_value": snap_val},
                    headers=auth_headers, timeout=10,
                )
                if err:
                    st.error(f"Connection error: {err}")
                elif r.ok:
                    st.success(f"Snapshot saved: ${snap_val:,.2f}")
                    st.session_state.drawdown_result = None
                else:
                    st.error(f"API error: {r.text}")

        if st.button("Refresh Drawdown Analysis", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/drawdown/start",
                headers=auth_headers, timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.drawdown_job = r.json()["job_id"]
                st.session_state.drawdown_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        _run_job_widget("drawdown_job", "drawdown_result",
                        "Computing drawdown…", api_base, auth_headers)

        if st.session_state.drawdown_result:
            dd = st.session_state.drawdown_result
            ddm1, ddm2, ddm3 = st.columns(3)
            ddm1.metric("Current Value", f"${dd.get('current_value', 0):,.2f}")
            ddm2.metric(
                "Max Drawdown",
                f"{dd.get('max_drawdown_pct', 0):.2%}",
                delta=f"Current: {dd.get('current_drawdown_pct', 0):.2%}",
                delta_color="inverse",
            )
            ddm3.metric("HWM", f"${dd.get('high_water_mark', 0):,.2f}")

            series = dd.get("series", [])
            if series:
                import plotly.graph_objects as go  # noqa: PLC0415
                series_df = pd.DataFrame(series)
                series_df["recorded_at"] = pd.to_datetime(series_df["recorded_at"])

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=series_df["recorded_at"], y=series_df["portfolio_value"],
                    name="Portfolio Value", line=dict(color="#0d6efd"),
                ))
                fig_dd.add_trace(go.Scatter(
                    x=series_df["recorded_at"], y=series_df["high_water_mark"],
                    name="High-Water Mark", line=dict(color="#fd7e14", dash="dash"),
                ))
                fig_dd.update_layout(title="Portfolio Value vs. HWM",
                                     xaxis_title="Date", yaxis_title="Value ($)")
                st.plotly_chart(fig_dd, use_container_width=True)

                fig_pct = px.area(
                    series_df, x="recorded_at", y="drawdown_pct",
                    title="Drawdown %",
                    labels={"drawdown_pct": "Drawdown", "recorded_at": "Date"},
                    color_discrete_sequence=["#dc3545"],
                )
                st.plotly_chart(fig_pct, use_container_width=True)
            else:
                st.info(f"No snapshots yet. {dd.get('n_snapshots', 0)} records found.")


# ===========================================================================
# TAB 4 — Tax Intelligence
# ===========================================================================
with tab4:
    st.subheader("Tax-Loss Harvest Scanner")
    st.caption("Scans all taxable lots for unrealised losses above your threshold.")

    min_loss = st.slider("Min Unrealised Loss ($)", 0, 10000, 100, 50)
    if st.button("Scan for Opportunities", use_container_width=True):
        r, err = safe_post(
            f"{api_base}/analytics/harvest/start",
            params={"min_loss": min_loss},
            headers=auth_headers, timeout=10,
        )
        if err:
            st.error(f"Could not reach API: {err}")
        elif r.ok:
            st.session_state.harvest_job = r.json()["job_id"]
            st.session_state.harvest_result = None
            st.rerun()
        else:
            st.error(f"API error: {r.text}")

    _run_job_widget("harvest_job", "harvest_result",
                    "Scanning tax lots…", api_base, auth_headers)

    if st.session_state.harvest_result:
        hr = st.session_state.harvest_result
        hm1, hm2 = st.columns(2)
        hm1.metric("Total Harvestable Loss", f"${hr.get('total_harvestable_loss', 0):,.2f}")
        hm2.metric("Qualifying Lots", hr.get("n_lots", 0))

        lots = hr.get("lots", [])
        if lots:
            lots_df = pd.DataFrame(lots)

            def _colour_wash(row):
                if row.get("wash_sale_risk"):
                    return ["background-color: #f8d7da"] * len(row)
                return ["background-color: #d4edda"] * len(row)

            display_cols = [c for c in [
                "ticker", "shares", "term", "holding_days",
                "cost_basis_total", "current_value", "unrealised_loss", "wash_sale_risk",
            ] if c in lots_df.columns]

            st.dataframe(
                lots_df[display_cols].style.apply(_colour_wash, axis=1),
                use_container_width=True,
            )
            st.caption(
                "🟢 Green = safe to harvest  |  🔴 Red = wash-sale risk "
                "(purchased or sold a substantially identical security within 30 days)"
            )

            if "unrealised_loss" in lots_df.columns:
                fig_harvest = px.bar(
                    lots_df.sort_values("unrealised_loss"),
                    x="ticker", y="unrealised_loss",
                    color="wash_sale_risk",
                    title="Unrealised Losses by Lot",
                    labels={"unrealised_loss": "Unrealised Loss ($)", "wash_sale_risk": "Wash-Sale Risk"},
                    color_discrete_map={True: "#dc3545", False: "#28a745"},
                )
                st.plotly_chart(fig_harvest, use_container_width=True)
        else:
            st.info("No lots meet the minimum loss threshold.")


# ===========================================================================
# TAB 5 — Analytics: Attribution | Scenarios | Factors
# ===========================================================================
with tab5:
    st.subheader("Portfolio Analytics")

    _default_analytics_weights = (
        st.session_state.prefill_weights
        if st.session_state.prefill_weights
        else "VTI:0.55,VXUS:0.25,BND:0.20"
    )
    analytics_weights_text = st.text_input(
        "Portfolio Weights (TICKER:WEIGHT, comma-separated)",
        value=_default_analytics_weights,
        help="Auto-filled from Recommendation or enter manually.",
    )

    attr_tab, scen_tab, fact_tab = st.tabs([
        "📐 Performance Attribution",
        "🌊 Scenario Stress Tests",
        "🔎 Factor Exposure",
    ])

    with attr_tab:
        a1, a2 = st.columns([1, 1], gap="large")
        with a1:
            benchmark_ticker = st.text_input("Benchmark Ticker", value="SPY")
            lookback_days = st.select_slider(
                "Lookback (trading days)", options=[21, 42, 63, 126, 252], value=63,
            )
            if st.button("Run Attribution", use_container_width=True):
                r, err = safe_post(
                    f"{api_base}/analytics/attribution/start",
                    params={
                        "portfolio_weights": analytics_weights_text.replace(" ", ""),
                        "benchmark": benchmark_ticker.upper(),
                        "lookback_days": lookback_days,
                    },
                    headers=auth_headers, timeout=10,
                )
                if err:
                    st.error(f"Could not reach API: {err}")
                elif r.ok:
                    st.session_state.attribution_job = r.json()["job_id"]
                    st.session_state.attribution_result = None
                    st.rerun()
                else:
                    st.error(f"API error: {r.text}")

        _run_job_widget("attribution_job", "attribution_result",
                        "Computing Brinson attribution…", api_base, auth_headers)

        if st.session_state.attribution_result:
            ar = st.session_state.attribution_result
            with a2:
                am1, am2, am3 = st.columns(3)
                am1.metric("Portfolio Return", f"{ar.get('portfolio_return_pct', 0):.2%}")
                am2.metric(f"Benchmark ({ar.get('benchmark_ticker', 'SPY')})",
                           f"{ar.get('benchmark_return_pct', 0):.2%}")
                am3.metric("Active Return", f"{ar.get('active_return_pct', 0):.2%}")
                ef1, ef2 = st.columns(2)
                ef1.metric("Selection Effect", f"{ar.get('total_selection_effect', 0):.2%}")
                ef2.metric("Interaction Effect", f"{ar.get('total_interaction_effect', 0):.2%}")

            attr_rows = ar.get("rows", [])
            if attr_rows:
                attr_df = pd.DataFrame(attr_rows)
                st.dataframe(attr_df, use_container_width=True)
                fig_attr = px.bar(
                    attr_df, x="ticker",
                    y=["selection_effect", "interaction_effect"],
                    title="Brinson Attribution by Ticker",
                    barmode="group",
                    labels={"value": "Effect", "variable": "Component"},
                )
                st.plotly_chart(fig_attr, use_container_width=True)

    with scen_tab:
        if st.button("Run Stress Tests", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/scenarios/start",
                params={"portfolio_weights": analytics_weights_text.replace(" ", "")},
                headers=auth_headers, timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.scenarios_job = r.json()["job_id"]
                st.session_state.scenarios_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        _run_job_widget("scenarios_job", "scenarios_result",
                        "Running scenarios…", api_base, auth_headers)

        if st.session_state.scenarios_result:
            scen_list = st.session_state.scenarios_result
            if scen_list:
                scen_df = pd.DataFrame([
                    {"scenario": s["scenario"],
                     "projected_return_pct": s["projected_return_pct"]}
                    for s in scen_list
                ])
                fig_scen = px.bar(
                    scen_df.sort_values("projected_return_pct"),
                    x="projected_return_pct", y="scenario", orientation="h",
                    title="Scenario Stress Test Results",
                    labels={"projected_return_pct": "Projected Return", "scenario": "Scenario"},
                    color="projected_return_pct",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                )
                fig_scen.update_layout(showlegend=False)
                st.plotly_chart(fig_scen, use_container_width=True)

                for s in scen_list:
                    with st.expander(
                        f"{s['scenario']}  →  {s['projected_return_pct']:.1%}", expanded=False
                    ):
                        st.dataframe(pd.DataFrame(s.get("detail", [])), use_container_width=True)

    with fact_tab:
        fact_lookback = st.select_slider(
            "Lookback (trading days)", options=[60, 126, 252, 504], value=252, key="fact_lb"
        )
        if st.button("Compute Factor Exposure", use_container_width=True):
            r, err = safe_post(
                f"{api_base}/analytics/factors/start",
                params={
                    "portfolio_weights": analytics_weights_text.replace(" ", ""),
                    "lookback_days": fact_lookback,
                },
                headers=auth_headers, timeout=10,
            )
            if err:
                st.error(f"Could not reach API: {err}")
            elif r.ok:
                st.session_state.factors_job = r.json()["job_id"]
                st.session_state.factors_result = None
                st.rerun()
            else:
                st.error(f"API error: {r.text}")

        _run_job_widget("factors_job", "factors_result",
                        "Running OLS regression…", api_base, auth_headers)

        if st.session_state.factors_result:
            fr = st.session_state.factors_result
            fm1, fm2, fm3 = st.columns(3)
            fm1.metric("Alpha (annualised)", f"{fr.get('alpha_annualised', 0):.2%}")
            fm2.metric("R²", f"{fr.get('r_squared', 0):.3f}")
            fm3.metric("Observations", fr.get("observations", 0))

            betas = fr.get("factor_betas", {})
            if betas:
                betas_df = pd.DataFrame(
                    [{"factor": k, "beta": v} for k, v in betas.items()]
                ).sort_values("beta", ascending=False)
                fig_betas = px.bar(
                    betas_df, x="factor", y="beta",
                    title="Factor Betas",
                    labels={"beta": "Beta", "factor": "Factor"},
                    color="beta",
                    color_continuous_scale="RdBu", color_continuous_midpoint=0,
                )
                st.plotly_chart(fig_betas, use_container_width=True)
                st.dataframe(betas_df, use_container_width=True)

            if fr.get("note"):
                st.caption(fr["note"])


# ===========================================================================
# TAB 6 — Decision Journal
# ===========================================================================
with tab6:
    st.subheader("Investment Decision Journal")
    jnl_create_col, jnl_list_col = st.columns([1, 2], gap="large")

    with jnl_create_col:
        st.subheader("New Entry")
        with st.form("journal_form", clear_on_submit=True):
            j_ticker = st.text_input("Ticker", value="").upper()
            j_action = st.selectbox("Action", ["BUY", "SELL", "HOLD", "AVOID"])
            j_rationale = st.text_area("Rationale / Thesis", height=120)
            j_exp_ret = st.number_input(
                "Expected Return (%)", min_value=-50.0, max_value=200.0, value=0.0, step=0.5
            )
            j_exp_days = st.number_input(
                "Expected Holding (days)", min_value=1, max_value=3650, value=252
            )
            submitted = st.form_submit_button(
                "Log Entry", use_container_width=True, type="primary"
            )
            if submitted:
                if not j_ticker.strip():
                    st.error("Ticker is required.")
                elif not j_rationale.strip():
                    st.error("Rationale is required.")
                else:
                    body = {
                        "ticker": j_ticker.strip(),
                        "action": j_action,
                        "rationale": j_rationale.strip(),
                        "expected_return": j_exp_ret / 100.0,
                        "expected_holding_days": int(j_exp_days),
                    }
                    r, err = safe_post(
                        f"{api_base}/journal",
                        json=body, headers=auth_headers, timeout=10,
                    )
                    if err:
                        st.error(f"Connection error: {err}")
                    elif r.ok:
                        st.success(f"Entry logged for {j_ticker}.")
                    else:
                        st.error(f"API error: {r.text}")

    with jnl_list_col:
        st.subheader("Journal History")
        jnl_ticker_filter = st.text_input("Filter by Ticker", value="").upper().strip()

        if st.button("Refresh Journal", use_container_width=True):
            params = {}
            if jnl_ticker_filter:
                params["ticker"] = jnl_ticker_filter
            r, err = safe_get(
                f"{api_base}/journal",
                params=params, headers=auth_headers, timeout=10,
            )
            if err:
                st.error(f"Connection error: {err}")
            elif r.ok:
                entries = r.json()
                if not entries:
                    st.info("No journal entries found.")
                else:
                    entries_df = pd.DataFrame(entries)
                    has_outcome = "outcome" in entries_df.columns
                    open_entries = (
                        entries_df[entries_df["outcome"].isna()]
                        if has_outcome else entries_df
                    )
                    closed_entries = (
                        entries_df[entries_df["outcome"].notna()]
                        if has_outcome else pd.DataFrame()
                    )

                    if not open_entries.empty:
                        st.caption(f"**Open positions** ({len(open_entries)})")
                        display_cols = [c for c in [
                            "id", "ticker", "action", "rationale",
                            "expected_return", "expected_holding_days", "created_at",
                        ] if c in open_entries.columns]
                        st.dataframe(open_entries[display_cols], use_container_width=True)

                        with st.expander("Record Outcome", expanded=False):
                            entry_id_to_update = st.number_input(
                                "Entry ID to close", min_value=1, step=1, value=1
                            )
                            outcome_text = st.text_area("Outcome / Notes", height=80)
                            closed_at = st.date_input("Closed Date", value=date.today())
                            if st.button("Save Outcome", use_container_width=True):
                                body = {
                                    "outcome": outcome_text,
                                    "closed_at": str(closed_at),
                                }
                                r2, err2 = safe_put(
                                    f"{api_base}/journal/{entry_id_to_update}",
                                    json=body, headers=auth_headers, timeout=10,
                                )
                                if err2:
                                    st.error(f"Connection error: {err2}")
                                elif r2.ok:
                                    st.success("Outcome saved.")
                                else:
                                    st.error(f"API error: {r2.text}")

                    if not closed_entries.empty:
                        with st.expander(
                            f"Closed entries ({len(closed_entries)})", expanded=False
                        ):
                            cl_cols = [c for c in [
                                "id", "ticker", "action", "expected_return",
                                "outcome", "closed_at",
                            ] if c in closed_entries.columns]
                            st.dataframe(
                                closed_entries[cl_cols], use_container_width=True
                            )
            else:
                st.error(f"API error: {r.text}")

