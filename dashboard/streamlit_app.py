import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Investment App", layout="wide")
st.title("Personal Portfolio Intelligence")


def parse_target_weights(text: str) -> dict[str, float]:
    target_weights: dict[str, float] = {}
    for entry in text.split(","):
        chunk = entry.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid entry '{chunk}'. Use format TICKER:WEIGHT")
        ticker, value = chunk.split(":", 1)
        target_weights[ticker.strip().upper()] = float(value.strip())
    if not target_weights:
        raise ValueError("At least one ticker:weight pair is required")
    return target_weights

with st.sidebar:
    st.header("Universe")
    tickers_input = st.text_input("Tickers", value="SPY,QQQ,AGG,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

tab1, tab2 = st.tabs(["Signals & Allocation", "Import & Rebalance"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tactical Signals")
        if st.button("Refresh Signals"):
            r = requests.get(f"{API_BASE}/analytics/signals", params={"tickers": ",".join(tickers)}, timeout=30)
            if r.ok:
                signal_df = pd.DataFrame(r.json())
                st.dataframe(signal_df, use_container_width=True)
            else:
                st.error(f"API error: {r.text}")

    with col2:
        st.subheader("Allocation Recommendation")
        if st.button("Generate Recommendation"):
            r = requests.get(
                f"{API_BASE}/analytics/recommendation",
                params={"tickers": ",".join(tickers)},
                timeout=60,
            )
            if r.ok:
                payload = r.json()
                alloc_df = pd.DataFrame(payload["allocations"])
                st.metric("Expected Return", f"{payload['expected_return']:.2%}")
                st.metric("Expected Volatility", f"{payload['expected_volatility']:.2%}")
                fig = px.pie(alloc_df, names="ticker", values="final_weight", title="Final Allocation")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(alloc_df, use_container_width=True)
                st.caption(" | ".join(payload["notes"]))
            else:
                st.error(f"API error: {r.text}")

with tab2:
    st.subheader("Broker CSV Import")
    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    import_account_type = st.selectbox("Account Type", options=["taxable", "401k", "ira"], index=0)
    import_source = st.text_input("Broker Source Label", value="csv-import")

    if st.button("Import CSV"):
        if uploaded is None:
            st.warning("Please upload a CSV file first.")
        else:
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            params = {"account_type": import_account_type, "broker_source": import_source}
            r = requests.post(f"{API_BASE}/portfolio/holdings/import-csv", files=files, params=params, timeout=60)
            if r.ok:
                payload = r.json()
                st.success(
                    f"Imported {payload['created_holdings']} holdings and {payload['created_lots']} lots. "
                    f"Skipped {payload['skipped_rows']} rows."
                )
                if payload.get("warnings"):
                    st.caption("Warnings: " + " | ".join(payload["warnings"]))
            else:
                st.error(f"API error: {r.text}")

    st.subheader("Tax-Lot Rebalance Plan")
    targets_text = st.text_area(
        "Target Weights (TICKER:WEIGHT, comma-separated)",
        value="VTI:0.55,VXUS:0.25,BND:0.20",
        height=100,
    )
    short_rate = st.number_input("Short-Term Tax Rate", min_value=0.0, max_value=1.0, value=0.37, step=0.01)
    long_rate = st.number_input("Long-Term Tax Rate", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    min_trade_value = st.number_input("Minimum Trade Value ($)", min_value=0.0, value=50.0, step=10.0)

    if st.button("Generate Rebalance Plan"):
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
            r = requests.post(f"{API_BASE}/portfolio/rebalance/plan", json=body, timeout=60)
            if r.ok:
                payload = r.json()
                st.metric("Portfolio Value", f"${payload['portfolio_value']:,.2f}")
                st.metric("Estimated Tax Impact", f"${payload['estimated_total_tax_impact']:,.2f}")
                trades = pd.DataFrame(payload["trades"])
                if not trades.empty:
                    st.dataframe(trades, use_container_width=True)
                st.caption(" | ".join(payload["notes"]))
            else:
                st.error(f"API error: {r.text}")
