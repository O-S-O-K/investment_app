"""
Portfolio drift monitoring.

Compares current taxable portfolio weights (from DB holdings) against a target
weight dict and flags positions that have drifted beyond a configurable threshold.
"""
from __future__ import annotations

from app.db import SessionLocal
from app.models import Holding
from app.services.market_data import fetch_adjusted_close


def compute_drift(
    target_weights: dict[str, float],
    drift_threshold: float = 0.05,
) -> dict:
    """
    Compare current portfolio weights to *target_weights*.

    Uses a fresh DB session (safe to call from a background thread).
    Returns a drift table and breach summary.
    """
    db = SessionLocal()
    try:
        holdings = db.query(Holding).filter(Holding.account_type == "taxable").all()
    finally:
        db.close()

    if not holdings:
        return {
            "error": "No taxable holdings found in database.",
            "rows": [],
            "portfolio_value": 0.0,
            "n_breached": 0,
            "drift_threshold": drift_threshold,
        }

    ticker_shares: dict[str, float] = {}
    for h in holdings:
        ticker_shares[h.ticker] = ticker_shares.get(h.ticker, 0.0) + h.shares

    tickers = list(ticker_shares.keys())
    # 1-month lookback is enough to get a current price
    prices_df = fetch_adjusted_close(tickers=tickers, years=0, months=2)
    latest_prices = prices_df.iloc[-1].to_dict()

    values = {t: ticker_shares[t] * latest_prices.get(t, 0.0) for t in tickers}
    total_value = sum(values.values())
    if total_value <= 0:
        return {
            "error": "Could not fetch prices for current holdings.",
            "rows": [],
            "portfolio_value": 0.0,
            "n_breached": 0,
            "drift_threshold": drift_threshold,
        }

    current_weights = {t: v / total_value for t, v in values.items()}
    all_tickers = sorted(set(list(current_weights.keys()) + list(target_weights.keys())))

    rows = []
    for ticker in all_tickers:
        curr = current_weights.get(ticker, 0.0)
        tgt = target_weights.get(ticker, 0.0)
        drift = curr - tgt
        rows.append(
            {
                "ticker": ticker,
                "current_weight": round(curr, 4),
                "target_weight": round(tgt, 4),
                "drift": round(drift, 4),
                "drift_pct": round(drift * 100, 2),
                "current_value": round(values.get(ticker, 0.0), 2),
                "needs_rebalance": abs(drift) >= drift_threshold,
            }
        )

    n_breached = sum(1 for r in rows if r["needs_rebalance"])
    return {
        "portfolio_value": round(total_value, 2),
        "rows": rows,
        "n_breached": n_breached,
        "drift_threshold": drift_threshold,
    }
