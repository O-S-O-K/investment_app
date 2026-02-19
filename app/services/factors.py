"""
Factor exposure analysis using ETF-proxy regression.

Regresses portfolio returns against five factor-mimicking ETF return series
(all fetched via yfinance — no external data service required):

  Market (Beta)   : SPY   — broad US equity
  Size (Small)    : IWM   — Russell 2000 small-cap
  Value (HML)     : VTV   — Vanguard Value ETF
  Momentum        : MTUM  — iShares MSCI USA Momentum
  Low Volatility  : USMV  — iShares Edge MSCI Min Vol USA

Uses OLS regression:  r_port = α + β_mkt·r_mkt + β_size·r_size + … + ε

Alpha is annualised; betas are unitless loadings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.market_data import compute_returns, fetch_adjusted_close

# ---------------------------------------------------------------------------
# Factor definitions — label -> proxy ETF ticker
# ---------------------------------------------------------------------------
FACTOR_PROXIES: dict[str, str] = {
    "Market (Beta)": "SPY",
    "Size (Small-Cap)": "IWM",
    "Value": "VTV",
    "Momentum": "MTUM",
    "Low Volatility": "USMV",
}


def compute_factor_exposure(
    portfolio_weights: dict[str, float],
    lookback_days: int = 252,
) -> dict:
    """
    Estimate portfolio factor loadings via OLS regression over *lookback_days*.

    Parameters
    ----------
    portfolio_weights : dict  ticker -> weight (should sum to ~1)
    lookback_days     : int   trading-day lookback (default 252 = 1 year)

    Returns
    -------
    dict with keys:
        alpha_annualised  — intercept (skill / unexplained return, annualised)
        factor_betas      — dict label -> beta loading
        r_squared         — goodness of fit (0-1; higher = more factor-explained)
        observations      — number of daily return obs used
        factor_proxies    — which ETF was used for each factor
    """
    portfolio_tickers = list(portfolio_weights.keys())
    factor_tickers = list(FACTOR_PROXIES.values())
    all_tickers = list(set(portfolio_tickers + factor_tickers))

    prices = fetch_adjusted_close(tickers=all_tickers, years=2, months=0)
    if prices.empty or len(prices) < 30:
        return {"error": "Insufficient price history for factor analysis."}

    if len(prices) > lookback_days:
        prices = prices.iloc[-lookback_days:]

    returns = compute_returns(prices)

    # --- Portfolio daily return series ---
    w = pd.Series(portfolio_weights).reindex(portfolio_tickers).fillna(0.0)
    total_w = w.sum()
    if total_w > 0:
        w = w / total_w
    port_rets = (returns[portfolio_tickers].fillna(0.0) @ w).rename("portfolio")

    # --- Factor return matrix ---
    available_factors: dict[str, str] = {}
    factor_cols = {}
    for label, ticker in FACTOR_PROXIES.items():
        if ticker in returns.columns:
            factor_cols[label] = returns[ticker]
            available_factors[label] = ticker

    if not factor_cols:
        return {"error": "None of the factor proxy ETFs were fetchable."}

    factor_df = pd.DataFrame(factor_cols)
    aligned = pd.concat([port_rets, factor_df], axis=1).dropna()

    if len(aligned) < 20:
        return {"error": "Not enough overlapping data for factor regression."}

    factor_labels = list(factor_cols.keys())
    y = aligned["portfolio"].values
    X_raw = aligned[factor_labels].values

    # OLS with intercept
    X = np.column_stack([np.ones(len(y)), X_raw])
    try:
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "Matrix decomposition failed — check data quality."}

    alpha_daily = float(betas[0])
    alpha_annualised = alpha_daily * 252
    factor_betas = {label: round(float(betas[i + 1]), 4) for i, label in enumerate(factor_labels)}

    y_hat = X @ betas
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "alpha_annualised": round(alpha_annualised, 4),
        "factor_betas": factor_betas,
        "r_squared": round(r_squared, 4),
        "observations": len(aligned),
        "lookback_days": lookback_days,
        "factor_proxies": available_factors,
        "note": (
            "Alpha is the annualised daily intercept — unexplained return after "
            "accounting for factor exposures. Betas near 1.0 indicate strong loading."
        ),
    }
