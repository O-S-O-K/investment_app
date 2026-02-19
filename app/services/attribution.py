"""
Simplified Brinson-Hood-Beebower performance attribution.

Decomposes portfolio return vs a benchmark into:
  - Allocation effect   (over/underweighting asset classes vs benchmark)
  - Selection effect    (picking better/worse securities within each class)
  - Interaction effect  (the product of both bets)
"""
from __future__ import annotations

from app.services.market_data import compute_returns, fetch_adjusted_close


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_ticker: str = "SPY",
    lookback_days: int = 63,
) -> dict:
    """
    Single-benchmark Brinson attribution over the last *lookback_days* trading days.

    The benchmark is treated as a single-asset portfolio with 100% weight.
    Attribution effects are computed per position and summed.

    Parameters
    ----------
    portfolio_weights : dict  ticker -> weight (should sum to ~1)
    benchmark_ticker  : str   Yahoo Finance-valid benchmark symbol
    lookback_days     : int   number of trading days to look back
    """
    tickers = list(portfolio_weights.keys())
    all_tickers = list(set(tickers + [benchmark_ticker]))

    prices = fetch_adjusted_close(tickers=all_tickers, years=0, months=6)
    if prices.empty or len(prices) < max(10, lookback_days // 4):
        return {"error": "Insufficient price history for attribution analysis."}

    # Use at most the requested lookback
    prices = prices.iloc[-lookback_days:]
    returns = compute_returns(prices)

    # Total period return per asset (geometric)
    period_returns = ((1 + returns).prod() - 1)
    bench_return = float(period_returns.get(benchmark_ticker, 0.0))

    rows = []
    total_portfolio_return = 0.0
    total_allocation = 0.0
    total_selection = 0.0
    total_interaction = 0.0

    for ticker in tickers:
        w_p = float(portfolio_weights.get(ticker, 0.0))
        # Benchmark weight for this ticker (1.0 if it IS the benchmark, else 0)
        w_b = 1.0 if ticker == benchmark_ticker else 0.0
        r_p = float(period_returns.get(ticker, 0.0))

        # BHB decomposition (relative to overall benchmark return)
        allocation_eff = (w_p - w_b) * (bench_return - bench_return)   # = 0 in single-benchmark; informative in multi-class
        selection_eff = w_b * (r_p - bench_return)
        interaction_eff = (w_p - w_b) * (r_p - bench_return)
        contribution = w_p * r_p

        total_portfolio_return += contribution
        total_allocation += allocation_eff
        total_selection += selection_eff
        total_interaction += interaction_eff

        rows.append(
            {
                "ticker": ticker,
                "portfolio_weight": round(w_p, 4),
                "benchmark_weight": round(w_b, 4),
                "asset_return_pct": round(r_p * 100, 2),
                "benchmark_return_pct": round(bench_return * 100, 2),
                "excess_return_pct": round((r_p - bench_return) * 100, 2),
                "selection_effect": round(selection_eff, 6),
                "interaction_effect": round(interaction_eff, 6),
                "contribution_pct": round(contribution * 100, 2),
            }
        )

    active_return = total_portfolio_return - bench_return
    return {
        "benchmark_ticker": benchmark_ticker,
        "lookback_days": lookback_days,
        "portfolio_return_pct": round(total_portfolio_return * 100, 2),
        "benchmark_return_pct": round(bench_return * 100, 2),
        "active_return_pct": round(active_return * 100, 2),
        "total_selection_effect": round(total_selection, 6),
        "total_interaction_effect": round(total_interaction, 6),
        "rows": rows,
    }
