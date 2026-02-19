from datetime import datetime, timedelta
from time import monotonic

import pandas as pd
import yfinance as yf

# Simple in-process TTL cache — avoids re-downloading on back-to-back calls
_CACHE: dict[tuple, tuple] = {}   # key -> (timestamp, DataFrame)
_CACHE_TTL_SECONDS = 600          # 10 minutes


def fetch_adjusted_close(tickers: list[str], years: int = 5) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required")

    cache_key = (tuple(sorted(tickers)), years)
    cached = _CACHE.get(cache_key)
    if cached and (monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1].copy()

    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years)

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(data.columns, pd.MultiIndex):
        # multi-ticker: columns are (field, ticker)
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=1).copy()
    else:
        # single ticker: flat columns
        close_col = "Close" if "Close" in data.columns else data.columns[0]
        prices = data[[close_col]].rename(columns={close_col: tickers[0]})

    # ensure all requested tickers are present
    for t in tickers:
        if t not in prices.columns:
            prices[t] = float("nan")

    prices = prices[tickers].dropna(how="all")
    _CACHE[cache_key] = (monotonic(), prices.copy())
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
