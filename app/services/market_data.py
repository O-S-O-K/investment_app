import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from time import monotonic

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Simple in-process TTL cache — avoids re-downloading on back-to-back calls
_CACHE: dict[tuple, tuple] = {}   # key -> (timestamp, DataFrame)
_CACHE_TTL_SECONDS = 600          # 10 minutes

_PER_TICKER_TIMEOUT = 25          # seconds before giving up on a single ticker


def _fetch_one(ticker: str, start: str, end: str) -> tuple[str, pd.Series | None]:
    """Download a single ticker using Ticker.history() (thread-safe, own session)."""
    try:
        t = yf.Ticker(ticker)
        data = t.history(start=start, end=end, auto_adjust=True)
        if data.empty or "Close" not in data.columns:
            logger.warning("No data for %s", ticker)
            return ticker, None
        return ticker, data["Close"].rename(ticker)
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        return ticker, None


def fetch_adjusted_close(tickers: list[str], years: int = 5, months: int = 0) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required")

    cache_key = (tuple(sorted(tickers)), years, months)
    cached = _CACHE.get(cache_key)
    if cached and (monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1].copy()

    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years + 30 * months)
    start_str, end_str = start.isoformat(), end.isoformat()

    # Download each ticker in parallel; skip any that time out or error
    series_map: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 12)) as pool:
        futures = {pool.submit(_fetch_one, t, start_str, end_str): t for t in tickers}
        try:
            for future in as_completed(futures, timeout=_PER_TICKER_TIMEOUT * 2):
                t, series = future.result()
                if series is not None:
                    series_map[t] = series
                else:
                    logger.warning("No data returned for ticker %s — will be excluded", t)
        except TimeoutError:
            # Collect whatever completed before the deadline
            for future, t in futures.items():
                if future.done() and not future.exception():
                    _, series = future.result()
                    if series is not None:
                        series_map[t] = series
            logger.warning("Timed out waiting for some tickers; proceeding with %d/%d", len(series_map), len(tickers))

    if not series_map:
        raise ValueError("No market data could be fetched for any of the requested tickers.")

    prices = pd.DataFrame(series_map)

    # Ensure all requested tickers present (fill missing with NaN)
    for t in tickers:
        if t not in prices.columns:
            prices[t] = float("nan")

    prices = prices[tickers].dropna(how="all")
    _CACHE[cache_key] = (monotonic(), prices.copy())
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
