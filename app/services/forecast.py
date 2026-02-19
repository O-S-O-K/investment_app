from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.services.market_data import compute_returns, fetch_adjusted_close

# Cache forecast results for 10 minutes (same TTL as price cache)
_FORECAST_CACHE: dict[tuple, tuple] = {}
_FORECAST_TTL = 600


def _fit_one(ticker: str, series: pd.Series) -> tuple[str, float]:
    """Train RF for a single ticker and return (ticker, annualised_return)."""
    if len(series) < 260:
        return ticker, float(series.mean() * 252)

    feature_df = pd.DataFrame(
        {
            "lag_1": series.shift(1),
            "lag_5": series.rolling(5).mean().shift(1),
            "lag_21": series.rolling(21).mean().shift(1),
            "vol_21": series.rolling(21).std().shift(1),
            "target": series,
        }
    ).dropna()

    if len(feature_df) < 100:
        return ticker, float(series.mean() * 252)

    x = feature_df[["lag_1", "lag_5", "lag_21", "vol_21"]]
    y = feature_df["target"]

    model = RandomForestRegressor(
        n_estimators=50,   # was 200 — 4× faster, accuracy loss is minimal
        max_depth=4,
        n_jobs=-1,         # use all CPU cores within each tree build
        random_state=42,
    )
    model.fit(x.iloc[:-1], y.iloc[:-1])
    next_return_daily = model.predict(x.iloc[[-1]])[0]
    return ticker, float(np.clip(next_return_daily * 252, -0.35, 0.35))


def forecast_expected_returns(tickers: list[str]) -> pd.Series:
    cache_key = tuple(sorted(tickers))
    cached = _FORECAST_CACHE.get(cache_key)
    if cached and (monotonic() - cached[0]) < _FORECAST_TTL:
        return cached[1].copy()

    prices = fetch_adjusted_close(tickers=tickers, years=2)
    returns = compute_returns(prices)

    # Fit each ticker in parallel threads
    predictions: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as pool:
        futures = {
            pool.submit(_fit_one, t, returns[t].dropna()): t for t in tickers
        }
        for future in as_completed(futures):
            ticker, value = future.result()
            predictions[ticker] = value

    result = pd.Series(predictions)
    _FORECAST_CACHE[cache_key] = (monotonic(), result.copy())
    return result
