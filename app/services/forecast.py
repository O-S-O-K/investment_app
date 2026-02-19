import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.services.market_data import compute_returns, fetch_adjusted_close


def forecast_expected_returns(tickers: list[str]) -> pd.Series:
    prices = fetch_adjusted_close(tickers=tickers, years=5)
    returns = compute_returns(prices)

    predictions: dict[str, float] = {}
    for ticker in tickers:
        series = returns[ticker].dropna()
        if len(series) < 260:
            predictions[ticker] = float(series.mean() * 252)
            continue

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
            predictions[ticker] = float(series.mean() * 252)
            continue

        x = feature_df[["lag_1", "lag_5", "lag_21", "vol_21"]]
        y = feature_df["target"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=42,
        )
        model.fit(x.iloc[:-1], y.iloc[:-1])
        next_return_daily = model.predict(x.iloc[[-1]])[0]
        predictions[ticker] = float(np.clip(next_return_daily * 252, -0.35, 0.35))

    return pd.Series(predictions)
