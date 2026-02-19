import numpy as np
import pandas as pd

from app.services.market_data import fetch_adjusted_close


def _normalize(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_tactical_signals(tickers: list[str]) -> pd.DataFrame:
    # 1.5 years = ~375 trading days, covers MA200 + 12-month momentum with buffer
    prices = fetch_adjusted_close(tickers=tickers, years=2, months=6)

    momentum = prices.iloc[-1] / prices.shift(252).iloc[-1] - 1
    ma_50 = prices.rolling(50).mean().iloc[-1]
    ma_200 = prices.rolling(200).mean().iloc[-1]
    trend_strength = (ma_50 / ma_200) - 1

    score = 0.6 * _normalize(momentum) + 0.4 * _normalize(trend_strength)
    output = pd.DataFrame(
        {
            "ticker": prices.columns,
            "momentum": momentum.values,
            "trend_strength": trend_strength.values,
            "tactical_score": score.values,
        }
    )
    return output.fillna(0.0)
