from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_adjusted_close(tickers: list[str], years: int = 5) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required")

    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years)

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
