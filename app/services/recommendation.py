import numpy as np

from app.config import settings
from app.schemas import RecommendationItem
from app.services.forecast import forecast_expected_returns
from app.services.market_data import compute_returns, fetch_adjusted_close
from app.services.optimizer import apply_tactical_overlay, optimize_strategic_allocation
from app.services.signals import compute_tactical_signals


def build_recommendation(tickers: list[str]) -> dict:
    prices = fetch_adjusted_close(tickers=tickers, years=2, months=6)
    returns = compute_returns(prices)

    historical_mu = returns.mean() * settings.annualization
    model_mu = forecast_expected_returns(tickers)
    blended_mu = 0.6 * historical_mu + 0.4 * model_mu

    covariance = returns.cov() * settings.annualization
    strategic = optimize_strategic_allocation(blended_mu, covariance)

    tactical_df = compute_tactical_signals(tickers)
    tactical_scores = tactical_df.set_index("ticker")["tactical_score"]
    final_weights = apply_tactical_overlay(strategic, tactical_scores)

    portfolio_return = float((final_weights * blended_mu).sum())
    cov_matrix = covariance.loc[final_weights.index, final_weights.index].values
    weight_vector = final_weights.values
    portfolio_vol = float(np.sqrt(weight_vector @ cov_matrix @ weight_vector))

    allocations = []
    for ticker in final_weights.index:
        allocations.append(
            RecommendationItem(
                ticker=ticker,
                strategic_weight=float(strategic[ticker]),
                tactical_tilt=float(final_weights[ticker] - strategic[ticker]),
                final_weight=float(final_weights[ticker]),
            )
        )

    notes = [
        "Strategic weights blend historical and ML-forecast expected returns.",
        "Tactical overlay uses momentum and trend filters.",
        "Review recommendations against taxes, liquidity, and IPS constraints.",
    ]

    return {
        "expected_return": round(portfolio_return, 4),
        "expected_volatility": round(portfolio_vol, 4),
        "notes": notes,
        "allocations": [a.model_dump() for a in allocations],
    }
