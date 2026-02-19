from __future__ import annotations

import numpy as np

from app.config import settings
from app.schemas import RecommendationItem
from app.services.forecast import forecast_expected_returns
from app.services.market_data import compute_returns, fetch_adjusted_close
from app.services.optimizer import apply_tactical_overlay, optimize_strategic_allocation
from app.services.risk import compute_cvar, marginal_risk_contribution
from app.services.signals import compute_tactical_signals


def build_recommendation(
    tickers: list[str],
    params: dict | None = None,
) -> dict:
    """
    Full SAA/TAA/ML recommendation pipeline.

    Optional *params* dict keys (all override settings defaults if provided):
        max_weight          float  0-1   per-ticker upper bound
        min_weight          float  0-1   per-ticker lower bound
        risk_free_rate      float        Sharpe denominator hurdle
        target_volatility   float        soft vol penalty threshold
        mu_blend_factor     float  0-1   historical weight (1-x goes to ML)
        taa_tilt_strength   float  0-1   TAA overlay magnitude
        prohibited_tickers  list[str]    exclude these from the universe
    """
    p = params or {}
    mu_blend = float(p.get("mu_blend_factor", 0.6))
    taa_strength = float(p.get("taa_tilt_strength", 0.10))
    prohibited = [t.upper() for t in p.get("prohibited_tickers", [])]

    # Filter prohibited tickers before fetching data
    active_tickers = [t for t in tickers if t not in prohibited]
    if not active_tickers:
        active_tickers = tickers  # safeguard

    prices = fetch_adjusted_close(tickers=active_tickers, years=2, months=6)
    returns = compute_returns(prices)

    historical_mu = returns.mean() * settings.annualization
    model_mu = forecast_expected_returns(active_tickers)
    blended_mu = mu_blend * historical_mu + (1.0 - mu_blend) * model_mu

    covariance = returns.cov() * settings.annualization

    strategic = optimize_strategic_allocation(
        blended_mu,
        covariance,
        max_weight=p.get("max_weight"),
        min_weight=p.get("min_weight"),
        risk_free_rate=p.get("risk_free_rate"),
        target_volatility=p.get("target_volatility"),
    )

    tactical_df = compute_tactical_signals(active_tickers)
    tactical_scores = tactical_df.set_index("ticker")["tactical_score"]
    final_weights = apply_tactical_overlay(strategic, tactical_scores, tilt_strength=taa_strength)

    # --- Portfolio-level metrics ---
    portfolio_return = float((final_weights * blended_mu).sum())
    cov_matrix = covariance.loc[final_weights.index, final_weights.index].values
    weight_vector = final_weights.values
    _var = float(weight_vector @ cov_matrix @ weight_vector)
    portfolio_vol = float(np.sqrt(max(_var, 0.0)))

    if not np.isfinite(portfolio_return):
        portfolio_return = 0.0
    if not np.isfinite(portfolio_vol):
        portfolio_vol = 0.0

    # --- Marginal risk contribution ---
    mrc = marginal_risk_contribution(
        final_weights,
        covariance.loc[final_weights.index, final_weights.index],
    )

    # --- CVaR (95% confidence) ---
    cvar = compute_cvar(returns, final_weights, confidence=0.95)
    if not np.isfinite(cvar):
        cvar = 0.0

    # --- Sharpe (use blended mu) ---
    rfr = p.get("risk_free_rate", settings.risk_free_rate)
    sharpe = (portfolio_return - rfr) / (portfolio_vol + 1e-8)
    if not np.isfinite(sharpe):
        sharpe = 0.0

    allocations = []
    for ticker in final_weights.index:
        allocations.append(
            RecommendationItem(
                ticker=ticker,
                strategic_weight=float(strategic.get(ticker, 0.0)),
                tactical_tilt=float(final_weights[ticker] - strategic.get(ticker, 0.0)),
                final_weight=float(final_weights[ticker]),
            ).model_dump()
            | {"risk_contribution": round(float(mrc.get(ticker, 0.0)), 4)}
        )

    # Build human-readable parameter notes
    param_notes = []
    if p.get("max_weight") is not None:
        param_notes.append(f"Max weight cap: {p['max_weight']:.0%}")
    if p.get("min_weight") and p["min_weight"] > 0:
        param_notes.append(f"Min weight floor: {p['min_weight']:.1%}")
    if p.get("taa_tilt_strength") is not None:
        label = "TAA disabled" if taa_strength == 0 else f"TAA tilt: {taa_strength:.0%}"
        param_notes.append(label)
    if prohibited:
        param_notes.append(f"Excluded: {', '.join(prohibited)}")

    notes = [
        f"Blended mu: {mu_blend:.0%} historical + {1 - mu_blend:.0%} ML forecast.",
        "Tactical overlay uses momentum and trend signals.",
        "Review against taxes, liquidity, and IPS constraints.",
    ] + param_notes

    return {
        "expected_return": round(portfolio_return, 4),
        "expected_volatility": round(portfolio_vol, 4),
        "expected_cvar_95": round(cvar, 4),
        "sharpe_ratio": round(sharpe, 4),
        "notes": notes,
        "allocations": allocations,
    }
