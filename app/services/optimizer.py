import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.config import settings


def _clip_and_normalize(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, settings.min_single_weight, settings.max_single_weight)
    total = clipped.sum()
    if total <= 0:
        return np.repeat(1 / len(clipped), len(clipped))
    return clipped / total


def optimize_strategic_allocation(expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    tickers = expected_returns.index.tolist()
    mu = expected_returns.values
    cov = covariance.loc[tickers, tickers].values
    n = len(mu)

    def objective(weights: np.ndarray) -> float:
        port_return = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_return - settings.risk_free_rate) / (port_vol + 1e-8)
        vol_penalty = max(port_vol - settings.target_volatility, 0) * 4.0
        return -sharpe + vol_penalty

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((settings.min_single_weight, settings.max_single_weight) for _ in range(n))
    x0 = np.repeat(1 / n, n)

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x if result.success else x0
    weights = _clip_and_normalize(weights)
    return pd.Series(weights, index=tickers)


def apply_tactical_overlay(strategic_weights: pd.Series, tactical_scores: pd.Series) -> pd.Series:
    tactical_scores = tactical_scores.reindex(strategic_weights.index).fillna(0)
    tilt = tactical_scores * 0.10
    adjusted = strategic_weights * (1 + tilt)
    adjusted = adjusted.clip(lower=0)
    normalized = adjusted / adjusted.sum()
    return normalized.fillna(0)
