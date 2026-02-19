from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.config import settings


def _clip_and_normalize(
    weights: np.ndarray,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    clipped = np.clip(weights, min_w, max_w)
    total = clipped.sum()
    if total <= 0:
        return np.repeat(1.0 / len(clipped), len(clipped))
    return clipped / total


def optimize_strategic_allocation(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float | None = None,
    min_weight: float | None = None,
    risk_free_rate: float | None = None,
    target_volatility: float | None = None,
    prohibited_tickers: list[str] | None = None,
) -> pd.Series:
    """
    Max-Sharpe optimizer with per-position weight bounds and a soft
    volatility-penalty term.

    Optional parameter overrides take precedence over settings values;
    if not supplied the settings defaults are used.
    """
    max_w = max_weight if max_weight is not None else settings.max_single_weight
    min_w = min_weight if min_weight is not None else settings.min_single_weight
    rfr = risk_free_rate if risk_free_rate is not None else settings.risk_free_rate
    vol_target = target_volatility if target_volatility is not None else settings.target_volatility

    # Remove prohibited tickers from universe
    tickers = [
        t for t in expected_returns.index.tolist()
        if not (prohibited_tickers and t in prohibited_tickers)
    ]
    if not tickers:
        tickers = expected_returns.index.tolist()  # ignore invalid prohibition

    mu = expected_returns.loc[tickers].values
    cov = covariance.loc[tickers, tickers].values
    n = len(mu)

    def objective(weights: np.ndarray) -> float:
        port_return = float(weights @ mu)
        port_vol = float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))
        sharpe = (port_return - rfr) / (port_vol + 1e-8)
        vol_penalty = max(port_vol - vol_target, 0.0) * 4.0
        return -sharpe + vol_penalty

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((min_w, max_w) for _ in range(n))
    x0 = np.repeat(1.0 / n, n)

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x if result.success else x0
    weights = _clip_and_normalize(weights, min_w, max_w)
    return pd.Series(weights, index=tickers)


def apply_tactical_overlay(
    strategic_weights: pd.Series,
    tactical_scores: pd.Series,
    tilt_strength: float | None = None,
) -> pd.Series:
    """
    Tilt strategic weights by normalised TAA scores.

    tilt_strength=0.10 means a +1 sigma score gives a +10% tilt to the weight.
    Pass tilt_strength=0.0 to disable TAA entirely.
    """
    strength = tilt_strength if tilt_strength is not None else 0.10
    tactical_scores = tactical_scores.reindex(strategic_weights.index).fillna(0.0)
    tilt = tactical_scores * strength
    adjusted = strategic_weights * (1.0 + tilt)
    adjusted = adjusted.clip(lower=0.0)
    total = adjusted.sum()
    normalized = adjusted / total if total > 0 else adjusted
    return normalized.fillna(0.0)
