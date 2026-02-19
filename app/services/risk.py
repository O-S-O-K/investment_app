"""
Portfolio risk analytics:
  - Marginal risk contribution (% of total variance per position)
  - Historical CVaR / Expected Shortfall
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def marginal_risk_contribution(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Percentage of total portfolio variance attributable to each asset.

    mrc_i = w_i * (Σ w)_i  /  (w' Σ w)

    Returns values that sum to 1.0 (each is a fraction of total portfolio risk).
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    sigma_w = cov.values @ w
    port_var = float(w @ sigma_w)
    if port_var <= 1e-12:
        n = len(weights)
        return pd.Series(1.0 / n, index=weights.index)
    mrc = w * sigma_w / port_var
    return pd.Series(mrc, index=cov.index)


def compute_cvar(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Historical CVaR (Expected Shortfall) — annualised.

    Returns the expected annualised loss on the worst (1 - confidence) fraction
    of trading days, expressed as a negative number (e.g. -0.18 = -18% per year).
    """
    aligned_w = weights.reindex(returns.columns).fillna(0.0)
    aligned_w = aligned_w / aligned_w.sum() if aligned_w.sum() > 0 else aligned_w
    port_daily = returns @ aligned_w
    cutoff = port_daily.quantile(1.0 - confidence)
    tail = port_daily[port_daily <= cutoff]
    if tail.empty:
        return 0.0
    return float(tail.mean() * 252)  # annualise from daily
