"""
Portfolio drawdown and high-water mark tracking.

Stores periodic portfolio snapshots in the DB and computes drawdown metrics
from snapshot history.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import PortfolioSnapshot


def record_snapshot(portfolio_value: float) -> None:
    """
    Persist a portfolio value snapshot.  Call whenever the user wants to
    capture the current portfolio value for drawdown tracking.
    Safe to call from a background thread (opens its own session).
    """
    db = SessionLocal()
    try:
        snap = PortfolioSnapshot(
            portfolio_value=portfolio_value,
            recorded_at=datetime.utcnow(),
        )
        db.add(snap)
        db.commit()
    finally:
        db.close()


def get_drawdown_analysis() -> dict:
    """
    Retrieve all snapshots and compute drawdown summary metrics.
    Returns current drawdown, max drawdown, high-water mark, and the full series.
    """
    db = SessionLocal()
    try:
        snaps = (
            db.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.recorded_at)
            .all()
        )
    finally:
        db.close()

    if not snaps:
        return {
            "error": "No portfolio snapshots recorded yet.",
            "series": [],
            "current_value": None,
            "high_water_mark": None,
            "current_drawdown_pct": None,
            "max_drawdown_pct": None,
            "n_snapshots": 0,
        }

    values = pd.Series(
        [s.portfolio_value for s in snaps],
        index=[s.recorded_at for s in snaps],
    )
    hwm = values.cummax()
    drawdown = (values - hwm) / hwm  # negative or zero

    series = [
        {
            "timestamp": str(snaps[i].recorded_at)[:19],
            "value": round(float(values.iloc[i]), 2),
            "hwm": round(float(hwm.iloc[i]), 2),
            "drawdown_pct": round(float(drawdown.iloc[i]) * 100, 2),
        }
        for i in range(len(snaps))
    ]

    return {
        "current_value": round(float(values.iloc[-1]), 2),
        "high_water_mark": round(float(hwm.iloc[-1]), 2),
        "current_drawdown_pct": round(float(drawdown.iloc[-1]) * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "n_snapshots": len(snaps),
        "series": series,
    }
