"""
Tax-loss harvesting scanner.

Scans all taxable tax lots for unrealised losses exceeding a minimum threshold
and flags 30-day wash-sale risk windows.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.db import SessionLocal
from app.models import TaxLot
from app.services.market_data import fetch_adjusted_close


def scan_harvest_opportunities(
    min_loss_threshold: float = 100.0,
) -> dict:
    """
    Identify taxable lots with unrealised losses >= *min_loss_threshold* dollars.

    Flags wash-sale risk when the same ticker has lots acquired within the last
    30 days (buying a substantially identical position soon before or after
    harvesting a loss disallows the deduction under wash-sale rules).

    Uses a fresh DB session — safe to call from a background thread.
    """
    db = SessionLocal()
    try:
        lots = (
            db.query(TaxLot)
            .filter(TaxLot.account_type == "taxable")
            .order_by(TaxLot.acquired_at)
            .all()
        )
    finally:
        db.close()

    if not lots:
        return {
            "lots": [],
            "total_harvestable_loss": 0.0,
            "n_lots": 0,
            "min_loss_threshold": min_loss_threshold,
            "message": "No taxable lots found in database.",
        }

    tickers = list({lot.ticker for lot in lots})
    prices_df = fetch_adjusted_close(tickers=tickers, years=0, months=3)
    latest_prices = prices_df.iloc[-1].to_dict()

    today = date.today()
    wash_window = timedelta(days=30)

    # Tickers where a lot was acquired within the last 30 days (wash-sale flag)
    recently_acquired: set[str] = {
        lot.ticker
        for lot in lots
        if (today - lot.acquired_at.date()) <= wash_window
    }

    candidates = []
    total_loss = 0.0

    for lot in lots:
        current_price = latest_prices.get(lot.ticker, 0.0)
        if current_price <= 0:
            continue

        cost_total = lot.cost_basis_per_share * lot.shares
        current_value = current_price * lot.shares
        unrealised_pnl = current_value - cost_total

        if unrealised_pnl >= -min_loss_threshold:
            continue  # not enough loss to bother

        holding_days = (today - lot.acquired_at.date()).days
        candidates.append(
            {
                "lot_id": lot.id,
                "ticker": lot.ticker,
                "shares": round(lot.shares, 4),
                "acquired_at": str(lot.acquired_at.date()),
                "holding_days": holding_days,
                "term": "long" if holding_days >= 365 else "short",
                "cost_basis_per_share": round(lot.cost_basis_per_share, 4),
                "current_price": round(current_price, 4),
                "cost_basis_total": round(cost_total, 2),
                "current_value": round(current_value, 2),
                "unrealised_loss": round(unrealised_pnl, 2),
                "wash_sale_risk": lot.ticker in recently_acquired,
            }
        )
        total_loss += unrealised_pnl

    candidates.sort(key=lambda r: r["unrealised_loss"])  # worst loss first

    return {
        "lots": candidates,
        "total_harvestable_loss": round(total_loss, 2),
        "n_lots": len(candidates),
        "min_loss_threshold": min_loss_threshold,
    }
