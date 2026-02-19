from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import TaxLot
from app.schemas import RebalanceRequest, RebalanceTrade
from app.services.market_data import fetch_adjusted_close


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    prices = fetch_adjusted_close(tickers, years=1)
    last = prices.iloc[-1]
    return {ticker: float(last[ticker]) for ticker in tickers}


def _tax_rate_for_lot(acquired_at: datetime, as_of: datetime, short_rate: float, long_rate: float) -> float:
    is_long_term = acquired_at <= (as_of - timedelta(days=365))
    return long_rate if is_long_term else short_rate


def build_tax_lot_rebalance_plan(db: Session, payload: RebalanceRequest) -> dict:
    target_weights = {ticker.upper(): weight for ticker, weight in payload.target_weights.items() if weight > 0}
    total_target = sum(target_weights.values())
    if total_target <= 0:
        raise ValueError("Target weights must have positive total weight")

    target_weights = {ticker: weight / total_target for ticker, weight in target_weights.items()}
    tickers = sorted(target_weights.keys())
    lots = db.query(TaxLot).filter(TaxLot.ticker.in_(tickers)).all()

    if not lots:
        return {
            "generated_at": datetime.utcnow(),
            "portfolio_value": 0.0,
            "estimated_total_tax_impact": 0.0,
            "trades": [],
            "notes": ["No tax lots found for requested target tickers."],
        }

    prices = _latest_prices(tickers)
    market_values: dict[str, float] = {ticker: 0.0 for ticker in tickers}
    for lot in lots:
        market_values[lot.ticker] += lot.shares * prices[lot.ticker]

    portfolio_value = sum(market_values.values())
    if portfolio_value <= 0:
        return {
            "generated_at": datetime.utcnow(),
            "portfolio_value": 0.0,
            "estimated_total_tax_impact": 0.0,
            "trades": [],
            "notes": ["Portfolio market value is zero or unavailable."],
        }

    target_values = {ticker: target_weights[ticker] * portfolio_value for ticker in tickers}
    deltas = {ticker: target_values[ticker] - market_values[ticker] for ticker in tickers}

    now = datetime.utcnow()
    trades: list[RebalanceTrade] = []
    total_tax_impact = 0.0

    for ticker in tickers:
        delta_value = deltas[ticker]
        price = prices[ticker]

        if abs(delta_value) < payload.min_trade_value:
            continue

        if delta_value > 0:
            shares_to_buy = delta_value / price
            trades.append(
                RebalanceTrade(
                    ticker=ticker,
                    action="BUY",
                    shares=round(shares_to_buy, 6),
                    estimated_trade_value=round(delta_value, 2),
                    estimated_tax_impact=0.0,
                    reason="Move toward target weight.",
                )
            )
            continue

        sell_value_needed = abs(delta_value)
        ticker_lots = [lot for lot in lots if lot.ticker == ticker and lot.shares > 0]

        ranked_lots = sorted(
            ticker_lots,
            key=lambda lot: (
                ((price - lot.cost_basis_per_share) * payload.long_term_tax_rate)
                if price < lot.cost_basis_per_share
                else (
                    (price - lot.cost_basis_per_share)
                    * _tax_rate_for_lot(
                        acquired_at=lot.acquired_at,
                        as_of=now,
                        short_rate=payload.short_term_tax_rate,
                        long_rate=payload.long_term_tax_rate,
                    )
                ),
                lot.acquired_at,
            ),
        )

        for lot in ranked_lots:
            if sell_value_needed <= 0:
                break

            lot_price = price
            lot_available_value = lot.shares * lot_price
            trade_value = min(sell_value_needed, lot_available_value)
            trade_shares = trade_value / lot_price

            gain_per_share = lot_price - lot.cost_basis_per_share
            realized_gain = gain_per_share * trade_shares
            tax_rate = _tax_rate_for_lot(
                acquired_at=lot.acquired_at,
                as_of=now,
                short_rate=payload.short_term_tax_rate,
                long_rate=payload.long_term_tax_rate,
            )
            tax_impact = max(realized_gain, 0) * tax_rate
            total_tax_impact += tax_impact

            lot_term = "long-term" if tax_rate == payload.long_term_tax_rate else "short-term"
            reason = (
                "Harvesting loss lot first."
                if realized_gain < 0
                else f"Selling {lot_term} lot with lower tax drag."
            )

            trades.append(
                RebalanceTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=round(trade_shares, 6),
                    estimated_trade_value=round(trade_value, 2),
                    estimated_tax_impact=round(tax_impact, 2),
                    lot_id=lot.id,
                    reason=reason,
                )
            )

            sell_value_needed -= trade_value

    return {
        "generated_at": datetime.utcnow(),
        "portfolio_value": round(portfolio_value, 2),
        "estimated_total_tax_impact": round(total_tax_impact, 2),
        "trades": trades,
        "notes": [
            "Plan is tax-aware but does not include wash-sale enforcement.",
            "Verify account restrictions and bid/ask liquidity before execution.",
        ],
    }
