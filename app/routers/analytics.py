from datetime import datetime

from fastapi import APIRouter, Query

from app.schemas import RecommendationResponse, SignalResponse
from app.services.recommendation import build_recommendation
from app.services.signals import compute_tactical_signals

router = APIRouter()


@router.get("/signals", response_model=list[SignalResponse])
def signals(tickers: str = Query(..., description="Comma-separated list, e.g. SPY,QQQ,AGG")):
    ticker_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    signals_df = compute_tactical_signals(ticker_list)
    records = []
    for _, row in signals_df.iterrows():
        records.append(
            SignalResponse(
                ticker=row["ticker"],
                momentum=float(row["momentum"]),
                trend_strength=float(row["trend_strength"]),
                tactical_score=float(row["tactical_score"]),
            )
        )
    return records


@router.get("/recommendation", response_model=RecommendationResponse)
def recommendation(tickers: str = Query(..., description="Comma-separated list, e.g. SPY,QQQ,AGG")):
    ticker_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    result = build_recommendation(ticker_list)
    return RecommendationResponse(
        generated_at=datetime.utcnow(),
        expected_return=result["expected_return"],
        expected_volatility=result["expected_volatility"],
        notes=result["notes"],
        allocations=result["allocations"],
    )
