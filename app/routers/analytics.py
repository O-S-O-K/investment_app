import threading
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import app.job_store as jobs
from app.schemas import RecommendationResponse, SignalResponse
from app.services.recommendation import build_recommendation
from app.services.signals import compute_tactical_signals

router = APIRouter()


# ---------------------------------------------------------------------------
# Job status response
# ---------------------------------------------------------------------------

class JobStarted(BaseModel):
    job_id: str
    status: str


class JobResult(BaseModel):
    job_id: str
    status: str
    result: dict | list | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Signals  (async job pattern)
# ---------------------------------------------------------------------------

def _run_signals(job_id: str, ticker_list: list[str]) -> None:
    try:
        jobs.set_running(job_id)
        signals_df = compute_tactical_signals(ticker_list)
        records = [
            {
                "ticker": row["ticker"],
                "momentum": float(row["momentum"]),
                "trend_strength": float(row["trend_strength"]),
                "tactical_score": float(row["tactical_score"]),
            }
            for _, row in signals_df.iterrows()
        ]
        jobs.set_done(job_id, records)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/signals/start", response_model=JobStarted)
def start_signals(tickers: str = Query(..., description="Comma-separated list, e.g. SPY,QQQ,AGG")):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    job = jobs.create_job()
    threading.Thread(target=_run_signals, args=(job.job_id, ticker_list), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Recommendation  (async job pattern)
# ---------------------------------------------------------------------------

def _run_recommendation(job_id: str, ticker_list: list[str]) -> None:
    try:
        jobs.set_running(job_id)
        result = build_recommendation(ticker_list)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/recommendation/start", response_model=JobStarted)
def start_recommendation(tickers: str = Query(..., description="Comma-separated list, e.g. SPY,QQQ,AGG")):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    job = jobs.create_job()
    threading.Thread(target=_run_recommendation, args=(job.job_id, ticker_list), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Shared job poll endpoint
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobResult)
def poll_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    jobs.purge_old()
    result = None
    if job.status == jobs.JobStatus.DONE:
        result = job.result
    return JobResult(job_id=job.job_id, status=job.status, result=result, error=job.error)


# ---------------------------------------------------------------------------
# Kept for backwards compatibility (direct sync calls) — not used by dashboard
# ---------------------------------------------------------------------------

@router.get("/signals", response_model=list[SignalResponse])
def signals(tickers: str = Query(...)):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    signals_df = compute_tactical_signals(ticker_list)
    return [
        SignalResponse(
            ticker=row["ticker"],
            momentum=float(row["momentum"]),
            trend_strength=float(row["trend_strength"]),
            tactical_score=float(row["tactical_score"]),
        )
        for _, row in signals_df.iterrows()
    ]


@router.get("/recommendation", response_model=RecommendationResponse)
def recommendation(tickers: str = Query(...)):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    result = build_recommendation(ticker_list)
    return RecommendationResponse(
        generated_at=datetime.utcnow(),
        expected_return=result["expected_return"],
        expected_volatility=result["expected_volatility"],
        notes=result["notes"],
        allocations=result["allocations"],
    )
