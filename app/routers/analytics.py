import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import app.job_store as jobs
from app.schemas import RecommendationResponse, SignalResponse
from app.services.attribution import brinson_attribution
from app.services.drawdown import get_drawdown_analysis, record_snapshot
from app.services.drift import compute_drift
from app.services.factors import compute_factor_exposure
from app.services.harvest import scan_harvest_opportunities
from app.services.recommendation import build_recommendation
from app.services.scenario import run_scenarios
from app.services.signals import compute_tactical_signals

router = APIRouter()


# ---------------------------------------------------------------------------
# Common response models
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
# Helpers
# ---------------------------------------------------------------------------

def _parse_tickers(tickers_str: str) -> list[str]:
    return [t.strip().upper() for t in tickers_str.split(",") if t.strip()]


def _parse_weights(weights_str: str) -> dict[str, float]:
    """Parse 'TICKER:WEIGHT,...' string into a dict."""
    result: dict[str, float] = {}
    for chunk in weights_str.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        ticker, val = chunk.split(":", 1)
        try:
            result[ticker.strip().upper()] = float(val.strip())
        except ValueError:
            pass
    return result


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
def start_signals(
    tickers: str = Query(..., description="Comma-separated, e.g. SPY,QQQ,AGG"),
):
    ticker_list = _parse_tickers(tickers)
    job = jobs.create_job()
    threading.Thread(target=_run_signals, args=(job.job_id, ticker_list), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Recommendation  (async job pattern, fully parameterised)
# ---------------------------------------------------------------------------

def _run_recommendation(job_id: str, ticker_list: list[str], params: dict) -> None:
    try:
        jobs.set_running(job_id)
        result = build_recommendation(ticker_list, params=params)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/recommendation/start", response_model=JobStarted)
def start_recommendation(
    tickers: str = Query(..., description="Comma-separated, e.g. SPY,QQQ,AGG"),
    max_weight: Optional[float] = Query(None, ge=0.01, le=1.0, description="Max weight per ticker (0-1)"),
    min_weight: Optional[float] = Query(None, ge=0.0, le=0.5, description="Min weight per ticker (0-1)"),
    risk_free_rate: Optional[float] = Query(None, ge=0.0, le=0.2, description="Risk-free rate for Sharpe"),
    target_volatility: Optional[float] = Query(None, ge=0.01, le=0.5, description="Portfolio vol target"),
    mu_blend_factor: Optional[float] = Query(None, ge=0.0, le=1.0, description="Historical return weight (0=all ML, 1=all hist)"),
    taa_tilt_strength: Optional[float] = Query(None, ge=0.0, le=0.5, description="TAA overlay magnitude (0=off)"),
    prohibited: Optional[str] = Query(None, description="Comma-separated tickers to exclude"),
):
    ticker_list = _parse_tickers(tickers)
    params: dict = {}
    if max_weight is not None:
        params["max_weight"] = max_weight
    if min_weight is not None:
        params["min_weight"] = min_weight
    if risk_free_rate is not None:
        params["risk_free_rate"] = risk_free_rate
    if target_volatility is not None:
        params["target_volatility"] = target_volatility
    if mu_blend_factor is not None:
        params["mu_blend_factor"] = mu_blend_factor
    if taa_tilt_strength is not None:
        params["taa_tilt_strength"] = taa_tilt_strength
    if prohibited:
        params["prohibited_tickers"] = _parse_tickers(prohibited)

    job = jobs.create_job()
    threading.Thread(
        target=_run_recommendation,
        args=(job.job_id, ticker_list, params),
        daemon=True,
    ).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Drift monitoring  (async job pattern)
# ---------------------------------------------------------------------------

def _run_drift(job_id: str, target_weights: dict[str, float], threshold: float) -> None:
    try:
        jobs.set_running(job_id)
        result = compute_drift(target_weights=target_weights, drift_threshold=threshold)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/drift/start", response_model=JobStarted)
def start_drift(
    target_weights: str = Query(..., description="TICKER:WEIGHT,... e.g. VTI:0.6,BND:0.4"),
    drift_threshold: float = Query(0.05, ge=0.0, le=0.5, description="Alert threshold (5% = 0.05)"),
):
    weights = _parse_weights(target_weights)
    if not weights:
        raise HTTPException(status_code=422, detail="Could not parse target_weights.")
    job = jobs.create_job()
    threading.Thread(target=_run_drift, args=(job.job_id, weights, drift_threshold), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Tax-loss harvest scanner  (async job pattern)
# ---------------------------------------------------------------------------

def _run_harvest(job_id: str, min_loss: float) -> None:
    try:
        jobs.set_running(job_id)
        result = scan_harvest_opportunities(min_loss_threshold=min_loss)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/harvest/start", response_model=JobStarted)
def start_harvest(
    min_loss: float = Query(100.0, ge=0.0, description="Minimum unrealised loss in $ to flag"),
):
    job = jobs.create_job()
    threading.Thread(target=_run_harvest, args=(job.job_id, min_loss), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Drawdown tracking
# ---------------------------------------------------------------------------

@router.post("/snapshot")
def post_snapshot(portfolio_value: float = Query(..., gt=0, description="Current portfolio value in $")):
    """Record a portfolio value snapshot for drawdown tracking."""
    record_snapshot(portfolio_value=portfolio_value)
    return {"status": "recorded", "portfolio_value": portfolio_value}


def _run_drawdown(job_id: str) -> None:
    try:
        jobs.set_running(job_id)
        result = get_drawdown_analysis()
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/drawdown/start", response_model=JobStarted)
def start_drawdown():
    """Start a background job to compute drawdown metrics from stored snapshots."""
    job = jobs.create_job()
    threading.Thread(target=_run_drawdown, args=(job.job_id,), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Performance attribution  (async job pattern)
# ---------------------------------------------------------------------------

def _run_attribution(
    job_id: str,
    portfolio_weights: dict[str, float],
    benchmark_ticker: str,
    lookback_days: int,
) -> None:
    try:
        jobs.set_running(job_id)
        result = brinson_attribution(portfolio_weights, benchmark_ticker, lookback_days)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/attribution/start", response_model=JobStarted)
def start_attribution(
    portfolio_weights: str = Query(..., description="TICKER:WEIGHT,... e.g. SPY:0.6,AGG:0.4"),
    benchmark: str = Query("SPY", description="Benchmark ticker symbol"),
    lookback_days: int = Query(63, ge=10, le=504, description="Trading-day lookback"),
):
    weights = _parse_weights(portfolio_weights)
    if not weights:
        raise HTTPException(status_code=422, detail="Could not parse portfolio_weights.")
    job = jobs.create_job()
    threading.Thread(
        target=_run_attribution,
        args=(job.job_id, weights, benchmark.upper(), lookback_days),
        daemon=True,
    ).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Scenario stress testing  (async job pattern)
# ---------------------------------------------------------------------------

def _run_scenarios(job_id: str, portfolio_weights: dict[str, float]) -> None:
    try:
        jobs.set_running(job_id)
        result = run_scenarios(portfolio_weights)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/scenarios/start", response_model=JobStarted)
def start_scenarios(
    portfolio_weights: str = Query(..., description="TICKER:WEIGHT,... e.g. VTI:0.6,BND:0.4"),
):
    weights = _parse_weights(portfolio_weights)
    if not weights:
        raise HTTPException(status_code=422, detail="Could not parse portfolio_weights.")
    job = jobs.create_job()
    threading.Thread(target=_run_scenarios, args=(job.job_id, weights), daemon=True).start()
    return JobStarted(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# Factor exposure  (async job pattern)
# ---------------------------------------------------------------------------

def _run_factors(
    job_id: str,
    portfolio_weights: dict[str, float],
    lookback_days: int,
) -> None:
    try:
        jobs.set_running(job_id)
        result = compute_factor_exposure(portfolio_weights, lookback_days=lookback_days)
        jobs.set_done(job_id, result)
    except Exception as exc:
        jobs.set_error(job_id, str(exc))


@router.post("/factors/start", response_model=JobStarted)
def start_factors(
    portfolio_weights: str = Query(..., description="TICKER:WEIGHT,... e.g. VTI:0.6,BND:0.4"),
    lookback_days: int = Query(252, ge=60, le=756, description="Trading-day lookback"),
):
    weights = _parse_weights(portfolio_weights)
    if not weights:
        raise HTTPException(status_code=422, detail="Could not parse portfolio_weights.")
    job = jobs.create_job()
    threading.Thread(
        target=_run_factors,
        args=(job.job_id, weights, lookback_days),
        daemon=True,
    ).start()
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
# Backwards-compatible sync routes (for scripting / direct API calls)
# ---------------------------------------------------------------------------

@router.get("/signals", response_model=list[SignalResponse])
def signals(tickers: str = Query(...)):
    ticker_list = _parse_tickers(tickers)
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
    ticker_list = _parse_tickers(tickers)
    result = build_recommendation(ticker_list)
    return result
