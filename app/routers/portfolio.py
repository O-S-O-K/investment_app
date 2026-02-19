import csv
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import FourOhOneKAllocation, FourOhOneKOption, Holding, TaxLot
from app.schemas import (
    FourOhOneKAllocationCreate,
    FourOhOneKAllocationOut,
    FourOhOneKOptionCreate,
    FourOhOneKOptionOut,
    HoldingCreate,
    HoldingOut,
    ImportCSVResponse,
    RebalancePlanResponse,
    RebalanceRequest,
)
from app.services.rebalance import build_tax_lot_rebalance_plan

router = APIRouter()


@router.post("/holdings", response_model=HoldingOut)
def create_holding(payload: HoldingCreate, db: Session = Depends(get_db)):
    item = Holding(**payload.model_dump(), ticker=payload.ticker.upper())
    db.add(item)
    db.flush()

    db.add(
        TaxLot(
            holding_id=item.id,
            ticker=item.ticker,
            account_type=item.account_type,
            acquired_at=datetime.utcnow(),
            shares=item.shares,
            cost_basis_per_share=item.cost_basis / item.shares,
            broker_source="manual",
        )
    )

    db.commit()
    db.refresh(item)
    return item


@router.get("/holdings", response_model=list[HoldingOut])
def list_holdings(db: Session = Depends(get_db)):
    return db.query(Holding).order_by(Holding.added_at.desc()).all()


@router.post("/401k/options", response_model=FourOhOneKOptionOut)
def create_401k_option(payload: FourOhOneKOptionCreate, db: Session = Depends(get_db)):
    option = FourOhOneKOption(**payload.model_dump(), symbol=payload.symbol.upper())
    db.add(option)
    db.commit()
    db.refresh(option)
    return option


@router.get("/401k/options", response_model=list[FourOhOneKOptionOut])
def list_401k_options(db: Session = Depends(get_db)):
    return db.query(FourOhOneKOption).all()


@router.post("/401k/allocation", response_model=FourOhOneKAllocationOut)
def set_401k_allocation(payload: FourOhOneKAllocationCreate, db: Session = Depends(get_db)):
    allocation = FourOhOneKAllocation(**payload.model_dump())
    db.add(allocation)
    db.commit()
    db.refresh(allocation)
    return allocation


@router.get("/401k/allocation", response_model=list[FourOhOneKAllocationOut])
def list_401k_allocation(db: Session = Depends(get_db)):
    return db.query(FourOhOneKAllocation).order_by(FourOhOneKAllocation.as_of.desc()).all()


@router.post("/holdings/import-csv", response_model=ImportCSVResponse)
async def import_holdings_csv(
    file: UploadFile = File(...),
    account_type: str = "taxable",
    broker_source: str = "csv-import",
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    raw = await file.read()
    decoded = raw.decode("utf-8-sig")
    reader = csv.DictReader(decoded.splitlines())

    required_columns = {"ticker", "shares", "cost_basis"}
    if not reader.fieldnames or not required_columns.issubset({col.strip() for col in reader.fieldnames}):
        raise HTTPException(
            status_code=400,
            detail="CSV must include columns: ticker, shares, cost_basis. Optional: acquired_at (YYYY-MM-DD).",
        )

    imported_rows = 0
    created_holdings = 0
    created_lots = 0
    skipped_rows = 0
    warnings: list[str] = []

    for idx, row in enumerate(reader, start=2):
        imported_rows += 1
        try:
            ticker = row["ticker"].strip().upper()
            shares = float(row["shares"])
            cost_basis = float(row["cost_basis"])
            acquired_raw = (row.get("acquired_at") or "").strip()
            acquired_at = datetime.strptime(acquired_raw, "%Y-%m-%d") if acquired_raw else datetime.utcnow()
            # per-row account_type overrides the query param if column is present
            row_account_type = (row.get("account_type") or "").strip() or account_type

            if shares <= 0 or cost_basis <= 0:
                raise ValueError("shares and cost_basis must be positive")

            holding = Holding(
                ticker=ticker,
                account_type=row_account_type,
                shares=shares,
                cost_basis=cost_basis,
            )
            db.add(holding)
            db.flush()
            created_holdings += 1

            lot = TaxLot(
                holding_id=holding.id,
                ticker=ticker,
                account_type=row_account_type,
                acquired_at=acquired_at,
                shares=shares,
                cost_basis_per_share=cost_basis / shares,
                broker_source=broker_source,
            )
            db.add(lot)
            created_lots += 1
        except Exception as exc:
            skipped_rows += 1
            warnings.append(f"Row {idx} skipped: {exc}")

    db.commit()

    return ImportCSVResponse(
        imported_rows=imported_rows,
        created_holdings=created_holdings,
        created_lots=created_lots,
        skipped_rows=skipped_rows,
        warnings=warnings[:20],
    )


@router.post("/rebalance/plan", response_model=RebalancePlanResponse)
def rebalance_plan(payload: RebalanceRequest, db: Session = Depends(get_db)):
    try:
        return build_tax_lot_rebalance_plan(db=db, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
