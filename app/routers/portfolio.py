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



# ---------------------------------------------------------------------------
# Broker column-name aliases   (lower-cased, stripped for matching)
# ---------------------------------------------------------------------------
_COL_TICKER = ["ticker", "symbol", "security"]
_COL_SHARES = ["shares", "quantity", "qty", "number of shares", "share quantity"]
# Total cost basis (preferred)
_COL_CB_TOTAL = ["cost_basis", "cost basis total", "cost basis", "total cost basis",
                 "total cost", "basis"]
# Per-share cost basis (fallback – multiplied by shares to get total)
_COL_CB_PER_SHARE = ["average cost basis", "avg cost basis", "cost basis/share",
                     "unit cost"]
_COL_ACQUIRED = ["acquired_at", "acquired date", "date acquired",
                 "acquisition date", "trade date", "open date"]
_COL_ACCOUNT  = ["account_type", "account type", "acct type"]


def _clean_num(val: str) -> float:
    """Strip $, commas, spaces and cast to float. Raises ValueError if empty/dash."""
    cleaned = val.strip().replace("$", "").replace(",", "").replace(" ", "")
    if cleaned in ("", "--", "N/A", "n/a"):
        raise ValueError(f"no numeric value: {val!r}")
    return float(cleaned)


def _map_col(fieldnames: list[str], aliases: list[str]) -> str | None:
    """Return the first fieldname that matches any alias (case-insensitive)."""
    lower_fields = {f.strip().lower(): f for f in fieldnames}
    for alias in aliases:
        if alias.lower() in lower_fields:
            return lower_fields[alias.lower()]
    return None


def _find_header_line(lines: list[str]) -> int:
    """Scan for the first line that looks like a real CSV header (has 'symbol' or 'ticker')."""
    for i, line in enumerate(lines):
        lower = line.lower()
        if "symbol" in lower or "ticker" in lower:
            return i
    return 0


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
    lines = decoded.splitlines()

    # Skip any broker preamble rows (Fidelity adds account/date headers above data)
    header_idx = _find_header_line(lines)
    reader = csv.DictReader(lines[header_idx:])

    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Could not parse CSV — no header row found.")

    fields: list[str] = list(reader.fieldnames)

    # Map column aliases → actual column names in this file
    col_ticker   = _map_col(fields, _COL_TICKER)
    col_shares   = _map_col(fields, _COL_SHARES)
    col_cb_total = _map_col(fields, _COL_CB_TOTAL)
    col_cb_ps    = _map_col(fields, _COL_CB_PER_SHARE)
    col_acquired = _map_col(fields, _COL_ACQUIRED)
    col_account  = _map_col(fields, _COL_ACCOUNT)

    missing = []
    if not col_ticker: missing.append("ticker/symbol")
    if not col_shares: missing.append("shares/quantity")
    if not col_cb_total and not col_cb_ps: missing.append("cost basis")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot find required columns: {', '.join(missing)}. "
                f"Detected columns: {', '.join(fields[:15])}. "
                "Supported brokers: Fidelity, Schwab, Vanguard, or any CSV with "
                "ticker/symbol, shares/quantity, and a cost basis column."
            ),
        )

    imported_rows = 0
    created_holdings = 0
    created_lots = 0
    skipped_rows = 0
    warnings: list[str] = []

    for idx, row in enumerate(reader, start=2):
        imported_rows += 1
        try:
            ticker = (row.get(col_ticker) or "").strip().upper()

            # Skip blank tickers, cash sweep funds (SPAXX**), summary rows
            if not ticker or ticker.startswith("--") or ticker.upper() in ("TOTAL", "TOTALS"):
                skipped_rows += 1
                imported_rows -= 1
                continue
            if ticker.endswith("**") or "PENDING" in ticker.upper():
                skipped_rows += 1
                imported_rows -= 1
                continue

            shares = _clean_num(row.get(col_shares) or "")

            # Cost basis: prefer total column, fall back to per-share × shares
            if col_cb_total:
                cost_basis = _clean_num(row.get(col_cb_total) or "")
            else:
                cb_ps = _clean_num(row.get(col_cb_ps) or "")
                cost_basis = cb_ps * shares

            acquired_raw = (row.get(col_acquired) if col_acquired else None) or ""
            acquired_raw = acquired_raw.strip()
            acquired_at = datetime.utcnow()
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
                try:
                    acquired_at = datetime.strptime(acquired_raw, fmt)
                    break
                except ValueError:
                    pass

            row_account_type = (
                (row.get(col_account).strip() if col_account and row.get(col_account) else "")
                or account_type
            )

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
