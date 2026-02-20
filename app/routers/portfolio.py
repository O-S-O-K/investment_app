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
from app.services.market_data import fetch_adjusted_close
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
# Total cost basis (preferred) — ordered most-specific first to avoid wrong matches
_COL_CB_TOTAL = ["cost_basis", "cost basis total", "total cost basis", "total cost",
                 "cost basis", "adjusted cost basis total"]
# Per-share cost basis (fallback – multiplied by shares to get total)
_COL_CB_PER_SHARE = ["average cost basis", "avg cost basis", "average cost",
                     "cost basis/share", "unit cost", "cost/share"]
_COL_ACQUIRED = ["acquired_at", "acquired date", "date acquired",
                 "acquisition date", "trade date", "open date"]
_COL_ACCOUNT  = ["account_type", "account type", "acct type"]

# Tickers that are never real positions (various broker footer/summary rows)
_SKIP_TICKERS = {
    "TOTAL", "TOTALS", "TOTAL ACCOUNT VALUE", "ACCOUNT TOTAL",
    "CASH", "PENDING ACTIVITY", "OTHER", "--", "N/A",
}


def _clean_num(val: str) -> float:
    """Strip $, commas, +, spaces and cast to float. Raises ValueError if empty/dash."""
    cleaned = val.strip().replace("$", "").replace(",", "").replace("+", "").replace(" ", "")
    if cleaned in ("", "--", "-", "N/A", "n/a", "n/a*"):
        raise ValueError(f"no numeric value: {val!r}")
    return float(cleaned)


def _map_col(fieldnames: list[str], aliases: list[str]) -> str | None:
    """Return the first fieldname that matches any alias (case-insensitive, exact match)."""
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


def _is_skip_ticker(ticker: str) -> bool:
    """Return True if this ticker value represents a non-position row."""
    t = ticker.strip().upper()
    if not t or t.startswith("--"):
        return True
    if t in _SKIP_TICKERS:
        return True
    if t.endswith("**"):          # cash sweep like SPAXX**
        return True
    if "PENDING" in t:
        return True
    if " " in t and len(t) > 5:  # real tickers have no spaces; descriptions do
        return True
    return False


def _parse_csv_rows(
    lines: list[str],
    account_type: str,
    broker_source: str,
) -> tuple[dict, list[dict], list[str]]:
    """
    Parse CSV lines into holding rows without touching the DB.

    Returns:
        column_map  — which column name was detected for each field
        parsed      — list of dicts ready to write
        warnings    — list of skipped-row messages
    """
    header_idx = _find_header_line(lines)
    reader = csv.DictReader(lines[header_idx:])

    if not reader.fieldnames:
        raise ValueError("Could not parse CSV — no header row found.")

    fields: list[str] = list(reader.fieldnames)

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
        raise ValueError(
            f"Cannot find required columns: {', '.join(missing)}. "
            f"Detected columns: {', '.join(fields[:20])}."
        )

    column_map = {
        "ticker_col":       col_ticker,
        "shares_col":       col_shares,
        "cost_basis_col":   col_cb_total or f"{col_cb_ps} (per-share × shares)",
        "acquired_col":     col_acquired or "(none — will use today)",
        "account_type_col": col_account  or f"(none — will use '{account_type}')",
        "all_columns":      fields,
    }

    parsed: list[dict] = []
    warnings: list[str] = []

    for idx, row in enumerate(reader, start=2):
        try:
            ticker = (row.get(col_ticker) or "").strip().upper()
            if _is_skip_ticker(ticker):
                warnings.append(f"Row {idx} skipped: non-position ticker {ticker!r}")
                continue

            shares = _clean_num(row.get(col_shares) or "")

            # Cost basis: prefer total column; fall back to per-share if total is "--"/missing
            cost_basis: float | None = None
            cb_method = ""
            if col_cb_total:
                raw_cb = (row.get(col_cb_total) or "").strip()
                if raw_cb and raw_cb not in ("--", "-", "N/A", "n/a", "n/a*", ""):
                    try:
                        cost_basis = _clean_num(raw_cb)
                        cb_method = f"from '{col_cb_total}'"
                    except ValueError:
                        pass  # fall through to per-share
            if cost_basis is None and col_cb_ps:
                raw_ps = (row.get(col_cb_ps) or "").strip()
                cb_ps_val = _clean_num(raw_ps)
                cost_basis = cb_ps_val * shares
                cb_method = f"per-share '{col_cb_ps}' × shares"
            if cost_basis is None:
                raise ValueError("cost basis unavailable (both columns are empty/--)")

            acquired_raw = (row.get(col_acquired) if col_acquired else None) or ""
            acquired_at_str = "today (no date column)"
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
                try:
                    datetime.strptime(acquired_raw.strip(), fmt)
                    acquired_at_str = acquired_raw.strip()
                    break
                except ValueError:
                    pass

            row_account_type = (
                (row.get(col_account).strip() if col_account and row.get(col_account) else "")
                or account_type
            )

            if shares <= 0 or cost_basis <= 0:
                raise ValueError(f"shares={shares}, cost_basis={cost_basis} must be positive")

            parsed.append({
                "ticker": ticker,
                "shares": round(shares, 6),
                "cost_basis_total": round(cost_basis, 2),
                "cost_basis_per_share": round(cost_basis / shares, 4),
                "acquired_at": acquired_at_str,
                "account_type": row_account_type,
                "broker_source": broker_source,
                "cb_method": cb_method,
            })

        except Exception as exc:
            warnings.append(f"Row {idx} skipped: {exc}")

    return column_map, parsed, warnings


@router.post("/holdings/preview-csv")
async def preview_csv(
    file: UploadFile = File(...),
    account_type: str = "taxable",
    broker_source: str = "csv-import",
):
    """
    Dry-run the CSV import and return exactly what would be written.
    Nothing is saved to the database.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    raw = await file.read()
    decoded = raw.decode("utf-8-sig")
    lines = decoded.splitlines()
    try:
        column_map, parsed, warnings = _parse_csv_rows(lines, account_type, broker_source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_cost_basis = sum(r["cost_basis_total"] for r in parsed)
    return {
        "column_map": column_map,
        "rows": parsed,
        "row_count": len(parsed),
        "skipped_count": len(warnings),
        "total_cost_basis": round(total_cost_basis, 2),
        "warnings": warnings,
    }


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

    try:
        _column_map, parsed, parse_warnings = _parse_csv_rows(lines, account_type, broker_source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not parsed:
        raise HTTPException(
            status_code=400,
            detail=(
                "No importable rows found. "
                + (" Warnings: " + "; ".join(parse_warnings[:5]) if parse_warnings else "")
            ),
        )

    created_holdings = 0
    created_lots = 0

    # UPSERT: for every (ticker, account_type) in this import, delete any existing
    # Holdings and TaxLots first so re-importing the same file is idempotent.
    incoming_combos = {(r["ticker"], r["account_type"]) for r in parsed}
    for ticker, acct in incoming_combos:
        old_holdings = (
            db.query(Holding)
            .filter(Holding.ticker == ticker, Holding.account_type == acct)
            .all()
        )
        for h in old_holdings:
            db.query(TaxLot).filter(TaxLot.holding_id == h.id).delete()
            db.delete(h)
        # Also remove any orphaned TaxLots for this (ticker, account_type)
        db.query(TaxLot).filter(
            TaxLot.ticker == ticker, TaxLot.account_type == acct
        ).delete()
    db.flush()

    for r in parsed:
        acquired_at = datetime.utcnow()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                acquired_at = datetime.strptime(r["acquired_at"], fmt)
                break
            except ValueError:
                pass

        holding = Holding(
            ticker=r["ticker"],
            account_type=r["account_type"],
            shares=r["shares"],
            cost_basis=r["cost_basis_total"],
        )
        db.add(holding)
        db.flush()
        created_holdings += 1

        lot = TaxLot(
            holding_id=holding.id,
            ticker=r["ticker"],
            account_type=r["account_type"],
            acquired_at=acquired_at,
            shares=r["shares"],
            cost_basis_per_share=r["cost_basis_per_share"],
            broker_source=r["broker_source"],
        )
        db.add(lot)
        created_lots += 1

    db.commit()

    return ImportCSVResponse(
        imported_rows=len(parsed),
        created_holdings=created_holdings,
        created_lots=created_lots,
        skipped_rows=len(parse_warnings),
        warnings=parse_warnings[:20],
    )



@router.post("/rebalance/plan", response_model=RebalancePlanResponse)
def rebalance_plan(payload: RebalanceRequest, db: Session = Depends(get_db)):
    try:
        return build_tax_lot_rebalance_plan(db=db, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/holdings/clear")
def clear_all_holdings(db: Session = Depends(get_db)):
    """Delete all Holding rows (and their TaxLot children via CASCADE)."""
    deleted = db.query(Holding).delete()
    db.commit()
    return {"deleted_holdings": deleted, "message": "All holdings and tax lots cleared."}


@router.get("/holdings/summary")
def holdings_summary(db: Session = Depends(get_db)):
    """
    Return a per-(ticker, account_type) summary of all tax lots with current
    market prices so the user can verify totals before running drift/rebalance.
    """
    lots = db.query(TaxLot).all()
    if not lots:
        return {"rows": [], "total_market_value": 0.0, "total_cost_basis": 0.0,
                "total_unrealized_gain": 0.0, "note": "No holdings in database."}

    # Aggregate per (ticker, account_type)
    agg: dict[tuple, dict] = {}
    for lot in lots:
        key = (lot.ticker, lot.account_type)
        if key not in agg:
            agg[key] = {"ticker": lot.ticker, "account_type": lot.account_type,
                        "shares": 0.0, "cost_basis_total": 0.0, "lot_count": 0}
        agg[key]["shares"] += lot.shares
        agg[key]["cost_basis_total"] += lot.shares * lot.cost_basis_per_share
        agg[key]["lot_count"] += 1

    tickers = list({k[0] for k in agg})
    try:
        prices_df = fetch_adjusted_close(tickers=tickers, years=0, months=2)
        latest_prices = prices_df.iloc[-1].to_dict()
    except Exception:
        latest_prices = {}

    rows = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    for (ticker, acct), data in sorted(agg.items()):
        price = latest_prices.get(ticker, 0.0)
        market_value = data["shares"] * price
        cost_basis = data["cost_basis_total"]
        unrealized = market_value - cost_basis
        unrealized_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0.0
        total_market_value += market_value
        total_cost_basis += cost_basis
        rows.append({
            "ticker": ticker,
            "account_type": acct,
            "lot_count": data["lot_count"],
            "shares": round(data["shares"], 6),
            "current_price": round(price, 4),
            "market_value": round(market_value, 2),
            "cost_basis_total": round(cost_basis, 2),
            "unrealized_gain": round(unrealized, 2),
            "unrealized_gain_pct": round(unrealized_pct, 2),
        })

    total_unrealized = total_market_value - total_cost_basis
    return {
        "rows": rows,
        "total_market_value": round(total_market_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealized_gain": round(total_unrealized, 2),
        "note": "Prices are delayed ~15 min. This total should match Drift and Rebalance.",
    }
