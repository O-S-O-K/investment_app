"""Decision journal CRUD router."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import JournalEntry
from app.schemas import JournalEntryCreate, JournalEntryOut, JournalEntryUpdate

router = APIRouter()


@router.post("", response_model=JournalEntryOut, status_code=201)
def create_entry(payload: JournalEntryCreate, db: Session = Depends(get_db)):
    entry = JournalEntry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("", response_model=list[JournalEntryOut])
def list_entries(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    db: Session = Depends(get_db),
):
    q = db.query(JournalEntry)
    if ticker:
        q = q.filter(JournalEntry.ticker == ticker.upper())
    return q.order_by(JournalEntry.created_at.desc()).all()


@router.put("/{entry_id}", response_model=JournalEntryOut)
def update_entry(
    entry_id: int,
    payload: JournalEntryUpdate,
    db: Session = Depends(get_db),
):
    entry = db.get(JournalEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/{entry_id}", status_code=204)
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.get(JournalEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    db.delete(entry)
    db.commit()
