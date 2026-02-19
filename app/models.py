from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    account_type: Mapped[str] = mapped_column(String(32), default="taxable")
    shares: Mapped[float] = mapped_column(Float)
    cost_basis: Mapped[float] = mapped_column(Float)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    lots = relationship("TaxLot", back_populates="holding", cascade="all, delete-orphan")


class TaxLot(Base):
    __tablename__ = "tax_lots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    holding_id: Mapped[int] = mapped_column(ForeignKey("holdings.id", ondelete="CASCADE"), index=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    account_type: Mapped[str] = mapped_column(String(32), default="taxable")
    acquired_at: Mapped[datetime] = mapped_column(DateTime)
    shares: Mapped[float] = mapped_column(Float)
    cost_basis_per_share: Mapped[float] = mapped_column(Float)
    broker_source: Mapped[str] = mapped_column(String(64), default="manual")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    holding = relationship("Holding", back_populates="lots")


class FourOhOneKOption(Base):
    __tablename__ = "four01k_options"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(128))
    asset_class: Mapped[str] = mapped_column(String(64))
    expense_ratio: Mapped[float] = mapped_column(Float, default=0.0)


class FourOhOneKAllocation(Base):
    __tablename__ = "four01k_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    option_id: Mapped[int] = mapped_column(ForeignKey("four01k_options.id", ondelete="CASCADE"))
    weight: Mapped[float] = mapped_column(Float)
    as_of: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    option = relationship("FourOhOneKOption")


class PortfolioSnapshot(Base):
    """Periodic portfolio value record used for drawdown / HWM tracking."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    portfolio_value: Mapped[float] = mapped_column(Float)


class JournalEntry(Base):
    """Decision journal — records investment decisions and their outcomes."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ticker: Mapped[str] = mapped_column(String(12), index=True)
    action: Mapped[str] = mapped_column(String(32))          # BUY / SELL / HOLD / REBALANCE
    rationale: Mapped[str] = mapped_column(Text, default="")
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_holding_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
