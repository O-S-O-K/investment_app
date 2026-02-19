from datetime import datetime

from pydantic import BaseModel, Field


class HoldingCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    account_type: str = "taxable"
    shares: float = Field(gt=0)
    cost_basis: float = Field(gt=0)


class HoldingOut(HoldingCreate):
    id: int
    added_at: datetime

    class Config:
        from_attributes = True


class FourOhOneKOptionCreate(BaseModel):
    symbol: str
    name: str
    asset_class: str
    expense_ratio: float = 0.0


class FourOhOneKOptionOut(FourOhOneKOptionCreate):
    id: int

    class Config:
        from_attributes = True


class FourOhOneKAllocationCreate(BaseModel):
    option_id: int
    weight: float = Field(ge=0, le=1)


class FourOhOneKAllocationOut(BaseModel):
    id: int
    option_id: int
    weight: float
    as_of: datetime

    class Config:
        from_attributes = True


class SignalResponse(BaseModel):
    ticker: str
    momentum: float
    trend_strength: float
    tactical_score: float


class RecommendationItem(BaseModel):
    ticker: str
    strategic_weight: float
    tactical_tilt: float
    final_weight: float


class RecommendationResponse(BaseModel):
    generated_at: datetime
    expected_return: float
    expected_volatility: float
    notes: list[str]
    allocations: list[RecommendationItem]


class ImportCSVResponse(BaseModel):
    imported_rows: int
    created_holdings: int
    created_lots: int
    skipped_rows: int
    warnings: list[str]


class RebalanceRequest(BaseModel):
    target_weights: dict[str, float]
    short_term_tax_rate: float = Field(default=0.37, ge=0, le=1)
    long_term_tax_rate: float = Field(default=0.20, ge=0, le=1)
    min_trade_value: float = Field(default=25.0, ge=0)


class RebalanceTrade(BaseModel):
    ticker: str
    action: str
    shares: float
    estimated_trade_value: float
    estimated_tax_impact: float
    lot_id: int | None = None
    reason: str


class RebalancePlanResponse(BaseModel):
    generated_at: datetime
    portfolio_value: float
    estimated_total_tax_impact: float
    trades: list[RebalanceTrade]
    notes: list[str]


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class JournalEntryCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    action: str = Field(description="BUY / SELL / HOLD / REBALANCE")
    rationale: str = ""
    expected_return: float | None = None
    expected_holding_days: int | None = None


class JournalEntryUpdate(BaseModel):
    outcome: str | None = None
    closed_at: datetime | None = None


class JournalEntryOut(BaseModel):
    id: int
    created_at: datetime
    ticker: str
    action: str
    rationale: str
    expected_return: float | None
    expected_holding_days: int | None
    outcome: str | None
    closed_at: datetime | None

    class Config:
        from_attributes = True
