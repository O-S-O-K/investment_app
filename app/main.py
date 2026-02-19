from fastapi import FastAPI

from app.db import Base, engine
from app.routers import analytics, portfolio

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Investment App API", version="0.1.0")

app.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
