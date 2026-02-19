from fastapi import Depends, FastAPI

from app.db import Base, engine
from app.routers import analytics, journal, portfolio
from app.security import require_api_key

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Investment App API", version="0.1.0")

app.include_router(
    portfolio.router,
    prefix="/portfolio",
    tags=["portfolio"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    journal.router,
    prefix="/journal",
    tags=["journal"],
    dependencies=[Depends(require_api_key)],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
