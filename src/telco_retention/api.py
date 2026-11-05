"""FastAPI service: /health, /churn/score, /tickets/classify (D-025).

Artifacts are loaded once at start-up from the artifacts directory (`TELCO_ARTIFACTS` overrides the config):
`churn_model.joblib` (`telco train`) and `ticket_models.joblib` (`telco package-tickets`). A missing artifact turns
its endpoint into 503 and /health into "degraded", so the service still starts.

Run: `uvicorn --factory telco_retention.api:create_app --host 0.0.0.0 --port 8000` (Dockerfile).
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from telco_retention.config import Config, load_config
from telco_retention.scoring import prepare, score

log = logging.getLogger("telco.api")
YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer in IBM's vocabulary (the values of the public Telco file)."""

    customer_id: str | None = None
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="months with the company")
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: AddOn
    online_backup: AddOn
    device_protection: AddOn
    tech_support: AddOn
    streaming_tv: AddOn
    streaming_movies: AddOn
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(gt=0, le=1000)
    cltv: float | None = Field(default=None, ge=0, description="customer lifetime value, if known")


class ChurnRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class ChurnScore(BaseModel):
    customer_id: str | None
    p_churn: float
    value: float
    value_source: str
    expected_value: float
    contact: bool = Field(description="expected value of an offer is positive")
    top_drivers: list[str]
    message: str


class TicketRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=500)


class TicketLabel(BaseModel):
    clean_text: str
    theme_code: int
    theme: str
    theme_probability: float
    sentiment: Literal["neg", "neu", "pos"]
    urgency: Literal["high", "normal"]
    p_urgent: float


def _load(path: Path) -> Any:
    if not path.exists():
        log.warning("artifact missing: %s", path)
        return None
    return joblib.load(path)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    artifacts = Path(os.environ.get("TELCO_ARTIFACTS") or cfg.path("artifacts_dir"))
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["churn"] = _load(artifacts / "churn_model.joblib")
        state["tickets"] = _load(artifacts / "ticket_models.joblib")
        yield

    app = FastAPI(title="Telco retention", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        churn, tickets = state.get("churn"), state.get("tickets")
        return {
            "status": "ok" if churn is not None and tickets is not None else "degraded",
            "churn_model": None if churn is None else f"{churn['name']} ({churn['calibrator'].method} calibration)",
            "ticket_models": None if tickets is None else tickets["trained_on"],
        }

    @app.post("/churn/score")
    def churn_score(req: ChurnRequest) -> list[ChurnScore]:
        bundle = state.get("churn")
        if bundle is None:
            raise HTTPException(503, "churn model not loaded")
        records = pd.DataFrame([c.model_dump() for c in req.customers])
        ids = records.pop("customer_id")
        records["customer_id"] = ids.where(ids.notna(), None)
        scored = score(prepare(records), bundle, cfg)
        out = []
        for r in scored.to_dict("records"):
            out.append(
                ChurnScore(
                    customer_id=r["customer_id"],
                    p_churn=float(r["p_churn"]),
                    value=float(r["value"]),
                    value_source=str(r["value_source"]),
                    expected_value=float(r["ev"]),
                    contact=float(r["ev"]) > 0,
                    top_drivers=[d for d in str(r["top_drivers"]).split("; ") if d],
                    message=str(r["message"]),
                )
            )
        return out

    @app.post("/tickets/classify")
    def tickets_classify(req: TicketRequest) -> list[TicketLabel]:
        from telco_retention.nlp.tickets import classify_texts

        bundle = state.get("tickets")
        if bundle is None:
            raise HTTPException(503, "ticket models not loaded")
        return [TicketLabel(**r) for r in classify_texts(req.texts, bundle)]

    return app
