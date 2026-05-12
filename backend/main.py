"""
main.py
FastAPI application – Fraud Detection System
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.fraud_model import analyze_transactions
from backend.utils import parse_csv_bytes, summarise, validate_single

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Fraud Detection API is starting up …")
    yield
    logger.info("🛑 Fraud Detection API is shutting down …")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Detect suspicious financial transactions using Z-score analysis "
        "and Isolation Forest machine learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=1, examples=["TXN001"])
    amount: float = Field(..., gt=0, examples=[250.00])

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return v


class BatchRequest(BaseModel):
    transactions: List[TransactionIn] = Field(..., min_length=1)


class TransactionResult(BaseModel):
    transaction_id: str
    amount: float
    z_score: float
    anomaly_flag: bool
    risk_score: float
    risk_level: str


class BatchResponse(BaseModel):
    results: List[TransactionResult]
    summary: dict


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Fraud Detection API v1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ──────────────────────────────────────────────
# POST /predict  – JSON body
# ──────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze transactions (JSON)",
    tags=["Prediction"],
)
def predict_json(payload: BatchRequest):
    """
    Accept a JSON array of transactions and return fraud-detection results.

    **Example request body**
    ```json
    {
      "transactions": [
        {"transaction_id": "TXN001", "amount": 150.0},
        {"transaction_id": "TXN002", "amount": 9500.0}
      ]
    }
    ```
    """
    logger.info("POST /predict (JSON) — %d transactions", len(payload.transactions))
    try:
        records = [t.model_dump() for t in payload.transactions]
        results = analyze_transactions(records)
        return BatchResponse(results=results, summary=summarise(results))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────
# POST /predict/csv  – file upload
# ──────────────────────────────────────────────

@app.post(
    "/predict/csv",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze transactions (CSV upload)",
    tags=["Prediction"],
)
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns `transaction_id` and `amount`.
    Returns fraud-detection results for every row.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only .csv files are accepted.",
        )

    content = await file.read()
    logger.info("POST /predict/csv — file: %s (%d bytes)", file.filename, len(content))

    try:
        records = parse_csv_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        results = analyze_transactions(records)
        return BatchResponse(results=results, summary=summarise(results))
    except Exception as exc:
        logger.exception("Prediction error (CSV)")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────
# POST /predict/single  – convenience endpoint
# ──────────────────────────────────────────────

@app.post(
    "/predict/single",
    response_model=TransactionResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze a single transaction",
    tags=["Prediction"],
)
def predict_single(transaction: TransactionIn):
    """
    Convenience endpoint to check one transaction.
    *Note*: Z-score and Isolation Forest are computed on a dataset
    of one point, so anomaly detection is less reliable here.
    Use `/predict` for batch analysis.
    """
    logger.info(
        "POST /predict/single — id=%s amount=%.2f",
        transaction.transaction_id,
        transaction.amount,
    )
    try:
        validate_single(transaction.transaction_id, transaction.amount)
        results = analyze_transactions([transaction.model_dump()])
        return results[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error (single)")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────
# Dev runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
