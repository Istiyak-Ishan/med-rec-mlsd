"""
Medicine Recommendation System — FastAPI Application
Run with: uvicorn src.serving.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Required so joblib/pickle can deserialize the MedicineRecommender class.
# The model was saved while train.py ran as __main__, so pickle recorded the
# class as __main__.MedicineRecommender.  Importing it here under its real
# module path and aliasing it into sys.modules under __main__ fixes the lookup.
import sys
from src.models.train import MedicineRecommender as _MedicineRecommender  # noqa: E402
import types as _types

_fake_main = sys.modules.get("__main__")
if _fake_main is not None and not hasattr(_fake_main, "MedicineRecommender"):
    _fake_main.MedicineRecommender = _MedicineRecommender

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/recommender.pkl")
# The recommender is expected to expose a `.medicines_df` attribute (a
# DataFrame) or a `.data_path` attribute pointing to the processed CSV.
# We'll try both strategies when looking up individual medicines.
PROCESSED_CSV_GLOB = [
    Path("data/processed/medeasy_processed.csv"),   # canonical path from train.py
    Path("data/processed/medicines.csv"),            # fallback aliases
    Path("data/processed/medicine_data.csv"),
]

# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------
class AppState:
    recommender: Any = None
    medicines_df: pd.DataFrame | None = None


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — load model & data once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Loading recommender from %s …", MODEL_PATH)
    if MODEL_PATH.exists():
        try:
            state.recommender = joblib.load(MODEL_PATH)
            logger.info("Recommender loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load recommender: %s", exc)
    else:
        logger.warning("Model file not found at %s — endpoints will return errors.", MODEL_PATH)

    # Try to obtain a medicines DataFrame for the /medicines/{name} endpoint.
    # Priority 1: attribute on the recommender itself.
    if state.recommender is not None and hasattr(state.recommender, "medicines_df"):
        state.medicines_df = state.recommender.medicines_df
        logger.info("Medicines DataFrame loaded from recommender (%d rows).", len(state.medicines_df))
    else:
        # Priority 2: well-known CSV paths.
        for csv_path in PROCESSED_CSV_GLOB:
            if csv_path.exists():
                state.medicines_df = pd.read_csv(csv_path)
                logger.info("Medicines DataFrame loaded from %s (%d rows).", csv_path, len(state.medicines_df))
                break
        else:
            logger.warning("No processed CSV found — /medicines/{medicine_name} will return 404 for all names.")

    yield  # ── Application runs ─────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources.")
    state.recommender = None
    state.medicines_df = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Medicine Recommendation System",
    description="AI-powered medicine recommender — for informational use only.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request-logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "This system is for informational purposes only. "
    "Always consult a qualified physician before taking any medication."
)


class MedicineResult(BaseModel):
    medicine_name: str
    generic_name: str
    dosage_form: str
    manufacturer: str
    mrp: float
    discounted_price: float
    similarity_score: float


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["fever and headache"])
    top_k: int = Field(10, ge=1, le=50)


class RecommendResponse(BaseModel):
    query: str
    results: list[MedicineResult]
    count: int
    disclaimer: str


class AlternativesRequest(BaseModel):
    medicine_name: str = Field(..., min_length=1, examples=["Paracetamol 500mg Tablet"])
    top_k: int = Field(5, ge=1, le=50)


class AlternativesResponse(BaseModel):
    medicine_name: str
    alternatives: list[MedicineResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_medicines: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _assert_model_loaded() -> None:
    if state.recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender model is not loaded. Check server logs.",
        )


def _coerce_results(raw: list[dict]) -> list[MedicineResult]:
    """Safely coerce raw dicts returned by the recommender into MedicineResult."""
    return [MedicineResult(**item) for item in raw]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# 1. Health check
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns service health and basic statistics."""
    total = len(state.medicines_df) if state.medicines_df is not None else 0
    return HealthResponse(
        status="ok",
        model_loaded=state.recommender is not None,
        total_medicines=total,
    )


# 2. Symptom / query-based recommendation
@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(body: RecommendRequest) -> RecommendResponse:
    """Return medicines relevant to a free-text query (symptoms, conditions, etc.)."""
    _assert_model_loaded()
    try:
        raw: list[dict] = state.recommender.recommend_by_query(
            query_text=body.query,
            top_k=body.top_k,
        )
        results = _coerce_results(raw)
        return RecommendResponse(
            query=body.query,
            results=results,
            count=len(results),
            disclaimer=DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in /recommend: %s", exc)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc


# 3. Alternative medicines for a given medicine name
@app.post("/alternatives", response_model=AlternativesResponse, tags=["Recommendations"])
async def alternatives(body: AlternativesRequest) -> AlternativesResponse:
    """Return alternative medicines for a given medicine name."""
    _assert_model_loaded()
    try:
        raw: list[dict] = state.recommender.recommend_alternatives(
            medicine_name=body.medicine_name,
            top_k=body.top_k,
        )
        # Treat an empty result as "not found".
        if raw is None or len(raw) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Medicine '{body.medicine_name}' not found in the database.",
            )
        alternatives_list = _coerce_results(raw)
        return AlternativesResponse(
            medicine_name=body.medicine_name,
            alternatives=alternatives_list,
            count=len(alternatives_list),
        )
    except HTTPException:
        raise
    except KeyError as exc:
        # recommend_alternatives raises KeyError when the name is not in the index
        raise HTTPException(
            status_code=404,
            detail=f"Medicine '{body.medicine_name}' not found in the database.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in /alternatives: %s", exc)
        raise HTTPException(status_code=500, detail=f"Alternatives lookup failed: {exc}") from exc


# 4. Medicine detail lookup
@app.get("/medicines/{medicine_name}", tags=["Medicines"])
async def get_medicine(medicine_name: str) -> dict[str, Any]:
    """Return all columns for a medicine looked up by name (case-insensitive)."""
    if state.medicines_df is None:
        raise HTTPException(
            status_code=503,
            detail="Medicine data is not available. Check server logs.",
        )

    # Case-insensitive search against the first column that looks like a name.
    name_cols = [
        c for c in state.medicines_df.columns
        if "name" in c.lower() or "medicine" in c.lower()
    ]
    if not name_cols:
        # Fall back to the first column.
        name_cols = [state.medicines_df.columns[0]]

    mask = pd.Series([False] * len(state.medicines_df))
    for col in name_cols:
        mask = mask | (
            state.medicines_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            == medicine_name.strip().lower()
        )

    matches = state.medicines_df[mask]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Medicine '{medicine_name}' not found.",
        )

    # Return the first match; NaN → None for clean JSON.
    row = matches.iloc[0].where(matches.iloc[0].notna(), other=None)
    return row.to_dict()