"""
GenAI ML Service - FastAPI Application

Main FastAPI application for the prediction service.

DESIGN RULES:
- FastAPI is a thin delivery layer
- All prediction logic delegated to PredictionService
- No business logic in API routes
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.config.settings import get_settings
from ml.store.model_store import ModelStore
from ml.store.prediction_store import PredictionStore
from ml.prediction.predictor import PredictionService
from app.api.predict import router as predict_router
from app.api.training import router as training_router


# Global service instances
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get the global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        settings = get_settings()
        model_store = ModelStore(settings.model_store_dir)
        prediction_store = PredictionStore(settings.prediction_store_dir)
        _prediction_service = PredictionService(
            model_store=model_store,
            prediction_store=prediction_store,
        )
    return _prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, cleanup on shutdown."""
    # Startup: ensure service is initialized
    _ = get_prediction_service()
    yield
    # Shutdown: cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="GenAI ML Service",
    description=(
        "Prediction service for the GenAI platform. "
        "Generates predictive signals (cost, quality, risk) from historical data. "
        "All outputs are METADATA ONLY - advisory and offline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, tags=["Prediction"])
app.include_router(training_router, tags=["Training"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Service information."""
    return {
        "service": "genai-ml-service",
        "version": "0.1.0",
        "purpose": "Prediction layer for GenAI platform",
        "note": "All outputs are metadata only - predictions do not influence execution",
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    service = get_prediction_service()
    models = service.has_models()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": models,
    }


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models and their versions."""
    service = get_prediction_service()
    versions = service.get_model_versions()
    available = service.has_models()
    
    return {
        "models": {
            model_type: {
                "available": available[model_type],
                "version": versions[model_type],
            }
            for model_type in ["cost", "quality", "risk"]
        }
    }
