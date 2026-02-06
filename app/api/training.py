"""
Training Endpoints

FastAPI routes for offline training operations.

DESIGN RULES:
- Training is OFFLINE only
- No real-time model updates
- Produces versioned models
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.models.dataset import DataWindow, TrainingDataset
from ml.training.trainer import ModelTrainer
from ml.store.model_store import ModelStore
from ml.features.llmops_reader import get_reader
from ml.config.settings import get_settings


router = APIRouter()


class TrainRequest(BaseModel):
    """Request for training a model."""
    record_count: Optional[int] = Field(None, description="Number of records to use for training")


class TrainResponse(BaseModel):
    """Response after training a model."""
    model_type: str
    model_version: str
    records_used: int
    message: str


def _train_and_save(model_type: str, record_count: Optional[int]) -> TrainResponse:
    """Common training logic."""
    settings = get_settings()
    
    # Get data from LLMOps
    reader = get_reader()
    window = DataWindow(record_count=record_count) if record_count else None
    dataset = reader.read_all(window)
    
    if dataset.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No training data available. Ensure LLMOps has data.",
        )
    
    # Train model
    trainer = ModelTrainer()
    
    if model_type == "cost":
        model = trainer.train_cost_model(dataset)
    elif model_type == "quality":
        model = trainer.train_quality_model(dataset)
    elif model_type == "risk":
        model = trainer.train_risk_model(dataset)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
    
    # Save model
    store = ModelStore(settings.model_store_dir)
    version = store.save_model(model)
    
    # Reload models in prediction service
    from app.api.main import get_prediction_service
    service = get_prediction_service()
    service.reload_models()
    
    return TrainResponse(
        model_type=model_type,
        model_version=version,
        records_used=dataset.total_records(),
        message=f"Successfully trained {model_type} model",
    )


@router.post("/train/cost", response_model=TrainResponse)
async def train_cost_model(request: TrainRequest) -> TrainResponse:
    """
    Train a cost prediction model.
    
    This is an OFFLINE operation that reads from LLMOps and produces
    a new versioned model.
    """
    return _train_and_save("cost", request.record_count)


@router.post("/train/quality", response_model=TrainResponse)
async def train_quality_model(request: TrainRequest) -> TrainResponse:
    """
    Train a quality prediction model.
    
    This is an OFFLINE operation that reads from LLMOps and produces
    a new versioned model.
    """
    return _train_and_save("quality", request.record_count)


@router.post("/train/risk", response_model=TrainResponse)
async def train_risk_model(request: TrainRequest) -> TrainResponse:
    """
    Train a risk prediction model.
    
    This is an OFFLINE operation that reads from LLMOps and produces
    a new versioned model.
    """
    return _train_and_save("risk", request.record_count)


@router.post("/train/all", response_model=Dict[str, TrainResponse])
async def train_all_models(request: TrainRequest) -> Dict[str, TrainResponse]:
    """
    Train all prediction models.
    
    This is an OFFLINE operation that reads from LLMOps and produces
    new versioned models for cost, quality, and risk.
    """
    settings = get_settings()
    
    # Get data from LLMOps
    reader = get_reader()
    window = DataWindow(record_count=request.record_count) if request.record_count else None
    dataset = reader.read_all(window)
    
    if dataset.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No training data available. Ensure LLMOps has data.",
        )
    
    # Train all models
    trainer = ModelTrainer()
    models = trainer.train_all(dataset)
    
    # Save models
    store = ModelStore(settings.model_store_dir)
    results = {}
    
    for model_type, model in models.items():
        version = store.save_model(model)
        results[model_type] = TrainResponse(
            model_type=model_type,
            model_version=version,
            records_used=dataset.total_records(),
            message=f"Successfully trained {model_type} model",
        )
    
    # Reload models in prediction service
    from app.api.main import get_prediction_service
    service = get_prediction_service()
    service.reload_models()
    
    return results


@router.get("/train/status")
async def training_status() -> Dict[str, Any]:
    """Get information about available models and their versions."""
    settings = get_settings()
    store = ModelStore(settings.model_store_dir)
    
    return {
        "models": {
            "cost": {
                "available": store.has_model("cost"),
                "latest_version": store.get_latest_version("cost"),
                "versions": store.list_versions("cost"),
            },
            "quality": {
                "available": store.has_model("quality"),
                "latest_version": store.get_latest_version("quality"),
                "versions": store.list_versions("quality"),
            },
            "risk": {
                "available": store.has_model("risk"),
                "latest_version": store.get_latest_version("risk"),
                "versions": store.list_versions("risk"),
            },
        }
    }
