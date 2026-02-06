"""
Prediction Endpoints

FastAPI routes for prediction operations.

DESIGN RULES:
- Thin routing layer
- Delegates to PredictionService
- Returns structured responses
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()


# Request/Response Models

class CostPredictionRequest(BaseModel):
    """Request for cost prediction."""
    agent_name: str = Field(..., description="Name of the agent")
    model: str = Field(..., description="LLM model name (e.g., gpt-4)")
    expected_tokens: Optional[int] = Field(None, description="Expected token count (optional)")


class QualityPredictionRequest(BaseModel):
    """Request for quality prediction."""
    agent_name: str = Field(..., description="Name of the agent")
    evaluator: Optional[str] = Field(None, description="Evaluator type (optional)")


class RiskPredictionRequest(BaseModel):
    """Request for risk prediction."""
    agent_name: str = Field(..., description="Name of the agent")
    tier: str = Field("standard", description="SLA tier (free, standard, premium)")


class PredictionResponse(BaseModel):
    """Response containing a prediction."""
    prediction_id: str
    prediction_type: str
    predicted_value: float
    confidence: float
    model_version: str
    input_window: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


# Helper to get prediction service
def _get_service():
    from app.api.main import get_prediction_service
    return get_prediction_service()


@router.post("/predict/cost", response_model=PredictionResponse)
async def predict_cost(request: CostPredictionRequest) -> PredictionResponse:
    """
    Predict expected cost for a request.
    
    Returns predicted cost in USD based on historical patterns.
    
    NOTE: This is METADATA ONLY - does not affect execution.
    """
    service = _get_service()
    
    prediction = service.predict_cost(
        agent_name=request.agent_name,
        model=request.model,
        expected_tokens=request.expected_tokens,
    )
    
    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Cost prediction model not available. Train a model first.",
        )
    
    return PredictionResponse(**prediction.to_dict())


@router.post("/predict/quality", response_model=PredictionResponse)
async def predict_quality(request: QualityPredictionRequest) -> PredictionResponse:
    """
    Predict expected quality score for a request.
    
    Returns predicted quality score (0.0 to 1.0) based on historical evaluations.
    
    NOTE: This is METADATA ONLY - does not affect execution.
    """
    service = _get_service()
    
    prediction = service.predict_quality(
        agent_name=request.agent_name,
        evaluator=request.evaluator,
    )
    
    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Quality prediction model not available. Train a model first.",
        )
    
    return PredictionResponse(**prediction.to_dict())


@router.post("/predict/risk", response_model=PredictionResponse)
async def predict_risk(request: RiskPredictionRequest) -> PredictionResponse:
    """
    Predict policy violation risk for a request.
    
    Returns predicted risk score (0.0 to 1.0) based on historical policy outcomes.
    
    NOTE: This is METADATA ONLY - does not affect execution.
    """
    service = _get_service()
    
    prediction = service.predict_risk(
        agent_name=request.agent_name,
        tier=request.tier,
    )
    
    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Risk prediction model not available. Train a model first.",
        )
    
    return PredictionResponse(**prediction.to_dict())


@router.get("/predictions")
async def query_predictions(
    prediction_type: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Query prediction history.
    
    Returns recent predictions from the append-only store.
    """
    from ml.store.prediction_store import PredictionStore
    from ml.config.settings import get_settings
    
    settings = get_settings()
    store = PredictionStore(settings.prediction_store_dir)
    
    predictions = store.query(
        prediction_type=prediction_type,
        limit=limit,
    )
    
    return {
        "count": len(predictions),
        "predictions": [p.to_dict() for p in predictions],
    }
