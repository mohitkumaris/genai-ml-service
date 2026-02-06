"""
Prediction Service

Main prediction service with versioning.

DESIGN RULES:
- Uses trained models for prediction
- Deterministic outputs
- All predictions are metadata only
- No execution influence
"""

from typing import Any, Dict, List, Optional

from ml.models.prediction import PredictionResult
from ml.models.base import BasePredictionModel
from ml.store.model_store import ModelStore
from ml.store.prediction_store import PredictionStore


class PredictionService:
    """
    Main prediction service.
    
    Loads models from store and generates predictions.
    All predictions are persisted in append-only storage.
    
    IMPORTANT: This service ONLY produces metadata.
    It does NOT influence execution in any way.
    """
    
    def __init__(
        self,
        model_store: ModelStore,
        prediction_store: PredictionStore,
        persist_predictions: bool = True,
    ):
        self.model_store = model_store
        self.prediction_store = prediction_store
        self.persist_predictions = persist_predictions
        
        # Loaded models cache
        self._models: Dict[str, BasePredictionModel] = {}
    
    def load_model(self, model_type: str, version: Optional[str] = None) -> Optional[BasePredictionModel]:
        """Load a model (caches for reuse)."""
        cache_key = f"{model_type}:{version or 'latest'}"
        
        if cache_key not in self._models:
            model = self.model_store.load_model(model_type, version)
            if model:
                self._models[cache_key] = model
        
        return self._models.get(cache_key)
    
    def reload_models(self) -> None:
        """Clear model cache to reload from store."""
        self._models.clear()
    
    def predict_cost(
        self,
        agent_name: str,
        model: str,
        expected_tokens: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> Optional[PredictionResult]:
        """
        Predict expected cost.
        
        Returns PredictionResult with predicted_value = estimated USD cost.
        """
        cost_model = self.load_model("cost", model_version)
        if cost_model is None:
            return None
        
        features = {
            "agent_name": agent_name,
            "model": model,
            "expected_tokens": expected_tokens,
        }
        
        prediction = cost_model.predict(features)
        
        if self.persist_predictions:
            self.prediction_store.append(prediction)
        
        return prediction
    
    def predict_quality(
        self,
        agent_name: str,
        evaluator: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Optional[PredictionResult]:
        """
        Predict expected quality score.
        
        Returns PredictionResult with predicted_value = quality score (0.0 to 1.0).
        """
        quality_model = self.load_model("quality", model_version)
        if quality_model is None:
            return None
        
        features = {
            "agent_name": agent_name,
            "evaluator": evaluator,
        }
        
        prediction = quality_model.predict(features)
        
        if self.persist_predictions:
            self.prediction_store.append(prediction)
        
        return prediction
    
    def predict_risk(
        self,
        agent_name: str,
        tier: str = "standard",
        model_version: Optional[str] = None,
    ) -> Optional[PredictionResult]:
        """
        Predict policy violation risk.
        
        Returns PredictionResult with predicted_value = risk score (0.0 to 1.0).
        """
        risk_model = self.load_model("risk", model_version)
        if risk_model is None:
            return None
        
        features = {
            "agent_name": agent_name,
            "tier": tier,
        }
        
        prediction = risk_model.predict(features)
        
        if self.persist_predictions:
            self.prediction_store.append(prediction)
        
        return prediction
    
    def has_models(self) -> Dict[str, bool]:
        """Check which model types are available."""
        return {
            "cost": self.model_store.has_model("cost"),
            "quality": self.model_store.has_model("quality"),
            "risk": self.model_store.has_model("risk"),
        }
    
    def get_model_versions(self) -> Dict[str, Optional[str]]:
        """Get the latest version for each model type."""
        return {
            "cost": self.model_store.get_latest_version("cost"),
            "quality": self.model_store.get_latest_version("quality"),
            "risk": self.model_store.get_latest_version("risk"),
        }
