"""
Model Trainer

Offline training orchestration for ML models.

DESIGN RULES:
- Training is OFFLINE only
- No real-time updates
- Produces versioned models
"""

from typing import Dict

from ml.models.base import BasePredictionModel
from ml.models.cost_model import CostPredictionModel
from ml.models.quality_model import QualityPredictionModel
from ml.models.risk_model import RiskPredictionModel
from ml.models.dataset import TrainingDataset


class ModelTrainer:
    """
    Orchestrates offline training of ML models.
    
    Training does not affect any running predictions.
    New models are stored with new versions.
    """
    
    def train_cost_model(self, dataset: TrainingDataset) -> CostPredictionModel:
        """
        Train a cost prediction model.
        
        Returns a new versioned model.
        """
        model = CostPredictionModel()
        model.train(dataset)
        return model
    
    def train_quality_model(self, dataset: TrainingDataset) -> QualityPredictionModel:
        """
        Train a quality prediction model.
        
        Returns a new versioned model.
        """
        model = QualityPredictionModel()
        model.train(dataset)
        return model
    
    def train_risk_model(self, dataset: TrainingDataset) -> RiskPredictionModel:
        """
        Train a risk prediction model.
        
        Returns a new versioned model.
        """
        model = RiskPredictionModel()
        model.train(dataset)
        return model
    
    def train_all(self, dataset: TrainingDataset) -> Dict[str, BasePredictionModel]:
        """
        Train all model types.
        
        Returns a dictionary of model_type -> model.
        """
        return {
            "cost": self.train_cost_model(dataset),
            "quality": self.train_quality_model(dataset),
            "risk": self.train_risk_model(dataset),
        }
