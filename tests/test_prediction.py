"""
Integration tests for prediction service.

Tests:
- End-to-end prediction flow
- Model loading and prediction
"""

import pytest
from datetime import datetime

from ml.models.dataset import TrainingDataset
from ml.training.trainer import ModelTrainer
from ml.prediction.predictor import PredictionService
from ml.store.model_store import InMemoryModelStore
from ml.store.prediction_store import InMemoryPredictionStore


class TestPredictionService:
    """Tests for PredictionService."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample training dataset."""
        return TrainingDataset(
            costs=[
                {"agent_name": "general", "model": "gpt-4", "total_tokens": 500, "estimated_cost_usd": 0.015},
                {"agent_name": "general", "model": "gpt-4", "total_tokens": 600, "estimated_cost_usd": 0.018},
                {"agent_name": "retrieval", "model": "gpt-3.5-turbo", "total_tokens": 300, "estimated_cost_usd": 0.0004},
            ],
            evaluations=[
                {"score": 0.9, "passed": True, "evaluator": "critic"},
                {"score": 0.8, "passed": True, "evaluator": "critic"},
                {"score": 0.5, "passed": False, "evaluator": "llm_judge"},
            ],
            policies=[
                {"status": "pass", "violations": [], "warnings": []},
                {"status": "pass", "violations": [], "warnings": []},
                {"status": "fail", "violations": ["cost_limit"], "warnings": []},
            ],
            slas=[
                {"tier": "free"},
                {"tier": "standard"},
                {"tier": "premium"},
            ],
        )
    
    @pytest.fixture
    def trained_service(self, sample_dataset):
        """Create a prediction service with trained models."""
        model_store = InMemoryModelStore()
        prediction_store = InMemoryPredictionStore()
        
        # Train and save models
        trainer = ModelTrainer()
        models = trainer.train_all(sample_dataset)
        
        for model in models.values():
            model_store.save_model(model)
        
        return PredictionService(
            model_store=model_store,
            prediction_store=prediction_store,
            persist_predictions=True,
        )
    
    def test_predict_cost(self, trained_service):
        """Test cost prediction."""
        result = trained_service.predict_cost(
            agent_name="general",
            model="gpt-4",
        )
        
        assert result is not None
        assert result.prediction_type == "cost"
        assert result.predicted_value > 0
        assert 0 <= result.confidence <= 1
    
    def test_predict_quality(self, trained_service):
        """Test quality prediction."""
        result = trained_service.predict_quality(
            agent_name="general",
        )
        
        assert result is not None
        assert result.prediction_type == "quality"
        assert 0 <= result.predicted_value <= 1
    
    def test_predict_risk(self, trained_service):
        """Test risk prediction."""
        result = trained_service.predict_risk(
            agent_name="general",
            tier="standard",
        )
        
        assert result is not None
        assert result.prediction_type == "risk"
        assert 0 <= result.predicted_value <= 1
    
    def test_predictions_are_persisted(self, trained_service):
        """Test that predictions are saved to store."""
        # Make some predictions
        trained_service.predict_cost(agent_name="test", model="gpt-4")
        trained_service.predict_quality(agent_name="test")
        trained_service.predict_risk(agent_name="test", tier="free")
        
        # Check prediction store
        count = trained_service.prediction_store.count()
        assert count == 3
    
    def test_has_models(self, trained_service):
        """Test checking model availability."""
        models = trained_service.has_models()
        
        assert models["cost"] is True
        assert models["quality"] is True
        assert models["risk"] is True
    
    def test_prediction_without_models(self):
        """Test that prediction returns None without models."""
        model_store = InMemoryModelStore()
        prediction_store = InMemoryPredictionStore()
        
        service = PredictionService(
            model_store=model_store,
            prediction_store=prediction_store,
        )
        
        result = service.predict_cost(agent_name="test", model="gpt-4")
        assert result is None
    
    def test_prediction_determinism(self, trained_service):
        """Test that same inputs produce same outputs."""
        result1 = trained_service.predict_cost(
            agent_name="general",
            model="gpt-4",
            expected_tokens=500,
        )
        result2 = trained_service.predict_cost(
            agent_name="general",
            model="gpt-4",
            expected_tokens=500,
        )
        
        assert result1.predicted_value == result2.predicted_value
        assert result1.confidence == result2.confidence


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    def test_train_all(self):
        """Test training all models."""
        trainer = ModelTrainer()
        
        dataset = TrainingDataset(
            costs=[{"agent_name": "test", "model": "gpt-4", "total_tokens": 500, "estimated_cost_usd": 0.015}],
            evaluations=[{"score": 0.9, "passed": True, "evaluator": "critic"}],
            policies=[{"status": "pass", "violations": [], "warnings": []}],
        )
        
        models = trainer.train_all(dataset)
        
        assert "cost" in models
        assert "quality" in models
        assert "risk" in models
        
        for model in models.values():
            assert model.trained_at is not None
