"""
Unit tests for ML models.

Tests:
- Prediction determinism
- Model versioning
- Serialization/deserialization
"""

import pytest
from datetime import datetime

from ml.models.prediction import PredictionResult, InputWindow
from ml.models.dataset import DataWindow, TrainingDataset
from ml.models.cost_model import CostPredictionModel
from ml.models.quality_model import QualityPredictionModel
from ml.models.risk_model import RiskPredictionModel


class TestPredictionResult:
    """Tests for PredictionResult model."""
    
    def test_create_prediction_result(self):
        """Test creating a prediction result."""
        result = PredictionResult(
            prediction_type="cost",
            predicted_value=0.05,
            confidence=0.8,
            model_version="cost-v1-test",
            input_window=InputWindow(record_count=100),
        )
        
        assert result.prediction_type == "cost"
        assert result.predicted_value == 0.05
        assert result.confidence == 0.8
        assert result.model_version == "cost-v1-test"
        assert result.input_window.record_count == 100
    
    def test_prediction_result_immutable(self):
        """Test that PredictionResult is immutable."""
        result = PredictionResult(
            prediction_type="cost",
            predicted_value=0.05,
            confidence=0.8,
            model_version="cost-v1-test",
            input_window=InputWindow(),
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            result.predicted_value = 0.10  # type: ignore
    
    def test_prediction_result_serialization(self):
        """Test serialization/deserialization."""
        result = PredictionResult(
            prediction_type="quality",
            predicted_value=0.85,
            confidence=0.9,
            model_version="quality-v1-test",
            input_window=InputWindow(record_count=50),
            metadata={"agent_name": "general"},
        )
        
        data = result.to_dict()
        restored = PredictionResult.from_dict(data)
        
        assert restored.prediction_type == result.prediction_type
        assert restored.predicted_value == result.predicted_value
        assert restored.confidence == result.confidence
        assert restored.model_version == result.model_version


class TestCostPredictionModel:
    """Tests for CostPredictionModel."""
    
    def test_model_version(self):
        """Test that model has a version."""
        model = CostPredictionModel()
        assert model.model_version.startswith("cost-v1-")
        assert model.model_type == "cost"
    
    def test_prediction_without_training(self):
        """Test prediction without training (uses defaults)."""
        model = CostPredictionModel()
        
        result = model.predict({
            "agent_name": "general",
            "model": "gpt-4",
        })
        
        assert result.prediction_type == "cost"
        assert result.predicted_value > 0
        assert 0 <= result.confidence <= 1
        assert result.model_version == model.model_version
    
    def test_prediction_determinism(self):
        """Test that same inputs produce same outputs."""
        model = CostPredictionModel()
        
        features = {
            "agent_name": "general",
            "model": "gpt-4",
            "expected_tokens": 1000,
        }
        
        result1 = model.predict(features)
        result2 = model.predict(features)
        
        # Same model version and input should give same value
        assert result1.predicted_value == result2.predicted_value
        assert result1.confidence == result2.confidence
    
    def test_training(self):
        """Test model training."""
        model = CostPredictionModel()
        
        dataset = TrainingDataset(
            costs=[
                {"agent_name": "general", "model": "gpt-4", "total_tokens": 500, "estimated_cost_usd": 0.015},
                {"agent_name": "general", "model": "gpt-4", "total_tokens": 600, "estimated_cost_usd": 0.018},
                {"agent_name": "retrieval", "model": "gpt-3.5-turbo", "total_tokens": 300, "estimated_cost_usd": 0.0004},
            ],
        )
        
        model.train(dataset)
        
        assert model.trained_at is not None
        assert model.record_count == 3
        assert "general" in model.avg_tokens_by_agent
    
    def test_serialization(self):
        """Test model serialization/deserialization."""
        model = CostPredictionModel()
        model.train(TrainingDataset(
            costs=[
                {"agent_name": "test", "model": "gpt-4", "total_tokens": 500, "estimated_cost_usd": 0.015},
            ],
        ))
        
        data = model.to_dict()
        restored = CostPredictionModel.from_dict(data)
        
        assert restored.model_version == model.model_version
        assert restored.record_count == model.record_count


class TestQualityPredictionModel:
    """Tests for QualityPredictionModel."""
    
    def test_model_version(self):
        """Test that model has a version."""
        model = QualityPredictionModel()
        assert model.model_version.startswith("quality-v1-")
        assert model.model_type == "quality"
    
    def test_prediction_determinism(self):
        """Test that same inputs produce same outputs."""
        model = QualityPredictionModel()
        
        features = {"agent_name": "general"}
        
        result1 = model.predict(features)
        result2 = model.predict(features)
        
        assert result1.predicted_value == result2.predicted_value
    
    def test_training(self):
        """Test model training."""
        model = QualityPredictionModel()
        
        dataset = TrainingDataset(
            evaluations=[
                {"score": 0.9, "passed": True, "evaluator": "critic"},
                {"score": 0.8, "passed": True, "evaluator": "critic"},
                {"score": 0.5, "passed": False, "evaluator": "llm_judge"},
            ],
        )
        
        model.train(dataset)
        
        assert model.trained_at is not None
        assert model.record_count == 3
        assert 0.7 <= model.global_avg_score <= 0.8  # Average of 0.9, 0.8, 0.5


class TestRiskPredictionModel:
    """Tests for RiskPredictionModel."""
    
    def test_model_version(self):
        """Test that model has a version."""
        model = RiskPredictionModel()
        assert model.model_version.startswith("risk-v1-")
        assert model.model_type == "risk"
    
    def test_prediction_determinism(self):
        """Test that same inputs produce same outputs."""
        model = RiskPredictionModel()
        
        features = {"agent_name": "general", "tier": "standard"}
        
        result1 = model.predict(features)
        result2 = model.predict(features)
        
        assert result1.predicted_value == result2.predicted_value
    
    def test_tier_affects_risk(self):
        """Test that tier affects risk prediction."""
        model = RiskPredictionModel()
        model.global_violation_rate = 0.1  # Set some base risk
        
        free_result = model.predict({"tier": "free"})
        premium_result = model.predict({"tier": "premium"})
        
        # Free tier should have higher risk
        assert free_result.predicted_value >= premium_result.predicted_value
    
    def test_training(self):
        """Test model training."""
        model = RiskPredictionModel()
        
        dataset = TrainingDataset(
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
        
        model.train(dataset)
        
        assert model.trained_at is not None
        assert model.record_count == 3
        assert abs(model.global_violation_rate - 1/3) < 0.01  # 1 fail out of 3
