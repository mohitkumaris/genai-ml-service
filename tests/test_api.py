"""
API endpoint tests.

Tests:
- Prediction endpoints return expected structure
- Health endpoints work
"""

import pytest
from fastapi.testclient import TestClient

from ml.models.dataset import TrainingDataset
from ml.training.trainer import ModelTrainer
from ml.store.model_store import ModelStore
from ml.config.settings import get_settings


@pytest.fixture
def test_client(tmp_path):
    """Create a test client with temporary storage."""
    import os
    
    # Set up temp directories
    model_dir = tmp_path / "models"
    prediction_dir = tmp_path / "predictions"
    model_dir.mkdir()
    prediction_dir.mkdir()
    
    # Set environment variables
    os.environ["MODEL_STORE_DIR"] = str(model_dir)
    os.environ["PREDICTION_STORE_DIR"] = str(prediction_dir)
    
    # Reset global settings
    import ml.config.settings as settings_module
    settings_module._settings = None
    
    # Import app after setting env vars
    from app.api.main import app, _prediction_service
    import app.api.main as main_module
    main_module._prediction_service = None
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    os.environ.pop("MODEL_STORE_DIR", None)
    os.environ.pop("PREDICTION_STORE_DIR", None)
    settings_module._settings = None
    main_module._prediction_service = None


@pytest.fixture
def trained_client(test_client, tmp_path):
    """Create a test client with trained models."""
    import os
    
    # Get model store directory from env
    model_dir = os.environ.get("MODEL_STORE_DIR")
    
    # Train and save models
    trainer = ModelTrainer()
    dataset = TrainingDataset(
        costs=[
            {"agent_name": "general", "model": "gpt-4", "total_tokens": 500, "estimated_cost_usd": 0.015},
        ],
        evaluations=[
            {"score": 0.9, "passed": True, "evaluator": "critic"},
        ],
        policies=[
            {"status": "pass", "violations": [], "warnings": []},
        ],
    )
    
    models = trainer.train_all(dataset)
    store = ModelStore(model_dir)
    
    for model in models.values():
        store.save_model(model)
    
    # Reload prediction service
    import app.api.main as main_module
    main_module._prediction_service = None
    _ = main_module.get_prediction_service()
    
    return test_client


class TestHealthEndpoints:
    """Tests for health endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "genai-ml-service"
        assert "prediction" in data["purpose"].lower()
    
    def test_health_endpoint(self, test_client):
        """Test health endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_available" in data
    
    def test_models_endpoint(self, test_client):
        """Test models endpoint."""
        response = test_client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "cost" in data["models"]
        assert "quality" in data["models"]
        assert "risk" in data["models"]


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_cost_without_model(self, test_client):
        """Test cost prediction fails without trained model."""
        response = test_client.post(
            "/predict/cost",
            json={"agent_name": "general", "model": "gpt-4"},
        )
        
        assert response.status_code == 503
    
    def test_predict_cost_with_model(self, trained_client):
        """Test cost prediction with trained model."""
        response = trained_client.post(
            "/predict/cost",
            json={"agent_name": "general", "model": "gpt-4"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_type"] == "cost"
        assert "predicted_value" in data
        assert "confidence" in data
        assert "model_version" in data
    
    def test_predict_quality_with_model(self, trained_client):
        """Test quality prediction with trained model."""
        response = trained_client.post(
            "/predict/quality",
            json={"agent_name": "general"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_type"] == "quality"
    
    def test_predict_risk_with_model(self, trained_client):
        """Test risk prediction with trained model."""
        response = trained_client.post(
            "/predict/risk",
            json={"agent_name": "general", "tier": "standard"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_type"] == "risk"


class TestTrainingEndpoints:
    """Tests for training endpoints."""
    
    def test_training_status(self, test_client):
        """Test training status endpoint."""
        response = test_client.get("/train/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "cost" in data["models"]
