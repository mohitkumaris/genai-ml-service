"""
LLMOps Integration Compliance Tests

Tests proving architectural invariants:
1. Pull-only: No HTTP methods other than GET
2. Fail-open: LLMOps unavailable → empty datasets
3. Determinism: Same input → same prediction
4. No execution coupling: No orchestrator imports
5. Read-only: No mutation methods
"""

import ast
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml.features.llmops_reader import (
    LLMOpsReader,
    LLMOpsAPIReader,
    LLMOpsFileReader,
    InMemoryReader,
    get_reader,
)
from ml.features.extractors import (
    CostFeatureExtractor,
    QualityFeatureExtractor,
    RiskFeatureExtractor,
)
from ml.models.dataset import TrainingDataset
from ml.training.trainer import ModelTrainer


class TestPullOnlyCompliance:
    """Verify ML service only uses HTTP GET (read-only)."""
    
    def test_llmops_reader_uses_get_only(self):
        """Verify LLMOpsAPIReader only uses requests.get."""
        # Read the source file
        reader_path = Path(__file__).parent.parent / "ml" / "features" / "llmops_reader.py"
        source = reader_path.read_text()
        
        # Parse AST to find all method calls
        tree = ast.parse(source)
        
        forbidden_methods = ["post", "put", "patch", "delete"]
        found_forbidden = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_methods:
                    found_forbidden.append(node.attr)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in forbidden_methods:
                        found_forbidden.append(node.func.attr)
        
        assert len(found_forbidden) == 0, (
            f"Found forbidden HTTP methods in llmops_reader.py: {found_forbidden}. "
            "Only GET is allowed for read-only access."
        )
    
    def test_no_write_endpoints_in_reader(self):
        """Verify no POST/PUT/PATCH/DELETE URLs or methods defined."""
        reader_path = Path(__file__).parent.parent / "ml" / "features" / "llmops_reader.py"
        source = reader_path.read_text()
        
        # Check for any indication of write operations
        forbidden_patterns = [
            "requests.post",
            "requests.put",
            "requests.patch",
            "requests.delete",
            "/ingest",  # LLMOps ingest endpoints
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source.lower(), (
                f"Found forbidden pattern '{pattern}' in llmops_reader.py. "
                "ML service must be read-only."
            )


class TestFailOpenCompliance:
    """Verify ML service fails open when LLMOps is unavailable."""
    
    def test_api_reader_returns_empty_on_connection_error(self):
        """Test that connection errors return empty list, not crash."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()
            
            reader = LLMOpsAPIReader("http://nonexistent:9999")
            
            # All methods should return empty list, not raise
            assert reader.read_traces() == []
            assert reader.read_costs() == []
            assert reader.read_evaluations() == []
            assert reader.read_policies() == []
            assert reader.read_slas() == []
    
    def test_api_reader_returns_empty_on_timeout(self):
        """Test that timeout returns empty list, not crash."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout()
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
            assert reader.read_costs() == []
    
    def test_api_reader_returns_empty_on_http_error(self):
        """Test that HTTP errors return empty list, not crash."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
    
    def test_dataset_from_unavailable_llmops_is_empty(self):
        """Test that full dataset read returns empty on failure."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            dataset = reader.read_all()
            
            assert dataset.is_empty()
            assert dataset.total_records() == 0


class TestLLMOpsDisabledMode:
    """Verify ML service works correctly when LLMOPS_ENABLED=false."""
    
    def test_api_reader_returns_empty_when_disabled(self):
        """Test that disabled LLMOps returns empty without making requests."""
        # Save original settings
        import ml.config.settings as settings_module
        original_settings = settings_module._settings
        
        try:
            # Reset and configure disabled
            settings_module._settings = None
            os.environ["LLMOPS_ENABLED"] = "false"
            
            with patch("ml.features.llmops_reader.requests.get") as mock_get:
                reader = LLMOpsAPIReader("http://localhost:8100")
                
                result = reader.read_traces()
                
                # Should return empty without calling requests
                assert result == []
                mock_get.assert_not_called()
        finally:
            # Restore
            os.environ.pop("LLMOPS_ENABLED", None)
            settings_module._settings = original_settings
    
    def test_dataset_empty_when_disabled(self):
        """Test full dataset is empty when disabled."""
        import ml.config.settings as settings_module
        original_settings = settings_module._settings
        
        try:
            settings_module._settings = None
            os.environ["LLMOPS_ENABLED"] = "false"
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            dataset = reader.read_all()
            
            assert dataset.is_empty()
        finally:
            os.environ.pop("LLMOPS_ENABLED", None)
            settings_module._settings = original_settings


class TestTimeoutConfiguration:
    """Verify timeout is configurable and respects settings."""
    
    def test_uses_configured_timeout(self):
        """Test that reader uses LLMOPS_TIMEOUT_MS setting."""
        import ml.config.settings as settings_module
        original_settings = settings_module._settings
        
        try:
            settings_module._settings = None
            os.environ["LLMOPS_TIMEOUT_MS"] = "500"  # 500ms
            
            with patch("ml.features.llmops_reader.requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.json.return_value = []
                mock_get.return_value = mock_response
                
                reader = LLMOpsAPIReader("http://localhost:8100")
                reader.read_traces()
                
                # Verify timeout was set to 0.5 seconds
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args[1]
                assert call_kwargs["timeout"] == 0.5
        finally:
            os.environ.pop("LLMOPS_TIMEOUT_MS", None)
            settings_module._settings = original_settings


class TestDeterminismCompliance:
    """Verify predictions are deterministic."""
    
    def test_same_features_same_prediction(self):
        """Test that identical input produces identical output."""
        extractor = CostFeatureExtractor()
        
        costs = [
            {"agent_name": "general", "model": "gpt-4", "total_tokens": 500, 
             "input_tokens": 300, "output_tokens": 200, "estimated_cost_usd": 0.015},
        ]
        
        # Run extraction twice
        features1 = extractor.extract_from_costs(costs)
        features2 = extractor.extract_from_costs(costs)
        
        # Must be identical
        assert features1 == features2
    
    def test_quality_extractor_deterministic(self):
        """Test quality extractor is deterministic."""
        extractor = QualityFeatureExtractor()
        
        evals = [
            {"score": 0.9, "passed": True, "evaluator": "critic"},
            {"score": 0.7, "passed": True, "evaluator": "llm_judge"},
        ]
        
        features1 = extractor.extract_from_evaluations(evals)
        features2 = extractor.extract_from_evaluations(evals)
        
        assert features1 == features2
    
    def test_risk_extractor_deterministic(self):
        """Test risk extractor is deterministic."""
        extractor = RiskFeatureExtractor()
        
        policies = [
            {"status": "pass", "violations": [], "warnings": []},
            {"status": "fail", "violations": ["cost_limit"], "warnings": []},
        ]
        
        features1 = extractor.extract_from_policies(policies)
        features2 = extractor.extract_from_policies(policies)
        
        assert features1 == features2


class TestNoExecutionCoupling:
    """Verify ML service has no coupling to execution/orchestrator."""
    
    def test_no_orchestrator_imports(self):
        """Verify no imports from orchestrator in ML codebase."""
        ml_dir = Path(__file__).parent.parent / "ml"
        
        forbidden_imports = [
            "orchestrator",
            "langchain",
            "openai",
            "agent_orchestrator",
            "genai_agent_orchestrator",
        ]
        
        violations = []
        
        for py_file in ml_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text()
            
            for forbidden in forbidden_imports:
                if f"import {forbidden}" in source or f"from {forbidden}" in source:
                    violations.append(f"{py_file.name}: imports {forbidden}")
        
        assert len(violations) == 0, (
            f"Found forbidden imports in ML codebase: {violations}. "
            "ML service must not import orchestrator or LLM SDKs."
        )
    
    def test_no_llm_sdk_dependencies(self):
        """Verify ML models don't depend on LLM SDKs."""
        models_dir = Path(__file__).parent.parent / "ml" / "models"
        
        if not models_dir.exists():
            pytest.skip("Models directory not found")
        
        forbidden = ["langchain", "openai", "anthropic", "azure.openai"]
        
        for py_file in models_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text()
            
            for lib in forbidden:
                assert lib not in source, (
                    f"Found forbidden library '{lib}' in {py_file.name}. "
                    "ML models must not use LLM SDKs."
                )


class TestReadOnlyCompliance:
    """Verify all data access is read-only."""
    
    def test_training_dataset_immutable_source(self):
        """Test that TrainingDataset doesn't mutate source data."""
        original_costs = [{"agent_name": "test", "total_tokens": 100}]
        costs_copy = original_costs.copy()
        
        dataset = TrainingDataset(costs=original_costs)
        
        # Dataset should not have modified original
        assert original_costs == costs_copy
    
    def test_in_memory_reader_returns_copies(self):
        """Test InMemoryReader returns copies, not references."""
        original = [{"key": "value"}]
        reader = InMemoryReader(traces=original)
        
        result = reader.read_traces()
        
        # Modifying result should not affect original
        result.append({"new": "data"})
        
        assert len(original) == 1
        assert len(reader.read_traces()) == 1


class TestWrappedResponseHandling:
    """Test handling of LLMOps wrapped response format."""
    
    def test_handles_wrapped_response(self):
        """Test reader extracts 'data' from wrapped response."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "meta": {"count": 1, "limit": 100, "has_more": False},
                "data": [{"trace_id": "123", "agent_name": "general"}]
            }
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            result = reader.read_traces()
            
            assert len(result) == 1
            assert result[0]["trace_id"] == "123"
    
    def test_handles_raw_list_response(self):
        """Test reader handles raw list response (backward compatibility)."""
        with patch("ml.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {"trace_id": "123", "agent_name": "general"}
            ]
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            result = reader.read_traces()
            
            assert len(result) == 1
            assert result[0]["trace_id"] == "123"
