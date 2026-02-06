"""
Unit tests for feature extractors.

Tests:
- Feature extraction from sample data
- Edge cases (empty data, missing fields)
"""

import pytest

from ml.features.extractors import (
    CostFeatureExtractor,
    QualityFeatureExtractor,
    RiskFeatureExtractor,
    LatencyFeatureExtractor,
)


class TestCostFeatureExtractor:
    """Tests for CostFeatureExtractor."""
    
    def test_extract_from_empty_costs(self):
        """Test extraction from empty data."""
        extractor = CostFeatureExtractor()
        features = extractor.extract_from_costs([])
        
        assert features["avg_total_tokens"] == 0.0
        assert features["record_count"] == 0
    
    def test_extract_from_costs(self):
        """Test extraction with sample data."""
        extractor = CostFeatureExtractor()
        
        costs = [
            {"agent_name": "general", "model": "gpt-4", "total_tokens": 500, "input_tokens": 300, "output_tokens": 200, "estimated_cost_usd": 0.015},
            {"agent_name": "general", "model": "gpt-4", "total_tokens": 600, "input_tokens": 400, "output_tokens": 200, "estimated_cost_usd": 0.018},
            {"agent_name": "retrieval", "model": "gpt-3.5-turbo", "total_tokens": 300, "input_tokens": 200, "output_tokens": 100, "estimated_cost_usd": 0.0004},
        ]
        
        features = extractor.extract_from_costs(costs)
        
        assert features["record_count"] == 3
        assert features["avg_total_tokens"] == (500 + 600 + 300) / 3
        assert "general" in features["tokens_by_agent"]
        assert "gpt-4" in features["tokens_by_model"]
    
    def test_extract_for_prediction(self):
        """Test extracting features for prediction."""
        extractor = CostFeatureExtractor()
        
        historical = {
            "avg_total_tokens": 500,
            "avg_cost_usd": 0.01,
            "tokens_by_agent": {"general": 550},
            "tokens_by_model": {"gpt-4": 600},
        }
        
        features = extractor.extract_for_prediction("general", "gpt-4", historical)
        
        assert features["agent_name"] == "general"
        assert features["model"] == "gpt-4"
        assert features["historical_avg_tokens_for_agent"] == 550


class TestQualityFeatureExtractor:
    """Tests for QualityFeatureExtractor."""
    
    def test_extract_from_empty_evaluations(self):
        """Test extraction from empty data."""
        extractor = QualityFeatureExtractor()
        features = extractor.extract_from_evaluations([])
        
        assert features["avg_score"] == 0.5
        assert features["pass_rate"] == 0.5
        assert features["record_count"] == 0
    
    def test_extract_from_evaluations(self):
        """Test extraction with sample data."""
        extractor = QualityFeatureExtractor()
        
        evaluations = [
            {"score": 0.9, "passed": True, "evaluator": "critic"},
            {"score": 0.8, "passed": True, "evaluator": "critic"},
            {"score": 0.5, "passed": False, "evaluator": "llm_judge"},
        ]
        
        features = extractor.extract_from_evaluations(evaluations)
        
        assert features["record_count"] == 3
        assert 0.7 <= features["avg_score"] <= 0.8
        assert 0.6 <= features["pass_rate"] <= 0.7
        assert "critic" in features["scores_by_evaluator"]


class TestRiskFeatureExtractor:
    """Tests for RiskFeatureExtractor."""
    
    def test_extract_from_empty_policies(self):
        """Test extraction from empty data."""
        extractor = RiskFeatureExtractor()
        features = extractor.extract_from_policies([])
        
        assert features["violation_rate"] == 0.0
        assert features["pass_rate"] == 1.0
        assert features["record_count"] == 0
    
    def test_extract_from_policies(self):
        """Test extraction with sample data."""
        extractor = RiskFeatureExtractor()
        
        policies = [
            {"status": "pass", "violations": [], "warnings": []},
            {"status": "pass", "violations": [], "warnings": []},
            {"status": "fail", "violations": ["cost_limit"], "warnings": []},
            {"status": "warn", "violations": [], "warnings": ["approaching_limit"]},
        ]
        
        features = extractor.extract_from_policies(policies)
        
        assert features["record_count"] == 4
        assert features["violation_rate"] == 0.25  # 1 out of 4
        assert features["warning_rate"] == 0.25   # 1 out of 4
        assert features["pass_rate"] == 0.5       # 2 out of 4
    
    def test_extract_from_slas(self):
        """Test SLA tier distribution."""
        extractor = RiskFeatureExtractor()
        
        slas = [
            {"tier": "free"},
            {"tier": "free"},
            {"tier": "standard"},
            {"tier": "premium"},
        ]
        
        features = extractor.extract_from_slas(slas)
        
        assert features["record_count"] == 4
        assert features["tier_distribution"]["free"] == 0.5
        assert features["tier_distribution"]["standard"] == 0.25


class TestLatencyFeatureExtractor:
    """Tests for LatencyFeatureExtractor."""
    
    def test_extract_from_empty_traces(self):
        """Test extraction from empty data."""
        extractor = LatencyFeatureExtractor()
        features = extractor.extract_from_traces([])
        
        assert features["avg_latency_ms"] == 0.0
        assert features["success_rate"] == 0.0
        assert features["record_count"] == 0
    
    def test_extract_from_traces(self):
        """Test extraction with sample data."""
        extractor = LatencyFeatureExtractor()
        
        traces = [
            {"agent_name": "general", "latency_ms": 100, "success": True},
            {"agent_name": "general", "latency_ms": 200, "success": True},
            {"agent_name": "retrieval", "latency_ms": 150, "success": False},
        ]
        
        features = extractor.extract_from_traces(traces)
        
        assert features["record_count"] == 3
        assert features["avg_latency_ms"] == 150.0
        assert features["success_rate"] == 2/3
        assert "general" in features["latency_by_agent"]
