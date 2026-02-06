"""
Feature Extractors

Extract features from LLMOps data for ML models.

DESIGN RULES:
- Read-only transformations
- Deterministic feature extraction
- No side effects
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict


class CostFeatureExtractor:
    """
    Extract features for cost prediction.
    
    Features:
    - Average tokens per agent
    - Average tokens per model
    - Session token trends
    """
    
    def extract_from_costs(self, costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated cost features."""
        if not costs:
            return {
                "avg_total_tokens": 0.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_cost_usd": 0.0,
                "tokens_by_agent": {},
                "tokens_by_model": {},
                "record_count": 0,
            }
        
        total_tokens = sum(c.get("total_tokens", 0) for c in costs)
        input_tokens = sum(c.get("input_tokens", 0) for c in costs)
        output_tokens = sum(c.get("output_tokens", 0) for c in costs)
        total_cost = sum(c.get("estimated_cost_usd", 0) for c in costs)
        
        # Group by agent
        tokens_by_agent: Dict[str, List[int]] = defaultdict(list)
        for c in costs:
            agent = c.get("agent_name", "unknown")
            tokens_by_agent[agent].append(c.get("total_tokens", 0))
        
        avg_by_agent = {
            agent: sum(tokens) / len(tokens)
            for agent, tokens in tokens_by_agent.items()
        }
        
        # Group by model
        tokens_by_model: Dict[str, List[int]] = defaultdict(list)
        for c in costs:
            model = c.get("model", "unknown")
            tokens_by_model[model].append(c.get("total_tokens", 0))
        
        avg_by_model = {
            model: sum(tokens) / len(tokens)
            for model, tokens in tokens_by_model.items()
        }
        
        n = len(costs)
        return {
            "avg_total_tokens": total_tokens / n,
            "avg_input_tokens": input_tokens / n,
            "avg_output_tokens": output_tokens / n,
            "avg_cost_usd": total_cost / n,
            "tokens_by_agent": avg_by_agent,
            "tokens_by_model": avg_by_model,
            "record_count": n,
        }
    
    def extract_for_prediction(
        self,
        agent_name: str,
        model: str,
        historical_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract features for a single cost prediction.
        
        Uses historical aggregates plus request-specific info.
        """
        tokens_by_agent = historical_features.get("tokens_by_agent", {})
        tokens_by_model = historical_features.get("tokens_by_model", {})
        
        return {
            "agent_name": agent_name,
            "model": model,
            "historical_avg_tokens_for_agent": tokens_by_agent.get(agent_name, historical_features.get("avg_total_tokens", 0)),
            "historical_avg_tokens_for_model": tokens_by_model.get(model, historical_features.get("avg_total_tokens", 0)),
            "global_avg_cost": historical_features.get("avg_cost_usd", 0),
        }


class QualityFeatureExtractor:
    """
    Extract features for quality prediction.
    
    Features:
    - Historical evaluation scores per agent
    - Pass rates per evaluator
    """
    
    def extract_from_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated quality features."""
        if not evaluations:
            return {
                "avg_score": 0.5,
                "pass_rate": 0.5,
                "scores_by_evaluator": {},
                "record_count": 0,
            }
        
        total_score = sum(e.get("score", 0.5) for e in evaluations)
        passes = sum(1 for e in evaluations if e.get("passed", False))
        
        # Group by evaluator
        scores_by_evaluator: Dict[str, List[float]] = defaultdict(list)
        for e in evaluations:
            evaluator = e.get("evaluator", "unknown")
            scores_by_evaluator[evaluator].append(e.get("score", 0.5))
        
        avg_by_evaluator = {
            evaluator: sum(scores) / len(scores)
            for evaluator, scores in scores_by_evaluator.items()
        }
        
        n = len(evaluations)
        return {
            "avg_score": total_score / n,
            "pass_rate": passes / n,
            "scores_by_evaluator": avg_by_evaluator,
            "record_count": n,
        }
    
    def extract_for_prediction(
        self,
        agent_name: str,
        historical_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract features for a single quality prediction."""
        return {
            "agent_name": agent_name,
            "historical_avg_score": historical_features.get("avg_score", 0.5),
            "historical_pass_rate": historical_features.get("pass_rate", 0.5),
        }


class RiskFeatureExtractor:
    """
    Extract features for risk (policy violation) prediction.
    
    Features:
    - Historical violation rates
    - Warning patterns
    - Tier-based risk factors
    """
    
    def extract_from_policies(self, policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated risk features."""
        if not policies:
            return {
                "violation_rate": 0.0,
                "warning_rate": 0.0,
                "pass_rate": 1.0,
                "avg_violations": 0.0,
                "avg_warnings": 0.0,
                "record_count": 0,
            }
        
        violations = sum(1 for p in policies if p.get("status") == "fail")
        warnings = sum(1 for p in policies if p.get("status") == "warn")
        passes = sum(1 for p in policies if p.get("status") == "pass")
        
        total_violations = sum(len(p.get("violations", [])) for p in policies)
        total_warnings = sum(len(p.get("warnings", [])) for p in policies)
        
        n = len(policies)
        return {
            "violation_rate": violations / n,
            "warning_rate": warnings / n,
            "pass_rate": passes / n,
            "avg_violations": total_violations / n,
            "avg_warnings": total_warnings / n,
            "record_count": n,
        }
    
    def extract_from_slas(self, slas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract SLA tier distribution."""
        if not slas:
            return {
                "tier_distribution": {},
                "record_count": 0,
            }
        
        tier_counts: Dict[str, int] = defaultdict(int)
        for s in slas:
            tier = s.get("tier", "unknown")
            tier_counts[tier] += 1
        
        n = len(slas)
        tier_distribution = {
            tier: count / n
            for tier, count in tier_counts.items()
        }
        
        return {
            "tier_distribution": tier_distribution,
            "record_count": n,
        }
    
    def extract_for_prediction(
        self,
        agent_name: str,
        tier: str,
        historical_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract features for a single risk prediction."""
        return {
            "agent_name": agent_name,
            "tier": tier,
            "historical_violation_rate": historical_features.get("violation_rate", 0.0),
            "historical_warning_rate": historical_features.get("warning_rate", 0.0),
        }


class LatencyFeatureExtractor:
    """
    Extract features for latency prediction.
    
    Features:
    - Historical latency per agent
    - Success/failure latency patterns
    """
    
    def extract_from_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated latency features."""
        if not traces:
            return {
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "latency_by_agent": {},
                "success_rate": 0.0,
                "record_count": 0,
            }
        
        latencies = [t.get("latency_ms", 0) for t in traces]
        successes = sum(1 for t in traces if t.get("success", False))
        
        # Group by agent
        latency_by_agent: Dict[str, List[int]] = defaultdict(list)
        for t in traces:
            agent = t.get("agent_name", "unknown")
            latency_by_agent[agent].append(t.get("latency_ms", 0))
        
        avg_by_agent = {
            agent: sum(lats) / len(lats)
            for agent, lats in latency_by_agent.items()
        }
        
        sorted_latencies = sorted(latencies)
        n = len(traces)
        
        return {
            "avg_latency_ms": sum(latencies) / n,
            "p50_latency_ms": sorted_latencies[n // 2] if n > 0 else 0,
            "p95_latency_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "latency_by_agent": avg_by_agent,
            "success_rate": successes / n,
            "record_count": n,
        }
    
    def extract_for_prediction(
        self,
        agent_name: str,
        historical_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract features for a single latency prediction."""
        latency_by_agent = historical_features.get("latency_by_agent", {})
        return {
            "agent_name": agent_name,
            "historical_avg_latency": latency_by_agent.get(
                agent_name,
                historical_features.get("avg_latency_ms", 0)
            ),
            "historical_p95_latency": historical_features.get("p95_latency_ms", 0),
        }
