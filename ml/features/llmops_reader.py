"""
LLMOps Data Reader

Read-only access to LLMOps data for ML training and prediction.

DESIGN RULES:
- READ-ONLY access only
- Never modifies LLMOps data
- Supports both API and file-based reading
- Batch/snapshot operations only, no streaming
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import requests

from ml.models.dataset import DataWindow, TrainingDataset
from ml.config.settings import get_settings


class LLMOpsReader(ABC):
    """
    Abstract base class for LLMOps data readers.
    
    All readers are READ-ONLY.
    """
    
    @abstractmethod
    def read_traces(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read trace records from LLMOps."""
        ...
    
    @abstractmethod
    def read_costs(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read cost records from LLMOps."""
        ...
    
    @abstractmethod
    def read_evaluations(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read evaluation records from LLMOps."""
        ...
    
    @abstractmethod
    def read_policies(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read policy outcome records from LLMOps."""
        ...
    
    @abstractmethod
    def read_slas(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read SLA records from LLMOps."""
        ...
    
    def read_all(self, window: Optional[DataWindow] = None) -> TrainingDataset:
        """
        Read all data types into a TrainingDataset.
        
        This is the primary method for batch data ingestion.
        """
        return TrainingDataset(
            traces=self.read_traces(window),
            costs=self.read_costs(window),
            evaluations=self.read_evaluations(window),
            policies=self.read_policies(window),
            slas=self.read_slas(window),
            window=window,
        )


class LLMOpsAPIReader(LLMOpsReader):
    """
    Read LLMOps data via HTTP API.
    
    Pulls data from LLMOps query endpoints.
    
    DESIGN RULES:
    - HTTP GET only (read-only)
    - Short timeout (≤1s) for fail-fast
    - Fail-open: return empty on any error
    - Respects LLMOPS_ENABLED setting
    """
    
    def __init__(self, base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = base_url or settings.llmops_base_url
    
    def _build_params(self, window: Optional[DataWindow], limit: int = 1000) -> Dict[str, str]:
        """Build query parameters from DataWindow."""
        params: Dict[str, str] = {"limit": str(limit)}
        if window:
            if window.start_time:
                params["start_time"] = window.start_time.isoformat()
            if window.end_time:
                params["end_time"] = window.end_time.isoformat()
        return params
    
    def _fetch(self, endpoint: str, window: Optional[DataWindow], limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch data from LLMOps API endpoint.
        
        Returns empty list on any error (fail-open).
        """
        settings = get_settings()
        
        # Check if LLMOps integration is disabled
        if not settings.llmops_enabled:
            return []
        
        url = f"{self.base_url}{endpoint}"
        params = self._build_params(window, limit)
        
        try:
            # Use configurable timeout (convert ms to seconds)
            timeout_seconds = settings.llmops_timeout_ms / 1000.0
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            # Handle wrapped response format from LLMOps API
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
        except requests.RequestException:
            # Fail-open: return empty on any network/HTTP error
            return []
    
    def read_traces(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read traces from LLMOps API."""
        return self._fetch("/query/traces", window)
    
    def read_costs(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read costs from LLMOps API."""
        return self._fetch("/query/costs", window)
    
    def read_evaluations(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read evaluations from LLMOps API."""
        return self._fetch("/query/evaluations", window)
    
    def read_policies(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read policy outcomes from LLMOps API."""
        return self._fetch("/query/policies", window)
    
    def read_slas(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read SLAs from LLMOps API."""
        return self._fetch("/query/slas", window)


class LLMOpsFileReader(LLMOpsReader):
    """
    Read LLMOps data from exported JSONL files.
    
    Reads from local filesystem for offline processing.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.llmops_data_dir or "data")
    
    def _read_jsonl(self, filename: str, window: Optional[DataWindow]) -> List[Dict[str, Any]]:
        """Read records from a JSONL file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return []
        
        records = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append(record)
        
        # Apply simple time filtering if window specified
        if window and window.start_time:
            records = [
                r for r in records
                if r.get("ingested_at") and r["ingested_at"] >= window.start_time.isoformat()
            ]
        if window and window.end_time:
            records = [
                r for r in records
                if r.get("ingested_at") and r["ingested_at"] <= window.end_time.isoformat()
            ]
        
        # Apply record count limit
        if window and window.record_count:
            records = records[:window.record_count]
        
        return records
    
    def read_traces(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read traces from JSONL file."""
        return self._read_jsonl("traces.jsonl", window)
    
    def read_costs(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read costs from JSONL file."""
        return self._read_jsonl("costs.jsonl", window)
    
    def read_evaluations(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read evaluations from JSONL file."""
        return self._read_jsonl("evaluations.jsonl", window)
    
    def read_policies(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read policy outcomes from JSONL file."""
        return self._read_jsonl("policies.jsonl", window)
    
    def read_slas(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        """Read SLAs from JSONL file."""
        return self._read_jsonl("slas.jsonl", window)


class InMemoryReader(LLMOpsReader):
    """
    In-memory reader for testing.
    
    Holds data directly in memory.
    """
    
    def __init__(
        self,
        traces: Optional[List[Dict[str, Any]]] = None,
        costs: Optional[List[Dict[str, Any]]] = None,
        evaluations: Optional[List[Dict[str, Any]]] = None,
        policies: Optional[List[Dict[str, Any]]] = None,
        slas: Optional[List[Dict[str, Any]]] = None,
    ):
        self._traces = traces or []
        self._costs = costs or []
        self._evaluations = evaluations or []
        self._policies = policies or []
        self._slas = slas or []
    
    def read_traces(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        return self._traces.copy()
    
    def read_costs(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        return self._costs.copy()
    
    def read_evaluations(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        return self._evaluations.copy()
    
    def read_policies(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        return self._policies.copy()
    
    def read_slas(self, window: Optional[DataWindow] = None) -> List[Dict[str, Any]]:
        return self._slas.copy()


def get_reader() -> LLMOpsReader:
    """
    Get the appropriate reader based on configuration.
    
    Prefers file-based if LLMOPS_DATA_DIR is set, otherwise uses API.
    """
    settings = get_settings()
    
    if settings.llmops_data_dir:
        return LLMOpsFileReader(settings.llmops_data_dir)
    else:
        return LLMOpsAPIReader(settings.llmops_base_url)
