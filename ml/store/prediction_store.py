"""
Prediction Store

Append-only persistence for prediction records.

DESIGN RULES:
- Append-only (no updates or deletes)
- JSONL format for efficient append
- Supports querying by type and time window
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ml.models.prediction import PredictionResult
from ml.models.dataset import DataWindow
from ml.config.settings import get_settings


class PredictionStore:
    """
    Append-only persistence for prediction records.
    
    Predictions are stored in JSONL format for efficient append operations.
    No updates or deletes are allowed.
    """
    
    def __init__(self, store_dir: Optional[str] = None):
        settings = get_settings()
        self.store_dir = Path(store_dir or settings.prediction_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_filepath(self, prediction_type: str) -> Path:
        """Get filepath for a prediction type."""
        return self.store_dir / f"{prediction_type}_predictions.jsonl"
    
    def append(self, prediction: PredictionResult) -> None:
        """
        Append a prediction record.
        
        This is an append-only operation.
        """
        filepath = self._get_filepath(prediction.prediction_type)
        
        with open(filepath, "a") as f:
            f.write(json.dumps(prediction.to_dict()) + "\n")
    
    def append_batch(self, predictions: List[PredictionResult]) -> None:
        """Append multiple predictions."""
        for prediction in predictions:
            self.append(prediction)
    
    def query(
        self,
        prediction_type: Optional[str] = None,
        window: Optional[DataWindow] = None,
        limit: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Query prediction records.
        
        Args:
            prediction_type: Filter by type (cost, quality, risk, latency)
            window: Time window filter
            limit: Maximum number of records to return
        """
        results = []
        
        # Determine which files to read
        if prediction_type:
            filepaths = [self._get_filepath(prediction_type)]
        else:
            filepaths = list(self.store_dir.glob("*_predictions.jsonl"))
        
        for filepath in filepaths:
            if not filepath.exists():
                continue
            
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    # Apply time window filter
                    if window:
                        created_at = datetime.fromisoformat(data["created_at"])
                        if window.start_time and created_at < window.start_time:
                            continue
                        if window.end_time and created_at > window.end_time:
                            continue
                    
                    results.append(PredictionResult.from_dict(data))
                    
                    if limit and len(results) >= limit:
                        return results
        
        return results
    
    def count(self, prediction_type: Optional[str] = None) -> int:
        """Count prediction records."""
        count = 0
        
        if prediction_type:
            filepaths = [self._get_filepath(prediction_type)]
        else:
            filepaths = list(self.store_dir.glob("*_predictions.jsonl"))
        
        for filepath in filepaths:
            if not filepath.exists():
                continue
            
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1
        
        return count


class InMemoryPredictionStore(PredictionStore):
    """
    In-memory prediction store for testing.
    """
    
    def __init__(self):
        self._predictions: List[PredictionResult] = []
    
    def append(self, prediction: PredictionResult) -> None:
        self._predictions.append(prediction)
    
    def query(
        self,
        prediction_type: Optional[str] = None,
        window: Optional[DataWindow] = None,
        limit: Optional[int] = None,
    ) -> List[PredictionResult]:
        results = []
        
        for prediction in self._predictions:
            # Filter by type
            if prediction_type and prediction.prediction_type != prediction_type:
                continue
            
            # Filter by window
            if window:
                if window.start_time and prediction.created_at < window.start_time:
                    continue
                if window.end_time and prediction.created_at > window.end_time:
                    continue
            
            results.append(prediction)
            
            if limit and len(results) >= limit:
                break
        
        return results
    
    def count(self, prediction_type: Optional[str] = None) -> int:
        if prediction_type:
            return sum(
                1 for p in self._predictions
                if p.prediction_type == prediction_type
            )
        return len(self._predictions)
