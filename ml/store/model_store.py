"""
Model Store

Persistence layer for trained ML models.

DESIGN RULES:
- Append-only (new versions, never overwrite)
- Supports versioned model retrieval
- JSON-based serialization
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from ml.models.base import BasePredictionModel
from ml.models.cost_model import CostPredictionModel
from ml.models.quality_model import QualityPredictionModel
from ml.models.risk_model import RiskPredictionModel
from ml.config.settings import get_settings


# Model type to class mapping
MODEL_CLASSES: Dict[str, Type[BasePredictionModel]] = {
    "cost": CostPredictionModel,
    "quality": QualityPredictionModel,
    "risk": RiskPredictionModel,
}


class ModelStore:
    """
    Persistence layer for trained prediction models.
    
    Models are stored as JSON files with versioned filenames.
    Append-only: new versions are added, never overwritten.
    """
    
    def __init__(self, store_dir: Optional[str] = None):
        settings = get_settings()
        self.store_dir = Path(store_dir or settings.model_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: BasePredictionModel) -> str:
        """
        Save a trained model to disk.
        
        Returns the model version.
        """
        model_type = model.model_type
        version = model.model_version
        
        # Create model type directory
        type_dir = self.store_dir / model_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model as JSON
        filepath = type_dir / f"{version}.json"
        with open(filepath, "w") as f:
            json.dump(model.to_dict(), f, indent=2)
        
        # Update latest pointer
        latest_filepath = type_dir / "latest.txt"
        with open(latest_filepath, "w") as f:
            f.write(version)
        
        return version
    
    def load_model(
        self,
        model_type: str,
        version: Optional[str] = None,
    ) -> Optional[BasePredictionModel]:
        """
        Load a model from disk.
        
        If version is None, loads the latest version.
        """
        if model_type not in MODEL_CLASSES:
            return None
        
        type_dir = self.store_dir / model_type
        if not type_dir.exists():
            return None
        
        # Get version to load
        if version is None:
            version = self.get_latest_version(model_type)
            if version is None:
                return None
        
        # Load model JSON
        filepath = type_dir / f"{version}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Deserialize
        model_class = MODEL_CLASSES[model_type]
        return model_class.from_dict(data)
    
    def list_versions(self, model_type: str) -> List[str]:
        """List all versions for a model type."""
        type_dir = self.store_dir / model_type
        if not type_dir.exists():
            return []
        
        versions = []
        for filepath in type_dir.glob("*.json"):
            versions.append(filepath.stem)
        
        return sorted(versions)
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        """Get the latest version for a model type."""
        type_dir = self.store_dir / model_type
        latest_filepath = type_dir / "latest.txt"
        
        if latest_filepath.exists():
            with open(latest_filepath, "r") as f:
                return f.read().strip()
        
        # Fallback: get most recent from list
        versions = self.list_versions(model_type)
        return versions[-1] if versions else None
    
    def has_model(self, model_type: str) -> bool:
        """Check if any model exists for a type."""
        return self.get_latest_version(model_type) is not None


class InMemoryModelStore(ModelStore):
    """
    In-memory model store for testing.
    """
    
    def __init__(self):
        self._models: Dict[str, Dict[str, BasePredictionModel]] = {}
        self._latest: Dict[str, str] = {}
    
    def save_model(self, model: BasePredictionModel) -> str:
        model_type = model.model_type
        version = model.model_version
        
        if model_type not in self._models:
            self._models[model_type] = {}
        
        self._models[model_type][version] = model
        self._latest[model_type] = version
        
        return version
    
    def load_model(
        self,
        model_type: str,
        version: Optional[str] = None,
    ) -> Optional[BasePredictionModel]:
        if model_type not in self._models:
            return None
        
        if version is None:
            version = self._latest.get(model_type)
            if version is None:
                return None
        
        return self._models[model_type].get(version)
    
    def list_versions(self, model_type: str) -> List[str]:
        if model_type not in self._models:
            return []
        return sorted(self._models[model_type].keys())
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        return self._latest.get(model_type)
    
    def has_model(self, model_type: str) -> bool:
        return model_type in self._latest
