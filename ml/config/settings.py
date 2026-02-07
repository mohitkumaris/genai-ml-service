"""
GenAI ML Service Configuration

Environment-based configuration for the prediction service.

DESIGN RULES:
- Read-only configuration
- No secrets in defaults
- Explicit over magic
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration for the ML service.
    
    All configuration via environment variables.
    """
    
    # LLMOps data source
    llmops_base_url: str = "http://localhost:8100"
    llmops_data_dir: Optional[str] = None  # For file-based data
    llmops_enabled: bool = True  # Toggle LLMOps integration
    llmops_timeout_ms: int = 1000  # Timeout in milliseconds (≤1s for fail-fast)
    
    # Storage directories
    model_store_dir: str = "models"
    prediction_store_dir: str = "predictions"
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8200
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            llmops_base_url=os.getenv("LLMOPS_BASE_URL", "http://localhost:8100"),
            llmops_data_dir=os.getenv("LLMOPS_DATA_DIR"),
            llmops_enabled=os.getenv("LLMOPS_ENABLED", "true").lower() == "true",
            llmops_timeout_ms=int(os.getenv("LLMOPS_TIMEOUT_MS", "1000")),
            model_store_dir=os.getenv("MODEL_STORE_DIR", "models"),
            prediction_store_dir=os.getenv("PREDICTION_STORE_DIR", "predictions"),
            host=os.getenv("ML_HOST", "0.0.0.0"),
            port=int(os.getenv("ML_PORT", "8200")),
            debug=os.getenv("ML_DEBUG", "false").lower() == "true",
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
