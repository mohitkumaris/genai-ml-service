# GenAI ML Service

**Prediction service for the GenAI platform - generates predictive signals from historical data.**

This service generates predictions (cost, quality, risk) based on historical LLMOps data. All outputs are **METADATA ONLY** - advisory and offline.

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Prediction Only** | ML predicts outcomes; it does NOT control behavior |
| **Read-Only Input** | Consumes LLMOps data read-only via batch pull |
| **Deterministic** | Same input data → same prediction |
| **Versioned** | Every prediction includes `model_version` |
| **Append-Only** | Predictions and models stored immutably |

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Run the Server

```bash
# Default configuration
uvicorn app.api.main:app --reload --port 8200

# With custom LLMOps URL
LLMOPS_BASE_URL=http://localhost:8100 uvicorn app.api.main:app --reload --port 8200
```

### Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

### Prediction (POST)

| Endpoint | Description |
|----------|-------------|
| `POST /predict/cost` | Predict expected cost (USD) |
| `POST /predict/quality` | Predict expected quality score (0.0-1.0) |
| `POST /predict/risk` | Predict policy violation risk (0.0-1.0) |
| `GET /predictions` | Query prediction history |

### Training (POST)

| Endpoint | Description |
|----------|-------------|
| `POST /train/cost` | Train cost prediction model |
| `POST /train/quality` | Train quality prediction model |
| `POST /train/risk` | Train risk prediction model |
| `POST /train/all` | Train all models |
| `GET /train/status` | Get training status and model versions |

### Health

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /` | Service info |
| `GET /models` | List available models |
| `GET /docs` | OpenAPI documentation |

## Example Usage

### Train Models

```bash
# Train all models from LLMOps data
curl -X POST http://localhost:8200/train/all \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Make Predictions

```bash
# Predict cost
curl -X POST http://localhost:8200/predict/cost \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "general", "model": "gpt-4"}'

# Predict quality
curl -X POST http://localhost:8200/predict/quality \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "general"}'

# Predict risk
curl -X POST http://localhost:8200/predict/risk \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "general", "tier": "standard"}'
```

### Prediction Response

All predictions return:
```json
{
  "prediction_id": "uuid",
  "prediction_type": "cost",
  "predicted_value": 0.015,
  "confidence": 0.85,
  "model_version": "cost-v1-20260206",
  "input_window": {"record_count": 100},
  "metadata": {"agent_name": "general", "model": "gpt-4"},
  "created_at": "2026-02-06T23:00:00"
}
```

## Architecture

```
genai-ml-service/
├── app/
│   └── api/
│       ├── main.py          # FastAPI application
│       ├── predict.py       # Prediction endpoints
│       └── training.py      # Training endpoints
├── ml/
│   ├── config/              # Configuration
│   ├── models/              # ML model definitions
│   ├── features/            # Feature extraction (read-only)
│   ├── training/            # Offline training logic
│   ├── prediction/          # Prediction service
│   └── store/               # Model & prediction persistence
└── tests/
```

## Integration Model

```
LLMOps (read-only) ───► ML Service (this service)
                              │
                              ▼
                        Predictions
                        (metadata only)
```

**Key invariant**: This service reads from LLMOps but NEVER writes back or influences execution.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLMOPS_BASE_URL` | `http://localhost:8100` | LLMOps service URL |
| `LLMOPS_DATA_DIR` | (none) | Local data directory for file-based input |
| `LLMOPS_ENABLED` | `true` | Enable/disable LLMOps integration (fail-open) |
| `LLMOPS_TIMEOUT_MS` | `1000` | HTTP timeout for LLMOps calls (milliseconds) |
| `MODEL_STORE_DIR` | `models` | Directory for trained models |
| `PREDICTION_STORE_DIR` | `predictions` | Directory for prediction records |
| `ML_HOST` | `0.0.0.0` | API host |
| `ML_PORT` | `8200` | API port |
| `ML_DEBUG` | `false` | Enable debug mode |

## Design Constraints

### This service will NEVER:
- Invoke agents or make LLM calls
- Route requests or influence execution
- Enforce policies or block traffic
- Call the orchestrator
- Modify runtime behavior

### This service ONLY:
- Reads historical data from LLMOps (batch/snapshot)
- Trains prediction models offline
- Generates predictions as metadata
- Stores predictions (append-only)
