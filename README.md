# 🎭 To AI or Not to AI

> *"That's the question."* — Shakespeare (probably would have asked this in 2025)
*(Insert Demo GIF here)* 
### [🔗 Live API Demo](#) | [🔗 Chrome Extension](#) | [📊 Grafana Dashboard](#)

A production-grade MLOps microservice that looks at an image and tells you whether a human or an AI made it. Not just a yes or no, it tells you **how confident** it is, and when it genuinely isn't sure, it admits it.

Built as a solo end-to-end project to learn what "production-ready" actually means beyond Jupyter notebooks.

---

## What Does It Actually Do?

You send it an image. It sends back one of these:

| Label | Meaning |
|---|---|
| `REAL` | Confidently human-made |
| `AI_GENERATED` | Confidently machine-made |
| `UNCERTAIN_LEANING_AI` | Probably AI, but hedging |
| `UNCERTAIN_LEANING_REAL` | Probably real, but hedging |
| `UNCERTAIN_NEUTRAL` | Genuinely no idea — flagged for review |

That last category is intentional. Most detectors give you a binary answer even when the model is basically guessing. Last one doesn't. Predictions that fall in the confidence gray zone get flagged as `UNCERTAIN` rather than quietly misfiring. That distinction matters a lot when false positives have real consequences.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client / Browser                  │
└───────────────────────┬─────────────────────────────┘
                        │ HTTPS
                        ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI  (src/main.py)                 │
│  • API key auth      • Rate limiting (5 req/s)      │
│  • Request ID trace  • Prometheus metrics           │
└──────┬─────────────────────────┬────────────────────┘
       │                         │
       ▼                         ▼
┌─────────────┐         ┌────────────────┐
│  Inference  │         │    Feedback    │
│  Service    │         │    Service     │
│             │         │                │
│ pHash cache │         │  Drift tracker │
│ ONNX engine │         │  Repo pattern  │
│ 5-label     │         │                │
│ uncertainty │         └───────┬────────┘
└──────┬──────┘                 │
       │                        │
       ▼                        ▼
┌─────────────────────────────────────────────────────┐
│                      PostgreSQL                     │
│         prediction_logs │ feedback_logs             │
│              (Alembic versioned migrations)         │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐        ┌──────────────────────────────┐
│    Redis    │◄──────►│        ARQ Worker            │
│  • Job queue│        │  • Model retraining pipeline │
│  • pHash    │        │  • Isolated subprocess       │
│    cache    │        │  • Auto-retry on failure     │
└─────────────┘        └──────────────────────────────┘
```

---

## Features

### 🧠 Intelligent Inference
- **MobileNetV3-Small** converted to **ONNX** for CPU/GPU agnostic, low-latency predictions
- **5-label uncertainty system** is not just binary, but granular confidence zones
- **Startup warm-up** model is pre-heated on boot so the first request is as fast as the tenth
- **Batch inference** with thread-safe concurrent preprocessing

### ⚡ Performance
- **Perceptual hash caching (pHash + Redis)** helps to  identical images never hit the model twice
- `asyncio.to_thread` for non-blocking ONNX execution
- Response times under 200ms for cached results, ~400ms for fresh inference

### 🔒 Security
- **API key authentication** on every endpoint
- **python-magic** for file validation. It checks not just file extensions also actual byte signatures.
- **Path traversal protection** filenames sanitized with `PurePosixPath`
- **Rate limiting** 5 requests/second per IP

### 📊 Observability
- **Prometheus metrics** via `/metrics` — request latency, prediction distribution, error rates
- **Structured JSON logging** with `X-Request-ID` tracing across every log line
- **Deep health check** at `/v1/inference/health` — verifies model, database, and Redis are all actually alive (not just "the process is running")

### 🔄 The MLOps Loop
- **Prediction logging** — every inference stored with confidence score and label
- **Feedback endpoint** — clients report misclassifications (false positives / false negatives)
- **Drift detection** — `/v1/inference/admin/check-drift` calculates real error rate and triggers retraining via ARQ when it crosses the threshold
- **Alembic migrations** — schema changes without data loss

### 🏗️ Engineering Patterns
- Layered architecture: routes → services → repositories → database
- Dependency injection throughout (FastAPI `Depends`)
- Abstract repository interface for loose coupling
- Soft deletes, nothing is ever permanently gone
- Pydantic v2 settings with startup validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ML Inference | ONNX Runtime + MobileNetV3 |
| Training | PyTorch + Transfer Learning |
| Database | PostgreSQL + SQLAlchemy (async) |
| Migrations | Alembic |
| Task Queue | ARQ + Redis |
| Caching | Redis (pHash keys) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Testing | pytest + pytest-asyncio |
| CI/CD | GitHub Actions |

---

## Getting Started

### Prerequisites
- Docker + Docker Compose
- A trained ONNX model at `models/model_v1.onnx` (see [Training](#training))

### 1. Clone and configure

```bash
git clone https://github.com/gismal/to-ai-or-not-to-ai.git
cd to-ai-or-not-to-ai
cp .env.example .env
```

Open `.env` and fill in your values:

```env
API_KEY=your_secret_key_here
DATABASE_URL=postgresql+asyncpg://pgadmin:yourpassword@db:5432/ai_detector
POSTGRES_USER=pgadmin
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=ai_detector
REDIS_URL=redis://redis:6379/0
MODEL_THRESHOLD=0.75
GRAY_AREA_MARGIN=0.35
DRIFT_THRESHOLD=0.15
DEBUG=false
ALLOWED_ORIGINS=["*"]
```

### 2. Run everything

```bash
make build
# or without Make:
docker compose up --build
```

This starts four services: `api`, `worker`, `redis`, `db`. Wait for the health checks to pass.

### 3. Run migrations

```bash
make migrate
# or:
docker compose exec api alembic upgrade head
```

### 4. Try it

```bash
# Health check
curl http://localhost:8000/v1/inference/health \
  -H "X-API-Key: your_secret_key_here"

# Classify an image
curl -X POST http://localhost:8000/v1/inference/predict \
  -H "X-API-Key: your_secret_key_here" \
  -F "file=@your_image.jpg"
```

Or open the interactive docs at **http://localhost:8000/docs**

---

## API Reference

### `POST /v1/inference/predict`
Classify an image as real or AI-generated.

**Request:** `multipart/form-data` with a `file` field (JPEG or PNG, max 10MB)

**Response:**
```json
{
  "filename": "photo.jpg",
  "confidence": 0.8821,
  "prediction": "AI_GENERATED",
  "status": "SUCCESS",
  "processing_time_ms": 143.7,
  "cached": false
}
```

### `POST /v1/feedback`
Tell the model it was wrong.

```json
{
  "filename": "photo.jpg",
  "model_prediction": "AI_GENERATED",
  "confidence": 0.8821,
  "user_correction": "REAL",
  "client_source": "API_v1"
}
```

### `POST /v1/inference/admin/check-drift`
Check whether model error rate has exceeded the threshold. Triggers retraining if it has.

### `GET /v1/inference/health`
Deep system health — model, database, and Redis all checked live.

### `GET /metrics`
Prometheus metrics endpoint.

---

## Training

If you want to train your own model instead of using a pre-trained one:

```bash
# Prepare your dataset
data/
├── train/
│   ├── REAL/
│   └── AI_GENERATED/
└── val/
    ├── REAL/
    └── AI_GENERATED/

# Run training
make train
# or:
python scripts/train.py
```

The training pipeline uses MobileNetV3-Small with frozen backbone, early stopping, cosine annealing LR, and exports to ONNX automatically. Training config lives in `scripts/train.py` as a Pydantic model — override with a YAML file for experiment tracking.

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-test.txt

# Run tests
make test

# Lint
make lint

# Run locally without Docker
uvicorn src.main:app --reload
```

### Running Tests

```bash
pytest -v --tb=short --cov=src --cov-report=term-missing
```

Tests use an in-memory SQLite database — no external services needed to run the test suite.

---

## Makefile Commands

```
make build     → docker compose up --build
make test      → run pytest with coverage
make lint      → ruff check src/ tests/
make migrate   → alembic upgrade head
make train     → python scripts/train.py
```

---

## Project Structure

```
to-ai-or-not-to-ai/
├── src/
│   ├── api/             # Routes, dependencies, rate limiting
│   ├── core/            # Enums, exceptions (no external deps)
│   ├── infra/           # Database engine, ORM models, limiter
│   ├── repositories/    # Data access layer
│   ├── schemas/         # Pydantic request/response models
│   ├── services/        # Business logic
│   ├── worker/          # ARQ task definitions
│   ├── config.py        # Pydantic settings
│   ├── inference.py     # ONNX engine + preprocessing
│   ├── logger.py        # JSON structured logger
│   ├── main.py          # FastAPI app + lifespan
│   └── utils.py         # pHash generation
├── scripts/
│   └── train.py         # Offline training pipeline
├── migrations/          # Alembic migration files
├── tests/               # pytest suite
├── models/              # ONNX model + metadata
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## What's Coming

This is an active project. On the roadmap:

- **GradCAM explainability** — `/v1/inference/explain` returns a heatmap showing *which pixels* drove the decision, not just what the decision was
- **Web Playground** — drag-and-drop frontend with animated confidence meter and side-by-side heatmap visualization  
- **Browser Extension** — right-click any image on any website, get an instant classification in a popup
- **ONNX INT8 Quantization** — 4x model size reduction, 2-3x CPU speedup

---

## Why I Built This

Most ML portfolio projects are a model in a notebook with a Flask endpoint bolted on. I wanted to understand what actually happens between "the model works locally" and "the model serves real traffic reliably."

The answer turns out to involve a lot of things notebooks never teach: dependency injection, session management, async task queues, database migrations, structured logging with request tracing, Docker networking, and what happens when the model starts drifting because the world changed.

This project is my attempt to build all of that from scratch, make every mistake, and document what I learned.

---

## License

MIT — use it, break it, learn from it.
