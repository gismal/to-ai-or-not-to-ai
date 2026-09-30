*"That's the question."* — Shakespeare (probably would have asked this in 2026)

### [🔗 Live API Demo](https://to-ai-or-not-to-ai.onrender.com/) | [🔗 Chrome Extension](#)

Have you ever wondered if a photo is real or AI generated in 2026? And if you suspected something was off, would you know exactly why? This doesn't just give you a yes or no. It tells you how confident it is, and if it is unsure, it admits that right away.

A solo end-to-end project to deep dive into "production ready" ML, MLOps, and microservices.

---

## What Does It Actually Do?

You send it an image. It sends back one of these:

| Label | Meaning |
|---|---|
| `REAL` | Confidently human-made |
| `AI_GENERATED` | Confidently machine-made |
| `UNCERTAIN_LEANING_AI` | Probably AI, but ambiguous |
| `UNCERTAIN_LEANING_REAL` | Probably real, but ambiguous |
| `UNCERTAIN_NEUTRAL` | Genuinely no idea. Flagged for review |

The uncertain category is intentional. Personally I don't like when AI models pretend to be confident when they're not. Most detectors give you a binary answer even when the model is basically guessing. This one doesn't. Predictions that fall in the confidence gray zone get flagged as `UNCERTAIN` rather than quietly misfiring. That distinction matters a lot when false positives have real consequences.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client / Browser                  │
│         Web Playground  │  Chrome Extension         │
└───────────────────────┬─────────────────────────────┘
                        │ HTTPS
                        ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI  (src/main.py)                 │
│  • API key auth      • Rate limiting (5 req/s)      │
│  • Request ID trace  • Prometheus metrics           │
└──────┬─────────────────┬──────────────┬─────────────┘
       │                 │              │
       ▼                 ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌────────────────┐
│  Inference  │  │Explainabilit│  │    Feedback    │
│  Service    │  │y Service    │  │    Service     │
│             │  │             │  │                │
│ pHash cache │  │ GradCAM     │  │  Drift tracker │
│ ONNX engine │  │ Heatmap     │  │  Repo pattern  │
│ 5-label     │  │ (PyTorch)   │  │                │
│ uncertainty │  │             │  └───────┬────────┘
└──────┬──────┘  └──────┬──────┘          │
       └────────────────┼─────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                      PostgreSQL                     │
│   prediction_logs │ feedback_logs │ task_results    │
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
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Prometheus + Grafana                   │
│   request rate │ latency p95 │ prediction dist.     │
└─────────────────────────────────────────────────────┘
```

---

## Features

### Inference
- **MobileNetV3-Small** converted to **ONNX** for CPU/GPU agnostic, low-latency predictions
- **5-label uncertainty system** granular confidence zones, not just binary
- **GradCAM explainability** drag-to-reveal heatmap showing which regions of the image drove the decision
- **Batch inference** up to 10 images per request with thread-safe concurrent preprocessing
- **Startup warm-up** model is pre-heated on boot so the first request is as fast as the tenth

### Performance
- **Perceptual hash caching (pHash + Redis)** identical images never hit the model twice
- `asyncio.to_thread` for non-blocking ONNX execution
- Response times under 200ms for cached results, around 400ms for fresh inference

### Security
- **API key authentication** on every endpoint
- **python-magic** for file validation — checks actual byte signatures, not just extensions
- **Path traversal protection** filenames sanitized with `PurePosixPath`
- **Rate limiting** 5 requests per second per IP

### Observability
- **Prometheus metrics** via `/metrics` — request latency, prediction distribution, error rates, cache hit rate
- **Structured JSON logging** with `X-Request-ID` tracing across every log line
- **Deep health check** at `/v1/inference/health` — verifies model, database, and Redis are all actually alive
- **10-panel Grafana dashboard** request rate, latency percentiles, prediction distribution, ONNX vs GradCAM latency comparison

### The MLOps Loop
- **Prediction logging** every inference stored with confidence score and label
- **Feedback endpoint** clients report misclassifications (false positives, false negatives)
- **Drift detection** `/v1/admin/check-drift` calculates the real error rate from prediction and feedback logs, triggers retraining via ARQ when it crosses the threshold
- **Active learning** `UNCERTAIN` predictions flagged for human review via `/v1/admin/uncertain`, labeled samples routed back into the training pipeline
- **MLflow experiment tracking** every training run logged with parameters, metrics, and model artifacts
- **Alembic migrations** schema changes without data loss

### Training
- **Two-phase transfer learning** frozen backbone for warmup, then full fine-tuning at a lower learning rate
- **Category-balanced dataset** faces, animals, nature, objects, and architecture equally represented
- **Augmentation** blur, color jitter, random rotation to improve generalization on real-world photos
- **Temperature scaling evaluated** Brier score on this model was already 0.03, calibration was skipped

### Browser Extension
- Right-click any image on any website to analyze it
- GradCAM heatmap shown in the popup, not just the verdict
- Passive badge mode scans the page and marks AI-generated images as you browse
- Deep-links to the web playground for full analysis

### Engineering Patterns
- Layered architecture: routes → services → repositories → database
- Dependency injection throughout via FastAPI `Depends`
- Abstract repository interface for loose coupling
- `BaseMLEngine` abstract base class shared by ONNX and GradCAM engines
- Soft deletes, nothing is ever permanently gone
- Pydantic v2 settings with startup validation

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ML Inference | ONNX Runtime + MobileNetV3-Small |
| Explainability | pytorch-grad-cam + PyTorch |
| Training | PyTorch + Transfer Learning |
| Experiment Tracking | MLflow |
| Database | PostgreSQL + SQLAlchemy (async) |
| Migrations | Alembic |
| Task Queue | ARQ + Redis |
| Caching | Redis (pHash keys) |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |
| Browser Extension | Chrome Manifest V3 |
| Testing | pytest + pytest-asyncio + Hypothesis |
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

This starts six services: `api`, `worker`, `redis`, `db`, `prometheus`, and `grafana`. Wait for the health checks to pass.

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

# Get a GradCAM heatmap
curl -X POST http://localhost:8000/v1/inference/explain \
  -H "X-API-Key: your_secret_key_here" \
  -F "file=@your_image.jpg"
```

Or open the interactive docs at **http://localhost:8000/docs**

Grafana is at **http://localhost:3001** (admin / your `GRAFANA_PASSWORD`)

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

### `POST /v1/inference/explain`
Same as predict, but also returns a base64-encoded GradCAM heatmap showing which pixels the model looked at.

```json
{
  "filename": "photo.jpg",
  "confidence": 0.8821,
  "prediction": "AI_GENERATED",
  "status": "SUCCESS",
  "processing_time_ms": 412.3,
  "heatmap_base64": "...",
  "cached": false
}
```

### `POST /v1/inference/predict-batch`
Send up to 10 images in one request. Each file is validated and processed concurrently. Failed images return an error field instead of crashing the whole batch.

### `POST /v1/feedback/`
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

### `POST /v1/admin/check-drift`
Check whether the model error rate has exceeded the threshold. Triggers retraining if it has.

### `GET /v1/admin/uncertain`
Returns recent predictions the model was uncertain about, ready for human review.

### `POST /v1/admin/label`
Submit a human-verified label for an uncertain prediction. Labeled samples feed back into the next training run.

### `GET /v1/inference/health`
Deep system health — model, database, and Redis all checked live.

### `GET /metrics`
Prometheus metrics endpoint.

---

## Training

If you want to train your own model instead of using the pre-trained one:

```
data/
├── train/
│   ├── REAL/
│   └── AI_GENERATED/
└── val/
    ├── REAL/
    └── AI_GENERATED/
```

```bash
python scripts/train.py
```

The pipeline uses MobileNetV3-Small with a frozen backbone in phase one, then unfreezes for full fine-tuning at a lower learning rate. Early stopping, cosine annealing LR, and ONNX export (opset 12) are all handled automatically. Training config lives in `scripts/train_config.py` as a Pydantic model.

For experiment tracking, set `MLFLOW_TRACKING_URI` before running. If MLflow is not installed, training continues without it.

The dataset used for the released model: CIFAKE, ArtiFact, FFHQ, LFW, CelebA, Open Images, and DeepFake Detection. Balanced across faces, animals, nature, objects, and architecture to avoid the face-detection bias most detectors have.

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

Tests use an in-memory SQLite database. No external services needed.

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
├── extension/           # Chrome Extension (Manifest V3)
├── frontend/            # Web playground (Shakespearean theme)
├── monitoring/          # Grafana dashboards and Prometheus config
├── scripts/             # Training, calibration, and dataset scripts
├── src/
│   ├── api/             # Routes, dependencies, rate limiting
│   ├── core/            # Enums, custom exceptions
│   ├── infra/           # Database engine, ORM models, Redis client
│   ├── repositories/    # Data access layer (Feedback, Logs, Tasks)
│   ├── schemas/         # Pydantic request/response models
│   ├── services/        # Business logic (Inference, Explainability, Cache, etc.)
│   ├── worker/          # ARQ task definitions and cron jobs
│   ├── config.py        # Pydantic settings and environment variables
│   ├── inference.py     # ONNX engine and image preprocessing
│   ├── logger.py        # Structured JSON logging
│   ├── main.py          # FastAPI application and lifespan events
│   └── utils.py         # pHash generation and utilities
├── migrations/          # Alembic migration files
├── tests/
│   ├── test_api.py      # Endpoint and integration tests
│   ├── test_database.py # Repository layer tests
│   └── test_model.py    # Inference logic and property-based tests
├── models/              # ONNX model weights and metadata
├── Dockerfile
├── docker-compose.yml   # Full stack (API, Worker, Redis, PostgreSQL, Prometheus, Grafana)
└── Makefile
```

---

## Why I Built This

I wanted to challenge myself beyond notebook projects. I focused on understanding what actually happens between "the model works locally" and "the model serves real traffic reliably."

My hands got dirty: dependency injection, session management, async task queues, database migrations, structured logging with request tracing, Docker networking, and what happens when the model starts drifting.

I took my sweet time to build all of that from scratch, make every mistake, and document what I learned.

---

## License
