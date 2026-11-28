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
│  • API key auth      • Rate limiting on /           |
|                        predict (5 req/s)            │
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
- **API key authentication** covers for two scenarios, one for the real key opens everything (admin included) and second a public demo key (demo-public) that is rate limited per IP to use inference and feedback. /health, /metrics, the playground and the privacy page are public.
- **python-magic** for file validation — checks actual byte signatures, not just extensions
- **Path traversal protection** filenames sanitized with `PurePosixPath`
- **Rate limiting** 5 requests per second per IP on /predict additionally the demo key's per-minute limit per IP. Inference and feedback have seperate demo budgets.

### Observability
- **Prometheus metrics** via `/metrics` — request latency, prediction distribution, error rates, cache hit rate
- **Structured JSON logging** with `X-Request-ID` tracing across every log line
- **Deep health check** at `/health` — verifies model, database, and Redis are all actually alive
- **10-panel Grafana dashboard** request rate, latency percentiles, prediction distribution, ONNX vs GradCAM latency comparison

### The MLOps Loop
- **Prediction logging** every inference stored with confidence score and label
- **Feedback endpoint** clients report misclassifications (false positives, false negatives)
- **Drift detection** `/v1/admin/check-drift` calculates the real error rate from prediction and feedback logs, which are rated by the users. Only decided answers count and absent answers are ignored. A threshold samples is defined to calculate the drift. They trigger retraining via ARQ when it crosses the threshold. It now uses rated feedback only within a window that starts when the current model was trained. It's also worth one sentence that feedback is voluntary, so the rate detects change rather than measuring accuracy. 
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
copy .env.example and fill it in
```

### 2. Run everything

```bash
make build
# or without Make:
docker compose up --build
```

This starts seven services: `api`, `worker`, `redis`, `db`, `prometheus`, `mlflow` and `grafana`. Wait for the health checks to pass.

### 3. Run migrations
Migrations run automatically at startup.
```bash
make migrate
# or:
docker compose exec api alembic upgrade head
```
is only needed if you run the app outside Docker.

### 4. Try it

```bash
# Health check
curl http://localhost:8000/health \
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
Tell the model it was wrong. It also accepts "agree" votes and UNKNOWN

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

### `GET /health`
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
pip install -r requirements/test.txt

# Run tests
make test

# Lint
make lint

# Run locally without Docker
uvicorn src.main:app --reload
```
### Deployment
Rendered with Docker. Neon is used for Postgres. For Upstash, Redis is selected. After each deployment smoke test applied and CI with branch protection. 

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
make format    → ruff format src/ tests/
make train     → python scripts/train.py
make up        → docker compose up -d 
make down      → docker compose down 
make psql      → docker compose exec db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'
make rebuild   → docker compose build --no-cache api worker
make logs      → docker compose logs -f api worker
make check     → lint typecheck test
make ps        → docker compose ps
make smoke     → @scripts/smoke.sh $(or $(URL),http://localhost:8000)
make clean     → docker compose down -v --remove-orphans
  
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

|── requirements.txt
  |── CONTRIBUTING.md
  |── Makefile
  |── ai-or-not
  |  |── public
  |  |  |── favicon.svg
  |  |  |── icons.svg
  |  |── vite.config.js
  |  |── index.html
  |  |── package.json
  |  |── src
  |  |  |── assets
  |  |  |── App.jsx
  |  |  |── index.css
  |  |  |── App.css
  |  |  |── main.jsx
  |  |── README.md
  |── tests
  |  |── integration
  |  |  |── test_feedback_votes.py
  |  |  |── test_api.py
  |  |  |── test_admin_review.py
  |  |  |── test_drift.py
  |  |  |── __init__.py
  |  |  |── test_database.py
  |  |  |── conftest.cpython-311-pytest-9.0.3.pyc
  |  |  |── test_model.cpython-314-pytest-9.0.3.pyc
  |  |  |── test_database.cpython-314-pytest-9.0.3.pyc
  |  |  |── conftest.cpython-314-pytest-9.0.3.pyc
  |  |  |── __init__.cpython-311.pyc
  |  |  |── test_api.cpython-314-pytest-9.0.3.pyc
  |  |── unit
  |  |  |── test_model_age.py
  |  |  |── test_auth_scopes.py
  |  |  |── test_rate_limit.py
  |  |  |── test_feedback_classification.py
  |  |  |── test_model.py
  |  |  |── __init__.py
  |  |── conftest.py
  |  |── __init__.py
  |── monitoring
  |  |── grafana
  |  |  |── provisioning
  |  |── prometheus.yml
  |── pytest.ini
  |── scripts
  |  |── calibrate.py
  |  |  |── train_config.cpython-314.pyc
  |  |── quantize.py
  |  |── smoke.sh
  |  |── store_screenshots.py
  |  |── train_config.py
  |  |── prepare_dataset.py
  |  |── train.py
  |── data
  |  |── train
  |  |  |── REAL
  |  |  |── FAKE
  |  |── val
  |  |  |── REAL
  |  |  |── FAKE
  |── migrations
  |  |  |── env.cpython-314.pyc
  |  |── script.py.mako
  |  |── README
  |  |── versions
  |  |  |── 5b7e2c91d0a4_drop_feedback_unique_constraint.py
  |  |  |── cf9613fa0af8_baseline_schema.py
  |  |── env.py
  |── frontend
  |  |── favicon.png
  |  |── og.png
  |  |── index.html
  |  |── images.png
  |  |── samples
  |  |  |── unsure.jpg
  |  |  |── real.jpg
  |  |  |── ai.jpg
  |  |── privacy.html
  |── MODEL_CARD.md
  |── docs
  |  |── store
  |  |  |── screenshots
  |  |  |── samples
  |  |── ENGINEERING-DECISIONS.md
  |── LICENSE
  |── src
  |  |── worker
  |  |  |── tasks.py
  |  |  |── __init__.py
  |  |── repositories
  |  |  |── __init__.py
  |  |  |── task_result_repo.py
  |  |  |── prediction_log_repository.py
  |  |  |── feedback_repo.py
  |  |── infra
  |  |  |── task_results.py
  |  |  |── metrics.py
  |  |  |── api_keys.py
  |  |  |── feedback_item.py
  |  |  |── limiter.py
  |  |  |── __init__.py
  |  |  |── database.py
  |  |  |── redis_client.py
  |  |  |── prediction_log.py
  |  |  |── labeled_sample.py
  |  |── inference.py
  |  |  |── inference.cpython-311.pyc
  |  |  |── main.cpython-311.pyc
  |  |  |── inference.cpython-314.pyc
  |  |  |── utils.cpython-314.pyc
  |  |  |── utils.cpython-311.pyc
  |  |  |── config.cpython-311.pyc
  |  |  |── logger.cpython-314.pyc
  |  |  |── __init__.cpython-311.pyc
  |  |  |── config.cpython-314.pyc
  |  |  |── main.cpython-314.pyc
  |  |  |── logger.cpython-311.pyc
  |  |── config.py
  |  |── api
  |  |  |── inference_routes.py
  |  |  |── feedback_routes.py
  |  |  |── rate_limits.py
  |  |  |── __init__.py
  |  |  |── admin_routes.py
  |  |  |── deps.py
  |  |── utils.py
  |  |── services
  |  |  |── feedback_service.py
  |  |  |── cache_service.py
  |  |  |── inference_service.py
  |  |  |── __init__.py
  |  |  |── explainability_service.py
  |  |  |── base_engine.py
  |  |  |── retrain_service.py
  |  |── __init__.py
  |  |── schemas
  |  |  |── batch.py
  |  |  |── feedback.py
  |  |  |── __init__.py
  |  |  |── predict.py
  |  |  |── label.py
  |  |  |── explain.py
  |  |── logger.py
  |  |── main.py
  |  |── core
  |  |  |── exceptions.py
  |  |  |── __init__.py
  |  |  |── enums.py
  |── Dockerfile
  |── requirements-test.txt
  |── alembic.ini
  |── requirements-train.txt
  |── docker-compose.yml
  |── extension
  |  |── icons
  |  |  |── icon128.png
  |  |  |── icon16.png
  |  |  |── icon48.png
  |  |── config.js
  |  |── results.js
  |  |── popup.js
  |  |── popup.html
  |  |── content.js
  |  |── results.html
  |  |── background.js
  |  |── manifest.json
  |── README.md
  |── models
  |  |── model_v1.onnx
  |  |── metadata.json
  |  |── best_checkpoint.pt
  |  |── model_v1.onnx.data
  |── pyproject.toml
  |── railway.json

```

---

## Why I Built This

I wanted to challenge myself beyond notebook projects. Rather than small projects and notebook, I want to find myself in a complex, production environment to deal with.  I focused on understanding what actually happens between "the model works locally" and "the model serves real traffic reliably."

My hands got dirty: dependency injection, session management, async task queues, database migrations, structured logging with request tracing, Docker networking, and what happens when the model starts drifting.

I took my sweet time to build all of that from scratch, make every mistake, and document what I learned.

---

## Future work (v3)

Things I'd build next, roughly in order of how much they'd help.

**Look at images the way detectors in the field do.** Right now every image is shrunk to 224×224 before the model sees it, which throws away much of the fine pixel detail that generators leave behind. The ideas I'd try:
- **Native-resolution patches:** judge several full-resolution crops instead of one shrunken image.
- **Frequency-domain features:** many generators leave regular patterns that show up clearly in an image's frequency spectrum (FFT/DCT).
- **Noise residuals:** filter out the image content and look only at the sensor-or-generator "noise" that remains.
- **Face alignment:** for face images, crop and align the face first so the model compares like with like.

**Check provenance before pixels.** More and more cameras and AI tools sign their images with Content Credentials (C2PA). When a signature is present, it's a stronger answer than any model's guess, so the API should read it first.

**Keep up with new generators.** Detectors go stale as new image models appear. Each retraining should add images from the newest generators, and the held-out test should always include at least one generator the model has never seen.

**Close the feedback loop properly.**
- A dashboard of votes and disagreement rate per model version.
- A fixed "canary" image set evaluated after every deploy.
- Retraining that's triggered by drift but approved by a human before it ships.

**Per-client API keys.** Hashed keys stored in the database, each with its own rate limit and usage history, so keys can be rotated one client at a time.

**No more cold starts.** A paid instance (or a scheduled keep-warm ping) would remove the up-to-a-minute wait after the free server sleeps.

**Test the background worker.** The retraining worker has the least test coverage in the project, and it's the part that changes the model.

---

## Contributing
Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License
MIT