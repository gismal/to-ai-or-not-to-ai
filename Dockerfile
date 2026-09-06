# ── Builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use CPU-only PyTorch index — reduces image by ~1.7GB
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── Runner ────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*

COPY . .

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]