"""
Limits the request rate  per-IP rate for the public demo key

Real keys are untouched here. The global slowapi limits apply to everyone.
Counts live in Redis when it is available and fall back to process memory, so a Redis
outage never switches the demo limit off.

"""

import time

from fastapi import Depends, HTTPException, Request, status

from src.api.deps import _matches, verify_public_key
from src.config import settings
from src.logger import logger

_memory: dict[str, int] = {}


def _incr_memo(key: str) -> int:
    """
    Responsible for counting the request and cleaning stale entries if needed

    Args:
        key (str): request key to search in  memo and increment

    Returns:
        int: Updated request count
    """
    if len(_memory) > 10_000:
        current = key.rsplit(":", 1)[1]
        for stale in [k for k in _memory if k.rsplit(":", 1)[1] != current]:
            # deletes the old requests
            del _memory[stale]
    # increases the memory counter
    _memory[key] = _memory.get(key, 0) + 1

    return _memory[key]


async def _increment(request: Request, key: str, window: int) -> int:
    """
    Increases the counter from Redis. If Redis is unavailable, it logs a warning and runs the
    _incr_memo in fallback situation.

    Args:
        request (Request): received client requests
        key (str): unique rate-limit key for the IP and time window
        window (int): Rate limit window in seconds

    Returns:
        int: Updated request count

    """
    client = getattr(request.app.state, "cache_client", None)
    if client is not None:
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window + 1)
            count, _ = await pipe.execute()
            return int(count)
        except Exception as e:
            logger.warning(f"Demo rate limit: Redis unavailable ({e}), using memory")

    return _incr_memo(key)


def demo_limit(scope: str, limit: int | None = None, window: int = 60):
    """
    FastAPI dependency that rate limits requests per IP for the demo key

    Args:
            scope (str): Context identifier for the route (inference, feedback etc.)
            limit (int|None): Max allowed request limit defined per window in config.py
            window (int): Time window size in seconds

    Raises:
        HTTPExceptipon: returns 429 status code if the rate limit is exceeded
    """

    async def dependency(
        request: Request, api_key: str = Depends(verify_public_key)
    ) -> None:
        if not _matches(api_key, settings.DEMO_API_KEY):
            return  # pass for real keys
        capacity = limit if limit is not None else settings.DEMO_RATE_LIMIT_PER_MIN
        ip = request.client.host if request.client else "unknown"
        count = await _increment(
            request, f"demo_rl:{scope}:{ip}:{int(time.time() // window)}", window
        )
        if count > capacity:
            retry = window - int(time.time() % window)
            logger.warning(f"Demo limit hit: scope = {scope} ip = {ip}")

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Demo rate limit reached. Wait a moment or use your own key.",
                headers={"Retry-After": str(retry)},
            )

    return dependency
