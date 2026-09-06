import redis.asyncio as aioredis

from src.config import settings
from src.logger import logger


async def create_cache_client() -> aioredis.Redis:
    """
    Creates a standalone async Redis client for the phash cache
    uses database 1 to keep cache keys isolated from ARQ job data (db 0)
    """
    url = settings.REDIS_URL.rsplit("/", 1)[0] + "/1"

    client = aioredis.from_url(
        url, decode_responses=False, socket_connect_timeout=5, socket_timeout=5
    )

    # verify connection at start
    try:
        await client.ping()
        logger.info("Redis cache client connected")

    except Exception as e:
        logger.warning(f"Redis cache client can't connect: {e}")

    return client
