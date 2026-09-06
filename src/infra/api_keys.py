"""
Multi-key authentication system.

Why multiple API keys?
  - Rotate one key without breaking other clients
  - Per-client rate limits (public API vs trusted partners)
  - Track which client is responsible for which traffic
  - Audit trail: last_used_at per key

The actual key is never stored, only its SHA256 hash.
Verification: hash(incoming_key) == stored_hash
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy import Enum as SQLEnum

from src.infra.database import Base


class APIKeyTier(str, enum.Enum):
    PUBLIC = "PUBLIC"  # low rate limit, public access
    PARTNER = "PARTNER"  # higher rate limit, trusted clients
    ADMIN = "ADMIN"  # no rate limit, internal tools


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, unique=True, nullable=False, index=True)
    client_name = Column(String, nullable=False)
    tier = Column(SQLEnum(APIKeyTier), default=APIKeyTier.PUBLIC, nullable=False)  # type: ignore
    is_active = Column(Boolean, default=True, nullable=False)
    rate_limit = Column(Integer, default=60)  # requests per minute
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
