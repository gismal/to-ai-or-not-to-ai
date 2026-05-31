from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from src.config import settings

engine = create_async_engine(
    settings.str(settings.DATABASE_URL),
    echo= settings.DEBUG,
    pool_size = 20,
    max_overflow = 10,
    pool_pre_ping = True
    )

# creates new session each request
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush = False)

class Base(DeclarativeBase):
    pass