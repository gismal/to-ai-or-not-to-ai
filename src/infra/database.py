from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from src.config import settings

is_sqlite = str(settings.DATABASE_URL).startswith("sqlite")

engine_kwargs = {"echo": settings.DEBUG}

if not is_sqlite:
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": 20,
        "max_overflow": 10,
    })

engine = create_async_engine(
    str(settings.DATABASE_URL),
    **engine_kwargs
)
    
# creates new session each request
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush = False)

class Base(DeclarativeBase):
    pass

