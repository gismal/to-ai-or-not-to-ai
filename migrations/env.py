import asyncio
import sys
from pathlib import Path
from logging.config import fileConfig

# Proje kök dizinini Python yoluna ekle
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Kendi ayarlarımızı ve modellerimizi içeri aktarıyoruz
from src.config import settings
from src.infra.database import Base
from src.infra.feedbacks import FeedbackItem, PredictionLog

# Alembic Config nesnesi
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    """Synchronous migration runner — called by run_sync inside the async engine."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch = True   #SQLite ALTER TABLE support
        )
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """Creates an async engine and runs migrations via run_sync."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())
    
def run_mitigations_offline() -> None:
    context.configure(
        url = url,
        target_metadata = target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True
    )