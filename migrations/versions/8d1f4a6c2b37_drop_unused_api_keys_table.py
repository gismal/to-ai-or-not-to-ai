"""Drop the unused api_keys table.

Auth uses two keys from settings (the real key and the demo key). The api_keys table
came from the baseline migration, for a per-client key system that was never built.

Revision ID: 8d1f4a6c2b37
Revises: 5b7e2c91d0a4
"""

import sqlalchemy as sa
from alembic import op

revision = "8d1f4a6c2b37"
down_revision = "5b7e2c91d0a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS api_keys")
    op.execute(
        "DROP TYPE IF EXISTS apikeytier"
    )  # Postgres keeps enum types after their table is gone


def downgrade() -> None:
    """Recreate the table exactly as the baseline migration did."""
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key_hash", sa.String(), nullable=False),
        sa.Column("client_name", sa.String(), nullable=False),
        sa.Column(
            "tier",
            sa.Enum("PUBLIC", "PARTNER", "ADMIN", name="apikeytier"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("rate_limit", sa.Integer(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_api_keys_id"), ["id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_api_keys_key_hash"), ["key_hash"], unique=True
        )
