"""add_task_results_table

Revision ID: 22f7235b02a8
Revises: e61879474172
Create Date: 2026-06-23 13:37:31.697304

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "22f7235b02a8"
down_revision: str | Sequence[str] | None = "e61879474172"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
