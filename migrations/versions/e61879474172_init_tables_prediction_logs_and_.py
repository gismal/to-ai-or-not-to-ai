"""Init tables: prediction_logs and feedback_logs

Revision ID: e61879474172
Revises:
Create Date: 2026-06-19 13:30:30.522706

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "e61879474172"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
