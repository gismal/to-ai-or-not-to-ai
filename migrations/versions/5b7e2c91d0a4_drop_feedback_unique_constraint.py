"""drop feedback unique constraint

Revision ID: 5b7e2c91d0a4
Revises: cf9613fa0af8
Create Date: 2026-11-19 14:10:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "5b7e2c91d0a4"
down_revision: str | None = "cf9613fa0af8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Allow more than one vote per filename and client."""
    op.execute(
        "ALTER TABLE feedback_logs DROP CONSTRAINT IF EXISTS uq_feedback_filename_source"
    )


def downgrade() -> None:
    """Restores the constraint. Fails if duplicate votes exist by then."""
    op.create_unique_constraint(
        "uq_feedback_filename_source", "feedback_logs", ["filename", "client_source"]
    )
