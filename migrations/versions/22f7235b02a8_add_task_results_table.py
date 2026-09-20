"""add_task_results_table

Revision ID: 22f7235b02a8
Revises: e61879474172
Create Date: 2026-06-23 13:37:31.697304

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "22f7235b02a8"
down_revision: Union[str, Sequence[str], None] = "e61879474172"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
