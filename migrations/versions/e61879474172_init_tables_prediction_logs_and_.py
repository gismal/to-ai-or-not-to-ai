"""Init tables: prediction_logs and feedback_logs

Revision ID: e61879474172
Revises:
Create Date: 2026-06-19 13:30:30.522706

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "e61879474172"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
