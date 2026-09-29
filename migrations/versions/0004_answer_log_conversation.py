"""answer log: small talk and the standalone form of follow-up questions

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-29 20:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "answers",
        sa.Column("conversation", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column("answers", sa.Column("standalone_question", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("answers", "standalone_question")
    op.drop_column("answers", "conversation")
