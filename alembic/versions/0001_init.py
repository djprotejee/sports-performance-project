"""init predictions table

Revision ID: 0001_init
Revises: 
Create Date: 2025-24-11 00:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("athlete_id", sa.String(), nullable=True),
        sa.Column("sport", sa.String(), nullable=True),
        sa.Column("gender", sa.String(), nullable=True),
        sa.Column("performance_score", sa.Float(), nullable=False),
        sa.Column("performance_class", sa.String(), nullable=False),
        sa.Column("model_regressor", sa.String(), nullable=False),
        sa.Column("model_classifier", sa.String(), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("extra_info", sa.JSON(), nullable=True),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")
