from sqlalchemy.ext.asyncio import AsyncEngine

from hexrl_platform.db.models import Base


async def create_schema(engine: AsyncEngine) -> None:
    # осознанная замена Alembic: см. docs/backpressure/worker.md
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
