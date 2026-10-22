from sqlalchemy import Connection, inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

from hexrl_platform.db.models import Base


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            # API и несколько воркеров могут стартовать одновременно: DDL сериализуем.
            await connection.execute(text("SELECT pg_advisory_xact_lock(1212504140)"))
        await connection.run_sync(_upgrade_schema)


def _upgrade_schema(connection: Connection) -> None:
    Base.metadata.create_all(connection)
    columns = {item["name"] for item in inspect(connection).get_columns("inference_results")}
    if "model_name" not in columns:
        connection.execute(text("ALTER TABLE inference_results ADD COLUMN model_name VARCHAR(256)"))
