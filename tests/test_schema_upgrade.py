from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hexrl_platform.db.schema import create_schema


async def test_old_result_table_is_upgraded_without_losing_rows(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'old.db').as_posix()}")
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE inference_results (task_id CHAR(32) PRIMARY KEY, action INTEGER, "
                    "model_version VARCHAR(64), inference_ms FLOAT, "
                    "delay_ms FLOAT, created_at DATETIME)"
                )
            )
            await connection.execute(
                text(
                    "INSERT INTO inference_results VALUES "
                    "('legacy', 3, '7', 1.0, 500.0, '2026-01-01')"
                )
            )
        await create_schema(engine)
        await create_schema(engine)
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text(
                        "SELECT action, model_version, model_name "
                        "FROM inference_results WHERE task_id='legacy'"
                    )
                )
            ).one()
        assert row == (3, "7", None)
    finally:
        await engine.dispose()
