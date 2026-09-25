import pytest
from pydantic import ValidationError

from hexrl_platform.core.config import Settings


def test_db_dsn():
    s = Settings(
        postgres_user="u", postgres_password="p", postgres_db="d", postgres_host="h", _env_file=None
    )
    assert s.db_dsn == "postgresql+asyncpg://u:p@h:5432/d"


def test_required_db_settings(monkeypatch):
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(name)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
