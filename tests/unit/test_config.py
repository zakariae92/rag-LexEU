import pytest

from lexeu.core.config import Settings


def test_defaults_point_to_local_stack() -> None:
    s = Settings(_env_file=None)
    assert s.env == "local"
    assert s.qdrant.url == "http://127.0.0.1:6333"
    assert s.postgres.dsn == "postgresql+asyncpg://lexeu:lexeu@127.0.0.1:5432/lexeu"


def test_nested_env_vars_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("POSTGRES__HOST", "db.internal")
    monkeypatch.setenv("POSTGRES__PASSWORD", "s3cret")
    monkeypatch.setenv("QDRANT__URL", "http://qdrant:6333")

    s = Settings(_env_file=None)

    assert s.env == "prod"
    assert s.postgres.dsn == "postgresql+asyncpg://lexeu:s3cret@db.internal:5432/lexeu"
    assert s.qdrant.url == "http://qdrant:6333"


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES__PASSWORD", "s3cret")
    s = Settings(_env_file=None)
    assert "s3cret" not in repr(s)
