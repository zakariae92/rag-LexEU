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


def test_langfuse_standard_variable_names_configure_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    t = Settings(_env_file=None).tracing
    assert t.langfuse_public_key == "pk-lf-test"
    assert t.langfuse_secret_key is not None
    assert t.langfuse_secret_key.get_secret_value() == "sk-lf-test"
    assert t.langfuse_host == "https://us.cloud.langfuse.com"


def test_explicit_tracing_langfuse_keys_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-standard")
    monkeypatch.setenv("TRACING__LANGFUSE_PUBLIC_KEY", "pk-lf-explicit")

    assert Settings(_env_file=None).tracing.langfuse_public_key == "pk-lf-explicit"


def test_managed_postgres_uses_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES__SSL", "true")
    assert Settings(_env_file=None).postgres.dsn.endswith("?ssl=require")
    assert "ssl" not in Settings(_env_file=None, postgres={"ssl": False}).postgres.dsn
