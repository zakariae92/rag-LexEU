from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexeu.api.main import create_app
from lexeu.core.config import Settings
from lexeu.core.tls import use_system_trust

# Model downloads (fastembed BM25) must trust the OS store, as the CLI does.
use_system_trust()


@pytest.fixture
def settings() -> Settings:
    # Ignore any developer .env so tests are hermetic.
    return Settings(_env_file=None, env="ci", probe_timeout_s=0.2)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:  # runs lifespan
        yield c
