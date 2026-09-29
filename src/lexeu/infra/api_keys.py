"""API keys: created once (the secret is shown a single time), stored as a hash, revocable."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from lexeu.infra.db import ApiKeyRow

KEY_PREFIX = "lx_"


@dataclass(frozen=True)
class ApiKey:
    id: int
    name: str
    prefix: str
    rate_limit_per_min: int
    created_at: datetime | None = None
    revoked_at: datetime | None = None


def hash_key(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def new_secret() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)  # 256 bits of randomness


def _key(row: ApiKeyRow) -> ApiKey:
    return ApiKey(
        row.id, row.name, row.prefix, row.rate_limit_per_min, row.created_at, row.revoked_at
    )


class ApiKeyStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    async def create(self, name: str, rate_limit_per_min: int) -> tuple[str, ApiKey]:
        """Returns the secret (to hand over once, it cannot be recovered) and the key record."""
        secret = new_secret()
        row = ApiKeyRow(
            name=name,
            prefix=secret[:10],
            key_hash=hash_key(secret),
            rate_limit_per_min=rate_limit_per_min,
        )
        async with self._session.begin() as s:
            s.add(row)
        return secret, _key(row)

    async def authenticate(self, secret: str) -> ApiKey | None:
        """The active key matching this secret. Lookup is by hash (indexed, constant time)."""
        if not secret.startswith(KEY_PREFIX):
            return None
        async with self._session() as s:
            row = await s.scalar(
                select(ApiKeyRow).where(
                    ApiKeyRow.key_hash == hash_key(secret), ApiKeyRow.revoked_at.is_(None)
                )
            )
        return _key(row) if row else None

    async def list(self) -> list[ApiKey]:
        async with self._session() as s:
            return [_key(r) for r in await s.scalars(select(ApiKeyRow).order_by(ApiKeyRow.id))]

    async def revoke(self, name: str) -> bool:
        async with self._session.begin() as s:
            row = await s.scalar(select(ApiKeyRow).where(ApiKeyRow.name == name))
            if row is None or row.revoked_at is not None:
                return False
            row.revoked_at = datetime.now(UTC)
        return True
