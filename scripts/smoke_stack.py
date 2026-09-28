"""Smoke-test every service of the local stack with a real write -> read -> cleanup.

Usage (stack running, from the repo root):
    uv run --with boto3 python scripts/smoke_stack.py
"""

import asyncio
import os
import struct
import time
import uuid
from collections.abc import Awaitable, Callable

import asyncpg
import boto3
import httpx
from qdrant_client import AsyncQdrantClient, models
from redis.asyncio import Redis

from lexeu.core.config import get_settings

s = get_settings()
API = f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}"


async def check_qdrant() -> str:
    q = AsyncQdrantClient(url=s.qdrant.url)
    name = "smoke_test"
    try:
        if await q.collection_exists(name):
            await q.delete_collection(name)
        await q.create_collection(
            name, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )
        await q.upsert(
            name,
            points=[
                models.PointStruct(
                    id=1, vector=[1, 0, 0, 0], payload={"art": "GDPR Art. 6", "lang": "en"}
                ),
                models.PointStruct(
                    id=2, vector=[0, 1, 0, 0], payload={"art": "RGPD Art. 6", "lang": "fr"}
                ),
            ],
        )
        hits = await q.query_points(
            name,
            query=[0.9, 0.1, 0, 0],
            query_filter=models.Filter(
                must=[models.FieldCondition(key="lang", match=models.MatchValue(value="en"))]
            ),
            limit=1,
        )
        top = hits.points[0]
        assert top.payload and top.payload["art"] == "GDPR Art. 6"
        return f"vector search + filter OK -> {top.payload['art']} (score {top.score:.3f})"
    finally:
        await q.delete_collection(name)
        await q.close()


async def check_postgres() -> str:
    p = s.postgres
    conn = await asyncpg.connect(
        host=p.host, port=p.port, user=p.user, password=p.password.get_secret_value(), database=p.db
    )
    try:
        version = await conn.fetchval("SHOW server_version")
        await conn.execute("CREATE TEMP TABLE smoke (celex text, title text)")
        await conn.execute("INSERT INTO smoke VALUES ('32016R0679', 'GDPR')")
        title = await conn.fetchval("SELECT title FROM smoke WHERE celex = '32016R0679'")
        assert title == "GDPR"
        return f"PostgreSQL {version}, write/read OK"
    finally:
        await conn.close()


async def check_redis() -> str:
    r = Redis.from_url(s.redis.url)
    idx, key = "smoke_idx", "smoke:1"
    try:
        await r.set("smoke:kv", "ok", ex=10)
        assert await r.get("smoke:kv") == b"ok"

        # Vector search = what the semantic cache (RedisVL) needs.
        await r.execute_command(
            "FT.CREATE", idx, "ON", "HASH", "PREFIX", "1", "smoke:", "SCHEMA",
            "question", "TEXT",
            "vec", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32", "DIM", "4", "DISTANCE_METRIC", "COSINE",
        )  # fmt: skip
        await r.hset(key, mapping={
            "question": "What is a DPIA?",
            "vec": struct.pack("4f", 1, 0, 0, 0),
        })  # fmt: skip
        res = await r.execute_command(
            "FT.SEARCH", idx, "*=>[KNN 1 @vec $q AS dist]",
            "PARAMS", "2", "q", struct.pack("4f", 0.95, 0.05, 0, 0),
            "RETURN", "2", "question", "dist", "DIALECT", "2",
        )  # fmt: skip
        # redis-py 8 speaks RESP3: FT.SEARCH returns a dict, not a flat list.
        fields = res[b"results"][0][b"extra_attributes"]
        return (
            f"key/value OK, vector search OK -> '{fields[b'question'].decode()}' "
            f"(distance {float(fields[b'dist']):.3f})"
        )
    finally:
        await r.execute_command("FT.DROPINDEX", idx, "DD")
        await r.delete("smoke:kv")
        await r.aclose()


async def check_object_store() -> str:
    o = s.object_store
    s3 = boto3.client(
        "s3",
        endpoint_url=o.endpoint,
        aws_access_key_id=o.access_key,
        aws_secret_access_key=o.secret_key.get_secret_value(),
        region_name="us-east-1",
    )
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    if o.bucket_raw not in existing:
        s3.create_bucket(Bucket=o.bucket_raw)
    key = f"smoke/{uuid.uuid4().hex}.html"
    body = b"<p>Article 6 - Lawfulness of processing</p>"
    s3.put_object(Bucket=o.bucket_raw, Key=key, Body=body, ContentType="text/html")
    try:
        got = s3.get_object(Bucket=o.bucket_raw, Key=key)["Body"].read()
        assert got == body
        return f"bucket '{o.bucket_raw}' ready, upload/download OK ({len(body)} bytes)"
    finally:
        s3.delete_object(Bucket=o.bucket_raw, Key=key)


async def check_api() -> str:
    async with httpx.AsyncClient(base_url=API, timeout=5) as c:
        health = (await c.get("/health")).json()
        ready = await c.get("/ready")
        deps = ready.json()["dependencies"]
        up = [n for n, d in deps.items() if d["status"] == "up"]
        assert ready.status_code == 200, ready.text
        return f"{API} v{health['version']}, /ready 200 with {len(up)}/{len(deps)} deps up"


CHECKS: dict[str, Callable[[], Awaitable[str]]] = {
    "Qdrant": check_qdrant,
    "PostgreSQL": check_postgres,
    "Redis": check_redis,
    "Object store (RustFS)": check_object_store,
    "API (FastAPI)": check_api,
}


async def main() -> int:
    failures = 0
    for name, check in CHECKS.items():
        start = time.perf_counter()
        try:
            detail = await check()
            status = "PASS"
        except Exception as exc:
            detail, status = f"{type(exc).__name__}: {exc}", "FAIL"
            failures += 1
        ms = (time.perf_counter() - start) * 1000
        print(f"[{status}] {name:<22} {ms:7.1f} ms  {detail}")
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
