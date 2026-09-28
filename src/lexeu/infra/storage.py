"""Raw-document store on any S3-compatible backend (RustFS locally, S3/R2 in the cloud).

Objects are content-addressed (`raw/{celex}/{lang}/{sha256}.xhtml`): a new version of an act
never overwrites the previous one, so every chunk can be traced to the exact bytes it came from.
"""

import asyncio
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

from lexeu.core.config import ObjectStoreSettings

if TYPE_CHECKING:  # stubs are a dev dependency, absent from the runtime image
    from types_boto3_s3 import S3Client


class S3RawStore:
    def __init__(self, settings: ObjectStoreSettings) -> None:
        self._bucket = settings.bucket_raw
        self._s3: S3Client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key.get_secret_value(),
            region_name="us-east-1",
        )

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            try:
                self._s3.head_bucket(Bucket=self._bucket)
            except ClientError:
                self._s3.create_bucket(Bucket=self._bucket)

        await asyncio.to_thread(_ensure)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self._s3.put_object, Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )

    async def get(self, key: str) -> bytes:
        resp = await asyncio.to_thread(self._s3.get_object, Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._s3.head_object, Bucket=self._bucket, Key=key)
        except ClientError:
            return False
        return True
