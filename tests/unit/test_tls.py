import ssl

import boto3

from lexeu.core.tls import system_ssl_context, use_system_trust


def test_injection_is_idempotent_and_contexts_still_build() -> None:
    use_system_trust()
    use_system_trust()  # the CLI callback and the tests both call it
    ctx = system_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_boto3_still_works_after_injection() -> None:
    # truststore + botocore recursed on Linux (CI ingest failure); the raw store uses boto3.
    use_system_trust()
    s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9", region_name="us-east-1")
    assert s3.meta.region_name == "us-east-1"
