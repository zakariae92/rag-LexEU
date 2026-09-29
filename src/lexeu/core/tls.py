"""Trust the OS certificate store instead of certifi.

Needed on the Windows/macOS dev machine, where antivirus or corporate proxies inspect TLS with a
CA that only the OS store knows. On Linux (CI, containers) OpenSSL already reads the system store,
and truststore's injection breaks botocore there (RecursionError when boto3 builds its
SSLContext), so it is skipped.
"""

import ssl
import sys

import truststore

_injected = False


def use_system_trust() -> None:
    """Make every `ssl` client in the process (httpx, HuggingFace, Modal) use the OS store."""
    global _injected
    if not _injected and sys.platform in ("win32", "darwin"):
        truststore.inject_into_ssl()
        _injected = True


def system_ssl_context() -> ssl.SSLContext:
    use_system_trust()
    return ssl.create_default_context()
