# 0003: RustFS as the local S3-compatible object store

**Status:** Accepted

## Context

Ingestion follows a *raw → processed* layout: original EUR-Lex HTML and EDPB PDFs are kept
untouched, so documents can be re-parsed without re-downloading and every chunk traces back to
its source file. Code must talk to the **S3 API**, so production can use any S3 provider without
changes.

The initial plan was MinIO. In 2025 MinIO stopped distributing community-edition binaries and
Docker images, so the `minio/minio` image is no longer maintained.

## Decision

Use **RustFS** (Apache-2.0, S3-compatible, single container, with a health endpoint and a web console) locally.
The application only uses the generic S3 API, configured through `OBJECT_STORE__*` settings.

## Consequences

- Swapping to AWS S3, Cloudflare R2, Garage or SeaweedFS is a config change.
- RustFS reached 1.0 recently. If it causes problems, **SeaweedFS** is the fallback.
- Rejected: plain local folders (no S3 parity with production), an unmaintained MinIO image.
