"""OpenTelemetry tracing: one trace per request, one span per stage.

Spans go over OTLP to any backend: Jaeger locally, and Langfuse (which reads the GenAI semantic
conventions: model, tokens, cost, input and output) when its keys are set. Instrumenting with the
standard rather than a vendor SDK keeps the backend a configuration choice.

`capture_content=False` keeps questions, prompts and answers out of traces, for deployments where
user text must not leave the service (GDPR: questions can contain personal data).
"""

import base64

import structlog
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from lexeu import __version__
from lexeu.core.config import Settings
from lexeu.observability.spans import set_capture_content

log = structlog.get_logger(__name__)

EXPORT_TIMEOUT_S = 3  # a monitoring backend that is down must not hold the API (e.g. at shutdown)


def langfuse_otlp(host: str, public_key: str, secret_key: str) -> tuple[str, dict[str, str]]:
    """Langfuse's OTLP endpoint and Basic-auth header (public key as user, secret as password)."""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"{host.rstrip('/')}/api/public/otel/v1/traces", {"Authorization": f"Basic {token}"}


def langfuse_exporter(host: str, public_key: str, secret_key: str) -> OTLPSpanExporter:
    endpoint, headers = langfuse_otlp(host, public_key, secret_key)
    return OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=EXPORT_TIMEOUT_S)


def setup_tracing(
    app: FastAPI, settings: Settings, extra_exporter: SpanExporter | None = None
) -> TracerProvider | None:
    """Install the tracer provider and instrument the app. Returns None when tracing is off."""
    cfg = settings.tracing
    if not cfg.enabled:
        return None
    set_capture_content(cfg.capture_content)
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": cfg.service_name,
                "service.version": __version__,
                "deployment.environment.name": settings.env,
            }
        ),
        sampler=ParentBased(TraceIdRatioBased(cfg.sample_ratio)),
    )
    exporters: list[tuple[str, SpanExporter]] = []
    if cfg.otlp_endpoint:
        exporters.append(
            ("otlp", OTLPSpanExporter(endpoint=cfg.otlp_endpoint, timeout=EXPORT_TIMEOUT_S))
        )
    if cfg.langfuse_public_key and cfg.langfuse_secret_key:
        secret = cfg.langfuse_secret_key.get_secret_value()
        exporters.append(
            ("langfuse", langfuse_exporter(cfg.langfuse_host, cfg.langfuse_public_key, secret))
        )
    if extra_exporter is not None:
        exporters.append(("extra", extra_exporter))
    for _, exporter in exporters:
        # Batched and asynchronous: exporting never adds latency to a request.
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="health,ready,metrics"
    )
    log.info("tracing_enabled", exporters=[name for name, _ in exporters], sample=cfg.sample_ratio)
    return provider
