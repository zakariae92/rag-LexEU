# syntax=docker/dockerfile:1
# Embedding server for ARM64 (and x86): BGE-M3's ONNX export behind TEI's API, model baked in.
# TEI publishes x86 CPU images only; see src/lexeu/embed_server.py. The model is part of the image
# so a (re)start never depends on the Hugging Face Hub.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS model
ARG MODEL_ID=BAAI/bge-m3
ARG REVISION=5617a9f61b028005a4858fdac845db406aefb181
RUN --mount=type=cache,target=/root/.cache/pip pip install "huggingface_hub==1.*"
# Only what the ONNX runtime path reads (not the 2.2 GB PyTorch weights). HF_HOME holds the
# download tool's own chunk cache, kept out of the image.
RUN HF_HOME=/tmp/hf hf download "$MODEL_ID" --revision "$REVISION" --cache-dir /data \
      --include "onnx/*" --include "tokenizer.json" --include "1_Pooling/*" \
 && rm -rf /data/.locks /tmp/hf
# A plain directory: -L copies the files behind the Hub cache's symlinks.
RUN cp -rL "/data/models--$(echo "$MODEL_ID" | sed 's|/|--|')/snapshots/$REVISION" /model


FROM python:${PYTHON_VERSION}-slim AS runtime
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --no-create-home app
WORKDIR /app
COPY docker/embeddings.requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-compile -r requirements.txt
COPY --from=model --chown=app:app /model /model
COPY --chown=app:app src/lexeu/embed_server.py ./
ENV MODEL_DIR=/model \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:80/health', timeout=2)"]
CMD ["uvicorn", "embed_server:from_env", "--factory", "--host", "0.0.0.0", "--port", "80", "--no-access-log"]
