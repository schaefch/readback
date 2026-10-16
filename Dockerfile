# syntax=docker/dockerfile:1
# Default image: API + Postgres checkpointer, cloud engines via env.
# Local engines (Piper / faster-whisper) add ~1 GB: --build-arg EXTRAS=local,postgres
FROM python:3.13-slim AS build
ARG EXTRAS=postgres,openai
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"

FROM python:3.13-slim
# Pre-create /data/audio so named volumes inherit the non-root owner.
RUN useradd --create-home --uid 10001 app && mkdir -p /data/audio && chown -R app /data
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    READBACK_AUDIO_DIR=/data/audio \
    READBACK_CHECKPOINTER=sqlite:////data/checkpoints.sqlite \
    READBACK_CORPUS_ROOT=/app/corpus
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY corpus ./corpus
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "readback.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
