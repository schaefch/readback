"""Prometheus metrics and JSON logging. LangSmith tracing is enabled purely via env
(``LANGSMITH_TRACING=true``, ``LANGSMITH_API_KEY``) and needs no code here."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

CHUNKS = Counter("readback_chunks_total", "Chunks finished, by outcome", ["status"])
STAGE_SECONDS = Histogram(
    "readback_stage_seconds",
    "Wall time per pipeline stage",
    ["stage", "provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
REVIEWS = Counter("readback_reviews_total", "Review verdicts", ["reviewer", "decision"])
REGENERATIONS = Counter("readback_regenerations_total", "TTS re-synthesis attempts")
LLM_TOKENS = Counter("readback_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST = Counter("readback_llm_cost_usd_total", "Estimated LLM cost in USD", ["reviewer"])
AUDIO_SECONDS = Counter("readback_audio_seconds_total", "Synthesized audio", ["provider"])


@contextmanager
def timed(stage: str, provider: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_SECONDS.labels(stage, provider).observe(time.perf_counter() - start)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
