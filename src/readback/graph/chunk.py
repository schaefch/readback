"""Per-chunk subgraph: synthesize -> transcribe -> score -> (review -> regenerate)*.

Mirrors the escalation model of the original pipeline: each stage only runs when the
cheaper stage before it could not clear the chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from readback import observability as obs
from readback.graph.state import ChunkOutput, ChunkResult, ChunkState, Status, Take
from readback.normalize import expand
from readback.providers.base import AudioArtifact
from readback.review import ReviewInput
from readback.scoring import Score, score
from readback.services import Services, resolve

log = logging.getLogger(__name__)


def _audio_path(services: Services, config: RunnableConfig, chunk_id: str, attempt: int) -> Path:
    run = config.get("configurable", {}).get("thread_id", "adhoc")
    return services.settings.audio_dir / str(run) / chunk_id / f"take{attempt}"


async def synthesize(state: ChunkState, runtime: Runtime[Services], config: RunnableConfig) -> dict:
    s = await resolve(runtime.context)
    chunk, attempt = state["chunk"], state.get("attempt", 0) + 1
    text = chunk["text"]
    if attempt > 1:
        obs.REGENERATIONS.inc()
        if s.settings.regeneration_strategy == "speakable":
            text = expand(text, chunk["lang"])
    with obs.timed("tts", s.tts.name):
        audio = await s.tts.synthesize(
            text,
            lang=chunk["lang"],
            out_path=_audio_path(s, config, chunk["id"], attempt),
            attempt=attempt,
        )
    obs.AUDIO_SECONDS.labels(s.tts.name).inc(audio.duration_s or 0)
    audio.meta["tts_input"] = text
    return {"attempt": attempt, "audio": audio.model_dump()}


async def transcribe_and_score(state: ChunkState, runtime: Runtime[Services]) -> dict:
    s = await resolve(runtime.context)
    chunk, audio = state["chunk"], AudioArtifact.model_validate(state["audio"])
    with obs.timed("stt", s.stt.name):
        transcript = await s.stt.transcribe(audio, lang=chunk["lang"])
    result = score(chunk["text"], transcript.text, chunk["lang"])
    take = Take(
        attempt=state["attempt"],
        tts_input=audio.meta.get("tts_input"),
        audio_path=audio.path,
        audio_seconds=audio.duration_s,
        transcript=transcript.text,
        wer=result.wer,
        diffs=[op.describe() for op in result.ops],
        injected_faults=audio.meta.get("injected_faults", []),
        injected_noise=transcript.meta.get("injected_noise", []),
        spoken_text=audio.meta.get("spoken_text"),
    )
    return {"score": result.model_dump(), "takes": [take.model_dump()]}


async def route_after_score(
    state: ChunkState, runtime: Runtime[Services]
) -> Literal["review", "finalize"]:
    settings = (await resolve(runtime.context)).settings
    return "finalize" if state["score"]["wer"] <= settings.wer_threshold else "review"


async def review(state: ChunkState, runtime: Runtime[Services], config: RunnableConfig) -> dict:
    s = await resolve(runtime.context)
    chunk, attempt = state["chunk"], state["attempt"]
    item = ReviewInput(
        chunk_id=chunk["id"],
        lang=chunk["lang"],
        reference=chunk["text"],
        transcript=state["takes"][-1]["transcript"],
        score=Score.model_validate(state["score"]),
        audio=AudioArtifact.model_validate(state["audio"]),
        attempt=attempt,
        attempts_left=max(0, s.settings.max_regenerations - (attempt - 1)),
    )
    with obs.timed("review", s.reviewer.name):
        record = await s.reviewer.review(item, config)
    obs.REVIEWS.labels(record.reviewer, record.decision).inc()
    if record.cost_usd:
        obs.LLM_COST.labels(record.reviewer).inc(record.cost_usd)
    for model, usage in record.usage.items():
        for kind in ("input_tokens", "output_tokens"):
            obs.LLM_TOKENS.labels(model, kind).inc(usage.get(kind, 0))
    log.info("review %s attempt=%d -> %s", chunk["id"], attempt, record.decision)
    return {"reviews": [record.model_dump()]}


async def route_after_review(
    state: ChunkState, runtime: Runtime[Services]
) -> Literal["synthesize", "finalize"]:
    settings = (await resolve(runtime.context)).settings
    last = state["reviews"][-1]
    if last["decision"] == "tts_error" and state["attempt"] <= settings.max_regenerations:
        return "synthesize"
    return "finalize"


async def finalize(state: ChunkState, runtime: Runtime[Services]) -> dict:
    settings = (await resolve(runtime.context)).settings
    chunk, reviews = state["chunk"], state.get("reviews", [])
    status: Status
    if state["score"]["wer"] <= settings.wer_threshold:
        status = "passed"
    elif reviews and reviews[-1]["decision"] == "accept":
        status = "accepted_by_review"
    else:
        status = "needs_human"
    obs.CHUNKS.labels(status).inc()
    result = ChunkResult(
        chunk_id=chunk["id"],
        source=chunk["source"],
        category=chunk["category"],
        expect=chunk.get("expect", ""),
        lang=chunk["lang"],
        text=chunk["text"],
        status=status,
        takes=[Take.model_validate(t) for t in state["takes"]],
        reviews=reviews,
    )
    return {"results": [result.model_dump()]}


def build_chunk_graph():
    g = StateGraph(ChunkState, context_schema=Services, output_schema=ChunkOutput)
    g.add_node("synthesize", synthesize)
    g.add_node("transcribe_and_score", transcribe_and_score)
    g.add_node("review", review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "synthesize")
    g.add_edge("synthesize", "transcribe_and_score")
    g.add_conditional_edges("transcribe_and_score", route_after_score)
    g.add_conditional_edges("review", route_after_review)
    g.add_edge("finalize", END)
    return g.compile(name="check_chunk")
