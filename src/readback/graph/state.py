from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from readback.corpus import Chunk
from readback.lexicon import Lang
from readback.results import ChunkResult, HumanDecision, Status, Take

__all__ = [
    "ChunkOutput",
    "ChunkResult",
    "ChunkState",
    "HumanDecision",
    "PipelineInput",
    "PipelineState",
    "Status",
    "Take",
    "merge_results",
]


def merge_results(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append new chunk results; a result with a known ``chunk_id`` replaces the old one."""
    merged = {r["chunk_id"]: r for r in left}
    merged.update({r["chunk_id"]: r for r in right})
    return list(merged.values())


class ChunkState(TypedDict, total=False):
    chunk: Chunk
    attempt: int
    takes: Annotated[list[dict[str, Any]], operator.add]
    audio: dict[str, Any]
    score: dict[str, Any]
    reviews: Annotated[list[dict[str, Any]], operator.add]
    results: Annotated[list[dict[str, Any]], merge_results]


class ChunkOutput(TypedDict):
    results: Annotated[list[dict[str, Any]], merge_results]


class PipelineInput(TypedDict, total=False):
    sources: list[str]
    """Corpus files or directories (YAML / *.de.txt / *.en.txt)."""
    text: str
    """Alternatively: free text to check."""
    lang: Lang
    limit: int
    chunks: list[Chunk]


class PipelineState(PipelineInput, total=False):
    results: Annotated[list[dict[str, Any]], merge_results]
    report: dict[str, Any]
