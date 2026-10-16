"""Per-chunk results: what the pipeline produces and what reports consume."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from readback.lexicon import Lang

Status = Literal["passed", "accepted_by_review", "needs_human", "human_approved", "human_rejected"]


class Take(BaseModel):
    """One synthesis attempt."""

    attempt: int
    tts_input: str | None = None
    audio_path: str
    audio_seconds: float | None
    transcript: str
    wer: float
    diffs: list[str]
    injected_faults: list[dict[str, Any]] = Field(default_factory=list)
    injected_noise: list[dict[str, Any]] = Field(default_factory=list)
    spoken_text: str | None = None  # only known for fake TTS (ground truth)


class ChunkResult(BaseModel):
    chunk_id: str
    source: str
    category: str
    expect: str = ""
    lang: Lang
    text: str
    status: Status
    takes: list[Take]
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    human: dict[str, Any] | None = None

    @property
    def final(self) -> Take:
        return self.takes[-1]


class HumanDecision(BaseModel):
    approved: bool
    note: str = ""
    reviewer: str = "human"
