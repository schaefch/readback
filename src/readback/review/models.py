from __future__ import annotations

from typing import Any, Literal, Protocol

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from readback.lexicon import Lang
from readback.providers.base import AudioArtifact
from readback.scoring import Score

OpCategory = Literal["stt_artifact", "homophone", "normalization_gap", "tts_error", "unclear"]


class OpAssessment(BaseModel):
    """Judgement on one difference between reference and transcript."""

    diff: str = Field(description="The difference as shown in the diff list, e.g. \"'a' -> 'b'\"")
    category: OpCategory
    explanation: str = Field(description="One short sentence")


class Verdict(BaseModel):
    """Final decision for one audio chunk."""

    decision: Literal["accept", "tts_error", "uncertain"] = Field(
        description=(
            "accept: every difference is explained by transcription/normalization; "
            "tts_error: the audio itself deviates from the text; "
            "uncertain: a human must listen"
        )
    )
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="Two sentences at most")
    issues: list[OpAssessment] = Field(default_factory=list)


class ReviewRecord(Verdict):
    """A verdict plus bookkeeping added by the pipeline (not produced by the model)."""

    reviewer: str
    attempt: int
    usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0


class ReviewInput(BaseModel):
    chunk_id: str
    lang: Lang
    reference: str
    transcript: str
    score: Score
    audio: AudioArtifact | None = None
    attempt: int = 1
    attempts_left: int = 0


class Reviewer(Protocol):
    name: str

    async def review(
        self, item: ReviewInput, config: RunnableConfig | None = None
    ) -> ReviewRecord: ...
