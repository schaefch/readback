"""LLM reviewer: a small LangChain agent with language tools and structured output."""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import BaseTool

from readback.review.models import ReviewInput, ReviewRecord, Verdict

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You review audiobook recordings. A TTS engine read a text aloud and an STT engine \
transcribed the audio back. You receive the original text, the transcript and the \
word differences that remain after both were normalized.

Decide whether the audio itself deviates from the text. A difference may be accepted \
only if it belongs to one of these categories (use the identifier as `category`):
- homophone: same sound, different spelling (das/dass, tale/tail)
- compound: words split or merged (Donaudampf schifffahrt, book shelves/bookshelves)
- formatting: number, date, unit or abbreviation written differently
- spelling_variant: British/American or old/modern German orthography
- numeral_inflection: inflected numeral (dritte/dritten)
- stt_hallucination: a known STT hallucination phrase ("Untertitel im Auftrag des ZDF")
Otherwise use tts_error or unclear.

STT engines do not reorder words and almost never repeat phrases. Therefore these are \
tts_error, never STT artifacts: a word or phrase that occurs more often than in the \
original; the same word shown as both "extra" and "missing" (changed word order); \
missing words; extra content words; a different word that fits none of the categories.

Use the tools to check expansions, abbreviations and homophones instead of guessing. \
If a difference might be either an STT mishearing or a TTS mispronunciation (e.g. a \
rare name or foreign word), answer uncertain - a human will listen. Accepting a real \
error costs far more than an unnecessary human check.\
"""


AUDIO_ADDENDUM = """

The recording itself is attached. Listen to it: it is the ground truth for what was \
spoken. Judge the audio against the original text, use the transcript and the diff only \
to know where to listen, and accept a difference only if the audio matches the text \
there. Report mispronunciations you hear (wrong vowel, wrong stress, letters spelled \
out) as tts_error.\
"""


def _price(
    model_name: str, prices: dict[str, tuple[float, float]], usage: Mapping[str, Any]
) -> float:
    for key, (inp, out) in prices.items():
        if key in model_name:
            return (usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out) / 1e6
    return 0.0


class LLMReviewer:
    name = "llm"

    def __init__(
        self,
        model: str | BaseChatModel,
        tools: list[BaseTool],
        *,
        effort: str | None = None,
        attach_audio: bool = False,
        prices: dict[str, tuple[float, float]] | None = None,
    ):
        response_format: Any = Verdict
        if isinstance(model, str):
            kwargs: dict[str, Any] = {}
            provider, name = model.split(":", 1) if ":" in model else ("", model)
            audio_model = "audio" in name  # gpt-audio*: chat completions, no reasoning
            if effort and provider in {"anthropic", "openai", "azure_openai"} and not audio_model:
                kwargs["reasoning_effort"] = effort
            if provider in {"openai", "azure_openai"} and not audio_model:
                # Reasoning models only combine tools with reasoning_effort on /v1/responses.
                kwargs["use_responses_api"] = True
            if audio_model:
                # Audio models lack native structured output; return the verdict as a tool call.
                response_format = ToolStrategy(Verdict)
            self.model_id, chat_model = model, init_chat_model(model, **kwargs)
        else:
            self.model_id, chat_model = type(model).__name__, model
        self.attach_audio = attach_audio
        self.prices = prices or {}
        prompt = SYSTEM_PROMPT + (AUDIO_ADDENDUM if attach_audio else "")
        self.agent = create_agent(
            chat_model,
            tools=tools,
            system_prompt=prompt,
            response_format=response_format,
            name="readback_reviewer",
        )

    def _message(self, item: ReviewInput) -> HumanMessage:
        diffs = "\n".join(f"- {item.score.in_context(op)}" for op in item.score.ops) or "- (none)"
        text = (
            f"Language: {item.lang}\n"
            f"Attempt: {item.attempt} (regenerations left: {item.attempts_left})\n\n"
            f"<original>\n{item.reference}\n</original>\n\n"
            f"<transcript>\n{item.transcript}\n</transcript>\n\n"
            f"Normalized word error rate: {item.score.wer:.1%}\nDifferences:\n{diffs}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        audio = item.audio
        if self.attach_audio and audio and audio.mime_type.startswith("audio/"):
            data = base64.b64encode(Path(audio.path).read_bytes()).decode()
            content.append({"type": "audio", "base64": data, "mime_type": audio.mime_type})
        return HumanMessage(content=content)  # type: ignore[arg-type]

    async def review(self, item: ReviewInput, config: RunnableConfig | None = None) -> ReviewRecord:
        usage = UsageMetadataCallbackHandler()
        run_config = merge_configs(
            config or {},
            {
                "callbacks": [usage],
                "run_name": "llm_review",
                "metadata": {"chunk_id": item.chunk_id},
            },
        )
        try:
            result = await self.agent.ainvoke({"messages": [self._message(item)]}, run_config)
            verdict: Verdict = result["structured_response"]
        except Exception as e:  # fail safe: a broken review sends the chunk to a human
            log.warning("LLM review failed for %s: %s", item.chunk_id, e)
            verdict = Verdict(
                decision="uncertain",
                confidence=0.0,
                reasoning=f"Reviewer error ({type(e).__name__}); needs a human.",
            )
        cost = sum((_price(name, self.prices, u) for name, u in usage.usage_metadata.items()), 0.0)
        return ReviewRecord(
            **verdict.model_dump(),
            reviewer=f"llm:{self.model_id}",
            attempt=item.attempt,
            usage={name: dict(u) for name, u in usage.usage_metadata.items()},
            cost_usd=cost,
        )
