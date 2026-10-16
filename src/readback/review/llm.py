"""LLM reviewer: a small LangChain agent with language tools and structured output."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import BaseTool

from readback.review.models import ReviewInput, ReviewRecord, Verdict

SYSTEM_PROMPT = """\
You review audiobook recordings. A TTS engine read a text aloud and an STT engine \
transcribed the audio back. You receive the original text, the transcript and the \
word differences that remain after both were normalized.

Decide whether the audio itself deviates from the text (tts_error), or whether every \
difference is explained by how the transcript was written down (accept): homophones, \
split or merged compounds, number and abbreviation formats, inflected numerals, old \
versus modern spelling, or known STT hallucinations. Use the tools to check number \
expansions, abbreviations and homophones rather than relying on memory.

Missing, repeated, extra or different content words mean tts_error. When the text \
alone cannot settle it - for example an inserted word that could be either an STT \
hallucination or a TTS insertion - answer uncertain and a human will listen. \
Accepting a real error costs far more than an unnecessary human check.\
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
        if isinstance(model, str):
            kwargs: dict[str, Any] = {}
            if effort and model.split(":", 1)[0] in {"anthropic", "openai", "azure_openai"}:
                kwargs["reasoning_effort"] = effort
            self.model_id, chat_model = model, init_chat_model(model, **kwargs)
        else:
            self.model_id, chat_model = type(model).__name__, model
        self.attach_audio = attach_audio
        self.prices = prices or {}
        self.agent = create_agent(
            chat_model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=Verdict,
            name="readback_reviewer",
        )

    def _message(self, item: ReviewInput) -> HumanMessage:
        diffs = "\n".join(f"- {op.describe()}" for op in item.score.ops) or "- (none)"
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
        result = await self.agent.ainvoke({"messages": [self._message(item)]}, run_config)
        verdict: Verdict = result["structured_response"]
        cost = sum((_price(name, self.prices, u) for name, u in usage.usage_metadata.items()), 0.0)
        return ReviewRecord(
            **verdict.model_dump(),
            reviewer=f"llm:{self.model_id}",
            attempt=item.attempt,
            usage={name: dict(u) for name, u in usage.usage_metadata.items()},
            cost_usd=cost,
        )
