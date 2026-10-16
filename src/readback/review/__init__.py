"""Second-stage review: decide what a non-zero WER actually means."""

from __future__ import annotations

import sys

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from readback.config import Settings
from readback.review.models import Reviewer, ReviewInput, ReviewRecord, Verdict
from readback.review.rules import RuleReviewer

__all__ = ["ReviewInput", "ReviewRecord", "Reviewer", "RuleReviewer", "Verdict", "build_reviewer"]


class CascadeReviewer:
    """Rules first; the LLM only sees what the rules cannot settle cheaply."""

    name = "cascade"

    def __init__(self, rules: RuleReviewer, llm: Reviewer):
        self.rules, self.llm = rules, llm

    async def review(self, item: ReviewInput, config: RunnableConfig | None = None) -> ReviewRecord:
        first = self.rules.judge(item)
        if first.decision != "uncertain":
            # accept -> done; tts_error -> regenerate or escalate to a human. The LLM
            # must not overrule a rule-detected defect (evals showed it explains away
            # repetitions and word swaps as "STT artifacts").
            return first
        return await self.llm.review(item, config)


async def load_review_tools(settings: Settings) -> list[BaseTool]:
    from readback.toolbox import REVIEW_TOOL_NAMES, REVIEW_TOOLS

    if not settings.llm_use_mcp and not settings.mcp_servers:
        return [StructuredTool.from_function(f) for f in REVIEW_TOOLS]

    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections = dict(settings.mcp_servers)
    if settings.llm_use_mcp:
        connections["readback"] = {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "readback.mcp_server"],
        }
    tools = await MultiServerMCPClient(connections).get_tools()  # type: ignore[arg-type]
    # The readback server also exposes pipeline tools; the reviewer only needs the lexicon.
    own = [t for t in tools if t.name in REVIEW_TOOL_NAMES]
    others = [t for t in tools if t.name not in REVIEW_TOOL_NAMES and t.name != "run_qa"]
    if not settings.llm_use_mcp:
        own = [StructuredTool.from_function(f) for f in REVIEW_TOOLS]
    return own + others


async def build_reviewer(settings: Settings) -> Reviewer:
    if settings.reviewer == "rules":
        return RuleReviewer()
    from readback.review.llm import LLMReviewer

    llm = LLMReviewer(
        settings.llm_model,
        await load_review_tools(settings),
        effort=settings.llm_effort,
        attach_audio=settings.attach_audio,
        prices=settings.llm_prices_per_mtok,
    )
    return llm if settings.reviewer == "llm" else CascadeReviewer(RuleReviewer(), llm)
