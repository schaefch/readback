"""MCP server exposing the lexicon tools and the QA pipeline.

* The LLM reviewer consumes the lexicon tools through this server (stdio).
* Claude Code (see ``.mcp.json``) or any other MCP client can call ``run_qa`` to
  proof-listen a text, or ``compare_texts`` to diff a transcript.

Run: ``python -m readback.mcp_server`` (stdio) or ``... --http`` (streamable HTTP).
"""

from __future__ import annotations

import sys
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from readback import toolbox
from readback.lexicon import Lang

mcp = FastMCP(
    "readback",
    instructions=(
        "Tools for checking text-to-speech output against its source text: "
        "number/abbreviation expansion, homophone checks, transcript diffing, "
        "and a full synthesize-transcribe-compare QA run."
    ),
)

for fn in toolbox.REVIEW_TOOLS:
    mcp.tool()(fn)


@mcp.tool()
async def run_qa(text: str, lang: Lang = "de") -> dict[str, Any]:
    """Synthesize the text, transcribe it back, compare and review every chunk.

    Uses the engines configured via READBACK_* environment variables. Chunks that
    need a human are reported under 'escalations' (no interactive review here).
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from readback.config import get_settings
    from readback.graph import build_graph
    from readback.services import Services

    services = await Services.from_settings(get_settings())
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": f"mcp-{uuid.uuid4().hex[:8]}"}}
    state = await graph.ainvoke({"text": text, "lang": lang}, config, context=services)
    if "report" not in state:  # paused for human review; report what we have
        from readback.report import build_report

        return build_report(state.get("results", []))
    return state["report"]


def main() -> None:
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
