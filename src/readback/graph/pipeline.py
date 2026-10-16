"""Top-level graph: prepare -> fan out per chunk -> human review (interrupt) -> report."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from readback.corpus import chunks_from_text, load_corpus
from readback.graph.chunk import build_chunk_graph
from readback.graph.state import HumanDecision, PipelineInput, PipelineState
from readback.report import build_report
from readback.services import Services, resolve


async def prepare(state: PipelineState, runtime: Runtime[Services]) -> dict:
    settings = (await resolve(runtime.context)).settings
    if state.get("chunks"):
        chunks = state["chunks"]
    elif state.get("sources"):
        chunks = load_corpus(state["sources"], settings.chunk_max_chars)
    elif state.get("text"):
        chunks = chunks_from_text(state["text"], state.get("lang", "de"), settings.chunk_max_chars)
    else:
        raise ValueError("Provide 'sources', 'text' or 'chunks'.")
    if limit := state.get("limit"):
        chunks = chunks[:limit]
    return {"chunks": chunks}


def dispatch(state: PipelineState) -> list[Send] | Literal["report"]:
    if not state["chunks"]:
        return "report"
    return [Send("check_chunk", {"chunk": chunk}) for chunk in state["chunks"]]


def collect(state: PipelineState) -> dict:
    """Join point: runs once after every parallel chunk branch has finished."""
    return {}


def triage(state: PipelineState) -> Literal["human_review", "report"]:
    pending = [r for r in state.get("results", []) if r["status"] == "needs_human"]
    return "human_review" if pending else "report"


def human_review(state: PipelineState) -> dict:
    """Pause the run until a person has listened to the escalated chunks.

    Resume with ``Command(resume={chunk_id: {"approved": bool, "note": str}})``.
    Chunks without a decision stay in ``needs_human``.
    """
    pending = [r for r in state["results"] if r["status"] == "needs_human"]
    decisions: dict[str, Any] = interrupt(
        {
            "type": "human_review",
            "instructions": "Listen to each take and approve it or reject it.",
            "items": [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "transcript": r["takes"][-1]["transcript"],
                    "audio_path": r["takes"][-1]["audio_path"],
                    "diffs": r["takes"][-1]["diffs"],
                    "verdict": r["reviews"][-1] if r["reviews"] else None,
                }
                for r in pending
            ],
        }
    )
    updated = []
    for r in pending:
        if (raw := decisions.get(r["chunk_id"])) is None:
            continue
        decision = HumanDecision.model_validate(raw if isinstance(raw, dict) else {"approved": raw})
        status = "human_approved" if decision.approved else "human_rejected"
        updated.append(r | {"status": status, "human": decision.model_dump()})
    return {"results": updated}


def report(state: PipelineState) -> dict:
    return {"report": build_report(state.get("results", []))}


def build_graph(checkpointer: Any = None):
    g = StateGraph(PipelineState, context_schema=Services, input_schema=PipelineInput)
    g.add_node("prepare", prepare)
    g.add_node("check_chunk", build_chunk_graph())
    g.add_node("collect", collect)
    g.add_node("human_review", human_review)
    g.add_node("report", report)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", dispatch, ["check_chunk", "report"])
    g.add_edge("check_chunk", "collect")
    g.add_conditional_edges("collect", triage)
    g.add_edge("human_review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer, name="readback")


# Entry point for `langgraph dev` / LangGraph Studio (see langgraph.json).
graph = build_graph()
