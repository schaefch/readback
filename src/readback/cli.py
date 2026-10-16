"""Command line interface: ``readback run | eval | serve | mcp``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from readback.checkpointing import open_checkpointer
from readback.config import Settings, load_env_file
from readback.graph import build_graph
from readback.observability import setup_logging
from readback.report import render_markdown
from readback.services import Services


def _ask(items: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    print(f"\n{len(items)} chunk(s) need a human ear.\n", file=sys.stderr)
    for item in items:
        verdict = item.get("verdict") or {}
        print(f"── {item['chunk_id']}", file=sys.stderr)
        print(f"   text:       {item['text']}", file=sys.stderr)
        print(f"   transcript: {item['transcript']}", file=sys.stderr)
        print(f"   diffs:      {'; '.join(item['diffs'])}", file=sys.stderr)
        if verdict:
            print(f"   reviewer:   {verdict['decision']} – {verdict['reasoning']}", file=sys.stderr)
        print(f"   audio:      {item['audio_path']}", file=sys.stderr)
        answer = input("   approve? [y]es / [n]o / [s]kip: ").strip().lower()
        if answer in {"y", "n"}:
            decisions[item["chunk_id"]] = {"approved": answer == "y", "reviewer": "cli"}
    return decisions


async def _run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    services = await Services.from_settings(settings)
    payload: dict[str, Any] = {"lang": args.lang}
    if args.text:
        payload["text"] = args.text
    else:
        payload["sources"] = args.sources or ["corpus"]
    if args.limit:
        payload["limit"] = args.limit

    async with open_checkpointer(args.checkpointer) as saver:
        graph = build_graph(saver)
        thread_id = args.thread or uuid.uuid4().hex[:12]
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "max_concurrency": settings.max_concurrency,
        }
        state = await graph.ainvoke(payload, config, context=services)
        while "__interrupt__" in state:
            items = state["__interrupt__"][0].value["items"]
            if args.review == "interactive":
                decisions = _ask(items)
            elif args.review in {"approve-all", "reject-all"}:
                approved = args.review == "approve-all"
                decisions = {i["chunk_id"]: {"approved": approved} for i in items}
            else:  # defer: leave the run paused; report what we have
                from readback.report import build_report

                print(
                    f"Run {thread_id} paused with {len(items)} chunk(s) awaiting human review.",
                    file=sys.stderr,
                )
                return build_report(state["results"])
            state = await graph.ainvoke(Command(resume=decisions), config, context=services)
        return state["report"]


def _settings(args: argparse.Namespace, **defaults: Any) -> Settings:
    overrides = {
        "tts_provider": args.tts,
        "stt_provider": args.stt,
        "reviewer": args.reviewer,
        "llm_model": args.model,
        "fake_fault_rate": args.fault_rate,
        "fake_sticky_fault_rate": args.sticky_rate,
        "fake_noise_rate": args.noise_rate,
        "seed": args.seed,
        "max_concurrency": args.concurrency,
    }
    values = defaults | {k: v for k, v in overrides.items() if v is not None}
    return Settings(**values)


def _emit(report: dict[str, Any], args: argparse.Namespace) -> int:
    print(render_markdown(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.md:
        Path(args.md).write_text(render_markdown(report))
    # Quality gate for CI: no defect may slip through, few clean chunks escalated.
    gt = report.get("ground_truth")
    if (
        args.gate
        and gt
        and (gt["escaped_defects"] > 0 or gt["false_escalation_rate"] > args.max_false_escalations)
    ):
        print(f"quality gate failed: {gt}", file=sys.stderr)
        return 1
    return 0


def _add_run_args(p: argparse.ArgumentParser, review_default: str) -> None:
    p.add_argument("sources", nargs="*", help="corpus files/dirs (default: corpus/)")
    p.add_argument("--text", help="check this text instead of corpus files")
    p.add_argument("--lang", default="de", choices=["de", "en"], help="language for --text")
    p.add_argument("--limit", type=int, help="only the first N chunks")
    p.add_argument("--tts", choices=["fake", "piper", "openai"])
    p.add_argument("--stt", choices=["fake", "faster_whisper", "openai"])
    p.add_argument("--reviewer", choices=["rules", "llm", "cascade"])
    p.add_argument("--model", help="init_chat_model id for the LLM reviewer")
    p.add_argument("--fault-rate", type=float, help="fake TTS: transient fault probability")
    p.add_argument("--sticky-rate", type=float, help="fake TTS: persistent fault probability")
    p.add_argument("--noise-rate", type=float, help="fake STT: benign noise probability")
    p.add_argument("--seed", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument(
        "--review",
        default=review_default,
        choices=["interactive", "approve-all", "reject-all", "defer"],
        help="what to do when chunks need a human",
    )
    p.add_argument("--thread", help="thread id (reuse to inspect a run later)")
    p.add_argument(
        "--checkpointer", default="memory", help="memory | sqlite:///path | postgresql://…"
    )
    p.add_argument("--json", help="write the report as JSON")
    p.add_argument("--md", help="write the report as Markdown")
    p.add_argument("--gate", action="store_true", help="exit 1 if the ground-truth gate fails")
    p.add_argument("--max-false-escalations", type=float, default=0.05)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="readback", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_run_args(sub.add_parser("run", help="proof-listen a corpus or text"), "interactive")
    _add_run_args(
        sub.add_parser("eval", help="offline evaluation with injected, labelled faults"), "defer"
    )
    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    mcp = sub.add_parser("mcp", help="start the MCP server")
    mcp.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    args = parser.parse_args(argv)
    load_env_file()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("readback.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.cmd == "mcp":
        from readback.mcp_server import mcp as server

        server.run(transport="streamable-http" if args.http else "stdio")
    else:
        defaults: dict[str, Any] = {}
        if args.cmd == "eval":
            defaults = {
                "tts_provider": "fake",
                "stt_provider": "fake",
                "fake_fault_rate": 0.25,
                "fake_sticky_fault_rate": 0.08,
                "fake_noise_rate": 0.35,
            }
        settings = _settings(args, **defaults)
        setup_logging("WARNING", json_logs=False)
        sys.exit(_emit(asyncio.run(_run(args, settings)), args))


if __name__ == "__main__":
    main()
