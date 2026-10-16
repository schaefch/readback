"""HTTP API: start QA runs, poll their state, submit human review decisions.

Runs execute as background tasks in this process; all run state lives in the
LangGraph checkpointer, so status and results survive restarts (with SQLite/Postgres)
and any replica can serve reads.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field, model_validator

from readback import __version__
from readback.checkpointing import open_checkpointer
from readback.config import Settings, get_settings, load_env_file
from readback.graph import build_graph
from readback.graph.state import HumanDecision
from readback.lexicon import Lang
from readback.observability import setup_logging
from readback.report import render_markdown
from readback.services import Services

log = logging.getLogger(__name__)


class RunRequest(BaseModel):
    sources: list[str] = Field(default_factory=list, description="Paths below the corpus root")
    text: str | None = None
    lang: Lang = "de"
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_input(self) -> RunRequest:
        if bool(self.sources) == bool(self.text):
            raise ValueError("Provide either 'sources' or 'text'.")
        return self


class ReviewRequest(BaseModel):
    decisions: dict[str, HumanDecision]


class RunStatus(BaseModel):
    run_id: str
    status: Literal["running", "awaiting_review", "completed", "failed", "incomplete"]
    chunks_total: int = 0
    chunks_done: int = 0
    pending_review: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    error: str | None = None


class Runner:
    def __init__(self, graph: Any, services: Services):
        self.graph, self.services = graph, services
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.errors: dict[str, str] = {}

    def config(self, run_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": run_id},
            "max_concurrency": self.services.settings.max_concurrency,
            "run_name": "readback",
            "metadata": {"run_id": run_id},
        }

    def start(self, run_id: str, payload: dict[str, Any] | Command | None) -> None:
        self.errors.pop(run_id, None)
        self.tasks[run_id] = asyncio.create_task(self._run(run_id, payload))

    async def _run(self, run_id: str, payload: dict[str, Any] | Command | None) -> None:
        try:
            await self.graph.ainvoke(payload, self.config(run_id), context=self.services)
        except Exception as e:  # surfaced via GET /runs/{id}
            log.exception("run %s failed", run_id)
            self.errors[run_id] = f"{type(e).__name__}: {e}"

    def running(self, run_id: str) -> bool:
        task = self.tasks.get(run_id)
        return task is not None and not task.done()

    async def status(self, run_id: str) -> RunStatus:
        snap = await self.graph.aget_state(self.config(run_id))
        values = snap.values or {}
        if not values and not self.running(run_id):
            raise HTTPException(404, f"unknown run {run_id}")
        out = RunStatus(
            run_id=run_id,
            status="running",
            chunks_total=len(values.get("chunks", [])),
            chunks_done=len(values.get("results", [])),
            report=values.get("report"),
        )
        if self.running(run_id):
            return out
        if run_id in self.errors:
            out.status, out.error = "failed", self.errors[run_id]
        elif snap.interrupts:
            out.status = "awaiting_review"
            out.pending_review = snap.interrupts[0].value["items"]
        elif "report" in values:
            out.status = "completed"
        else:
            # Checkpointed but not running here: another replica is working on it, or
            # the process died mid-run. POST /retry continues from the last checkpoint.
            out.status = "incomplete"
        return out


def create_app(settings: Settings | None = None) -> FastAPI:
    load_env_file()
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        services = await Services.from_settings(settings)
        async with open_checkpointer(settings.checkpointer) as saver:
            app.state.runner = Runner(build_graph(saver), services)
            yield

    app = FastAPI(title="readback", version=__version__, lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    def runner(request: Request) -> Runner:
        active: Runner = request.app.state.runner
        return active

    def resolve_sources(sources: list[str]) -> list[str]:
        root = settings.corpus_root.resolve()
        resolved = []
        for raw in sources:
            path = (root / raw).resolve()
            if not path.is_relative_to(root) or not path.exists():
                raise HTTPException(422, f"source not found below corpus root: {raw}")
            resolved.append(str(path))
        return resolved

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "runner"):
            raise HTTPException(503, "starting")
        return {"status": "ready"}

    @app.post("/v1/runs", status_code=202)
    async def create_run(body: RunRequest, request: Request) -> RunStatus:
        payload: dict[str, Any] = {"lang": body.lang}
        if body.sources:
            payload["sources"] = resolve_sources(body.sources)
        else:
            payload["text"] = body.text
        if body.limit:
            payload["limit"] = body.limit
        run_id = uuid.uuid4().hex[:12]
        runner(request).start(run_id, payload)
        return RunStatus(run_id=run_id, status="running")

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> RunStatus:
        return await runner(request).status(run_id)

    @app.get("/v1/runs/{run_id}/report.md", response_class=PlainTextResponse)
    async def get_report(run_id: str, request: Request) -> str:
        status = await runner(request).status(run_id)
        if status.report is None:
            raise HTTPException(409, f"run is {status.status}")
        return render_markdown(status.report)

    @app.post("/v1/runs/{run_id}/retry", status_code=202)
    async def retry_run(run_id: str, request: Request) -> RunStatus:
        r = runner(request)
        status = await r.status(run_id)
        if status.status not in {"incomplete", "failed"}:
            raise HTTPException(409, f"run is {status.status}")
        r.start(run_id, None)  # resume from the last checkpoint; finished chunks are kept
        return RunStatus(run_id=run_id, status="running")

    @app.post("/v1/runs/{run_id}/review", status_code=202)
    async def submit_review(run_id: str, body: ReviewRequest, request: Request) -> RunStatus:
        r = runner(request)
        status = await r.status(run_id)
        if status.status != "awaiting_review":
            raise HTTPException(409, f"run is {status.status}")
        pending = {item["chunk_id"] for item in status.pending_review}
        if unknown := set(body.decisions) - pending:
            raise HTTPException(422, f"not pending review: {sorted(unknown)}")
        decisions = {k: v.model_dump() for k, v in body.decisions.items()}
        r.start(run_id, Command(resume=decisions))
        return RunStatus(run_id=run_id, status="running")

    return app
