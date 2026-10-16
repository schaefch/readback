from __future__ import annotations

import asyncio

import httpx
import pytest
from asgi_lifespan import LifespanManager

from readback.api import create_app


@pytest.fixture
async def client(settings):
    settings.fake_sticky_fault_rate = 0.5
    app = create_app(settings)
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def wait_for(client: httpx.AsyncClient, run_id: str, *states: str) -> dict:
    for _ in range(200):
        body = (await client.get(f"/v1/runs/{run_id}")).json()
        if body["status"] in states:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"run stuck in {body['status']}")


async def test_run_lifecycle_with_human_review(client):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    resp = await client.post("/v1/runs", json={"sources": ["edge_cases/de.yaml"], "limit": 8})
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    body = await wait_for(client, run_id, "awaiting_review", "completed")
    if body["status"] == "awaiting_review":
        decisions = {
            i["chunk_id"]: {"approved": False, "note": "audible error"}
            for i in body["pending_review"]
        }
        assert (
            await client.post(f"/v1/runs/{run_id}/review", json={"decisions": decisions})
        ).status_code == 202
        body = await wait_for(client, run_id, "completed")

    assert body["report"]["chunks"] == 8
    md = await client.get(f"/v1/runs/{run_id}/report.md")
    assert md.status_code == 200 and "readback run" in md.text
    assert "readback_chunks_total" in (await client.get("/metrics/")).text


async def test_rejects_paths_outside_corpus_root(client):
    resp = await client.post("/v1/runs", json={"sources": ["../pyproject.toml"]})
    assert resp.status_code == 422


async def test_unknown_run_is_404(client):
    assert (await client.get("/v1/runs/nope")).status_code == 404
