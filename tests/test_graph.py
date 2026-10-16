from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from readback.graph import build_graph
from tests.conftest import make_services

TEXT = (
    "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem "
    "Bett zu einem ungeheueren Ungeziefer verwandelt. Er lag auf seinem panzerartig harten "
    "Rücken.\n\nDas Fläschchen kostet 2,95 € und enthält 500 ml.\n\nEs war kein Traum."
)


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "max_concurrency": 2}


async def test_clean_run_passes_every_chunk_in_stage_one(settings):
    graph = build_graph(InMemorySaver())
    out = await graph.ainvoke(
        {"text": TEXT, "lang": "de"}, cfg("clean"), context=make_services(settings)
    )
    assert out["report"]["funnel"] == {
        "stage1_wer_zero": 3,
        "stage2_cleared_by_review": 0,
        "stage3_human": 0,
    }
    assert all(r["status"] == "passed" for r in out["results"])


async def test_transient_faults_are_repaired_by_regeneration(settings):
    services = make_services(settings, fault=0.6)
    graph = build_graph(InMemorySaver())
    out = await graph.ainvoke({"text": TEXT, "lang": "de"}, cfg("transient"), context=services)
    if "__interrupt__" in out:  # still broken after all retries: a human rejects it
        items = out["__interrupt__"][0].value["items"]
        resume = Command(resume={i["chunk_id"]: {"approved": False} for i in items})
        out = await graph.ainvoke(resume, cfg("transient"), context=services)
    regenerated = [r for r in out["results"] if len(r["takes"]) > 1]
    assert regenerated, "expected at least one chunk to be re-synthesized"
    # later takes are synthesized from the "speakable" text (numbers written out)
    assert any(
        "zwei Euro" in (t["tts_input"] or "") for r in regenerated for t in r["takes"][1:]
    ) or all("€" not in r["text"] for r in regenerated)
    assert out["report"]["ground_truth"]["escaped_defects"] == 0


async def test_sticky_fault_pauses_for_human_and_resumes(settings):
    services = make_services(settings, sticky=1.0)  # every chunk is systematically broken
    graph = build_graph(InMemorySaver())
    out = await graph.ainvoke({"text": TEXT, "lang": "de"}, cfg("sticky"), context=services)

    assert "__interrupt__" in out
    items = out["__interrupt__"][0].value["items"]
    assert items and all(i["diffs"] for i in items)
    for r in out["results"]:  # regenerations were attempted before escalating
        if r["status"] == "needs_human" and r["reviews"][-1]["decision"] == "tts_error":
            assert len(r["takes"]) == settings.max_regenerations + 1

    first, *rest = items
    decisions = {first["chunk_id"]: {"approved": True, "note": "fine on listen"}}
    decisions |= {i["chunk_id"]: {"approved": False} for i in rest}
    out = await graph.ainvoke(Command(resume=decisions), cfg("sticky"), context=services)

    status = {r["chunk_id"]: r["status"] for r in out["results"]}
    assert status[first["chunk_id"]] == "human_approved"
    assert all(status[i["chunk_id"]] == "human_rejected" for i in rest)
    assert out["report"]["ground_truth"]["escaped_defects"] == 0


async def test_benign_stt_noise_is_not_escalated(settings):
    text = "Ich weiß, dass das wahr ist. Die Donaudampfschifffahrtsgesellschaft fährt seit 1987."
    out = await build_graph(InMemorySaver()).ainvoke(
        {"text": text, "lang": "de"}, cfg("noise"), context=make_services(settings, noise=1.0)
    )
    assert "__interrupt__" not in out
    assert out["report"]["funnel"]["stage3_human"] == 0


async def test_corpus_sources_and_limit(settings):
    out = await build_graph(InMemorySaver()).ainvoke(
        {"sources": ["corpus/edge_cases/de.yaml"], "limit": 5},
        cfg("corpus"),
        context=make_services(settings),
    )
    assert out["report"]["chunks"] == 5
