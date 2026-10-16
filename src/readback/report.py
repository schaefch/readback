"""Run summary: escalation funnel, listening time saved, LLM cost and - for runs with
fake engines - precision/recall against the injected ground truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from readback.normalize import normalize
from readback.results import ChunkResult

AUTO_ACCEPTED = {"passed", "accepted_by_review"}
ESCALATED = {"needs_human", "human_approved", "human_rejected"}


def _has_defect(result: ChunkResult) -> bool | None:
    """Ground truth: does the *final* take really deviate from the text?"""
    spoken = result.final.spoken_text
    if spoken is None:
        return None
    return normalize(spoken, result.lang) != normalize(result.text, result.lang)


def build_report(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    results = [ChunkResult.model_validate(r) for r in raw_results]
    n = len(results)
    status: Counter[str] = Counter(r.status for r in results)
    audio = sum(t.audio_seconds or 0 for r in results for t in r.takes)
    final_audio = {r.chunk_id: r.final.audio_seconds or 0 for r in results}
    human_audio = sum(final_audio[r.chunk_id] for r in results if r.status in ESCALATED)
    cost = sum(rv.get("cost_usd", 0) for r in results for rv in r.reviews)
    llm_reviews = sum(1 for r in results for rv in r.reviews if rv["reviewer"].startswith("llm"))

    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for r in results:
        by_category[r.category][r.status] += 1

    report: dict[str, Any] = {
        "chunks": n,
        "status": dict(status),
        "funnel": {
            "stage1_wer_zero": status["passed"],
            "stage2_cleared_by_review": status["accepted_by_review"],
            "stage3_human": sum(status[s] for s in ESCALATED),
        },
        "regenerations": sum(len(r.takes) - 1 for r in results),
        "llm_reviews": llm_reviews,
        "llm_cost_usd": round(cost, 4),
        "audio_seconds_synthesized": round(audio, 1),
        "audio_seconds_final": round(sum(final_audio.values()), 1),
        "audio_seconds_for_human": round(human_audio, 1),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "escalations": [
            {
                "chunk_id": r.chunk_id,
                "status": r.status,
                "text": r.text,
                "transcript": r.final.transcript,
                "diffs": r.final.diffs,
                "last_verdict": r.reviews[-1]["reasoning"] if r.reviews else None,
            }
            for r in results
            if r.status in ESCALATED
        ],
    }

    truth = [(r, _has_defect(r)) for r in results]
    if n and all(d is not None for _, d in truth):
        defective = [r for r, d in truth if d]
        clean = [r for r, d in truth if not d]
        escaped = [r for r in defective if r.status in AUTO_ACCEPTED]
        false_alarms = [r for r in clean if r.status in ESCALATED]
        first_take_faulty = [r for r in results if r.takes[0].injected_faults]
        repaired = [r for r in first_take_faulty if not r.final.injected_faults]
        report["ground_truth"] = {
            "defective_final": len(defective),
            "escaped_defects": len(escaped),
            "escaped_ids": [r.chunk_id for r in escaped],
            "defect_recall": round(1 - len(escaped) / len(defective), 4) if defective else 1.0,
            "false_escalations": len(false_alarms),
            "false_escalation_rate": round(len(false_alarms) / len(clean), 4) if clean else 0.0,
            "faulty_first_takes": len(first_take_faulty),
            "repaired_by_regeneration": len(repaired),
            "stt_noise_chunks": sum(1 for r in results if any(t.injected_noise for t in r.takes)),
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    f = report["funnel"]
    n = report["chunks"] or 1
    total = report["audio_seconds_final"] or 1
    human = report["audio_seconds_for_human"]
    lines = [
        "## readback run",
        "",
        "| Stage | Chunks | Share |",
        "|---|---:|---:|",
        *(
            f"| {label} | {f[key]} | {f[key] / n:.0%} |"
            for label, key in (
                ("1 · WER = 0 after normalization", "stage1_wer_zero"),
                ("2 · cleared by review", "stage2_cleared_by_review"),
                ("3 · human listening", "stage3_human"),
            )
        ),
        "",
        f"- Regenerations: **{report['regenerations']}**, LLM reviews: **{report['llm_reviews']}** "
        f"(≈ ${report['llm_cost_usd']:.4f})",
        f"- Audio to listen to: **{human / 60:.1f} min of {total / 60:.1f} min** "
        f"({1 - human / total:.0%} saved)",
    ]
    if gt := report.get("ground_truth"):
        lines += [
            "",
            "### Against injected ground truth",
            "",
            f"- Defect recall: **{gt['defect_recall']:.1%}** "
            f"({gt['escaped_defects']} of {gt['defective_final']} defective chunks escaped)",
            f"- False escalations: **{gt['false_escalations']}** "
            f"({gt['false_escalation_rate']:.1%} of clean chunks)",
            f"- Faulty first takes repaired by regeneration: "
            f"{gt['repaired_by_regeneration']} / {gt['faulty_first_takes']}",
        ]
    if report["escalations"]:
        lines += [
            "",
            "### Escalated chunks",
            "",
            "| Chunk | Status | Differences |",
            "|---|---|---|",
        ]
        for e in report["escalations"][:25]:
            diffs = "; ".join(e["diffs"])[:120].replace("|", "\\|")
            lines.append(f"| `{e['chunk_id']}` | {e['status']} | {diffs} |")
    return "\n".join(lines) + "\n"
