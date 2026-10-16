"""The LLM reviewer wired through create_agent, with a scripted chat model."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from readback.review import CascadeReviewer, RuleReviewer
from readback.review.llm import LLMReviewer
from readback.review.models import ReviewInput
from readback.scoring import score
from readback.toolbox import REVIEW_TOOLS


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


def scripted(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def item(ref: str, hyp: str, attempts_left: int = 0) -> ReviewInput:
    return ReviewInput(
        chunk_id="c1",
        lang="de",
        reference=ref,
        transcript=hyp,
        score=score(ref, hyp, "de"),
        attempts_left=attempts_left,
    )


VERDICT = {
    "decision": "accept",
    "confidence": 0.8,
    "reasoning": "'ungeheueren' is the old spelling of 'ungeheuren'.",
    "issues": [
        {
            "diff": "'ungeheueren' -> 'ungeheuren'",
            "category": "stt_artifact",
            "explanation": "orthography",
        }
    ],
}


async def test_llm_reviewer_uses_tools_and_returns_structured_verdict():
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "normalize_text",
                    "args": {"text": "ungeheueren", "lang": "de"},
                    "id": "t1",
                }
            ],
        ),
        AIMessage(content="", tool_calls=[{"name": "Verdict", "args": VERDICT, "id": "t2"}]),
    )
    tools = [StructuredTool.from_function(f) for f in REVIEW_TOOLS]
    reviewer = LLMReviewer(model, tools)
    record = await reviewer.review(
        item("zu einem ungeheueren Ungeziefer", "zu einem ungeheuren Ungeziefer")
    )
    assert record.decision == "accept"
    assert record.reviewer == "llm:ScriptedModel"
    assert record.issues[0].category == "stt_artifact"


async def test_cascade_skips_the_llm_when_rules_are_sure():
    class Exploding:
        name = "llm"

        async def review(self, *_: Any, **__: Any):
            raise AssertionError("LLM must not be called")

    cascade = CascadeReviewer(RuleReviewer(), Exploding())  # type: ignore[arg-type]
    homophone = await cascade.review(item("Ich weiß, dass es so ist.", "Ich weiß, das es so ist."))
    assert homophone.decision == "accept"
    truncated = await cascade.review(
        item("Er lag auf seinem Rücken.", "Er lag auf", attempts_left=1)
    )
    assert truncated.decision == "tts_error"  # regenerate first, no LLM call yet
