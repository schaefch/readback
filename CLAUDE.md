# readback – notes for Claude Code

LangGraph agent that proof-listens TTS audio: synthesize → transcribe back → normalize →
WER → (rules/LLM review ↔ regenerate) → human review via `interrupt()` → report.

## Commands
- `make test` – pytest (offline, fake engines, ~1 s)
- `make lint` – ruff check + ruff format --check + mypy
- `make eval` – offline evaluation with injected faults; must keep **0 escaped defects**
- `.venv/bin/readback run <corpus> --tts piper --stt faster_whisper` – real audio (needs `make install-local`)

## Layout
- `src/readback/graph/` – `pipeline.py` (top graph, Send fan-out, human review), `chunk.py` (per-chunk subgraph), `state.py`
- `normalize.py` / `lexicon.py` – written → spoken form; both reference and transcript go through it
- `scoring.py` – jiwer alignment, merged into `DiffOp`s
- `review/` – `rules.py` (deterministic), `llm.py` (create_agent + tools + structured `Verdict`), cascade in `__init__.py`
- `providers/` – `fake.py` (labelled ground truth), `local.py` (Piper, faster-whisper), `openai_compat.py` (OpenAI/Azure OpenAI)
- `mcp_server.py` – lexicon tools + `run_qa`; also consumed by the LLM reviewer via langchain-mcp-adapters
- `api.py` – FastAPI; run state lives only in the checkpointer

## Conventions
- Graph state holds plain dicts (`model_dump()`), so checkpoints stay JSON-serializable.
- Nodes get dependencies from `Runtime[Services]` (LangGraph context) – never from globals.
- New normalization rule ⇒ add a pair to `tests/test_normalize.py` (written and spoken form must normalize identically) and check `make eval` still passes.
- A normalization rule that could hide a real TTS error (e.g. ß→ss) needs a comment and a `blind` corpus item.
- `mcp` is pinned `<2` because langchain-mcp-adapters does not support the 2.x API yet.
- Default LLM: `anthropic:claude-opus-5-5` at `reasoning_effort=low`. Opus 5.5 rejects forced tool choice, so structured output goes through `create_agent(response_format=Verdict)`, never forced tool calls.
