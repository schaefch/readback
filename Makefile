.PHONY: install install-local test lint eval demo serve voices docker

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

install-local:  ## Piper + faster-whisper + voices
	.venv/bin/pip install -e ".[dev,local]"
	$(MAKE) voices

voices:
	.venv/bin/python -m piper.download_voices --download-dir var/voices de_DE-thorsten-medium en_US-lessac-medium

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy

test:
	.venv/bin/pytest --cov=readback --cov-report=term-missing

eval:  ## offline evaluation with injected faults + quality gate
	.venv/bin/readback eval --gate --md var/eval.md

demo:  ## real audio: Piper -> faster-whisper on the German edge cases
	.venv/bin/readback run corpus/edge_cases/de.yaml --tts piper --stt faster_whisper

serve:
	.venv/bin/readback serve

docker:
	docker build -t readback:dev .
