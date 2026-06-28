.PHONY: setup bootstrap ocr-diff ocr-baseline test test-ocr lint clean help venv-2026 ocr-diff-2026 test-integration

# Cross-platform setup dispatch
ifeq ($(OS),Windows_NT)
setup:
	@powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
bootstrap: setup
else
setup:
	@if [ ! -d ".venv" ]; then python3.11 -m venv .venv; fi
	@.venv/bin/pip install --upgrade pip --upgrade-strategy only-if-needed
	@.venv/bin/pip install -e .
bootstrap: setup
endif

BASELINE ?= jobs/Tập 1-5/ocr/subtitles.json

VIDEO    ?= tests/fixtures/sample.mp4
OUTPUT   ?= tests/fixtures/ocr_diff_report.json
THRESH   ?= tests/fixtures/diff_thresholds.json

help:
	@echo "Targets:"
	@echo "  make ocr-diff          Run OCR regression harness"
	@echo "  make ocr-baseline      Overwrite baseline (requires CONFIRM=overwrite)"

ocr-diff:
	@if [ ! -f "$(BASELINE)" ]; then \
		echo "ERROR: baseline $(BASELINE) not found"; \
		echo "  Run 'vietdub run' on a video first, or set BASELINE=<path>"; \
		exit 2; \
	fi
	python3.11 tests/ocr_regression.py \
		--baseline "$(BASELINE)" \
		--video "$(VIDEO)" \
		--output "$(OUTPUT)" \
		--thresholds "$(THRESH)"

ocr-baseline:
	@if [ "$(CONFIRM)" != "overwrite" ]; then \
		echo "Refusing to overwrite baseline without CONFIRM=overwrite"; \
		exit 1; \
	fi
	@echo "Overwriting baseline at $(BASELINE)"

test:
	python3.11 -m pytest tests/ -v -m "not integration"

# Phase 1 OCR regression subset — fast loop for harness development.
test-ocr:
	python3.11 -m pytest tests/test_ocr_schema.py tests/test_ocr_regression_metrics.py \
	       tests/test_diff_thresholds.py tests/test_ocr_bridge.py \
	       tests/test_ocr_regression.py -v

lint:
	@echo "(no linter configured yet)"

clean:
	rm -rf tests/.tmp tests/fixtures/ocr_diff_report.json tests/fixtures/sample_diff.json

venv-2026:
	@if [ ! -f "requirements-2026.txt" ]; then \
		echo "ERROR: requirements-2026.txt not found"; exit 1; \
	fi
	@reqHash=$$(sha256sum requirements-2026.txt | awk '{print $$1}'); \
	if [ -d ".venv-2026" ] && [ "$$(cat build/installed-stack-2026.txt 2>/dev/null)" = "$$reqHash" ]; then \
		echo "venv-2026 already up to date. Skipping."; \
	else \
		rm -rf .venv-2026; \
		python3.11 -m venv .venv-2026; \
		.venv-2026/bin/pip install --upgrade pip; \
		.venv-2026/bin/pip install -r requirements-2026.txt; \
		mkdir -p build && echo $$reqHash > build/installed-stack-2026.txt; \
	fi

ocr-diff-2026:
	@if [ ! -d ".venv-2026" ]; then \
		echo "Run 'make venv-2026' (Linux) or '.\\scripts\\setup-2026.ps1' (Windows) first"; \
		exit 1; \
	fi
	.venv-2026/bin/python tests/ocr_regression_2026.py \
		--baseline "$(BASELINE)" \
		--video "$(VIDEO)" \
		--output "$(OUTPUT)" \
		--thresholds "$(THRESH)"

test-integration:
	python3.11 -m pytest tests/integration/ -v -m integration
