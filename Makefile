.PHONY: ocr-diff ocr-baseline test lint clean help

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
	python tests/ocr_regression.py \
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
	pytest tests/test_ocr_schema.py tests/test_ocr_regression_metrics.py \
	       tests/test_diff_thresholds.py tests/test_ocr_bridge.py \
	       tests/test_ocr_regression.py -v

lint:
	@echo "(no linter configured yet)"

clean:
	rm -rf tests/.tmp tests/fixtures/ocr_diff_report.json tests/fixtures/sample_diff.json
