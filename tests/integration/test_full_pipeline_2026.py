"""Integration smoke test for the vietdub pipeline against tests/fixtures/sample.mp4.

Marked with `@pytest.mark.integration` so `make test` skips it by default.
Run explicitly with `make test-integration` (or `pytest tests/integration/ -v -m integration`).

Skip behavior:
- `VIETDUB_SKIP_INTEGRATION=1` -> whole test skipped (CI default gate).
- `tests/fixtures/sample.mp4` missing -> skipped (manual-only environment).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_VIDEO = ROOT / "tests" / "fixtures" / "sample.mp4"
SERIES_NAME = "integration-smoke"
STATUS_PATH = ROOT / "jobs" / SERIES_NAME / "status.json"

SKIP_INTEGRATION = bool(os.environ.get("VIETDUB_SKIP_INTEGRATION"))

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason="VIETDUB_SKIP_INTEGRATION set",
)
@pytest.mark.skipif(
    not FIXTURE_VIDEO.exists(),
    reason=f"fixture video missing: {FIXTURE_VIDEO} (manual-only env)",
)
def test_full_pipeline_2026_smoke() -> None:
    """Run `vietdub run` against the sample fixture and assert success."""
    # Pre-clean to make this test idempotent.
    if STATUS_PATH.exists():
        # Best-effort: leave any prior artifacts but rely on the CLI to overwrite status.
        pass

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        # Explicitly clear any inherited skip flag for the subprocess.
        "VIETDUB_SKIP_INTEGRATION": "0",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vietdub",
            "run",
            str(FIXTURE_VIDEO),
            "--mode",
            "review",
            "--series",
            SERIES_NAME,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, (
        f"vietdub run exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert STATUS_PATH.exists(), f"status.json not written at {STATUS_PATH}"

    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    assert status, "status.json is empty"

    # Every recorded step must have state == "done".
    bad = {name: info for name, info in status.items() if info.get("state") != "done"}
    assert not bad, f"steps not done: {bad}"

    # Content assertions (M11): ensure each step actually wrote useful output,
    # not just that the step state is "done".
    job_root = STATUS_PATH.parent
    review_csv = job_root / "translation" / "review.csv"
    assert review_csv.exists(), f"review.csv not written at {review_csv}"
    review_text = review_csv.read_text(encoding="utf-8")
    row_count = max(0, len(review_text.splitlines()) - 1)
    assert row_count > 0, f"review.csv has 0 rows (header only): {review_csv}"

    ocr_json = job_root / "ocr" / "subtitles.json"
    assert ocr_json.exists(), f"subtitles.json not written at {ocr_json}"
    ocr_data = json.loads(ocr_json.read_text(encoding="utf-8"))
    assert isinstance(ocr_data, list), f"subtitles.json is not a list: {ocr_json}"
    assert len(ocr_data) > 0, f"subtitles.json has 0 segments: {ocr_json}"

    # Every recorded step must have a non-empty details dict.
    for step_name, info in status.items():
        assert info.get("details"), f"step {step_name} has no details: {info}"