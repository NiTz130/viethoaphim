# Spec: Upgrade to NumPy 2.x + paddlepaddle 3.x Stack

**Date:** 2026-06-29
**Status:** Draft (pending user review)
**Author:** Brainstorming session
**Related:** `docs/ocr-setup.md`, `requirements-ocr.txt`, `pyproject.toml`

## Context & Motivation

The current pinned stack in `requirements-ocr.txt` requires `torch==2.2.0` (compiled against NumPy 1.x C API). When NumPy 2.x is installed (e.g., pulled in by another package), `vietdub run` crashes during the STT step with:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6 as it may crash.
```

The previous "Tập 1-3" job (2026-06-27) succeeded when NumPy <2 was installed. The bug surfaced again on 2026-06-29 when running "Tập 1-5".

**Goal:** Migrate to a modern stack (`numpy>=2.0,<3`, `paddlepaddle>=3.0,<4`, `torch>=2.6,<3`) with structural OCR regression verification, while preserving the option to roll back.

## Architecture Overview

The system is organized into three independent layers:

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Source code (src/)                     │
│   - Unchanged. Already abstracts PaddleOCR and  │
│     torch behind interfaces, no version pins.   │
└─────────────────────────────────────────────────┘
              ↑
┌─────────────────────────────────────────────────┐
│ Layer 2: Dependency manifests                   │
│   - requirements.txt          (hybrid pins)     │
│   - requirements-2026.txt     (transient)       │
│   - pyproject.toml extras     (dev/test groups) │
└─────────────────────────────────────────────────┘
              ↑
┌─────────────────────────────────────────────────┐
│ Layer 3: Bootstrap scripts                      │
│   - setup.ps1                 (Windows user)    │
│   - Makefile                  (dev/CI)          │
│   - tests/ocr_regression.py   (regression check)│
└─────────────────────────────────────────────────┘
```

**Principle:** Source code does not pin specific versions. Manifests declare version ranges. Scripts enforce the contract and provide reproducible setup.

## Phase Plan

### Phase 1 — OCR Regression Harness

**Goal:** Build test infrastructure that compares OCR output across stacks without requiring any upgrade yet.

**Deliverables:**
- `tests/ocr_regression.py` — script that loads a baseline (`jobs/<job>/ocr/subtitles.json`), runs OCR on an input video, and emits a diff report.
- `tests/fixtures/ocr_diff_report.json` — golden-file output from the first run.
- `tests/fixtures/diff_thresholds.json` — configurable thresholds:
  - `segment_count_pct`: 5
  - `text_length_pct`: 10
  - `iou_min`: 0.95
  - `text_similarity_min`: 0.85
- `Makefile` — new targets:
  - `ocr-diff` — runs `tests/ocr_regression.py` against current stack.
  - `ocr-baseline` — overwrite baseline (requires confirmation prompt).
- `README.md` — new section "OCR Regression Testing" with one example.

**Acceptance criteria:**
- `make ocr-diff` succeeds against the current (un-upgraded) stack.
- Diff report prints as a table to stdout and writes JSON.
- Self-regression test passes (delta = 0 when baseline = current run).

**Rollback:** Delete `tests/ocr_regression.py`, `tests/fixtures/*`, Makefile targets, README section. Source code unchanged.

### Phase 2 — Test-Only Upgrade

**Goal:** Verify that upgrading to `paddlepaddle>=3.0,<4` + `numpy>=2.0,<3` + `torch>=2.6,<3` does not break OCR quality.

**Deliverables:**
- `requirements-2026.txt` — new file with soft pins:
  ```
  paddlepaddle>=3.0,<4
  onnxruntime>=1.18
  torch>=2.6,<3
  numpy>=2.0,<3
  ```
- `tests/ocr_regression_2026.py` — variant that runs in a separate `venv-2026` directory.
- `docs/ocr-upgrade-2026-results.md` — recorded diff report and conclusion (acceptable vs. regression).
- `Makefile` — new target `ocr-diff-2026`.

**Acceptance criteria:**
- `make venv-2026` creates an isolated venv.
- `make ocr-diff-2026` runs the new stack and produces a diff against `tests/fixtures/ocr_diff_report.json`.
- All four metrics within thresholds.
- `docs/ocr-upgrade-2026-results.md` records verdict.
- **Main `requirements.txt` and main venv untouched.**

**Rollback:** Delete `requirements-2026.txt`, the 2026 regression script, and the docs file. Main stack is unaffected.

### Phase 3 — Cutover

**Goal:** Switch the main stack to paddlepaddle 3.x + numpy 2.x.

**Deliverables:**
- `requirements.txt` — rewritten with hybrid pinning:
  - Soft floor pins for `torch`, `numpy`, `paddlepaddle`, `faster-whisper`, `paddleocr`, `onnxruntime`.
  - Exact pins only for protobuf (paddlepaddle 3.x hard-requires `protobuf>=4.25,<6`).
- `requirements-ocr.txt` — **deleted** (merged into `requirements.txt`).
- `setup.ps1` — idempotent bootstrap:
  1. Check Python version (3.11 only; paddlepaddle 3.x does not yet support 3.13).
  2. Create `.venv/` if missing.
  3. Upgrade pip.
  4. Install `requirements.txt`.
  5. Print activation instructions.
- `Makefile` — extended targets:
  - `setup` — wraps `setup.ps1` on Windows or creates `.venv` + pip install on Linux.
  - `venv`, `clean`, `test`, `lint`, `ocr-diff`, `ocr-diff-2026` (kept from earlier phases).
- `README.md` — "Setup" section rewritten as a single command: `.\setup.ps1`.

**Acceptance criteria:**
- `./setup.ps1` on a clean machine → `vietdub run --help` works with the new stack.
- `make setup` produces the same result on Linux/CI.
- `vietdub run` end-to-end on `C:\code\test\Tập 1.mp4` produces a `review.csv` with no errors.
- `make ocr-diff` runs against the new stack; diff within thresholds.

**Rollback:** `git revert` of the Phase 3 commit. Old stack remains in git history.

## Data Flow: OCR Regression Test

```
┌────────────────────────────────────────────────────────────────┐
│ ocr_regression.py                                               │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────────┐
   │ Step 1  │          │ Step 2  │          │ Step 3      │
   │ Load    │          │ Run OCR │          │ Compute     │
   │ baseline│          │ on input│          │ diff & emit │
   └─────────┘          └─────────┘          └─────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
  jobs/Tập 1-5/         PaddleOCR             diff_report.json
  ocr/subtitles         (current stack)       + stdout table
  .json (608 segs)
```

**Step 1 — Load baseline:** Parse baseline JSON into `list[OcrSegment]` with schema `{id, start_ms, end_ms, text, confidence}`.

**Step 2 — Run OCR:** Call `PaddleOCREngine.recognize(video_path, region="bottom_28pct")` (existing API). Output uses the same schema.

**Step 3 — Compute diff:**

| Metric | Formula | Threshold |
|--------|---------|-----------|
| `segment_count_delta_pct` | `abs(new_count - baseline_count) / baseline_count * 100` | ≤ 5 |
| `mean_text_length_delta_pct` | `abs(new_mean - baseline_mean) / baseline_mean * 100` | ≤ 10 |
| `time_range_overlap` | IoU of union time ranges | ≥ 0.95 |
| `text_similarity` | mean of `difflib.SequenceMatcher.ratio()` for matched pairs | ≥ 0.85 |
| `top_10_diff_segments` | segments with lowest text_similarity | (report only) |

**Matching algorithm:** Greedy match by closest `(start_ms + end_ms) / 2`, threshold ±200 ms. Unmatched segments accumulate in `unmatched_baseline_ids` and `unmatched_new_ids`.

**Output JSON schema:**
```json
{
  "baseline_path": "jobs/Tập 1-5/ocr/subtitles.json",
  "new_path": "tests/fixtures/ocr_run_2026.json",
  "metrics": {
    "segment_count": {"baseline": 608, "new": 612, "delta_pct": 0.66, "pass": true},
    "mean_text_length": {"baseline": 12.4, "new": 12.8, "delta_pct": 3.2, "pass": true},
    "time_range_overlap": 0.97,
    "text_similarity": 0.89
  },
  "thresholds": {"segment_count_pct": 5, "text_length_pct": 10, "iou_min": 0.95, "text_similarity_min": 0.85},
  "passed": true,
  "top_10_diff": [],
  "unmatched": {"baseline_ids": [], "new_ids": []}
}
```

**Exit codes:** `0` if `passed=true`, `1` if any metric fails threshold, `2` if baseline/input missing, `3` if PaddleOCR raises.

## Error Handling

### Phase 1 errors
- **Baseline file not found** → exit 2, message: "Run `vietdub run` on a video first to create baseline at `jobs/<name>/ocr/subtitles.json`."
- **Input video not found** → exit 2, message: "Pass `--video PATH` or set `VIETDUB_TEST_VIDEO` env var."
- **PaddleOCR raises during run** → log full traceback to stderr, exit 3, partial output saved to `tests/fixtures/ocr_run_partial.json`.

### Phase 2 errors
- **Diff > threshold** → save report, exit 1, message: "Regression detected — see `tests/fixtures/ocr_diff_report.json`. Do NOT proceed to Phase 3."
- **`requirements-2026.txt` install fails** → log pip output, exit 4. Main venv unchanged.
- **No venv-2026 yet** → message: "Run `make venv-2026` first to create test venv with new stack."

### Phase 3 errors (cutover)
- **`setup.ps1` fails mid-install** → script prints "ROLLBACK NEEDED" with last successful step. README documents manual rollback: `rm -rf .venv && git checkout requirements.txt requirements-ocr.txt`.
- **Paddlepaddle 3.x install fails on Windows** → `setup.ps1` checks Python version and suggests 3.11 (paddle 3.x doesn't yet support 3.13).
- **End-to-end pipeline fails** → log shows which step. User can: (a) `git revert` the Phase 3 commit, (b) report issue with `vietdub inspect` output.
- **OCR quality regression detected post-cutover** → rollback per Phase 3 docs. Note in CHANGELOG that downgrade to numpy<2 stack is supported via git checkout.

### Cross-cutting
- All scripts use `$ErrorActionPreference='Stop'` (PowerShell) / `set -euo pipefail` (bash) so any failure halts immediately.
- All scripts are idempotent — running twice gives the same state (no double-install, no orphan venvs).
- All scripts log to `build/setup-<date>.log` for postmortem.

## Testing

### Unit tests (pytest)

| Test | Verifies |
|------|----------|
| `test_load_baseline.py` | Loads baseline JSON correctly, handles missing fields |
| `test_match_segments.py` | Greedy match by center time, ±200 ms threshold |
| `test_compute_metrics.py` | All four metrics compute correctly with synthetic data |
| `test_threshold_logic.py` | `passed = all metrics within thresholds` |
| `test_diff_report_schema.py` | JSON output matches schema, all required keys present |
| `test_makefile_targets.py` | Each Makefile target runs (smoke test, mocked deps) |

### Integration tests

- `tests/integration/test_full_pipeline_2026.py` — runs `vietdub run` end-to-end on `Tập 1.mp4`. Asserts all `status.json` steps = `done` and `review.csv` has ≥ 1 row with non-empty `text_vi`. Marked `@pytest.mark.integration`.

### OCR regression tests

- `tests/test_ocr_regression.py::test_self_regression` — runs OCR diff against the SAME stack (baseline = current run). Must pass with delta = 0.
- `tests/test_ocr_regression.py::test_fixtures_present` — asserts `tests/fixtures/ocr_diff_report.json` exists.

### Manual smoke test (documented in README)

```powershell
.\setup.ps1
vietdub run .\sample.mp4 --mode review --series smoke-test
vietdub inspect jobs\smoke-test
# Expect: extract, stt, ocr, merge, context, translate all "done"
```

### CI

- `make test` → unit tests + self-regression (fast, ~30 s).
- `make test-integration` → full pipeline on sample (~25 min, nightly only).
- `make ocr-diff` → regression check on every PR (gated on `tests/fixtures/ocr_diff_report.json` existing).

### Coverage targets

- New code (`tests/ocr_regression.py`, scripts): ≥ 85 % line coverage.
- Existing pipeline code: no regression in coverage.

## Pinning Strategy

Hybrid:
- **Soft floor pins** in `requirements.txt` for `torch`, `numpy`, `paddlepaddle`, `faster-whisper`, `paddleocr`, `onnxruntime` (e.g., `torch>=2.6,<3`).
- **Exact pins** only where a hard ABI constraint exists (e.g., `protobuf>=4.25,<6` required by paddlepaddle 3.x).
- Soft pins let `pip` resolve within range while still preventing silent major-version drift.

## Out of Scope

- Upgrading Python from 3.11 to 3.13 (paddlepaddle 3.x doesn't yet support 3.13).
- Migrating to a faster-whisper alternative (e.g., whisperX).
- GPU detection and CUDA auto-configuration.
- Replacing the OCStep output schema.

## Open Questions

None at draft time.