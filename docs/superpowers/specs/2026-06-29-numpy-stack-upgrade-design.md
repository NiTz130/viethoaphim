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

**Goal:** Migrate to a modern stack (`numpy>=2.0,<2.4`, `paddlepaddle>=3.0,<4`, `torch>=2.6,<3`) with structural OCR regression verification, while preserving the option to roll back. The numpy upper bound `<2.4` is pinned because the original crash occurred against NumPy 2.4.6; pinning below that version prevents ABI drift back into the known-broken range.

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
- Self-regression test passes with realistic determinism thresholds: `segment_count_delta_pct <= 1` AND `text_similarity >= 0.98` when baseline = current run on the same hardware. PaddleOCR is non-deterministic across runs (model precision, thread count, batch order), so "delta = 0" is not achievable. A separate `tests/test_paddleocr_determinism.py` investigation test must validate acceptable run-to-run variance BEFORE Phase 1 is considered complete.

**Rollback:** Delete `tests/ocr_regression.py`, `tests/fixtures/*`, Makefile targets, README section. Source code unchanged.

### Phase 2 — Test-Only Upgrade

**Goal:** Verify that upgrading to `paddlepaddle>=3.0,<4` + `numpy>=2.0,<3` + `torch>=2.6,<3` does not break OCR quality.

**Deliverables:**
- `requirements-2026.txt` — new file with bounded compatibility-range pins (not soft floors — each pin has both a floor and a ceiling):
  ```
  paddlepaddle>=3.0,<4
  onnxruntime>=1.18,<1.20
  torch>=2.6,<3
  numpy>=2.0,<2.4
  ```
  The numpy upper bound is pinned to `<2.4` to keep the resolver out of the known-broken 2.4.x range. Run `pip install --dry-run -r requirements-2026.txt` in CI and treat resolver conflicts as a build failure.
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
  1. Check Python version. Accept `>=3.9,<3.14` (paddlepaddle 3.x publishes wheels for cp39–cp313). Use `py -3.X` launcher (not `python`, which on Windows may point to the Microsoft Store stub). If the resolved interpreter is under `WindowsApps`, print instructions to disable the App Execution Alias.
  2. Check architecture: refuse to proceed unless `platform.machine()` is `AMD64` (paddlepaddle 3.x Windows wheels are amd64-only).
  3. Check VC++ runtime: run `python -c "import ctypes; ctypes.CDLL('vcruntime140.dll'); ctypes.CDLL('msvcp140.dll')"`. If missing, print the Microsoft VC++ 2019/2022 redistributable download URL and exit.
  4. Record the installed stack into `build/installed-stack.txt` (hash of requirements.txt + pip freeze). On re-run, if the hash matches, skip install; otherwise nuke `.venv/` and recreate.
  5. Create `.venv/` if missing (or if previous hash mismatched).
  6. Upgrade pip only if its version is below the minimum required (`--upgrade-strategy only-if-needed`).
  7. Install `requirements.txt`. After every native command, check `$LASTEXITCODE -ne 0` and throw (PowerShell 5.1 does not propagate `$ErrorActionPreference='Stop'` to native exes).
  8. Warn if `$PSVersionTable.PSVersion -lt 7` (PowerShell 7+ recommended; 5.1 has weaker error propagation).
  9. Print shell-specific activation instructions (PowerShell: `.\\.venv\\Scripts\\Activate.ps1`; cmd: `.venv\\Scripts\\activate.bat`; bash: `source .venv/bin/activate`).
- `Makefile` — extended targets:
  - `setup` — wraps `setup.ps1` on Windows or creates `.venv` + pip install on Linux.
  - `venv`, `clean`, `test`, `lint`, `ocr-diff`, `ocr-diff-2026` (kept from earlier phases).
- `README.md` — "Setup" section rewritten as a single command: `.\setup.ps1`.

**Acceptance criteria:**
- `.\setup.ps1` on a clean machine (Windows 11, PowerShell 5.1+, no prior `.venv/`, fresh git clone) → `vietdub run --help` returns exit code 0 with no `ERROR`-level log lines in `build/setup-*.log`. Per-step assertions: after step 5, `.venv/` exists; after step 7, `pip show paddlepaddle` reports a version in `[3.0, 4)`.
- `make setup` produces the same pip freeze on Linux/CI (Ubuntu 22.04+, Python 3.11).
- `vietdub run` end-to-end on `tests/fixtures/sample.mp4` (committed ASCII-named fixture, not `C:\code\test\Tập 1.mp4`) → exit code 0, no `ERROR`-level log lines, `review.csv` row count within ±N% of the golden row count (`tests/fixtures/golden_review_row_count.txt`), all `status.json` steps = `done`. The Windows-specific smoke test on `C:\code\test\Tập 1.mp4` is documented as a manual local-only check, NOT a CI acceptance criterion.
- `make ocr-diff` runs against the new stack; all four SCORED metrics (segment_count_delta_pct, mean_text_length_delta_pct, time_range_overlap, text_similarity) within thresholds. `top_10_diff_segments` is informational only and does not gate pass/fail.

**Rollback:**
1. `git revert <phase-3-sha>` — restores `requirements.txt` and any deleted files (see sequencing note below).
2. `Remove-Item -Recurse -Force .venv` and `Remove-Item -Recurse -Force .venv-2026` (PowerShell) / `rm -rf .venv .venv-2026` (bash).
3. `py -3.11 -m venv .venv && .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` to reinstall the old stack.

**Sequencing note:** `requirements-ocr.txt` deletion and the `requirements.txt` rewrite MUST land in two separate commits to keep `git revert` self-contained:
- Commit 3a: copy `requirements-ocr.txt` content into `requirements.txt` while keeping `requirements-ocr.txt` as a re-export shim (`-r requirements.txt` + a "DEPRECATED" comment).
- Commit 3b (after ≥3 days soak): delete the shim.

Before 3b, run `git grep requirements-ocr` to surface any remaining references in README, Dockerfiles, or CI workflows.

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

**Step 1 — Load baseline:** Parse baseline JSON into `list[OcrSegment]`. Schema is enforced via pydantic models in `src/vietdub/ocr/schema.py`: `{id: str, start_ms: int >= 0, end_ms: int >= start_ms, text: str, confidence: float in [0,1]?}`. `confidence` is optional and not used by any diff metric in Step 3 (carried only for diagnostic logging). Segments are sorted by `start_ms` ascending before matching. Malformed segments raise `OcrSegmentError` and abort the run (do not silently skip).

**Step 2 — Run OCR:** Call `PaddleOCREngine.recognize(video_path, region="bottom_28pct")` (existing API). Output uses the same schema.

**Step 3 — Compute diff:**

| Metric | Formula | Threshold |
|--------|---------|-----------|
| `segment_count_delta_pct` | `abs(new_count - baseline_count) / baseline_count * 100` | ≤ 5 |
| `mean_text_length_delta_pct` | `abs(new_mean - baseline_mean) / baseline_mean * 100` | ≤ 10 |
| `time_range_overlap` | mean pairwise IoU of MATCHED pairs: `mean over matched (intersection_len / union_len)` | ≥ 0.95 |
| `text_similarity` | mean of `difflib.SequenceMatcher.ratio()` for matched pairs | ≥ 0.85 |
| `top_10_diff_segments` | segments with lowest text_similarity | (report only) |

**Matching algorithm:** Sort both lists by midpoint ascending before matching. Greedy match by closest midpoint, threshold ±200 ms. Tie-breaking: ascending `baseline.id` first, then smallest midpoint distance. Zero-width or negative-duration segments are excluded from matching. Unmatched segments accumulate in `unmatched.baseline_ids` and `unmatched.new_ids`. A fifth diagnostic metric `unmatched_pct = (unmatched_count / max(baseline_count, new_count)) * 100` is reported (informational; no threshold by default).

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

**Exit codes:**

| Code | Meaning | Writes partial JSON? | CI action |
|------|---------|----------------------|-----------|
| 0 | `passed=true`, all SCORED metrics within thresholds | No (full report written) | Pass |
| 1 | Any scored metric failed threshold (regression) | Yes (full report at `tests/fixtures/ocr_diff_report.json`) | Block PR |
| 2 | Baseline or input file missing | No | Page on-call (infra) |
| 3 | PaddleOCR raised during run | Yes (partial at `tests/fixtures/ocr_run_partial.json`) | Page on-call (infra) |
| 4 | `requirements-2026.txt` install failed (Phase 2 only) | No | Page on-call (infra) |
| 5 | Environment broken (venv missing, module not importable, missing fixture video) | No | Page on-call (infra) |

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
- All scripts use `$ErrorActionPreference='Stop'` (PowerShell) / `set -euo pipefail` (bash). On PowerShell 5.1 this does NOT propagate to native commands, so `setup.ps1` MUST follow every native invocation with `if ($LASTEXITCODE -ne 0) { throw "<cmd> failed with exit $LASTEXITCODE" }`. PowerShell 7+ (`pwsh`) is recommended; the script warns if `$PSVersionTable.PSVersion -lt 7`.
- Idempotency contract: see `setup.ps1` step 4 above. A re-run after a successful run produces the same `pip freeze` output. Re-runs are NOT skipped — pip install is re-invoked — but the end state is stable. Partial-failure detection via `build/installed-stack.txt` triggers nuke-and-recreate.
- All scripts log to `build/setup-yyyyMMdd-HHmmss-<pid>.log` (PowerShell: `Get-Date -Format 'yyyyMMdd-HHmmss'`). At the top of `setup.ps1`, prune logs older than 7 days. Use `Tee-Object` to mirror to stdout so the user sees progress live.

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

- `tests/test_ocr_regression.py::test_self_regression` — runs OCR diff against the SAME stack (baseline = current run). Must pass with `segment_count_delta_pct <= 1` and `text_similarity >= 0.98`. The self-regression loop copies the current run's output to `tests/.tmp/self_regression_baseline.json`, re-runs OCR, diffs against the tmpdir baseline, then cleans up. Does NOT pollute checked-in fixtures.
- `tests/test_ocr_regression.py::test_fixtures_present` — FAILS (not skips) if `tests/fixtures/ocr_diff_report.json` is missing, with an actionable error instructing the maintainer to run `make ocr-baseline CONFIRM=overwrite` on a known-good run.

### Manual smoke test (documented in README)

```powershell
.\setup.ps1
vietdub run .\sample.mp4 --mode review --series smoke-test
vietdub inspect jobs\smoke-test
# Expect: extract, stt, ocr, merge, context, translate all "done"
```

### CI (delivered as part of Phase 3)

CI infrastructure is added as an explicit Phase 3 deliverable, not assumed:
- `.github/workflows/test.yml` — runs `make test` on every PR. Runner: `windows-latest` + `ubuntu-latest`. Caches `~/.cache/pip` keyed on `requirements.txt` hash.
- `.github/workflows/integration.yml` — runs `make test-integration` nightly + on `main` merges. ~25 min; uses `windows-latest` with paddlepaddle cached. Failure retries once to distinguish flakes from real regressions.
- `.github/workflows/ocr-diff.yml` — runs `make ocr-diff` on every PR. FAILS (does not skip) if the golden baseline is missing. Updating the baseline requires a dedicated PR titled `chore: update ocr baseline` reviewed by a maintainer.

### Coverage targets

- New code (`tests/ocr_regression.py`, scripts): ≥ 85 % line coverage.
- Existing pipeline code: no regression in coverage.

## Pinning Strategy

Hybrid:
- **Bounded compatibility-range pins** in `requirements.txt` for `torch`, `numpy`, `paddlepaddle`, `faster-whisper`, `paddleocr`, `onnxruntime`. Each pin has both a floor and a ceiling (e.g., `torch>=2.6,<3`). The ceiling is mandatory and is what distinguishes these from floor-only pins.
- **ABI-constrained ranges** where a hard ABI relationship exists (e.g., a protobuf range dictated by paddlepaddle 3.x). These are still ranges, not exact pins.
- Compatibility ranges let `pip` resolve within range while preventing both major-version drift (floor) and silent regression into known-broken versions (ceiling). The numpy ceiling `<2.4` is set because the original crash occurred against `numpy==2.4.6`; the resolver must not be allowed to pick that version.

Glossary:
- **Compatibility range** — `>=X,<Y` with both bounds. Prevents drift in either direction.
- **Floor-only pin** — `>=X` only. Allows any newer version (use sparingly; ABI risk).
- **ABI-constrained range** — compatibility range dictated by a dependency's hard requirement; may shift across releases.
- **Exact pin** — `==X.Y.Z`. Use only for packages whose ABI changes on patch versions.

## Out of Scope

- Upgrading the team default Python to a version outside the `[3.9, 3.13)` window — paddlepaddle 3.x DOES publish cp39–cp313 wheels (verified against the 3.3.1 release, March 2026), so the prior exclusion was incorrect. A future migration to Python 3.14+ will require re-validating paddlepaddle wheel availability and rebuilding CI images.
- Migrating to a faster-whisper alternative (e.g., whisperX).
- GPU detection and CUDA auto-configuration. Note: on Windows, the default `paddlepaddle` install is CPU-only. Developers with NVIDIA GPUs who want acceleration must follow `docs/gpu-setup.md` (out of scope here).
- Replacing the OcrStep output schema.
- Adding a CHANGELOG.md file (delivered in Phase 3) — format and template TBD.

## Open Questions (resolved before Phase 3 ships)

- **OcrSegment schema contract** — finalized in `src/vietdub/ocr/schema.py`; see Step 1 above.
- **PaddleOCR determinism** — investigated via `tests/test_paddleocr_determinism.py`; acceptable thresholds documented.
- **CI infrastructure** — `.github/workflows/*.yml` added as Phase 3 deliverables (see CI section).
- **Baseline-update process** — dedicated PR convention `chore: update ocr baseline` with maintainer review.
- **Fixture video source** — committed as `tests/fixtures/sample.mp4` (≤10 MB, ASCII-named).
- **`requirements-2026.txt` final pin ranges** — onnxruntime narrowed to `<1.20`; numpy narrowed to `<2.4`; faster-whisper pinned to a known-compatible version after Phase 2 validation.
- **CHANGELOG.md format** — TBD by Phase 3 implementation.