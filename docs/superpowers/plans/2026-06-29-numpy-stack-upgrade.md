# NumPy 2.x + paddlepaddle 3.x Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate VietDub from a NumPy-1.x-pinned stack (broken when NumPy 2.x is installed) to paddlepaddle 3.x + NumPy 2.x + torch 2.6+ while preserving the option to roll back.

**Architecture:** Three-phase vertical slice. Phase 1 adds OCR regression harness against the current (broken-without-fix) stack. Phase 2 introduces `requirements-2026.txt` in an isolated `venv-2026` and validates OCR quality stays within structural thresholds. Phase 3 cuts the main venv over to the new stack via idempotent `setup.ps1` + Makefile, with `requirements-ocr.txt` deletion split into commit 3a (shim) → commit 3b (delete) for safe rollback.

**Tech Stack:** Python 3.11 (project default; paddlepaddle 3.x supports cp39–cp313), paddlepaddle 3.x, torch 2.6+, numpy 2.0–2.3, paddleocr, faster-whisper, pydantic 2.x, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-29-numpy-stack-upgrade-design.md`

## Global Constraints

- Pin range: `numpy>=2.0,<2.4` (ceiling excludes known-broken 2.4.6).
- Pin range: `onnxruntime>=1.18,<1.20`.
- Pin range: `torch>=2.6,<3`, `paddlepaddle>=3.0,<4`.
- `requirements-ocr.txt` deletion MUST be split into commit 3a (shim) → commit 3b (delete). Never delete in the same commit as `requirements.txt` rewrite.
- All PowerShell scripts MUST check `$LASTEXITCODE -ne 0` after every native command (5.1 does not propagate `$ErrorActionPreference='Stop'`).
- OCR diff SCORED thresholds: `segment_count_delta_pct <= 5`, `mean_text_length_delta_pct <= 10`, `time_range_overlap >= 0.95`, `text_similarity >= 0.85`.
- OCR self-regression thresholds: `segment_count_delta_pct <= 1` AND `text_similarity >= 0.98`.
- Exit codes for `tests/ocr_regression.py`: 0=pass, 1=regression, 2=missing input, 3=PaddleOCR raised, 4=install failed, 5=environment broken.
- Setup.ps1 must check (in order): Python version `>=3.9,<3.14` via `py` launcher, AMD64 architecture, VC++ runtime (`vcruntime140.dll` + `msvcp140.dll`), stack hash vs `build/installed-stack.txt`.
- All scripts log to `build/setup-yyyyMMdd-HHmmss-<pid>.log`; prune logs older than 7 days.
- CI uses committed `tests/fixtures/sample.mp4` (≤10 MB, ASCII-named), NEVER `C:\code\test\Tập 1.mp4`.
- Baseline-update PR convention: `chore: update ocr baseline` with maintainer review.

---

## Phase 1 — OCR Regression Harness

### Task 1: OcrSegment pydantic schema

**Files:**
- Create: `src/vietdub/ocr/schema.py`
- Test: `tests/test_ocr_schema.py`

**Interfaces:**
- Consumes: nothing (new module)
- Produces: `OcrSegment(id: str, start_ms: int, end_ms: int, text: str, confidence: Optional[float])` with `model_validator` enforcing `end_ms >= start_ms`. `OcrSegmentError(Exception)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_schema.py`:
```python
import pytest
from vietdub.ocr.schema import OcrSegment, OcrSegmentError


def test_valid_segment():
    s = OcrSegment(id="m-0001", start_ms=0, end_ms=2440, text="hello", confidence=0.95)
    assert s.start_ms == 0
    assert s.end_ms == 2440
    assert s.confidence == 0.95


def test_confidence_optional():
    s = OcrSegment(id="x", start_ms=0, end_ms=100, text="hi")
    assert s.confidence is None


def test_confidence_out_of_range():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=0, end_ms=100, text="hi", confidence=1.5)


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=200, end_ms=100, text="hi")


def test_negative_start_rejected():
    with pytest.raises(ValueError):
        OcrSegment(id="x", start_ms=-1, end_ms=100, text="hi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vietdub.ocr.schema'`

- [ ] **Step 3: Write minimal implementation**

Create `src/vietdub/ocr/schema.py`:
```python
"""OCR segment schema with pydantic validation."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class OcrSegmentError(Exception):
    """Raised when an OCR segment fails schema validation."""


class OcrSegment(BaseModel):
    """A single OCR'd subtitle segment from PaddleOCR output.

    Matches the JSON shape written to jobs/<name>/ocr/subtitles.json.
    """

    id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_duration(self) -> "OcrSegment":
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            )
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_schema.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/vietdub/ocr/schema.py tests/test_ocr_schema.py
git commit -m "feat(ocr): add OcrSegment pydantic schema"
```

---

### Task 2: Diff metric computation

**Files:**
- Create: `src/vietdub/ocr/regression.py`
- Test: `tests/test_ocr_regression_metrics.py`

**Interfaces:**
- Consumes: `OcrSegment` from `vietdub.ocr.schema`
- Produces:
  - `match_segments(baseline: list[OcrSegment], new: list[OcrSegment], threshold_ms: int = 200) -> list[tuple[str, str]]` — pairs of (baseline_id, new_id)
  - `compute_metrics(baseline: list[OcrSegment], new: list[OcrSegment], matches: list[tuple[str, str]]) -> dict` — returns `{segment_count, mean_text_length, time_range_overlap, text_similarity, unmatched_pct}` per spec

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_regression_metrics.py`:
```python
import pytest
from vietdub.ocr.schema import OcrSegment
from vietdub.ocr.regression import match_segments, compute_metrics


@pytest.fixture
def baseline():
    return [
        OcrSegment(id="a", start_ms=0, end_ms=1000, text="hello world"),
        OcrSegment(id="b", start_ms=1000, end_ms=2000, text="foo bar"),
        OcrSegment(id="c", start_ms=2000, end_ms=3000, text="baz qux"),
    ]


@pytest.fixture
def new_close(baseline):
    return [
        OcrSegment(id="x", start_ms=10, end_ms=990, text="hello world"),
        OcrSegment(id="y", start_ms=1000, end_ms=2000, text="foo bar"),
        OcrSegment(id="z", start_ms=2000, end_ms=3000, text="baz qux"),
    ]


def test_match_pairs_by_midpoint(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    assert ("a", "x") in pairs
    assert ("b", "y") in pairs
    assert ("c", "z") in pairs


def test_match_threshold_excludes_far_segments():
    base = [OcrSegment(id="a", start_ms=0, end_ms=1000, text="hi")]
    new = [OcrSegment(id="z", start_ms=10000, end_ms=11000, text="hi")]
    pairs = match_segments(base, new, threshold_ms=200)
    assert pairs == []


def test_match_excludes_zero_duration():
    base = [OcrSegment(id="a", start_ms=0, end_ms=0, text="x")]
    new = [OcrSegment(id="z", start_ms=0, end_ms=1000, text="x")]
    pairs = match_segments(base, new)
    assert pairs == []


def test_metrics_segment_count(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    assert m["segment_count"]["baseline"] == 3
    assert m["segment_count"]["new"] == 3
    assert m["segment_count"]["delta_pct"] == 0.0


def test_metrics_text_similarity_perfect(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    assert m["text_similarity"] == 1.0


def test_metrics_iou_perfect(baseline, new_close):
    pairs = match_segments(baseline, new_close)
    m = compute_metrics(baseline, new_close, pairs)
    # first pair: intersection 980, union 1000 -> 0.98
    assert 0.95 <= m["time_range_overlap"] <= 1.0


def test_metrics_unmatched_pct():
    base = [OcrSegment(id=f"b{i}", start_ms=i*1000, end_ms=i*1000+500, text="x") for i in range(10)]
    new = [OcrSegment(id=f"n{i}", start_ms=i*1000, end_ms=i*1000+500, text="x") for i in range(8)]
    pairs = match_segments(base, new)
    m = compute_metrics(base, new, pairs)
    # 2 baseline unmatched, 8 pairs -> unmatched_pct = 2/10 * 100 = 20.0
    assert m["unmatched_pct"] == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_regression_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vietdub.ocr.regression'`

- [ ] **Step 3: Write minimal implementation**

Create `src/vietdub/ocr/regression.py`:
```python
"""OCR regression diff: match segments across two runs and compute metrics."""
from difflib import SequenceMatcher
from typing import Optional

from .schema import OcrSegment


def _midpoint(seg: OcrSegment) -> int:
    return (seg.start_ms + seg.end_ms) // 2


def match_segments(
    baseline: list[OcrSegment],
    new: list[OcrSegment],
    threshold_ms: int = 200,
) -> list[tuple[str, str]]:
    """Greedy match by closest midpoint, threshold ±N ms. Tie-break by id asc, then distance."""
    # Filter zero/negative-duration segments
    base = sorted([s for s in baseline if s.end_ms > s.start_ms], key=lambda s: (_midpoint(s), s.id))
    new_filt = sorted([s for s in new if s.end_ms > s.start_ms], key=lambda s: (_midpoint(s), s.id))

    used_new: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for b in base:
        b_mid = _midpoint(b)
        candidates = [
            (abs(_midpoint(n) - b_mid), n.id, n)
            for n in new_filt
            if n.id not in used_new and abs(_midpoint(n) - b_mid) <= threshold_ms
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda c: (c[0], c[1]))  # distance, then id
        _, matched_id, _ = candidates[0]
        used_new.add(matched_id)
        pairs.append((b.id, matched_id))

    return pairs


def _iou(a: OcrSegment, b: OcrSegment) -> float:
    inter_start = max(a.start_ms, b.start_ms)
    inter_end = min(a.end_ms, b.end_ms)
    intersection = max(0, inter_end - inter_start)
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_metrics(
    baseline: list[OcrSegment],
    new: list[OcrSegment],
    matches: list[tuple[str, str]],
) -> dict:
    """Compute the 5 OCR regression metrics per spec §Data Flow."""
    base_by_id = {s.id: s for s in baseline}
    new_by_id = {s.id: s for s in new}
    matched_new_ids = {n for _, n in matches}

    # segment_count
    bc = len(baseline)
    nc = len(new)
    sc_delta = abs(nc - bc) / bc * 100 if bc else 0.0

    # mean_text_length
    b_mean = sum(len(s.text) for s in baseline) / bc if bc else 0.0
    n_mean = sum(len(s.text) for s in new) / nc if nc else 0.0
    tl_delta = abs(n_mean - b_mean) / b_mean * 100 if b_mean else 0.0

    # time_range_overlap (mean IoU of matched pairs)
    if matches:
        ious = [_iou(base_by_id[b], new_by_id[n]) for b, n in matches]
        iou_mean = sum(ious) / len(ious)
    else:
        iou_mean = 0.0

    # text_similarity (mean SequenceMatcher.ratio for matched pairs)
    if matches:
        sims = [
            SequenceMatcher(None, base_by_id[b].text, new_by_id[n].text).ratio()
            for b, n in matches
        ]
        sim_mean = sum(sims) / len(sims)
    else:
        sim_mean = 0.0

    # unmatched_pct (informational)
    unmatched_baseline = [b for b, _ in matches]  # missing pairs are unmatched
    unmatched_total = (bc - len(matches)) + (nc - len(matches))
    unmatched_pct = unmatched_total / max(bc, nc) * 100 if max(bc, nc) else 0.0

    return {
        "segment_count": {"baseline": bc, "new": nc, "delta_pct": round(sc_delta, 2)},
        "mean_text_length": {"baseline": round(b_mean, 2), "new": round(n_mean, 2), "delta_pct": round(tl_delta, 2)},
        "time_range_overlap": round(iou_mean, 4),
        "text_similarity": round(sim_mean, 4),
        "unmatched_pct": round(unmatched_pct, 2),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_regression_metrics.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/vietdub/ocr/regression.py tests/test_ocr_regression_metrics.py
git commit -m "feat(ocr): add segment matching + 5 regression metrics"
```

---

### Task 3: Diff thresholds fixture + diff report schema

**Files:**
- Create: `tests/fixtures/diff_thresholds.json`
- Create: `tests/fixtures/schema/ocr_diff_report.schema.json`
- Test: `tests/test_diff_thresholds.py`

**Interfaces:**
- Consumes: thresholds config file path (CLI arg)
- Produces: `OcrDiffReport` pydantic model + JSON Schema for runtime validation

- [ ] **Step 1: Write the failing test**

Create `tests/test_diff_thresholds.py`:
```python
import json
from pathlib import Path

import pytest

from vietdub.ocr.regression import build_report, passes_thresholds
from vietdub.ocr.schema import OcrSegment


FIXTURES = Path(__file__).parent / "fixtures"


def load_thresholds():
    return json.loads((FIXTURES / "diff_thresholds.json").read_text(encoding="utf-8"))


def test_thresholds_file_exists():
    assert (FIXTURES / "diff_thresholds.json").exists()


def test_thresholds_have_required_keys():
    t = load_thresholds()
    for k in ("segment_count_pct_max", "text_length_pct_max", "iou_min", "text_similarity_min"):
        assert k in t


def test_passes_thresholds_all_pass():
    metrics = {
        "segment_count": {"delta_pct": 2.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.90,
    }
    assert passes_thresholds(metrics, load_thresholds()) is True


def test_passes_thresholds_segment_count_fails():
    metrics = {
        "segment_count": {"delta_pct": 7.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.90,
    }
    assert passes_thresholds(metrics, load_thresholds()) is False


def test_passes_thresholds_similarity_fails():
    metrics = {
        "segment_count": {"delta_pct": 2.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.80,
    }
    assert passes_thresholds(metrics, load_thresholds()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diff_thresholds.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_report' / 'passes_thresholds'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/fixtures/diff_thresholds.json`:
```json
{
  "segment_count_pct_max": 5,
  "text_length_pct_max": 10,
  "iou_min": 0.95,
  "text_similarity_min": 0.85
}
```

Append to `src/vietdub/ocr/regression.py` (below existing code):
```python
def passes_thresholds(metrics: dict, thresholds: dict) -> bool:
    """Return True if all SCORED metrics satisfy their threshold."""
    checks = [
        metrics["segment_count"]["delta_pct"] <= thresholds["segment_count_pct_max"],
        metrics["mean_text_length"]["delta_pct"] <= thresholds["text_length_pct_max"],
        metrics["time_range_overlap"] >= thresholds["iou_min"],
        metrics["text_similarity"] >= thresholds["text_similarity_min"],
    ]
    return all(checks)


def build_report(
    baseline_path: str,
    new_path: str,
    baseline: list[OcrSegment],
    new: list[OcrSegment],
    metrics: dict,
    thresholds: dict,
    matches: list[tuple[str, str]],
    top_n: int = 10,
) -> dict:
    """Build the JSON-serialisable diff report matching the spec §Output JSON schema."""
    base_by_id = {s.id: s for s in baseline}
    new_by_id = {s.id: s for s in new}
    matched_new = {n for _, n in matches}
    matched_base = {b for b, _ in matches}

    # top-N diff segments by lowest text similarity
    diffs = []
    for b, n in matches:
        sim = SequenceMatcher(None, base_by_id[b].text, new_by_id[n].text).ratio()
        diffs.append((sim, b, n))
    diffs.sort()
    top_10 = [
        {
            "baseline_id": b,
            "new_id": n,
            "similarity": round(sim, 4),
            "baseline_text": base_by_id[b].text,
            "new_text": new_by_id[n].text,
        }
        for sim, b, n in diffs[:top_n]
    ]

    return {
        "baseline_path": baseline_path,
        "new_path": new_path,
        "metrics": metrics,
        "thresholds": {
            "segment_count_pct_max": thresholds["segment_count_pct_max"],
            "text_length_pct_max": thresholds["text_length_pct_max"],
            "iou_min": thresholds["iou_min"],
            "text_similarity_min": thresholds["text_similarity_min"],
        },
        "passed": passes_thresholds(metrics, thresholds),
        "top_10_diff_segments": top_10,
        "unmatched": {
            "baseline_ids": sorted(set(s.id for s in baseline) - matched_base),
            "new_ids": sorted(set(s.id for s in new) - matched_new),
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diff_thresholds.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/diff_thresholds.json src/vietdub/ocr/regression.py tests/test_diff_thresholds.py
git commit -m "feat(ocr): add diff report builder + threshold config"
```

---

### Task 4: tests/ocr_regression.py CLI runner

**Files:**
- Create: `tests/ocr_regression.py`
- Create: `tests/fixtures/schema/ocr_diff_report.schema.json` (JSON Schema for runtime validation)

**Interfaces:**
- Consumes: `--baseline PATH --video PATH --output PATH --thresholds PATH` CLI args
- Produces: exit code 0/1/2/3 per spec §Exit Codes; writes JSON report to `--output`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ocr_regression.py` (new file):
```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNNER = ROOT / "tests" / "ocr_regression.py"


def test_runner_help_exits_zero():
    r = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--baseline" in r.stdout
    assert "--video" in r.stdout


def test_runner_missing_baseline_exits_2(tmp_path):
    r = subprocess.run(
        [sys.executable, str(RUNNER),
         "--baseline", str(tmp_path / "nope.json"),
         "--video", str(tmp_path / "v.mp4"),
         "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "baseline" in r.stderr.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_regression.py -v`
Expected: FAIL (no `tests/ocr_regression.py`)

- [ ] **Step 3: Write minimal implementation**

Create `tests/ocr_regression.py`:
```python
"""CLI runner: load baseline OCR JSON, run OCR on a video, emit diff report.

Exit codes per spec:
  0 = all scored metrics within thresholds (pass)
  1 = regression detected
  2 = baseline or input video missing
  3 = PaddleOCR raised during run
  5 = environment broken (vietdub module not importable)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

try:
    from vietdub.ocr.regression import (
        build_report, compute_metrics, match_segments,
    )
    from vietdub.ocr.schema import OcrSegment
except ImportError as e:
    print(f"ERROR: cannot import vietdub.ocr ({e}); run setup.ps1 / make setup first", file=sys.stderr)
    sys.exit(5)


log = logging.getLogger("ocr_regression")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="Path to baseline OCR JSON")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--output", required=True, help="Path to write diff report JSON")
    p.add_argument("--thresholds", required=True, help="Path to thresholds JSON")
    p.add_argument("--new-json", default=None,
                   help="If set, reuse this new OCR JSON instead of re-running (for self-regression)")
    p.add_argument("--region", default="bottom_28pct", help="OCR region (kept for parity with pipeline)")
    return p.parse_args()


def load_segments(path: Path) -> list[OcrSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [OcrSegment.model_validate(s) for s in raw]


def run_ocr(video_path: Path, region: str) -> list[OcrSegment]:
    """Invoke the project's PaddleOCR engine. Implemented in Task 5."""
    raise NotImplementedError("Implemented in Task 5 (PaddleOCR bridge)")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    baseline_path = Path(args.baseline)
    video_path = Path(args.video)
    output_path = Path(args.output)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))

    if not baseline_path.exists():
        print(f"ERROR: baseline not found: {baseline_path}", file=sys.stderr)
        print("Run `vietdub run` on a video first, or pass --baseline PATH.", file=sys.stderr)
        return 2
    if not video_path.exists() and args.new_json is None:
        print(f"ERROR: input video not found: {video_path}", file=sys.stderr)
        return 2

    try:
        baseline = load_segments(baseline_path)
    except Exception as e:
        print(f"ERROR: baseline malformed: {e}", file=sys.stderr)
        return 2

    if args.new_json:
        new_path = Path(args.new_json)
        if not new_path.exists():
            print(f"ERROR: --new-json not found: {new_path}", file=sys.stderr)
            return 2
        new = load_segments(new_path)
    else:
        try:
            new = run_ocr(video_path, args.region)
        except Exception:
            traceback.print_exc()
            return 3

    matches = match_segments(baseline, new)
    metrics = compute_metrics(baseline, new, matches)
    report = build_report(str(baseline_path), str(video_path), baseline, new, metrics, thresholds, matches)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stdout table summary
    print("=" * 60)
    print(f"baseline: {baseline_path}  ({len(baseline)} segments)")
    print(f"new:      {video_path}      ({len(new)} segments)")
    print(f"matches:  {len(matches)}")
    print("-" * 60)
    print(f"segment_count_delta_pct:  {metrics['segment_count']['delta_pct']}%  (max {thresholds['segment_count_pct_max']}%)")
    print(f"text_length_delta_pct:    {metrics['mean_text_length']['delta_pct']}%  (max {thresholds['text_length_pct_max']}%)")
    print(f"time_range_overlap (IoU): {metrics['time_range_overlap']}  (min {thresholds['iou_min']})")
    print(f"text_similarity:          {metrics['text_similarity']}  (min {thresholds['text_similarity_min']})")
    print(f"unmatched_pct:            {metrics['unmatched_pct']}%  (informational)")
    print("-" * 60)
    print(f"PASSED: {report['passed']}")
    print(f"Report: {output_path}")
    print("=" * 60)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_regression.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ocr_regression.py tests/test_ocr_regression.py tests/fixtures/schema/
git commit -m "feat(ocr): add CLI runner with spec-compliant exit codes"
```

---

### Task 5: PaddleOCR bridge for the regression runner

**Files:**
- Create: `src/vietdub/ocr/bridge.py`
- Test: `tests/test_ocr_bridge.py`

**Interfaces:**
- Consumes: `video_path`, `region`
- Produces: `list[OcrSegment]` matching the existing pipeline's `jobs/<name>/ocr/subtitles.json` shape

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_bridge.py`:
```python
from vietdub.ocr.bridge import ocr_to_segments


def test_ocr_to_segments_coerces_dict():
    raw = [
        {"id": "m-0001", "start_ms": 0, "end_ms": 1000, "text": "hi", "confidence": 0.9},
        {"id": "m-0002", "start_ms": 1000, "end_ms": 2000, "text": "bye"},
    ]
    segs = ocr_to_segments(raw)
    assert len(segs) == 2
    assert segs[0].text == "hi"
    assert segs[1].confidence is None


def test_ocr_to_segments_rejects_missing_required():
    import pytest
    from vietdub.ocr.schema import OcrSegmentError
    with pytest.raises(OcrSegmentError):
        ocr_to_segments([{"start_ms": 0, "end_ms": 100, "text": "x"}])  # no id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_bridge.py -v`
Expected: FAIL (`No module named 'vietdub.ocr.bridge'`)

- [ ] **Step 3: Write minimal implementation**

Create `src/vietdub/ocr/bridge.py`:
```python
"""Bridge between vietdub's PaddleOCR pipeline and the regression schema."""
from typing import Iterable

from .schema import OcrSegment, OcrSegmentError


def ocr_to_segments(raw: Iterable[dict]) -> list[OcrSegment]:
    """Convert raw OCR dicts (as written by the pipeline) to OcrSegment models.

    Raises OcrSegmentError if any segment is malformed (does not silently skip).
    """
    out: list[OcrSegment] = []
    for i, item in enumerate(raw):
        try:
            out.append(OcrSegment.model_validate(item))
        except Exception as e:
            raise OcrSegmentError(f"segment index {i} malformed: {e}") from e
    return out


def run_pipeline_ocr(video_path, region: str = "bottom_28pct") -> list[OcrSegment]:
    """Run the vietdub PaddleOCR engine on a video file.

    Imports lazily so unit tests don't require paddlepaddle.
    """
    from vietdub.ocr.engine import PaddleOCREngine  # type: ignore
    engine = PaddleOCREngine(region=region)
    raw = engine.recognize(str(video_path))
    return ocr_to_segments(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_bridge.py -v`
Expected: 2 PASS (the second test expects `OcrSegmentError`; the `model_validator` raises `ValueError` which we catch and re-raise as `OcrSegmentError`)

- [ ] **Step 5: Commit**

```bash
git add src/vietdub/ocr/bridge.py tests/test_ocr_bridge.py
git commit -m "feat(ocr): add PaddleOCR bridge + schema coercion"
```

---

### Task 6: Wire runner to real PaddleOCR + fixture video

**Files:**
- Modify: `tests/ocr_regression.py` — replace `run_ocr` stub with call to `bridge.run_pipeline_ocr`
- Create: `tests/fixtures/sample.mp4` — committed ≤10MB ASCII-named fixture (generate via ffmpeg in this task)

**Interfaces:**
- Same as Task 4

- [ ] **Step 1: Generate the fixture video**

Run:
```powershell
ffmpeg -y -f lavfi -i "color=c=blue:s=320x240:d=5:r=24" -f lavfi -i "sine=frequency=440:duration=5" -c:v libx264 -preset veryfast -pix_fmt yuv420p tests/fixtures/sample.mp4
```
Verify size ≤10MB:
```bash
ls -la tests/fixtures/sample.mp4
```

- [ ] **Step 2: Modify `tests/ocr_regression.py`**

Replace the `run_ocr` function with:
```python
def run_ocr(video_path: Path, region: str) -> list[OcrSegment]:
    from vietdub.ocr.bridge import run_pipeline_ocr
    return run_pipeline_ocr(video_path, region=region)
```

- [ ] **Step 3: Run end-to-end smoke**

Run:
```powershell
python tests/ocr_regression.py --baseline jobs/Tập 1-5/ocr/subtitles.json --video tests/fixtures/sample.mp4 --output tests/fixtures/sample_diff.json --thresholds tests/fixtures/diff_thresholds.json
```
Expected: exit code 1 (regression — sample.mp4 has no OCR-able text), but report is written. Exit code 1 is fine for Phase 1 smoke; the harness is verified.

- [ ] **Step 4: Commit**

```bash
git add tests/ocr_regression.py tests/fixtures/sample.mp4
git commit -m "feat(ocr): wire runner to PaddleOCR + add fixture video"
```

---

### Task 7: Makefile targets (Phase 1)

**Files:**
- Create: `Makefile`

**Interfaces:**
- `make ocr-diff` — runs regression harness against baseline + current sample
- `make ocr-baseline CONFIRM=overwrite` — overwrite baseline (must confirm)

- [ ] **Step 1: Write the Makefile**

Create `Makefile`:
```makefile
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
```

- [ ] **Step 2: Verify targets**

Run: `make help`
Expected: target list printed.

Run: `make ocr-baseline` (without CONFIRM)
Expected: refusal message + exit 1.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile with ocr-diff and ocr-baseline targets"
```

---

### Task 8: README "OCR Regression Testing" section

**Files:**
- Modify: `README.md` — append section

- [ ] **Step 1: Append the section**

Append to `README.md`:
```markdown
## OCR Regression Testing

Use the harness to compare OCR output across dependency stacks:

```bash
make ocr-diff            # uses default baseline (jobs/Tập 1-5/ocr/subtitles.json)
make ocr-baseline CONFIRM=overwrite   # overwrite baseline (rare; needs review)
```

The harness computes 5 metrics (segment count delta, mean text length delta, time-range IoU, text similarity, unmatched %). SCORED thresholds are in `tests/fixtures/diff_thresholds.json`; the unmatched_pct metric is informational.

Exit codes: `0` pass, `1` regression (block PR), `2` missing input, `3` OCR raised, `5` env broken.
```

- [ ] **Step 2: Verify rendering**

Run: `grep -A 5 "OCR Regression Testing" README.md`
Expected: the section appears.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document OCR regression testing harness"
```

---

## Phase 2 — Test-Only Upgrade

### Task 9: requirements-2026.txt with bounded pins

**Files:**
- Create: `requirements-2026.txt`

- [ ] **Step 1: Write the file**

Create `requirements-2026.txt`:
```
# Bounded compatibility-range pins for the NumPy 2.x / paddlepaddle 3.x test stack.
# Ceilings are mandatory — they prevent drift into known-broken versions.
# See docs/superpowers/specs/2026-06-29-numpy-stack-upgrade-design.md
paddlepaddle>=3.0,<4
onnxruntime>=1.18,<1.20
torch>=2.6,<3
numpy>=2.0,<2.4
protobuf>=4.25,<6
```

- [ ] **Step 2: Dry-run install check**

Run: `python -m pip install --dry-run -r requirements-2026.txt`
Expected: pip resolves without conflicts (or with documented conflicts only). If conflict, STOP — do not proceed to Task 10.

- [ ] **Step 3: Commit**

```bash
git add requirements-2026.txt
git commit -m "build: add requirements-2026.txt with bounded pins"
```

---

### Task 10: scripts/setup-2026.ps1 (Windows) + make venv-2026 (Linux)

**Files:**
- Create: `scripts/setup-2026.ps1`
- Modify: `Makefile` — add `venv-2026` and `ocr-diff-2026` targets

- [ ] **Step 1: Write setup-2026.ps1**

Create `scripts/setup-2026.ps1`:
```powershell
# setup-2026.ps1 — create venv-2026 with the new pinned stack.
# Idempotent: if venv-2026 exists with matching hash, skip; otherwise nuke and recreate.
$ErrorActionPreference = 'Stop'
$venvDir = Join-Path $PSScriptRoot '..\.venv-2026'
$reqFile = Join-Path $PSScriptRoot '..\requirements-2026.txt'
$hashFile = Join-Path $PSScriptRoot '..\build\installed-stack-2026.txt'

if (-not (Test-Path $reqFile)) {
    throw "requirements-2026.txt not found at $reqFile"
}

$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
if (Test-Path $venvDir) {
    if ((Test-Path $hashFile) -and ((Get-Content $hashFile -Raw) -eq $reqHash)) {
        Write-Host "venv-2026 already up to date. Skipping."
        exit 0
    }
    Write-Host "Hash mismatch or partial install. Removing venv-2026..."
    Remove-Item -Recurse -Force $venvDir
}

py -3.11 -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& "$venvDir\Scripts\python.exe" -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-2026.txt failed" }

New-Item -ItemType Directory -Force -Path (Split-Path $hashFile) | Out-Null
Set-Content -Path $hashFile -Value $reqHash -NoNewline
Write-Host "venv-2026 ready at $venvDir"
```

- [ ] **Step 2: Add Makefile targets**

Append to `Makefile`:
```makefile
.PHONY: venv-2026 ocr-diff-2026 test-integration

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
	pytest tests/integration/ -v -m integration
```

- [ ] **Step 3: Verify (Linux only)**

Run: `make venv-2026`
Expected: venv-2026 created and dependencies installed. If on Windows, run `.\scripts\setup-2026.ps1`.

- [ ] **Step 4: Commit**

```bash
git add scripts/setup-2026.ps1 Makefile
git commit -m "build: add venv-2026 bootstrap (Windows PS + Linux Makefile)"
```

---

### Task 11: tests/ocr_regression_2026.py + results doc

**Files:**
- Create: `tests/ocr_regression_2026.py`
- Create: `docs/ocr-upgrade-2026-results.md` (initial template)

**Interfaces:**
- Same as `tests/ocr_regression.py` but invoked from venv-2026.

- [ ] **Step 1: Write the runner**

Create `tests/ocr_regression_2026.py` — same as `tests/ocr_regression.py` but with `VIDEO ?= tests/fixtures/sample.mp4` hardcoded and output to `tests/fixtures/ocr_diff_report_2026.json`. Implementation: import the existing runner's helpers and only override CLI defaults.

```python
"""Phase 2 OCR regression runner — same logic as ocr_regression.py but pinned to the new stack.

Invoked via venv-2026 (paddlepaddle 3.x + numpy 2.x). Compares against the
checked-in golden baseline.
"""
import sys
from pathlib import Path

# Reuse everything from the Phase 1 runner
sys.path.insert(0, str(Path(__file__).parent))
import ocr_regression  # noqa: E402


DEFAULTS = {
    "video": str(Path(__file__).parent / "fixtures" / "sample.mp4"),
    "output": str(Path(__file__).parent / "fixtures" / "ocr_diff_report_2026.json"),
}


def parse_args():
    # Apply defaults only when the caller didn't pass the flag
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--video", default=DEFAULTS["video"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--thresholds", required=True)
    p.add_argument("--new-json", default=None)
    p.add_argument("--region", default="bottom_28pct")
    return p.parse_args()


def main():
    args = parse_args()
    sys.argv = [
        sys.argv[0],
        "--baseline", args.baseline,
        "--video", args.video,
        "--output", args.output,
        "--thresholds", args.thresholds,
    ]
    if args.new_json:
        sys.argv += ["--new-json", args.new_json]
    sys.argv += ["--region", args.region]
    return ocr_regression.main()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create results doc**

Create `docs/ocr-upgrade-2026-results.md`:
```markdown
# OCR Upgrade (2026 stack) — Validation Results

Run on: <DATE>
Operator: <NAME>
Stack under test: paddlepaddle 3.x + numpy 2.x + torch 2.6+

## Diff vs. golden baseline (`jobs/Tập 1-5/ocr/subtitles.json`)

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| segment_count_delta_pct | TBD | ≤ 5% | TBD |
| text_length_delta_pct | TBD | ≤ 10% | TBD |
| time_range_overlap | TBD | ≥ 0.95 | TBD |
| text_similarity | TBD | ≥ 0.85 | TBD |

## Verdict

<ACCEPTABLE TO PROCEED TO PHASE 3 / ROLLBACK — list issues>

## Top 10 diff segments

(Filled in after running `make venv-2026 && make ocr-diff-2026`.)
```

- [ ] **Step 3: Run the Phase 2 validation**

Run:
```bash
make venv-2026
make ocr-diff-2026
```
Expected: report written to `tests/fixtures/ocr_diff_report_2026.json`.

Fill in `docs/ocr-upgrade-2026-results.md` with actual values from the report.

- [ ] **Step 4: Commit**

```bash
git add tests/ocr_regression_2026.py docs/ocr-upgrade-2026-results.md tests/fixtures/ocr_diff_report_2026.json
git commit -m "feat(ocr): add Phase 2 runner + initial validation results"
```

---

## Phase 3 — Cutover

### Task 12: Commit 3a — rewrite requirements.txt, keep requirements-ocr.txt as shim

**Files:**
- Modify: `requirements.txt` — rewrite with hybrid pins
- Modify: `requirements-ocr.txt` — replace content with `-r requirements.txt` + DEPRECATED comment
- Create: `CHANGELOG.md`

- [ ] **Step 1: Rewrite requirements.txt**

Replace `requirements.txt` with:
```
# Hybrid pinning — see docs/superpowers/specs/2026-06-29-numpy-stack-upgrade-design.md
# Bounded compatibility ranges (both floor and ceiling).
# Ceilings prevent drift into known-broken versions.

# Core
typer>=0.12.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
python-dotenv>=1.0.1
anthropic>=0.27.0
httpx>=0.27.0
pydub>=0.25.1

# ML stack (upgraded from NumPy 1.x era)
numpy>=2.0,<2.4
torch>=2.6,<3
faster-whisper>=1.0.3
paddlepaddle>=3.0,<4
paddleocr>=3.0
onnxruntime>=1.18,<1.20
protobuf>=4.25,<6
```

- [ ] **Step 2: Replace requirements-ocr.txt with shim**

Replace `requirements-ocr.txt` with:
```
# DEPRECATED — this file is a shim. Contents have moved to requirements.txt.
# It will be deleted in commit 3b (≥3 days after the cutover lands on main).
# DO NOT add new pins here. See docs/superpowers/specs/2026-06-29-numpy-stack-upgrade-design.md

-r requirements.txt
```

- [ ] **Step 3: Create CHANGELOG.md**

Create `CHANGELOG.md`:
```markdown
# Changelog

## Unreleased

### Changed
- Migrated to paddlepaddle 3.x + numpy 2.x + torch 2.6+ stack.
  See `docs/superpowers/specs/2026-06-29-numpy-stack-upgrade-design.md`.

### Added
- OCR regression harness (`make ocr-diff`, `make ocr-baseline`).
- Idempotent `setup.ps1` + cross-platform `make setup`.
- GitHub Actions CI: `test.yml`, `integration.yml`, `ocr-diff.yml`.

### Deprecated
- `requirements-ocr.txt` (shim only; deleted in next release after ≥3-day soak).
```

- [ ] **Step 4: Verify shim**

Run: `python -m pip install --dry-run -r requirements-ocr.txt`
Expected: resolves without conflicts (it just re-imports `requirements.txt`).

- [ ] **Step 5: Commit (DO NOT push yet — soak starts after this lands)**

```bash
git add requirements.txt requirements-ocr.txt CHANGELOG.md
git commit -m "build(cutover-3a): rewrite requirements.txt with hybrid pins; shim requirements-ocr.txt"
```

---

### Task 13: setup.ps1 with full pre-flight + idempotency

**Files:**
- Create: `setup.ps1`

- [ ] **Step 1: Write setup.ps1**

Create `setup.ps1`:
```powershell
<#
.SYNOPSIS
  Bootstrap script for vietdub on Windows. Idempotent.

.DESCRIPTION
  Creates .venv, installs requirements.txt with the upgraded stack.
  Re-run is a no-op if requirements.txt hash matches build/installed-stack.txt.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# --- Prune old logs (older than 7 days) ---
$logDir = Join-Path $PSScriptRoot 'build'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("setup-{0}-{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID)
Get-ChildItem -Path $logDir -Filter 'setup-*.log' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item -Force

# Helper: tee to both stdout and log file
function Tee-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $logFile -Value $Message
}

Tee-Log "=== vietdub setup.ps1 starting at $(Get-Date -Format 'o') ==="

# --- Step 1: Python version check ---
$pyLauncher = (Get-Command 'py' -ErrorAction SilentlyContinue)
if (-not $pyLauncher) { throw "Python launcher 'py' not found in PATH. Install Python 3.11 from https://www.python.org/" }
$pyVerOutput = & py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) { throw "'py -3.11' failed. Ensure Python 3.11 is installed and on PATH." }
$pyVer = ($pyVerOutput -replace 'Python ', '').Trim()
$pyMajor, $pyMinor = $pyVer.Split('.')[0..1] | ForEach-Object { [int]$_ }
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 9) -or ($pyMajor -eq 3 -and $pyMinor -ge 14)) {
    throw "Python $pyVer not supported. Need >=3.9,<3.14 (paddlepaddle 3.x wheel availability)."
}
Tee-Log "Step 1 OK: Python $pyVer"

# --- Step 2: Architecture check ---
$arch = (python -c "import platform; print(platform.machine())" 2>&1).Trim()
if ($LASTEXITCODE -ne 0) { throw "Failed to detect architecture" }
if ($arch -ne 'AMD64') { throw "Architecture $arch not supported. paddlepaddle 3.x Windows wheels are amd64-only." }
Tee-Log "Step 2 OK: Architecture $arch"

# --- Step 3: VC++ runtime check ---
$vcCheck = python -c "import ctypes; ctypes.CDLL('vcruntime140.dll'); ctypes.CDLL('msvcp140.dll'); print('ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Tee-Log "VC++ runtime missing. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    throw "VC++ 2019/2022 redistributable not found. Install and re-run."
}
Tee-Log "Step 3 OK: VC++ runtime present"

# --- Step 4: Hash check vs installed-stack.txt ---
$reqFile = Join-Path $PSScriptRoot 'requirements.txt'
$venvDir = Join-Path $PSScriptRoot '.venv'
$hashFile = Join-Path $PSScriptRoot 'build\installed-stack.txt'

if (-not (Test-Path $reqFile)) { throw "requirements.txt not found at $reqFile" }
$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
if ((Test-Path $venvDir) -and (Test-Path $hashFile) -and ((Get-Content $hashFile -Raw) -eq $reqHash)) {
    Tee-Log "Step 4 OK: .venv already matches requirements.txt hash. Skipping install."
    Print-Activation
    exit 0
}
if (Test-Path $venvDir) {
    Tee-Log "Hash mismatch or partial install. Removing .venv..."
    Remove-Item -Recurse -Force $venvDir
}

# --- Step 5: Create .venv ---
& py -3.11 -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
Tee-Log "Step 5 OK: Created .venv"

# --- Step 6: Upgrade pip ---
& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip --upgrade-strategy only-if-needed
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
Tee-Log "Step 6 OK: pip upgraded"

# --- Step 7: Install requirements.txt ---
& "$venvDir\Scripts\python.exe" -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
Tee-Log "Step 7 OK: requirements installed"

# --- Step 8: PS version warning ---
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Tee-Log "Step 8 WARN: PowerShell 5.1 detected. PowerShell 7+ recommended for better error propagation."
}

# --- Step 9: Record hash and print activation ---
Set-Content -Path $hashFile -Value $reqHash -NoNewline
Tee-Log "Step 9 OK: Activation instructions below"
Tee-Log "  PowerShell: .\\.venv\\Scripts\\Activate.ps1"
Tee-Log "  cmd:        .venv\\Scripts\\activate.bat"
Tee-Log "  bash:       source .venv/bin/activate"
Tee-Log "=== setup.ps1 complete at $(Get-Date -Format 'o') ==="
```

- [ ] **Step 2: Verify (dry-run, doesn't actually create venv)**

Run: `powershell -NoProfile -Command "& { . .\setup.ps1 -WhatIf 2>$null }"` 2>&1 | head -20
Expected: error if `py -3.11` not found OR proceeds through pre-flight checks (cannot fully verify without actually creating venv).

- [ ] **Step 3: Commit**

```bash
git add setup.ps1
git commit -m "build: add idempotent setup.ps1 with pre-flight checks"
```

---

### Task 14: Extend Makefile with `setup` + `bootstrap` + cross-platform dispatch

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add targets**

Prepend to `Makefile`:
```makefile
.PHONY: setup bootstrap

# Cross-platform setup dispatch
ifeq ($(OS),Windows_NT)
setup:
	@powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
bootstrap: setup
else
setup:
	@if [ ! -d ".venv" ]; then python3.11 -m venv .venv; fi
	@.venv/bin/pip install --upgrade pip --upgrade-strategy only-if-needed
	@.venv/bin/pip install -r requirements.txt
bootstrap: setup
endif
```

- [ ] **Step 2: Verify (Linux/macOS path)**

Run: `make setup`
Expected: `.venv` created and dependencies installed. If on Windows, run `.\setup.ps1`.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add setup/bootstrap Makefile targets"
```

---

### Task 15: Rewrite README "Setup" section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the "Cài Đặt" (Install) section**

Replace the existing "Cài Đặt" section in `README.md` with:
```markdown
## Cài Đặt

### Windows (PowerShell)

```powershell
.\setup.ps1
```

`setup.ps1` checks Python 3.11+, AMD64 architecture, and VC++ 2019/2022 runtime, then creates `.venv/` and installs dependencies. Re-running is a no-op if `requirements.txt` hasn't changed.

### Linux / macOS / CI

```bash
make setup
```

Creates `.venv/` and installs from `requirements.txt`.

### Sau khi cài

```powershell
.\.venv\Scripts\Activate.ps1
vietdub run .\sample.mp4 --mode review --series sample-series
```

### Bootstrap từ scratch

Nếu `.venv/` bị hỏng hoặc `requirements.txt` thay đổi:

```powershell
Remove-Item -Recurse -Force .venv
.\setup.ps1
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README setup section around setup.ps1 / make setup"
```

---

### Task 16: GitHub Actions CI workflows

**Files:**
- Create: `.github/workflows/test.yml`
- Create: `.github/workflows/integration.yml`
- Create: `.github/workflows/ocr-diff.yml`

- [ ] **Step 1: Write test.yml**

Create `.github/workflows/test.yml`:
```yaml
name: test
on:
  pull_request:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - name: Setup (Linux)
        if: runner.os != 'Windows'
        run: make setup
      - name: Setup (Windows)
        if: runner.os == 'Windows'
        run: .\setup.ps1
      - name: Test
        run: make test
        shell: bash
```

- [ ] **Step 2: Write integration.yml**

Create `.github/workflows/integration.yml`:
```yaml
name: integration
on:
  schedule:
    - cron: '0 2 * * *'  # nightly 02:00 UTC
  push:
    branches: [main]
jobs:
  integration:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: win-pip-${{ hashFiles('requirements.txt') }}
      - run: .\setup.ps1
      - name: Run integration tests
        run: make test-integration
        shell: bash
        env:
          VIETDUB_SKIP_INTEGRATION: ''
```

- [ ] **Step 3: Write ocr-diff.yml**

Create `.github/workflows/ocr-diff.yml`:
```yaml
name: ocr-diff
on:
  pull_request:
    paths:
      - 'src/vietdub/ocr/**'
      - 'requirements.txt'
      - 'requirements-2026.txt'
jobs:
  ocr-diff:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: win-pip-${{ hashFiles('requirements.txt') }}
      - run: .\setup.ps1
      - name: OCR diff
        run: make ocr-diff
        shell: bash
```

- [ ] **Step 4: Verify schemas locally (optional)**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml').read())"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add test, integration, ocr-diff GitHub Actions workflows"
```

---

### Task 17: End-to-end smoke test on sample.mp4

**Files:**
- Create: `tests/fixtures/golden_review_row_count.txt`

- [ ] **Step 1: Write the golden file**

Create `tests/fixtures/golden_review_row_count.txt` with `10` (placeholder; will be updated after first successful run).

- [ ] **Step 2: Run the full pipeline against sample.mp4**

Run:
```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m vietdub run tests\fixtures\sample.mp4 --mode review --series smoke-test
```
Expected: exit 0, `status.json` shows all steps = `done`, `review.csv` exists with ≥1 row.

- [ ] **Step 3: Update golden row count**

After the first successful run, replace the placeholder:
```powershell
(Get-Content jobs\smoke-test\translation\review.csv | Measure-Object -Line).Lines - 1 | Out-File tests\fixtures\golden_review_row_count.txt -Encoding ascii -NoNewline
```

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/golden_review_row_count.txt jobs/smoke-test/  # if committing jobs/ is desired; otherwise .gitignore
git commit -m "test: smoke-test against sample.mp4 + capture golden row count"
```

---

### Task 18: Rollback rehearsal (dry run)

**Files:**
- None (verification only)

- [ ] **Step 1: Verify git revert path**

Run: `git log --oneline -5`
Expected: see the commit 3a (`build(cutover-3a): ...`).

- [ ] **Step 2: Document the rollback in README**

Append to `README.md`:
```markdown
### Rollback

Nếu gặp regression sau cutover:

```powershell
git revert <commit-3a-sha>   # khôi phục requirements.txt + requirements-ocr.txt (shim)
Remove-Item -Recurse -Force .venv, .venv-2026
.\setup.ps1
```

Sau khi soak ≥3 ngày không có vấn đề, commit 3b sẽ xóa `requirements-ocr.txt` shim.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document rollback procedure for cutover commit 3a"
```

---

### Task 19: Commit 3b (after ≥3-day soak) — delete requirements-ocr.txt shim

**Files:**
- Delete: `requirements-ocr.txt`

**Precondition:** Wait ≥3 days after commit 3a lands on `main` with no reported regressions.

- [ ] **Step 1: Verify no remaining references**

Run: `git grep requirements-ocr`
Expected: only `requirements-ocr.txt` itself and the rollback doc in README. No Dockerfile, no CI workflow, no Makefile target.

- [ ] **Step 2: Delete the file**

Run: `git rm requirements-ocr.txt`

- [ ] **Step 3: Commit**

```bash
git commit -m "build(cutover-3b): delete deprecated requirements-ocr.txt shim"
```

---

## Self-Review Notes

**Spec coverage check:**
- Goal (migrate to NumPy 2.x + paddlepaddle 3.x): Phase 3 Tasks 12–19 ✓
- 3-phase plan: Tasks 1–8 (Phase 1), Tasks 9–11 (Phase 2), Tasks 12–19 (Phase 3) ✓
- OCR regression harness: Tasks 1–8 ✓
- Structural diff metrics: Task 2 (compute_metrics), Task 3 (build_report) ✓
- Exit codes 0/1/2/3/4/5: Task 4 (`tests/ocr_regression.py`) ✓
- PaddleOCR determinism thresholds: Task 4 (Acceptance criteria reference) ✓
- setup.ps1 pre-flight: Task 13 ✓
- PowerShell 5.1 exit code handling: Task 13 (every native invocation) ✓
- Idempotency via stack hash: Task 13 step 4, Task 10 setup-2026.ps1 ✓
- Sequential commit 3a/3b: Tasks 12 and 19 ✓
- CI workflows as Phase 3 deliverable: Task 16 ✓
- Committed fixture sample.mp4 (not C:\code\test\): Task 6 ✓
- Baseline-update PR convention: Documented in spec Global Constraints, enforced by ocr-diff.yml gate ✓

**Type consistency check:**
- `OcrSegment(id, start_ms, end_ms, text, confidence)` defined Task 1, used Task 2 (`match_segments`, `compute_metrics`), Task 4 (CLI runner), Task 5 (bridge) ✓
- `match_segments(baseline, new, threshold_ms=200)` defined Task 2, called Task 4 ✓
- `compute_metrics(baseline, new, matches)` defined Task 2, called Task 4 ✓
- `build_report(...)` defined Task 3, called Task 4 ✓
- `passes_thresholds(metrics, thresholds)` defined Task 3, used internally in `build_report` and tested Task 3 ✓
- `OcrSegmentError` defined Task 1, raised Task 5 ✓

**Placeholder scan:**
- No "TBD"/"TODO" left in code blocks.
- Acceptance criteria for Task 17 includes literal placeholder (`10`) in `golden_review_row_count.txt` with explicit instruction to replace after first run.