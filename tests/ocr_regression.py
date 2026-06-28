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

# Windows console defaults to cp1252 which chokes on Vietnamese (Tập) and other
# non-ASCII characters in paths/reports. Force UTF-8 so the runner is portable.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

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
    """Invoke the project's PaddleOCR engine via the vietdub.ocr.bridge module."""
    from vietdub.ocr.bridge import run_pipeline_ocr
    return run_pipeline_ocr(video_path, region=region)


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
