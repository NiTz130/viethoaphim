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
