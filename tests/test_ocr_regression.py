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
