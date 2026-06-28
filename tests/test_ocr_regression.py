import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNNER = ROOT / "tests" / "ocr_regression.py"

# Spawned subprocesses don't inherit pytest's pythonpath=["src"], so without
# this the subprocess imports the stale site-packages vietdub/ocr.py instead
# of the source tree's vietdub/ocr/ package. Phase 3 setup.ps1 reinstalls
# vietdub cleanly; until then, this keeps tests self-sufficient.
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def test_runner_help_exits_zero():
    r = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True, env=_SUBPROCESS_ENV)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "--baseline" in r.stdout
    assert "--video" in r.stdout


def test_runner_missing_baseline_exits_2(tmp_path):
    r = subprocess.run(
        [sys.executable, str(RUNNER),
         "--baseline", str(tmp_path / "nope.json"),
         "--video", str(tmp_path / "v.mp4"),
         "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True, env=_SUBPROCESS_ENV,
    )
    assert r.returncode == 2, f"stderr: {r.stderr}"
    assert "baseline" in r.stderr.lower()
