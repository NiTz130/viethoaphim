# Fix PaddleOCR Dependencies — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a reproducible set of pinned package versions that allow viethoaphimv4's PaddleOCR step to load and run on Windows, documented in `requirements-ocr.txt` + `docs/ocr-setup.md`.

**Architecture:** Discovery-driven approach — try pinning old protobuf-compatible versions (Approach A), fall back to paddlepaddle 3.x upgrade (Approach B), fall back to documenting Docker-only solution (Approach C). Document the winning combo in `requirements-ocr.txt` for reproducibility. Verify OCR works end-to-end via a small standalone script. Update the existing `C:\code\test\scripts\run_pipeline.py` to remove its OCR bypass once OCR works.

**Tech Stack:** Python 3.11, pip, paddlepaddle, paddleocr, protobuf, onnxruntime, tensorflow, torch.

## Global Constraints

- Spec: `C:\code\viethoaphimv4\docs\superpowers\specs\2026-06-27-fix-paddleocr-dependencies-design.md` (authoritative)
- **No source code changes to viethoaphimv4.** Only dependencies, documentation, and the OCR bypass script in `C:\code\test\`.
- New artifacts go to `viethoaphimv4/requirements-ocr.txt` (pinned deps) and `viethoaphimv4/docs/ocr-setup.md` (setup guide).
- New verification script goes to `C:\code\test/scripts/verify_ocr.py`.
- Discovery flow is sequential: try Approach A first, then B if A fails, then C if both fail.
- Each task ends with a documented outcome (either working pinned versions or documented failure with fallback).

---

### Task 1: Discovery — find a working PaddleOCR dependency combo

**Files:**
- Create: `C:\code\viethoaphimv4\requirements-ocr.txt`
- Read: `C:\code\viethoaphimv4\pyproject.toml` (already read; dependencies are listed there)

**Goal:** Determine which of the 3 approaches produces a working PaddleOCR stack.

- [ ] **Step 1: Try Approach A — pin old protobuf-compatible versions**

Run these commands in order:

```bash
pip install "protobuf==3.20.2"
pip install "onnxruntime==1.13.1"
pip install "tensorflow==2.13.0"
pip install "torch==2.2.0" --force-reinstall --no-cache-dir
```

(Reasoning: protobuf 3.20.2 is the highest version paddlepaddle 2.6.2 accepts. onnxruntime 1.13.1 is the last version supporting protobuf 3.x. tensorflow 2.13 is the last version supporting protobuf 3.x. torch 2.2.0 is forced to match paddlepaddle 2.6.2's expected torch version.)

- [ ] **Step 2: Test if PaddleOCR imports successfully**

Run:

```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR imports OK')"
```

- [ ] **Step 3: Test if viethoaphimv4's PaddleSubtitleOcrEngine loads**

Run:

```bash
python -c "from vietdub.ocr import PaddleSubtitleOcrEngine; e = PaddleSubtitleOcrEngine(); print('Engine instantiated OK')"
```

Expected:
- **If both succeed**: Approach A works. Record versions in `requirements-ocr.txt` (Step 5).
- **If PaddleOCR import fails** (protobuf/torch/dll error): Move to Approach B (Step 4).
- **If `PaddleSubtitleOcrEngine` instantiation fails** (but PaddleOCR import worked): There's a viethoaphimv4-side issue. Move to Approach B.

- [ ] **Step 4 (conditional): If Approach A failed, try Approach B — upgrade paddlepaddle to 3.x**

Run:

```bash
pip install "paddlepaddle==3.3.1" "paddleocr==3.0.1"
```

(Reasoning: paddlepaddle 3.3.1 is the latest stable. paddleocr 3.0.1 is a guess — verify by running `pip install paddleocr` first to see latest 3.x, then pin.)

If the exact version `3.0.1` doesn't exist, run `pip install paddlepaddle==3.3.1 paddleocr` (unpinned) to get latest matching 3.x, then capture versions with `pip freeze | grep paddle`.

Then re-test with:

```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR imports OK')"
python -c "from vietdub.ocr import PaddleSubtitleOcrEngine; e = PaddleSubtitleOcrEngine(); print('Engine instantiated OK')"
```

If both succeed: Approach B works. Pin versions in `requirements-ocr.txt`.
If still failing: Move to Approach C.

- [ ] **Step 5 (conditional): If both A and B failed, document Approach C fallback**

If neither approach worked:
- Do NOT create `requirements-ocr.txt` (or create with a single comment line: `# No compatible version found — see docs/ocr-setup.md`)
- Note the failure in the report
- The `verify_ocr.py` script will then test the OCR stack as documented (and likely fail)
- Task 3 will document this in `ocr-setup.md` and recommend Docker for production

- [ ] **Step 6: Write `requirements-ocr.txt` with the winning combo (or failure note)**

If A or B worked, create `C:\code\viethoaphimv4\requirements-ocr.txt`:

```
# Pinned OCR dependency stack for viethoaphimv4
# Generated 2026-06-27 — see docs/ocr-setup.md for context
# Install with: pip install -r requirements-ocr.txt
#
# Approach used: A (pinned old versions) | B (paddlepaddle 3.x)
# <insert approach label here>

<package>==<exact.version>
<package>==<exact.version>
...
```

Fill in with the exact versions found by Steps 3 or 4. Include ALL packages whose version you pinned (protobuf, onnxruntime OR paddlepaddle 3.x, tensorflow if relevant, torch if pinned).

- [ ] **Step 7: Commit `requirements-ocr.txt`**

```bash
cd C:\code\viethoaphimv4
git add requirements-ocr.txt
git commit -m "build: add requirements-ocr.txt with PaddleOCR-compatible pinned versions"
```

- [ ] **Step 8: Write report**

Write to `C:\code\viethoaphimv4\.superpowers\sdd\task-1-report.md` (the `.superpowers/sdd/` directory is gitignored):

- Approach attempted (A, B, or both)
- Final outcome (which approach worked, or both failed)
- Exact versions pinned
- Any errors encountered
- Decision: proceed to Task 2 with winning approach, OR escalate to user if both A and B failed

**Report back (under 15 lines):**
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- Commit SHA + subject
- Approach used (A, B, or NONE)
- One-line: "OCR stack works with pinned versions X.Y.Z / Y.Z / Z.Y" OR "OCR stack failed both A and B"
- Report file path

---

### Task 2: Document setup in `docs/ocr-setup.md`

**Files:**
- Create: `C:\code\viethoaphimv4\docs\ocr-setup.md`
- Read: `C:\code\viethoaphimv4\requirements-ocr.txt` (written in Task 1)

**Goal:** Provide reproducible setup instructions + troubleshooting for the OCR stack.

- [ ] **Step 1: If `requirements-ocr.txt` doesn't exist (Approach C fallback), create a minimal placeholder**

```markdown
# OCR Setup

No pip-only dependency combination was found to be compatible on this system.

For production deployment, use the official paddlepaddle Docker image:
```dockerfile
FROM paddlepaddle/paddle:3.3.1
# ... (rest of viethoaphimv4 setup)
```

For local development, use the OCR bypass in `C:\code\test\scripts\run_pipeline.py`.
```

- [ ] **Step 2: If `requirements-ocr.txt` exists, write the full `ocr-setup.md`**

```markdown
# OCR Setup

## Background

viethoaphimv4 uses PaddleOCR for Chinese subtitle extraction. The OCR stack
(paddlepaddle + paddleocr + their transitive deps) has known compatibility
constraints:

- paddlepaddle 2.6.2 requires `protobuf<=3.20.2`
- onnxruntime >=1.14 requires `protobuf>=4.25.8`
- tensorflow >=2.14 requires `protobuf>=4.x`

These constraints conflict in a single Python environment.

## Working combo

<insert "Approach A: pinned old versions" or "Approach B: paddlepaddle 3.x">

Versions (pinned in `requirements-ocr.txt`):
<list packages + versions>

## Install

```bash
pip install -r requirements-ocr.txt
```

## Verify

```bash
python -c "from vietdub.ocr import PaddleSubtitleOcrEngine; PaddleSubtitleOcrEngine()"
```

If this prints nothing and exits 0, OCR stack is compatible.

## Troubleshooting

If install fails:
- Ensure `pip` is recent (`pip install --upgrade pip`)
- Try installing packages individually to identify the conflict
- Check [paddlepaddle issues](https://github.com/PaddlePaddle/Paddle/issues) for Windows-specific bugs

If OCR still fails after install:
- Check torch version matches paddlepaddle's expected version
- On Windows, try `pip install torch==<paddlepaddle's expected version> --force-reinstall`
- Consider Docker-based deployment (see "Alternative" below)

## Alternative: Docker

For production deployment, paddlepaddle provides official Docker images:
```dockerfile
FROM paddlepaddle/paddle:3.3.1
```
These images have pre-resolved dependency graphs and avoid the version conflict.
```

Customize with the actual approach + versions used.

- [ ] **Step 3: Commit `docs/ocr-setup.md`**

```bash
cd C:\code\viethoaphimv4
git add docs/ocr-setup.md
git commit -m "docs: add OCR setup guide with dependency resolution rationale"
```

- [ ] **Step 4: Write report**

Write to `C:\code\viethoaphimv4\.superpowers\sdd\task-2-report.md`:
- File created with what content (approach, versions, troubleshooting)
- Self-review: did the docs cover install, verify, troubleshoot?

**Report back (under 15 lines):**
- Status: DONE | DONE_WITH_CONCERNS
- Commit SHA + subject
- One-line summary of `ocr-setup.md` content
- Report file path

---

### Task 3: Create `verify_ocr.py` and verify OCR works end-to-end

**Files:**
- Create: `C:\code\test\scripts\verify_ocr.py`

**Goal:** A standalone verification script that confirms the OCR stack loads correctly.

- [ ] **Step 1: Write `verify_ocr.py`**

Create `C:\code\test\scripts\verify_ocr.py`:

```python
"""Verify PaddleOCR stack loads correctly after installing requirements-ocr.txt.

Usage:
    python scripts/verify_ocr.py

Exits 0 if OCR stack is compatible. Non-zero otherwise.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout for Vietnamese / Chinese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    print("[1/3] Importing paddleocr...")
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        print(f"[FAIL] PaddleOCR import failed: {exc}")
        return 1

    print("[2/3] Importing viethoaphimv4's PaddleSubtitleOcrEngine...")
    try:
        from vietdub.ocr import PaddleSubtitleOcrEngine
    except Exception as exc:
        print(f"[FAIL] viethoaphimv4 OCR engine import failed: {exc}")
        return 1

    print("[3/3] Instantiating PaddleSubtitleOcrEngine...")
    try:
        engine = PaddleSubtitleOcrEngine()
    except Exception as exc:
        print(f"[FAIL] Engine instantiation failed: {exc}")
        return 1

    print("[OK] PaddleOCR stack loads correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run `verify_ocr.py`**

```bash
cd C:\code\viethoaphimv4
python C:\code\test\scripts\verify_ocr.py
```

Expected (if Task 1 succeeded): exit code 0, prints `[OK] PaddleOCR stack loads correctly`.

If exit code 1: OCR stack still not loading. Check error message. May need to revisit Task 1 (try different versions) or escalate to user.

- [ ] **Step 3: Commit `verify_ocr.py`**

```bash
cd C:\code\test
git add scripts/verify_ocr.py
git commit -m "test: add verify_ocr.py for OCR stack smoke test"
```

(Note: `C:\code\test` may not be a git repo. If not, skip the commit and just note in the report.)

- [ ] **Step 4: Write report**

Write to `C:\code\viethoaphimv4\.superpowers\sdd\task-3-report.md`:
- Output of `verify_ocr.py` run
- Pass/fail status
- If fail, the error message

**Report back (under 15 lines):**
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- Commit SHA + subject (if applicable)
- One-line: "OCR verification PASSED" OR "OCR verification FAILED with <error>"
- Report file path

---

### Task 4: Update `C:\code\test\scripts\run_pipeline.py` to remove OCR bypass

**Files:**
- Modify: `C:\code\test\scripts\run_pipeline.py`

**Goal:** Now that OCR works, remove the monkey-patch bypass so the pipeline runs with real OCR.

- [ ] **Step 1: Read current `run_pipeline.py`**

```bash
cat C:\code\test\scripts\run_pipeline.py
```

- [ ] **Step 2: If Task 3 passed (OCR works), remove the monkey-patch**

Remove the OCR-bypass section. The script becomes:

```python
"""Run viethoaphimv4 pipeline on Tập 1.mp4.

Prerequisites: install viethoaphimv4 and run `pip install -r requirements-ocr.txt`.

Usage:
    python scripts/run_pipeline.py [review|auto]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout for Vietnamese / Chinese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure viethoaphimv4 is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "viethoaphimv4" / "src"))


def main(mode: str = "review") -> None:
    from vietdub.config import Settings
    from vietdub.pipeline import run_review_pipeline

    video_path = Path(r"C:\code\test\Tập 1.mp4")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    settings = Settings()
    print(f"[CONFIG] LLM_MODEL={settings.llm_model}")
    print(f"[CONFIG] jobs_dir={settings.jobs_dir}")

    job = run_review_pipeline(
        video=video_path,
        jobs_dir=Path(settings.jobs_dir),
        series=None,
        settings=settings,
    )
    print(f"\n[OK] Review file written: {job.root / 'translation' / 'review.csv'}")

    if mode == "review":
        print("[MODE] review — stopped before TTS. Edit review.csv then resume.")
    else:
        from vietdub.pipeline import resume_tts_and_render
        srt_path = resume_tts_and_render(job, settings)
        print(f"[OK] Vietnamese subtitles: {srt_path}")
        print("[OK] TTS segment files written under tts/segments.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "review"
    if mode not in {"review", "auto"}:
        raise SystemExit(f"mode must be 'review' or 'auto', got: {mode}")
    main(mode)
```

- [ ] **Step 3: Run the updated `run_pipeline.py` end-to-end**

```bash
cd C:\code\viethoaphimv4
python C:\code\test\scripts\run_pipeline.py review
```

Expected:
- Pipeline runs from extract → STT → OCR (now real!) → merge → context → translate → review CSV
- Exit code 0
- Output mentions OCR segments (not "Skipping PaddleOCR")

If exit code 1: OCR still not working in the actual pipeline run (different from import-time check). Investigate further.

- [ ] **Step 4: Commit the updated script**

(Only if `C:\code\test` is a git repo. If not, skip and just report the change.)

- [ ] **Step 5: Write report**

Write to `C:\code\viethoaphimv4\.superpowers\sdd\task-4-report.md`:
- Whether OCR bypass was successfully removed
- Whether end-to-end pipeline ran successfully with real OCR
- Any new errors encountered

**Report back (under 15 lines):**
- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- Commit SHA + subject (if applicable)
- One-line: "OCR bypass removed; pipeline runs with real OCR" OR "Bypass NOT removed because OCR still failing"
- Report file path

---

## Self-review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Try Approach A (pin old versions) | Task 1 Steps 1-3 |
| Try Approach B (paddlepaddle 3.x) if A fails | Task 1 Step 4 |
| Approach C fallback (Docker docs) | Task 1 Step 5 + Task 2 |
| `requirements-ocr.txt` artifact | Task 1 Step 6 |
| `docs/ocr-setup.md` artifact | Task 2 |
| `verify_ocr.py` artifact | Task 3 |
| Update `run_pipeline.py` to remove OCR bypass | Task 4 |
| Fallback chain documented | Task 2 Step 2 |
| Rollback safety | All tasks additive only |

No gaps.

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later" — exact commands provided throughout.
- Code blocks shown for all code changes.
- Exact versions to try are specified (with fallback guidance if exact versions don't exist).

**3. Type consistency:**

- `requirements-ocr.txt` format consistent (pip-compatible with comments).
- `verify_ocr.py` uses consistent error reporting (exit codes 0/1).
- `run_pipeline.py` retains same function signature and behavior, just removes the bypass.

**4. Risk handling:**

- Task 1 has explicit fallback paths (A → B → C).
- Task 3 verifies before Task 4 removes bypass — if Task 3 fails, Task 4 keeps bypass.
- All tasks are additive (no destructive changes to viethoaphimv4).
