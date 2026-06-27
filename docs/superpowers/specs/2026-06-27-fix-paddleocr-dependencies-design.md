# Fix PaddleOCR Dependencies — Design

**Date:** 2026-06-27
**Scope:** Dependency version resolution for PaddleOCR (no source code changes to viethoaphimv4)
**Status:** Approved (design), pending implementation

## Context

The viethoaphimv4 OCR step (`PaddleSubtitleOcrEngine` in `src/vietdub/ocr.py`) currently fails at import time on this system due to a dependency conflict:

- `paddlepaddle==2.6.2` requires `protobuf<=3.20.2`
- `onnxruntime==1.27.0` requires `protobuf>=4.25.8`
- `tensorflow==2.21.0` requires `protobuf<8.0.0,>=6.31.1`

Additionally, paddlepaddle 2.6.2 triggers a Windows-specific `OSError: [WinError 127]` when loading `torch\lib\shm.dll`, indicating torch version incompatibility.

**Current workaround:** `C:\code\test\scripts\run_pipeline.py` monkey-patches `PaddleSubtitleOcrEngine.recognize` to return `[]`, bypassing OCR entirely. The pipeline runs (extract → STT → context → translate → review CSV) but loses Chinese subtitle extraction from video frames.

**Goal:** Find a reproducible set of pinned package versions that allow the OCR step to load and run on Windows, documented in a `requirements-ocr.txt` file. No viethoaphimv4 source code changes.

## Non-goals

- Modify viethoaphimv4 source code (per user constraint).
- Build production Docker images (out of scope; documented as future option).
- Fix paddlepaddle itself.
- Add OCR unit tests to viethoaphimv4's existing test suite (those use `FixtureOcrEngine`, not real PaddleOCR).

## Design

### Section 1: Discovery + Documentation Architecture

Create two new files in the viethoaphimv4 repo:

**`viethoaphimv4/requirements-ocr.txt`** — Pinned versions for the OCR stack that are known to work together. Format: pip-compatible requirements file with comments explaining why each version was chosen.

**`viethoaphimv4/docs/ocr-setup.md`** — Setup guide covering:
- Why a separate requirements file (vs. adding to `pyproject.toml`)
- The conflict history (paddlepaddle 2.6.2 vs onnxruntime 1.27.0)
- How to install: `pip install -r requirements-ocr.txt`
- Verification test (run the OCR engine, assert it loads without crashing)
- Fallback paths if the pinned combo fails on a future system
- Reference to the official paddlepaddle Docker image as the production-grade solution

### Section 2: Discovery flow

Try approaches in order of least disruption:

**Approach A (preferred):** Pin all packages to old protobuf 3.x compatible versions.

Hypothesis: paddlepaddle 2.6.2's constraint (`protobuf<=3.20.2`) is the hard one. Downgrade `onnxruntime` to last version supporting protobuf 3.x (probably ≤1.13), and `tensorflow` to ≤2.13.

Steps:
1. `pip install "protobuf==3.20.2"`
2. `pip install "onnxruntime<=1.13.1"`
3. `pip install "tensorflow<=2.13.0"` (or 2.15 if compatible)
4. `pip install "torch" --force-reinstall` (in case torch DLL needs refresh for paddlepaddle 2.6.2)
5. Run verification test
6. If works → pin exact versions in `requirements-ocr.txt`
7. If fails (still DLL error or other) → try Approach B

**Approach B (fallback):** Upgrade to paddlepaddle 3.x.

Hypothesis: paddlepaddle 3.3.1 (latest stable) uses newer protobuf compatible with onnxruntime 1.27.0 and tensorflow 2.21.0.

Steps:
1. `pip install paddlepaddle==3.3.1 paddleocr==3.x`
2. `pip install paddlepaddle-gpu==3.3.1` (if GPU needed)
3. Verify `PaddleSubtitleOcrEngine().recognize(...)` works (note: paddleocr 3.x API may differ — investigate)
4. If works → pin exact versions
5. If paddleocr 3.x API incompatible with viethoaphimv4's `PaddleSubtitleOcrEngine` → revert to A or fallback to bypass

**Approach C (last resort):** Document that dependency-only fix isn't feasible; recommend official paddlepaddle Docker image for production.

Steps:
1. Document in `ocr-setup.md` why A and B failed
2. Provide Dockerfile template using `paddlepaddle/paddle:3.3.1` base image
3. Keep the existing `scripts/run_pipeline.py` OCR bypass as the local-dev fallback

### Section 3: Verification test

A small standalone script that confirms OCR stack loads correctly:

**`C:\code\test\scripts\verify_ocr.py`:**
```python
"""Verify PaddleOCR stack loads correctly after installing requirements-ocr.txt."""
from vietdub.ocr import PaddleSubtitleOcrEngine


def main() -> None:
    engine = PaddleSubtitleOcrEngine()
    print("[OK] PaddleSubtitleOcrEngine instantiated successfully")
    # Optional: actually invoke recognize on a test image to verify end-to-end OCR
    # engine.recognize(Path("test_image.png"))


if __name__ == "__main__":
    main()
```

Exit code 0 = OCR stack compatible. Non-zero = something still broken.

### Section 4: requirements-ocr.txt format

```
# OCR dependency stack for viethoaphimv4
# Pinned to a known-working combination — see docs/ocr-setup.md for why.
# Install with: pip install -r requirements-ocr.txt

# Approach A target (if it works):
protobuf==3.20.2
paddlepaddle==2.6.2
paddleocr==2.10.0
onnxruntime==1.13.1
tensorflow==2.13.0
# torch version pinned to paddlepaddle 2.6.2's expected version
torch==2.2.0  # TBD after testing
```

(Actual versions filled in during implementation after discovery testing.)

### Section 5: Implementation order

1. **Discovery phase (manual, ~10-15 min):**
   - Try Approach A, test with `verify_ocr.py`
   - If A fails, try B
   - Document the winning combo's exact versions
   
2. **Documentation phase:**
   - Write `requirements-ocr.txt` with pinned versions
   - Write `docs/ocr-setup.md` with conflict history + setup instructions
   
3. **Verification phase:**
   - Run `verify_ocr.py` to confirm the pinned combo works
   - Run full viethoaphimv4 pipeline on `Tập 1.mp4` (in `C:\code\test\`) WITHOUT OCR bypass, confirm OCR step succeeds
   - Update existing `C:\code\test\scripts\run_pipeline.py` to remove the OCR bypass (since OCR now works)

### Section 6: Failure mode handling

**If Approach A fails:**
- Document A's failure in `ocr-setup.md` (e.g., "paddlepaddle 2.6.2 + onnxruntime 1.13 + tensorflow 2.13 still has torch DLL error on Windows 11")
- Move to Approach B

**If Approach B fails (paddlepaddle 3.x API break):**
- Document B's failure
- Revert to Approach A's failure state
- Recommend Docker approach in `ocr-setup.md`

**If both A and B fail:**
- Keep `scripts/run_pipeline.py` OCR bypass as documented workaround
- Document the Docker-based solution as the production-grade path
- Consider filing an issue against paddlepaddle for Windows compatibility

### Section 7: Rollback

All changes are additive:
- New file `requirements-ocr.txt` (no impact if not installed)
- New file `docs/ocr-setup.md` (documentation only)
- New file `C:\code\test\scripts\verify_ocr.py` (verification tool)

No existing files are modified. If the design doesn't work:
- Delete the 3 new files → state reverts to before this work
- viethoaphimv4's existing `pyproject.toml` and test suite are unchanged

## Risk

**Medium.** Dependency version compatibility is hard to predict in advance. May require multiple iterations. Mitigation: clear rollback path (no source changes), and the existing OCR bypass in `scripts/run_pipeline.py` continues to work as fallback.

## Implementation notes

- **Discover first, document second.** Don't try to predict exact compatible versions in the spec; verify locally and pin what works.
- **The Windows DLL issue is the highest-risk failure mode.** Paddlepaddle 2.6.2's bundled torch may not match the installed torch. May need to reinstall torch with `--force-reinstall --no-cache-dir`.
- **CPU vs GPU matters.** Windows paddlepaddle is CPU-only by default (GPU single-card only). Don't pull `paddlepaddle-gpu` unless GPU is the target.
- **onnxruntime and tensorflow may not be strictly required by viethoaphimv4's runtime.** They might be transitive deps from some other install (e.g., a different project). Verify before downgrading.
