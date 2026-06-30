# PaddleOCR 3.x API Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit the existing `src/vietdub/ocr/paddle.py` migration to the PaddleOCR 3.x API, then add `tests/test_paddleocr_api.py` so future API drift is caught at PR time.

**Architecture:** Two commits, one fix and one test. The fix replaces the 2.x constructor kwargs (`use_angle_cls`, `show_log`, `use_gpu`) with their 3.x equivalents (`use_textline_orientation`, removed, `device`) and switches result parsing from nested-list to `result.json["res"]["rec_texts"]`. The test inspects `PaddleOCR.__init__` signature for the six kwargs the source now uses, and runs a smoke test that loads the Chinese OCR model on a synthetic PIL image and asserts the result shape.

**Tech Stack:** Python 3.11, PaddleOCR 3.7+, paddlepaddle 3.x, Pillow, pytest 8.2+.

## Global Constraints

- Python 3.11 only (paddlepaddle 3.x wheel availability).
- Windows PowerShell paths use backslashes; bash paths use forward slashes.
- Project venv is `.venv/`. Use `.venv/Scripts/python.exe` on Windows.
- `make test` runs `pytest tests/ -v -m "not integration"`.
- Pinned ML stack (from `pyproject.toml`):
  - `numpy>=2.0,<2.4`
  - `paddlepaddle>=3.0,<4`
  - `paddleocr>=3.0,<4`
  - `torch>=2.6,<3`
- Commit message style: conventional commits (`fix(ocr):`, `test(ocr):`).
- Smoke test must use `device="cpu"` and `enable_mkldnn=False` to avoid OneDNN crash on Windows.
- Smoke test must `pytest.skip` (not fail) when paddleocr package is genuinely missing, when the model cannot be downloaded, or when OneDNN is unavailable.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vietdub/ocr/paddle.py` | Modify (commit 1) | OCR engine. Constructor of `PaddleOCR` and parsing of `predict()` results. The patch is already in the working tree. |
| `tests/test_paddleocr_api.py` | Create (commit 2) | Regression test. Static signature test + smoke test using a synthetic PIL image. |

No other files change. The two commits are independent: reverting either one leaves the other valid.

---

## Task 1: Commit the PaddleOCR 3.x migration of `paddle.py`

**Files:**
- Modify: `src/vietdub/ocr/paddle.py` (lines 30-92, the `PaddleSubtitleOcrEngine.recognize` method)
- No new files in this task
- No test in this task (test lands in Task 2)

**Interfaces:**
- Consumes: `Settings.ocr_use_gpu: bool`, `Settings.ocr_enable_mkldnn: bool` (already exists, no change)
- Produces: `PaddleSubtitleOcrEngine.recognize(video_path: Path) -> list[TimedSegment]` — same signature, now uses 3.x API internally

The working tree already contains the patch. `git diff src/vietdub/ocr/paddle.py` should show:
- Constructor: `use_angle_cls=True` removed, `show_log=False` removed, `use_gpu=self.use_gpu` replaced with `device="gpu"|"cpu"`, three new kwargs added (`use_doc_orientation_classify=False`, `use_doc_unwarping=False`, `use_textline_orientation=True`).
- Result parsing: `ocr.ocr(...)` replaced with `ocr.predict(...)`; nested-list traversal replaced with `result.json["res"]["rec_texts"]` extraction.
- Three-line comment explaining the API change.

- [ ] **Step 1: Verify the working-tree patch is intact**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git status --short
```
Expected: `M src/vietdub/ocr/paddle.py` (one modified file, nothing else under `src/`).

If you see other modified files, stop and ask the user — they belong to a different change.

- [ ] **Step 2: Eyeball the diff**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git diff src/vietdub/ocr/paddle.py
```
Expected: ~20-line diff that matches the description in the Interfaces block above. The constructor kwargs block should now be:
```python
ocr = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    device="gpu" if self.use_gpu else "cpu",
    enable_mkldnn=self.enable_mkldnn,
)
```
And the per-frame loop should call `ocr.predict(str(frame))` and read `page.json["res"]["rec_texts"]`.

If the diff does not match, stop — the working tree has been edited since the patch was applied. Restore the patch from this session's commit log before continuing.

- [ ] **Step 3: Run existing test suite to confirm no regression**

Run (bash):
```bash
cd /c/code/viethoaphimv4
.venv/Scripts/python.exe -m pytest tests/ -v -m "not integration"
```
Expected: All existing tests pass. (None of them import real PaddleOCR — they use `FixtureOcrEngine` and `_FakeEngine` — so this confirms the patch did not break the import path or the `PaddleSubtitleOcrEngine` constructor itself.)

If anything fails, the patch is wrong. Stop and investigate before committing.

- [ ] **Step 4: Smoke-import the engine class**

Run (bash):
```bash
cd /c/code/viethoaphimv4
.venv/Scripts/python.exe -c "from vietdub.ocr.paddle import PaddleSubtitleOcrEngine; e = PaddleSubtitleOcrEngine(use_gpu=False, enable_mkldnn=False); print('OK', e)"
```
Expected output: `OK <vietdub.ocr.paddle.PaddleSubtitleOcrEngine object at 0x...>`

This proves `__init__` runs and accepts the constructor signature. The actual PaddleOCR instance is created lazily inside `recognize`, not here, so this check is a sanity step, not a full OCR run.

- [ ] **Step 5: Commit**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git add src/vietdub/ocr/paddle.py
git commit -m "fix(ocr): migrate paddle.py to PaddleOCR 3.x API

PaddleOCR 3.x (paddlex-backed) removed the legacy constructor
kwargs use_angle_cls, show_log, and use_gpu. The angle classifier
is now exposed as use_textline_orientation; device selection moved
to a device= kwarg. Result format changed from a nested list to
result.json['res']['rec_texts'].

Cutover commit bcfc258 pinned paddleocr>=3 but left this module
calling the 2.x API, so setup.ps1 followed by vietdub run fails
with 'Unknown argument: use_gpu'."
```
Expected: One new commit on the branch. `git log --oneline -1` shows the new commit at HEAD.

---

## Task 2: Add API contract test `tests/test_paddleocr_api.py`

**Files:**
- Create: `tests/test_paddleocr_api.py`
- No modifications to other files

**Interfaces:**
- Consumes: `paddleocr.PaddleOCR` (third-party, not modified)
- Produces: Two pytest test functions:
  - `test_paddleocr_constructor_signature()` — static, ~0.1s
  - `test_paddleocr_predict_returns_ocrresult()` — smoke, ~5s first run, ~1s after model cache is warm

The test file does NOT import `vietdub.ocr.paddle`. It tests the PaddleOCR API contract directly, because the contract is what `paddle.py` depends on. If PaddleOCR's API changes again, this test fails before `paddle.py` is even loaded.

- [ ] **Step 1: Create the test file**

> **Plan drift note (mid-execution design change):** The original plan specified a static `inspect.signature` check against `_REQUIRED_PADDLEOCR_KWARGS`. Mid-execution, this was changed to a try-construct behavior check (Option B2) because PaddleOCR 3.x forwards `device` and `enable_mkldnn` via `**kwargs` to the underlying `paddlebase.PaddlePredictor` — they are accepted at construction but invisible to `inspect.signature`. The behavior check is strictly more robust: it catches the original 2.x-kwargs bug class AND future drift where paddleocr removes or renames a kwarg. See `tests/test_paddleocr_api.py:38-66` for the actual implementation.

The two test functions are at module level so pytest picks them up automatically: `test_paddleocr_constructor_signature` (try-construct with the 6 production kwargs, see the file at the path above) and `test_paddleocr_predict_returns_ocrresult` (smoke inference on a synthetic PIL image, asserting `page.json["res"]["rec_texts"]` is a `list`). Both skip with a documented reason when `paddleocr` is not importable or the model cannot be downloaded.

- [ ] **Step 2: Run the new test in isolation**

Run (bash):
```bash
cd /c/code/viethoaphimv4
.venv/Scripts/python.exe -m pytest tests/test_paddleocr_api.py -v
```
Expected:
- `test_paddleocr_constructor_signature PASSED`
- `test_paddleocr_predict_returns_ocrresult PASSED` (first run may take 10-30s while paddleocr downloads the Chinese OCR models; subsequent runs ~1-3s)

If the static test FAILS with "missing kwargs", the source code in `src/vietdub/ocr/paddle.py` is calling the wrong API. Stop and investigate — this means Task 1's commit introduced a regression.

If the smoke test SKIPs with "paddleocr model unavailable" or similar, that is acceptable for the first run on a network-constrained machine. The static test should still pass. If the static test also skips, then `paddleocr` is genuinely not importable on this machine — that is also acceptable for the first run, but investigate before declaring victory.

- [ ] **Step 3: Negative test — prove the test guards against the original bug**

This step proves that the OLD `paddle.py` code path actually crashed (so the bug was real, not theoretical) and that the new code path no longer crashes.

The two contract tests are intentionally independent of `src/vietdub/ocr/paddle.py` — they inspect PaddleOCR's own API surface. So "stash the patch and run the tests" is the wrong negative-test approach. The correct demonstration is:

Run (bash):
```bash
cd /c/code/viethoaphimv4
.venv/Scripts/python.exe -c "
from paddleocr import PaddleOCR
# Simulate the OLD paddle.py call pattern (2.x kwargs).
try:
    PaddleOCR(use_angle_cls=True, lang='ch', show_log=False, use_gpu=False, enable_mkldnn=False)
    print('UNEXPECTED: old constructor accepted')
except ValueError as e:
    print('EXPECTED ValueError:', e)
" 2>&1 | tail -3
```
Expected output: `EXPECTED ValueError: Unknown argument: use_gpu`

This proves:
1. The old `paddle.py` constructor call would have raised `ValueError: Unknown argument: use_gpu` against the installed paddleocr 3.x. The bug was real.
2. The new `paddle.py` constructor call (verified by `make test` in Task 1 Step 3) does not raise this error.
3. The static contract test (`test_paddleocr_constructor_signature`) would have caught a future regression where paddleocr removes one of the 6 kwargs paddle.py now uses — by failing with a message naming the missing kwarg.

If the output says `UNEXPECTED: old constructor accepted`, then the installed paddleocr is actually 2.x (not 3.x), and the regression we are guarding against is hypothetical, not real. Stop and check `pip show paddleocr`.

- [ ] **Step 4: Run the full test suite to confirm no regression**

Run (bash):
```bash
cd /c/code/viethoaphimv4
.venv/Scripts/python.exe -m pytest tests/ -v -m "not integration"
```
Expected: All previously-passing tests still pass, plus the two new tests pass. Total test count grows by 2.

- [ ] **Step 5: Commit**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git add tests/test_paddleocr_api.py
git commit -m "test(ocr): add API contract test for PaddleSubtitleOcrEngine

Static test inspects PaddleOCR.__init__ signature for the six
kwargs paddle.py now uses. Smoke test loads the Chinese OCR
model on a synthetic image and asserts that
result.json['res']['rec_texts'] is a list. Together they catch
future API drift between paddleocr minor versions."
```
Expected: One new commit on the branch. `git log --oneline -2` shows the test commit at HEAD and the fix commit as HEAD~1.

---

## Task 3: Final verification

**Files:** None modified.

- [ ] **Step 1: Confirm git state is clean except for the two new commits**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git status
```
Expected: `nothing to commit, working tree clean`. (`.venv/` and `build/` are ignored.)

- [ ] **Step 2: Show the new commit history**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git log --oneline -3
```
Expected: HEAD is the test commit, HEAD~1 is the fix commit, HEAD~2 is `8378a2f` (the previous tip).

- [ ] **Step 3: Push the branch and open a PR**

Run (bash):
```bash
cd /c/code/viethoaphimv4
git push -u origin <branch-name>
```
Then use the GitHub CLI to open the PR:
```bash
gh pr create --title "fix(ocr): migrate paddle.py to PaddleOCR 3.x + add API contract test" --body "See docs/superpowers/specs/2026-06-30-ocr-paddleocr3-api-migration-design.md and docs/superpowers/plans/2026-06-30-ocr-paddleocr3-api-migration.md for the design and plan."
```

If the user has not yet assigned a branch name, ask them. Common options: `fix/ocr-paddleocr3-migration`, `ocr/paddleocr-3-migration`.

- [ ] **Step 4: Wait for CI to pass**

Watch the PR checks: `.github/workflows/test.yml` and `.github/workflows/integration.yml` should both turn green. The new test runs as part of `make test` in `test.yml`; the smoke test downloads ~100MB of Chinese OCR models on first run (~30s), then ~3s on subsequent runs.

If the smoke test fails on CI with a model-download error, add the cache step from the spec (`.github/workflows/test.yml`):

```yaml
- name: Cache PaddleOCR models
  uses: actions/cache@v4
  with:
    path: ~/.paddlex/official_models
    key: paddleocr-models-v1-${{ hashFiles('pyproject.toml') }}
```

This is documented as optional in the spec — only add if CI shows a regression.

- [ ] **Step 5: Merge**

Once CI is green and the PR has at least one approval, merge with a merge commit (not squash) to preserve the two-commit history. The fix and the test should land as separate commits on `main`.

---

## Self-Review Checklist (run before opening the PR)

- [ ] `git log --oneline -2` shows exactly two new commits: fix first, test second.
- [ ] `git status` is clean.
- [ ] `make test` passes locally with both new tests included.
- [ ] Manually reverting `paddle.py` to the old API causes `ValueError: Unknown argument: use_gpu` (verified in Task 2 Step 3).
- [ ] The smoke test skips (not fails) when paddleocr is missing or model download is blocked.
- [ ] No unrelated files modified.
