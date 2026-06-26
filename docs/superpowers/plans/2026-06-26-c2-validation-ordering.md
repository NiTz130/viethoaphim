# C2 Validation Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder checks in `_validate_resume_segment_ids` so path-safety is validated before duplicate/membership, with a regression test that locks in the contract.

**Architecture:** Single function reorder in `pipeline.py`. New test follows existing path-traversal test patterns. Two commits: one for the test, one for the fix.

**Tech Stack:** Python 3.11, pytest 9, pydantic 2

## Global Constraints

- Run tests with: `python -m pytest 2>&1 | tail -20`
- All test files in `tests/`; all source files in `src/vietdub/`
- Commit messages use imperative mood, scoped prefix (`fix:`, `test:`, `docs:`)
- Test pattern for path-traversal rejection: use UTF-8-SIG encoding for CSV, monkeypatch `vietdub.tts.EdgeTtsEngine.synthesize_segment` to assert it never runs
- Helper: `_resume_settings()` exists in `tests/test_pipeline.py` to build a stub settings object
- Helper: `_assert_no_resume_render_artifacts(job)` exists for verifying cleanup after failure
- Do not touch `resume_tts_and_render` — only `_validate_resume_segment_ids` (lines 152-174)
- Do not modify other validation helpers (`_is_safe_segment_path_component`, `_is_safe_fallback_segment_id`, `_load_allowed_segment_ids`)

---

### Task 1: Add regression test for unsafe duplicate segment_id

**Files:**
- Modify: `tests/test_pipeline.py` (add new test at end of the `test_resume_tts_*` block, around line 528, after `test_resume_tts_rejects_transcript_path_traversal_segment_id`)

**Interfaces:**
- Consumes: existing `Job`, `resume_tts_and_render`, `_resume_settings`, monkeypatch fixture
- Produces: `test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message` test function

- [ ] **Step 1: Write the test**

Insert the following test function in `tests/test_pipeline.py` directly after `test_resume_tts_rejects_transcript_path_traversal_segment_id` (ends at line 527 with `assert not (job_root / "tts" / "evil.mp3").exists()`):

```python
def test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message(monkeypatch, tmp_path):
    """C2 regression: unsafe ID appearing twice must report 'Invalid segment_id'
    for the first occurrence, never 'Duplicate segment_id' which would mask the
    real issue (path traversal in the segment_id itself)."""
    job_root = tmp_path / "job"
    unsafe_id = "..\\evil"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        f"{unsafe_id},0,1000,,你好,Xin chao,,reviewed\n"
        f"{unsafe_id},1000,2000,,你好,Xin chao tiep,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": unsafe_id, "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    called = False

    async def fail_synthesize_segment(self, row, output):
        nonlocal called
        called = True
        raise AssertionError("synthesis should not run for unsafe segment_id")

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fail_synthesize_segment)

    with pytest.raises(RuntimeError) as exc_info:
        resume_tts_and_render(job, _resume_settings())

    message = str(exc_info.value)
    assert "Invalid segment_id" in message
    assert unsafe_id in message
    assert "Duplicate" not in message
    assert not called
    assert not (job_root / "tts" / "evil.mp3").exists()
```

- [ ] **Step 2: Run the new test**

Run: `python -m pytest tests/test_pipeline.py::test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message -v 2>&1 | tail -15`

Expected: PASS on current (pre-fix) code because the first occurrence of `..\\evil` is caught by `_is_safe_segment_path_component` before the duplicate check runs. This test is **regression coverage** that locks in the contract — it does not exhibit a behavior difference between old and new code, but it documents the expected error message for the unsafe-duplicate scenario and prevents future regressions if the check order is changed back.

- [ ] **Step 3: Run full test suite to confirm no collateral damage**

Run: `python -m pytest 2>&1 | tail -5`

Expected: 95 passed (94 existing + 1 new)

- [ ] **Step 4: Commit the test**

```bash
git add tests/test_pipeline.py
git commit -m "test: cover unsafe duplicate segment_id error message"
```

---

### Task 2: Reorder checks in _validate_resume_segment_ids

**Files:**
- Modify: `src/vietdub/pipeline.py` (replace the body of the `for row in rows` loop in `_validate_resume_segment_ids`, lines 156-174)

**Interfaces:**
- Consumes: `Job`, `list[TranslationRow]` (unchanged signature)
- Produces: reordered checks; same external behavior except cleaner internal control flow

- [ ] **Step 1: Read current implementation**

Read `src/vietdub/pipeline.py` lines 152-174 to confirm the current state.

Expected output: function body matches the pre-fix ordering described in Task 2 below. If it already matches the new ordering, abort and report — the fix may already be applied.

- [ ] **Step 2: Replace the loop body**

In `src/vietdub/pipeline.py`, replace the entire `for row in rows:` block (lines 156-174, from `for row in rows:` through the last `if not _is_safe_fallback_segment_id(...)` block) with:

```python
    for row in rows:
        if row.status == "skip" or not row.text_vi.strip():
            continue
        segment_id = row.segment_id
        if row.start_ms < 0 or row.end_ms <= row.start_ms:
            raise RuntimeError(
                f"Invalid timing in {review_path}: {segment_id!r} start_ms={row.start_ms} end_ms={row.end_ms}"
            )
        if not _is_safe_segment_path_component(segment_id):
            raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
        if allowed_ids is not None:
            if segment_id not in allowed_ids:
                raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r} is not in transcript/merged.json")
        elif not _is_safe_fallback_segment_id(segment_id):
            raise RuntimeError(f"Invalid segment_id in {review_path}: {segment_id!r}")
        if segment_id in active_ids:
            raise RuntimeError(f"Duplicate segment_id in {review_path}: {segment_id!r}")
        active_ids.add(segment_id)
```

Three changes from the old version:
1. The `_is_safe_segment_path_component` check (formerly at the old position) moved up to right after the timing check
2. The `if allowed_ids is not None: ... continue` followed by `if not _is_safe_fallback_segment_id(...)` is now structured as `if ... / elif ...` so the two branches are mutually exclusive
3. The duplicate check (formerly before the safety check) and `active_ids.add(...)` moved to the end

- [ ] **Step 3: Run the new test from Task 1**

Run: `python -m pytest tests/test_pipeline.py::test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message -v 2>&1 | tail -10`

Expected: PASS

- [ ] **Step 4: Run full test suite to confirm no regressions**

Run: `python -m pytest 2>&1 | tail -5`

Expected: 95 passed, 0 failed. All pre-existing tests should pass because the observable behavior for each scenario (unique safe ID, unique unsafe ID, unique ID not in transcript, duplicate safe ID, skipped rows) is unchanged.

- [ ] **Step 5: Commit the fix**

```bash
git add src/vietdub/pipeline.py
git commit -m "fix: reorder segment_id validation to check path-safety first"
```

---

## Self-Review Notes

- **Spec coverage:**
  - Code reorder → Task 2
  - New test → Task 1
  - All existing tests unaffected → Task 2 Step 4
  - Risk/rollback noted in spec, not re-implemented in plan
- **No placeholders:** all code blocks are complete and runnable
- **Type consistency:** function signature unchanged; no new types introduced
- **Test infrastructure reuse:** new test uses existing helpers (`_resume_settings`) and patterns (CSV encoding, monkeypatch) from `test_resume_tts_rejects_transcript_path_traversal_segment_id`
