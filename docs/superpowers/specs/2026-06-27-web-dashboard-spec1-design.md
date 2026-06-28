# Web Dashboard Spec 1: Job List + Job Detail — Design

**Date:** 2026-06-28
**Scope:** New `viethoaphimv4-dashboard/` repo: FastAPI backend + React frontend for monitoring viethoaphimv4 jobs
**Status:** Approved (design), pending implementation

## Context

viethoaphimv4 is currently a Windows-first CLI tool (Python with typer). Operators monitor jobs by:
- Tail the terminal output
- Manually browse the `jobs/` directory in Explorer/Finder
- Open `review.csv` in Excel/LibreOffice to verify translations

For non-technical operators and for batch processing, this workflow is cumbersome. They need a web dashboard that shows:
- What jobs are running / completed / failed
- How far along each job is (which step, how many segments translated)
- What the pipeline is logging in real-time
- Where to download the final outputs (review.csv, SRT, TTS segments)

This is **Spec 1 of N** for the web dashboard series:
- Spec 1: Job list + Job detail (MVP — this spec)
- Spec 2 (future): Video library, Settings page
- Spec 3 (future): Auth, multi-user
- Spec 4 (future): Analytics, charts
- Spec 5 (future): Real-time log streaming + WebSocket improvements

## Goal

Build a local-first web dashboard that:
- Wraps the existing viethoaphimv4 CLI (no source changes to viethoaphimv4 itself)
- Shows job list with status, progress, and metadata
- Shows job detail with step-by-step progress, real-time logs, and segment preview
- Provides download links for outputs (review.csv)

**Non-goals:**
- Multi-user / auth (single-user local)
- Remote deployment (local only, `localhost:8000`)
- Modifying viethoaphimv4 source code
- Real-time WebSocket (use polling + SSE for now)
- Settings page (uses existing `.env` file)
- Analytics / charts
- Production-grade security (local-only)

## Design

### Section 1: Architecture

**New repo:** `C:\code\viethoaphimv4-dashboard\`

**Backend** (Python 3.11+, FastAPI):
- `viethoaphimv4_dashboard/main.py`: FastAPI app, mounts static, includes routers
- `viethoaphimv4_dashboard/api/jobs.py`: Job endpoints (REST)
- `viethoaphimv4_dashboard/api/logs.py`: SSE log streaming
- `viethoaphimv4_dashboard/services/scanner.py`: Read `jobs/` directory, parse `status.json`
- `viethoaphimv4_dashboard/services/runner.py`: Subprocess wrapper for `vietdub run`
- `viethoaphimv4_dashboard/models.py`: Pydantic models

**Frontend** (Node 18+, React 18 + Vite + TypeScript):
- `frontend/src/App.tsx`: Router with `/jobs`, `/jobs/new`, `/jobs/:id`
- `frontend/src/pages/JobListPage.tsx`: Data-dense table
- `frontend/src/pages/JobDetailPage.tsx`: Header + progress + log + segments
- `frontend/src/pages/NewJobPage.tsx`: File upload + mode selector
- `frontend/src/components/`: JobRow, ProgressStep, LogStream, SegmentTable, StatusBadge
- `frontend/src/api/client.ts`: Fetch wrapper

**Build system:** Vite for dev server + production bundle.

**Process model:** FastAPI spawns viethoaphimv4 as subprocess per job. Jobs tracked in-memory (no DB; jobs/ directory is the source of truth). Dashboard reads jobs/ state via scanner.

### Section 2: REST API

```
GET  /api/jobs
  Returns: [{id, name, video_name, status, mode, created_at, updated_at, progress_pct}, ...]
  Reads: jobs/*/status.json files

GET  /api/jobs/{id}
  Returns: Job (full detail with progress per step)
  Reads: jobs/{id}/status.json

GET  /api/jobs/{id}/review.csv
  Returns: text/csv (Content-Disposition: attachment)
  Reads: jobs/{id}/translation/review.csv

GET  /api/jobs/{id}/segments
  Returns: [{segment_id, start_ms, end_ms, speaker, text_cn, text_vi, status}, ...]
  Reads: jobs/{id}/translation/review.csv (parsed)

GET  /api/jobs/{id}/logs
  Returns: text/event-stream (SSE)
  Reads: jobs/{id}/output/*.log (tail -f style)
  Note: Spec 1 uses polling fallback (1s interval), SSE-ready architecture for Spec 5

POST /api/jobs
  Body: {video_path: str, mode: "review"|"auto"}
  Returns: {id: str}
  Action: spawn `vietdub run` subprocess, return job id (video stem + timestamp)
  Note: video_path must be absolute, must exist, must be a video file

DELETE /api/jobs/{id}
  Action: kill running subprocess (if any)
  Returns: 204 No Content
```

### Section 3: Data model

```python
class StepProgress(BaseModel):
    name: str               # "extract" | "stt" | "ocr" | "merge" | "context" | "translate" | "tts" | "render"
    status: str             # "pending" | "running" | "done" | "failed" | "skipped"
    count: int | None = None
    total: int | None = None
    error: str | None = None

class JobSummary(BaseModel):
    id: str
    name: str
    video_name: str
    status: str             # "running" | "completed" | "failed" | "pending"
    mode: str               # "review" | "auto"
    created_at: datetime
    updated_at: datetime
    progress_pct: int       # 0-100
    segments_translated: int | None = None
    segments_total: int | None = None

class Job(JobSummary):
    steps: list[StepProgress]
    has_tts: bool
    has_video_output: bool
    has_review_csv: bool
    has_srt: bool
```

### Section 4: Frontend routes

- `/` → redirect to `/jobs`
- `/jobs` → JobListPage: data-dense table with status badges, progress bars
- `/jobs/new` → NewJobPage: video file path input (with file picker), mode selector (review/auto), submit
- `/jobs/:id` → JobDetailPage: header (status, name, video, timestamps, actions), progress steps with status icons, real-time log stream (polling 1s), segment preview table (virtualized), download buttons (review.csv, SRT)
- `/404` → NotFoundPage

### Section 5: Visual design

Per UI/UX Pro Max recommendations:
- **Style:** Data-Dense Dashboard (light + dark mode, navy primary `#1E3A5F`, green accent `#059669`)
- **Typography:** Fira Sans (UI) + Fira Code (data tables, logs)
- **Spacing:** 4dp/8dp rhythm
- **Status colors:** green (done), blue (running), red (failed), gray (pending)
- **Touch targets:** ≥44px (mobile compatibility, even though this is desktop)
- **Accessibility:** WCAG AA contrast, focus rings, keyboard nav

Status badge component:
- `done`: green background (`bg-emerald-100`), dark green text
- `running`: blue background with pulsing dot
- `failed`: red background
- `pending`: gray background

Progress bar: Fira Code numerals showing `42 / 100` segments on the right.

Log stream: monospace, dark background, auto-scroll on new lines, filter by level (debug/info/warn/error).

## Test plan

**Backend tests** (FastAPI TestClient, pytest):

1. `test_get_jobs_empty` — fresh state, no jobs → empty list
2. `test_post_job_starts_vietdub_subprocess` — POST with valid video, subprocess spawned, status.json appears
3. `test_post_job_with_invalid_video_returns_400` — missing path → 400
4. `test_post_job_with_unknown_mode_returns_400` — mode="invalid" → 400
5. `test_get_jobs_lists_running_job_with_status` — after start, list shows job with status="running"
6. `test_get_job_detail_returns_step_progress` — after translation step, status.json has step entries
7. `test_get_job_detail_404_for_unknown_id` — random ID → 404
8. `test_get_review_csv_returns_csv_content` — CSV download matches file on disk
9. `test_get_segments_parses_review_csv` — segment list with text_cn + text_vi populated
10. `test_get_logs_returns_text_event_stream` — content-type correct
11. `test_delete_running_job_kills_subprocess` — DELETE on running job, subprocess terminates
12. `test_progress_pct_reflects_segments_translated_ratio`

**Frontend tests** (Vitest + React Testing Library):

1. `JobListPage renders empty state` — no jobs → "No jobs yet" message
2. `JobListPage renders rows with status badges` — 3 jobs, 3 different status colors
3. `NewJobPage calls API on submit` — fill form, click submit, fetch POST called with correct body
4. `JobDetailPage shows progress steps` — 5 steps with mixed statuses
5. `LogStream auto-scrolls on new messages` — polls /api/jobs/{id}/logs, appends messages
6. `SegmentTable renders Vietnamese diacritics correctly`
7. `JobRow clicking navigates to detail page`

**E2E test** (in C:\code\test\ framework):
- New test: `test_dashboard_e2e_starts_job_and_shows_progress` — POST to local dashboard API, poll for status="completed", verify review.csv downloadable

**Visual smoke test** (manual):
- Run `uvicorn viethoaphimv4_dashboard.main:app --reload`
- Open `http://localhost:5173`
- Verify: light + dark mode, mobile viewport (responsive), keyboard nav

## Risk

**Medium.** New repo, new stack (FastAPI + React). Mitigations:
- No changes to existing viethoaphimv4 source code
- All jobs continue to run via CLI (dashboard is read-only + start jobs)
- Existing 149 unit tests still pass (no regressions expected)
- Dashboard uses subprocess to invoke viethoaphimv4, not Python imports

## Rollback

If dashboard causes issues, just stop the uvicorn process. viethoaphimv4 CLI continues to work standalone.

## Implementation notes

- **Storage:** No database. Jobs are read from `jobs/` directory on each API call. `status.json` is the source of truth (created by viethoaphimv4's `Job.mark_done`).
- **Polling vs SSE:** Spec 1 uses polling (1s interval for status, 2s for logs). SSE architecture is in place (route returns `text/event-stream`) but full streaming is deferred to Spec 5.
- **Auth:** None in Spec 1. Local-only single-user assumption. Spec 3 adds auth.
- **CORS:** FastAPI uses CORSMiddleware with `allow_origins=["http://localhost:5173"]` (Vite dev server).
- **Static serving:** In production, FastAPI serves the built `frontend/dist/` as static files. In dev, Vite runs separately on port 5173.
- **Bootstrap:** `viethoaphimv4-dashboard/scripts/bootstrap.sh` (or `.bat` for Windows) creates venv, installs deps, runs migrations (none for Spec 1), starts both uvicorn and vite.
- **Subprocess lifecycle:** `runner.py` keeps a registry of `subprocess.Popen` objects keyed by job id. On dashboard shutdown, all running subprocesses receive SIGTERM.
- **Polling efficiency:** Scanner caches `status.json` mtimes to avoid re-reading unchanged files. Cache invalidates on file mtime change.

## Out of scope (future specs)

- Spec 2: Video library page (browse past outputs by video file)
- Spec 3: Auth (single-user password → multi-user)
- Spec 4: Settings page (model selection, batch size, etc.)
- Spec 5: True SSE/WebSocket log streaming + chart-based analytics
