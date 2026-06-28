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
