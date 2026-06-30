"""Regenerate review.csv from translated.json + merged.json using the canonical export function.

This restores proper CSV quoting (DictWriter auto-quotes fields with commas/newlines)
that was lost when a manual rewrite stripped quoting.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vietdub.models import TimedSegment, TranslationRow  # noqa: E402
from vietdub.translate import export_review_csv  # noqa: E402

job_dir = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9")

merged = [TimedSegment.model_validate(s) for s in json.loads((job_dir / "transcript" / "merged.json").read_text(encoding="utf-8"))]
translations = [TranslationRow.model_validate(r) for r in json.loads((job_dir / "translation" / "translated.json").read_text(encoding="utf-8"))]

# Apply the same skip-on-fallback rule as before
for row in translations:
    if row.text_vi.startswith("[CHƯA DỊCH]"):
        row.status = "skip"

export_review_csv(job_dir / "translation" / "review.csv", merged, translations)
print(f"OK: rewrote {len(translations)} rows with proper CSV quoting")
print(f"  Output: {job_dir / 'translation' / 'review.csv'}")

# Validate it imports cleanly
from vietdub.translate.csv import import_review_csv  # noqa: E402
parsed = import_review_csv(job_dir / "translation" / "review.csv")
print(f"Re-imported OK: {len(parsed)} rows; first id={parsed[0].segment_id}, last id={parsed[-1].segment_id}")
skipped = sum(1 for r in parsed if r.status == "skip")
print(f"Skipped rows: {skipped}")