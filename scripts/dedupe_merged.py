"""Dedupe consecutive merged segments with identical text.

This fixes the overlap/repeat bug where OCR samples the same subtitle every 0.5s
(so a subtitle that stays on screen for 2-3s appears 4-6 times in merged.json
and gets spoken 4-6 times by TTS).

Groups consecutive merged segments whose text is identical AND whose time ranges
overlap or touch, then collapses each group into a single segment using the
group's earliest start_ms and latest end_ms. Maps old segment_ids to new ones,
and updates translated.json so existing translations carry over to the surviving
segments.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

job = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9")

merged = json.loads((job / "transcript" / "merged.json").read_text(encoding="utf-8"))
translations = json.loads((job / "translation" / "translated.json").read_text(encoding="utf-8"))
translations_by_id = {t["segment_id"]: t for t in translations}

print(f"Before: {len(merged)} merged segments, {len(translations)} translations")

# Sort by start_ms just in case
merged.sort(key=lambda s: (s["start_ms"], s["end_ms"]))

# Group consecutive segments with same text and overlapping/touching time ranges
def normalize(text: str) -> str:
    return (text or "").strip()

groups: list[list[dict]] = []
current: list[dict] = []
current_text = None
for seg in merged:
    t = normalize(seg.get("text", ""))
    if current and t == current_text and seg["start_ms"] <= current[-1]["end_ms"] + 100:
        current.append(seg)
    else:
        if current:
            groups.append(current)
        current = [seg]
        current_text = t
if current:
    groups.append(current)

# Build new merged list
new_merged = []
old_id_to_new_id: dict[str, str] = {}
for idx, group in enumerate(groups, start=1):
    new_id = f"m-{idx:04}"
    start_ms = min(s["start_ms"] for s in group)
    end_ms = max(s["end_ms"] for s in group)
    text = normalize(group[0].get("text", ""))
    if not text:
        # Skip empty groups
        continue
    new_merged.append({
        "id": new_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "source": group[0].get("source", "merged"),
        "meta": {"merged_from": [s["id"] for s in group]},
    })
    for s in group:
        old_id_to_new_id[s["id"]] = new_id

# Build new translations: pick first available translation in each group
new_translations = []
for new_seg in new_merged:
    group_ids = new_seg["meta"]["merged_from"]
    chosen = None
    for old_id in group_ids:
        if old_id in translations_by_id:
            chosen = translations_by_id[old_id]
            break
    if chosen is None:
        continue
    new_row = dict(chosen)
    new_row["segment_id"] = new_seg["id"]
    new_row["start_ms"] = new_seg["start_ms"]
    new_row["end_ms"] = new_seg["end_ms"]
    new_translations.append(new_row)

# Write back
(job / "transcript" / "merged.json").write_text(
    json.dumps(new_merged, ensure_ascii=False, indent=2), encoding="utf-8"
)
(job / "translation" / "translated.json").write_text(
    json.dumps(new_translations, ensure_ascii=False, indent=2), encoding="utf-8"
)

# Regenerate review.csv using the canonical exporter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vietdub.models import TimedSegment, TranslationRow
from vietdub.translate import export_review_csv

segs = [TimedSegment.model_validate(s) for s in new_merged]
trans = [TranslationRow.model_validate(t) for t in new_translations]

# Reapply skip on fallback
for r in trans:
    if r.text_vi.startswith("[CHƯA DỊCH]"):
        r.status = "skip"

export_review_csv(job / "translation" / "review.csv", segs, trans)

# Update status.json: rewind translate/render, but keep extract/stt/ocr/context/merge as done
status_path = job / "status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
now = datetime.now(timezone.utc).isoformat()
status["merge"] = {
    "state": "done",
    "finished_at": now,
    "details": {"segments": len(new_merged), "path": str(job / "transcript" / "merged.json"), "dedupe": True},
}
status["translate"] = {
    "state": "done",
    "finished_at": now,
    "details": {"rows": len(new_translations), "review_csv": str(job / "translation" / "review.csv"), "carried_from_previous": True},
}
for k in ("tts", "render"):
    status.pop(k, None)
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

# Clean stale TTS/render artifacts
for path in [
    job / "tts" / "final_vi.wav",
    job / "tts" / "sync_report.json",
    job / "output" / "subtitles_vi.srt",
    job / "output" / "preview_vi.mp4",
]:
    if path.exists():
        path.unlink()

# Clean all per-segment mp3 files (they'll be regenerated)
seg_dir = job / "tts" / "segments"
if seg_dir.exists():
    for p in seg_dir.glob("*.mp3"):
        p.unlink()

print(f"After: {len(new_merged)} merged segments, {len(new_translations)} translations")
print(f"Reduction: {len(merged)} -> {len(new_merged)} ({100 * (1 - len(new_merged) / len(merged)):.0f}% fewer)")
print(f"\nReady for: vietdub resume 'jobs\\Tập 1-9' --from tts")