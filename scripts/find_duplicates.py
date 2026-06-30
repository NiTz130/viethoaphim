"""Find overlapping or duplicate segments in merged transcript."""
import json
from collections import Counter
from pathlib import Path

job = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9")
merged = json.loads((job / "transcript" / "merged.json").read_text(encoding="utf-8"))
print(f"Total merged segments: {len(merged)}")

# Overlapping (start < previous end)
prev_end = -1
overlaps = []
for s in merged:
    if s["start_ms"] < prev_end and s.get("text", "").strip():
        overlaps.append((s["id"], prev_end, s["start_ms"], s["end_ms"], s.get("text", "")[:50]))
    if s["end_ms"] > prev_end:
        prev_end = s["end_ms"]

print(f"\nOverlapping segments: {len(overlaps)}")
for o in overlaps[:15]:
    print(f"  {o[0]}: prev_end={o[1]} this=[{o[2]}..{o[3]}] text={o[4]!r}")

# Duplicate exact text
texts = [s.get("text", "").strip() for s in merged if s.get("text", "").strip()]
dup = [(t, c) for t, c in Counter(texts).items() if c > 1]
dup.sort(key=lambda x: -x[1])
print(f"\nDuplicate texts: {len(dup)}")
for t, c in dup[:15]:
    print(f"  x{c}: {t[:60]!r}")

# Near-duplicates (consecutive identical STT/OCR)
print("\nNear-duplicate (start < prev_end within 500ms):")
near = []
prev = None
for s in merged:
    if prev and s["start_ms"] < prev["end_ms"] + 500 and abs(s["start_ms"] - prev["start_ms"]) < 1500 and s.get("text", "").strip() == prev.get("text", "").strip():
        near.append((prev["id"], s["id"], prev.get("text", "")[:40]))
    prev = s
print(f"  Found: {len(near)}")
for n in near[:10]:
    print(f"  {n[0]} == {n[1]}: {n[2]!r}")