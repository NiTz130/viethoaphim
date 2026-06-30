import json
from pathlib import Path

job = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9")
merged = json.loads((job / "transcript" / "merged.json").read_text(encoding="utf-8"))

print(f"Total: {len(merged)}")
print(f"\nFirst 20:")
for s in merged[:20]:
    text = s.get("text", "")
    if len(text) > 40:
        text = text[:40] + "..."
    print(f"  {s['id']}: [{s['start_ms']}..{s['end_ms']}] {text}")

# Check overlap rate
prev_end = -1
overlap_count = 0
for s in merged:
    if s["start_ms"] < prev_end:
        overlap_count += 1
    if s["end_ms"] > prev_end:
        prev_end = s["end_ms"]
print(f"\nOverlapping: {overlap_count}/{len(merged)} ({100*overlap_count/len(merged):.0f}%)")

# Check for segments with similar text nearby (case for STT noise)
from collections import Counter
c = Counter(s.get("text", "").strip() for s in merged if s.get("text", "").strip())
print(f"\nTop duplicates remaining:")
for t, n in c.most_common(8):
    print(f"  x{n}: {t[:50]}")