"""Verify the deduped final preview:
- Sync report: every segment placed at start_ms without major overlap
- SRT: timestamps non-overlapping within tolerance
- Audio: wav header + non-empty
- Video: probe with ffprobe for duration + streams
"""
import json
import re
import subprocess
import wave
from pathlib import Path

job = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9")

print("=" * 60)
print("VERIFICATION REPORT — Job Tập 1-9 (post-dedupe)")
print("=" * 60)

# 1. Sync report
sync = json.loads((job / "tts" / "sync_report.json").read_text(encoding="utf-8"))
segments = sync.get("segments", [])
print(f"\n[SYNCeport] {len(segments)} segments")
skipped = [s for s in segments if s.get("skipped") or s.get("synced_duration_ms", 0) == 0]
print(f"  Skipped: {len(skipped)}")
overlaps = []
for i in range(len(segments) - 1):
    cur_end = segments[i]["start_ms"] + segments[i].get("synced_duration_ms", 0)
    nxt_start = segments[i + 1]["start_ms"]
    if cur_end > nxt_start:
        overlaps.append((segments[i]["segment_id"], segments[i + 1]["segment_id"], cur_end - nxt_start))
print(f"  Audio overlaps (consecutive): {len(overlaps)}")
if overlaps:
    print("  Sample overlaps:")
    for o in overlaps[:8]:
        print(f"    {o[0]} -> {o[1]}: {o[2]}ms")

# 2. SRT
srt_text = (job / "output" / "subtitles_vi.srt").read_text(encoding="utf-8")
srt_blocks = re.split(r"\n\n+", srt_text.strip())
print(f"\n[SRT] {len(srt_blocks)} subtitle blocks")
prev_end = 0
srt_overlaps = []
for block in srt_blocks:
    lines = block.split("\n")
    if len(lines) < 2:
        continue
    m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
    if not m:
        continue
    h, mi, s, ms = map(int, m.groups()[:4])
    start = ((h * 60 + mi) * 60 + s) * 1000 + ms
    h, mi, s, ms = map(int, m.groups()[4:])
    end = ((h * 60 + mi) * 60 + s) * 1000 + ms
    if start < prev_end:
        srt_overlaps.append((start, end, prev_end))
    prev_end = end
print(f"  SRT timestamp overlaps: {len(srt_overlaps)}")

# 3. Audio
wav = job / "tts" / "final_vi.wav"
with wave.open(str(wav), "rb") as w:
    dur = w.getnframes() / w.getframerate()
    print(f"\n[AUDIO] {wav.name}: {dur:.2f}s, {w.getframerate()}Hz, {w.getnchannels()}ch, {w.getsampwidth()*8}-bit")

# 4. Video probe
mp4 = job / "output" / "preview_vi.mp4"
result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name", "-of", "json", str(mp4)],
    capture_output=True, text=True
)
probe = json.loads(result.stdout)
print(f"\n[VIDEO] {mp4.name}: duration={float(probe['format']['duration']):.2f}s")
for s in probe.get("streams", []):
    print(f"  Stream: {s['codec_type']} ({s['codec_name']})")

# 5. File sizes
print("\n[FILES]")
for p in [
    job / "output" / "preview_vi.mp4",
    job / "output" / "subtitles_vi.srt",
    job / "tts" / "final_vi.wav",
    job / "tts" / "sync_report.json",
]:
    print(f"  {p.name}: {p.stat().st_size / 1024:.1f} KB")