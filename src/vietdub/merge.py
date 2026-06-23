from __future__ import annotations

from .models import TimedSegment


def overlap_ms(left: TimedSegment, right: TimedSegment) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def overlaps_enough(left: TimedSegment, right: TimedSegment) -> bool:
    overlap = overlap_ms(left, right)
    shortest = min(left.duration_ms, right.duration_ms)
    return shortest > 0 and overlap / shortest >= 0.45


def merge_segments(stt_segments: list[TimedSegment], ocr_segments: list[TimedSegment]) -> list[TimedSegment]:
    merged: list[TimedSegment] = []
    used_stt_ids: set[str] = set()

    for index, ocr_segment in enumerate(sorted(ocr_segments, key=lambda item: item.start_ms), start=1):
        matching_stt = [segment for segment in stt_segments if overlaps_enough(ocr_segment, segment)]
        for segment in matching_stt:
            used_stt_ids.add(segment.id)
        start_ms = min([ocr_segment.start_ms, *[segment.start_ms for segment in matching_stt]])
        end_ms = max([ocr_segment.end_ms, *[segment.end_ms for segment in matching_stt]])
        merged.append(
            TimedSegment(
                id=f"m-{index:04}",
                start_ms=start_ms,
                end_ms=end_ms,
                text=ocr_segment.text,
                confidence=ocr_segment.confidence,
                source="ocr+stt" if matching_stt else "ocr",
                meta={"ocr_id": ocr_segment.id, "stt_ids": [segment.id for segment in matching_stt]},
            )
        )

    next_index = len(merged) + 1
    for stt_segment in sorted(stt_segments, key=lambda item: item.start_ms):
        if stt_segment.id in used_stt_ids:
            continue
        merged.append(
            TimedSegment(
                id=f"m-{next_index:04}",
                start_ms=stt_segment.start_ms,
                end_ms=stt_segment.end_ms,
                text=stt_segment.text,
                confidence=stt_segment.confidence,
                source="stt",
                meta={"stt_id": stt_segment.id},
            )
        )
        next_index += 1

    merged.sort(key=lambda item: item.start_ms)
    for index, segment in enumerate(merged, start=1):
        segment.id = f"m-{index:04}"
    return merged
