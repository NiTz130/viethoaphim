from .models import TimedSegment, millis_to_srt_time


def render_srt(segments: list[TimedSegment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{millis_to_srt_time(segment.start_ms)} --> {millis_to_srt_time(segment.end_ms)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")
