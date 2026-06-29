from __future__ import annotations

import logging

from ..models import TranslationRow

VIETNAMESE_SPEECH_CHARS_PER_SEC = 14
LENGTH_BUFFER_FACTOR = 1.1
LENGTH_TOLERANCE = 0.20  # ±20% of target_vi_chars
MIN_TARGET_VI_CHARS = 10

# Use the parent package's logger so test assertions on
# `r.name == "vietdub.translate"` continue to match.
logger = logging.getLogger("vietdub.translate")


def _length_target_vi_chars(start_ms: int, end_ms: int) -> int:
    """Compute target Vietnamese character count for a segment.

    Vietnamese conversational speech is ~14 chars/sec; add 10% buffer for
    punctuation and natural variation. Floor at MIN_TARGET_VI_CHARS for
    very short segments.
    """
    duration_s = (end_ms - start_ms) / 1000.0
    return max(
        MIN_TARGET_VI_CHARS,
        int(duration_s * VIETNAMESE_SPEECH_CHARS_PER_SEC * LENGTH_BUFFER_FACTOR),
    )


def _warn_oversized_translations(rows: list[TranslationRow]) -> None:
    """Warn if Vietnamese translation length is outside ±20% of target.

    Skips empty translations and segments in the floor region (target <= 10).
    Warnings go to the `vietdub.translate` logger at WARNING level.
    """
    for row in rows:
        text_len = len(row.text_vi)
        if text_len == 0:
            continue
        target = _length_target_vi_chars(row.start_ms, row.end_ms)
        if target <= MIN_TARGET_VI_CHARS:
            continue
        if text_len > target * (1 + LENGTH_TOLERANCE):
            over_pct = int((text_len / target - 1) * 100)
            logger.warning(
                f"[LEN WARNING] segment {row.segment_id}: {text_len} chars vs target {target} "
                f"({over_pct}% over). Vietnamese translation may exceed dubbing timing."
            )
        elif text_len < target * (1 - LENGTH_TOLERANCE):
            under_pct = int((1 - text_len / target) * 100)
            logger.warning(
                f"[LEN WARNING] segment {row.segment_id}: {text_len} chars vs target {target} "
                f"({under_pct}% under). Vietnamese translation may read too fast for dubbing."
            )