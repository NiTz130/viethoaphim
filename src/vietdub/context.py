from __future__ import annotations

from .models import DEFAULT_TONE, TimedSegment


EPISODE_SUMMARY = (
    "T\u1eadp phim ho\u1ea1t h\u00ecnh Trung Qu\u1ed1c ng\u1eafn, tho\u1ea1i nhanh, "
    "c\u00f3 y\u1ebfu t\u1ed1 h\u00e0i sa \u0111i\u00eau."
)
SCENE_SUMMARY = (
    "C\u1ea3nh m\u1edf \u0111\u1ea7u ho\u1eb7c \u0111o\u1ea1n tho\u1ea1i li\u00ean "
    "t\u1ee5c c\u1ea7n d\u1ecbch theo c\u00f9ng ng\u1eef c\u1ea3nh."
)


def build_context_bundle(segments: list[TimedSegment], series_context: dict) -> dict:
    joined = " ".join(segment.text for segment in segments[:20])
    scene_ids = [segment.id for segment in segments[:20]]
    return {
        "series_context": series_context,
        "episode_context": {
            "summary": EPISODE_SUMMARY,
            "source_excerpt": joined,
        },
        "characters": [],
        "glossary": {},
        "style_guide": {
            "tone": DEFAULT_TONE,
            "translation_rules": [
                "\u01afu ti\u00ean c\u00e2u tho\u1ea1i t\u1ef1 nhi\u00ean h\u01a1n d\u1ecbch s\u00e1t ch\u1eef.",
                "Gi\u1eef punchline ng\u1eafn \u0111\u1ec3 h\u1ee3p timing TTS.",
                "D\u1ecbch nh\u1ea5t qu\u00e1n t\u00ean ri\u00eang v\u00e0 c\u00e1ch x\u01b0ng h\u00f4 trong to\u00e0n b\u1ed9 job.",
            ],
        },
        "scene_context": [
            {
                "segment_ids": scene_ids,
                "summary": SCENE_SUMMARY,
                "characters": [],
                "tone": DEFAULT_TONE,
            }
        ],
    }
