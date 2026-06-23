from vietdub.context import build_context_bundle
from vietdub.models import DEFAULT_TONE, TimedSegment


def test_context_bundle_contains_episode_and_style():
    segments = [
        TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u5230\u5e95\u662f\u8c01"),
        TimedSegment(
            id="m-0002",
            start_ms=1000,
            end_ms=2000,
            text="\u90fd\u6000\u7591\u4f60\u6709\u4e00\u4e2a\u795e\u79d8\u7684\u8fc7\u5f80",
        ),
    ]

    bundle = build_context_bundle(segments, series_context={})

    assert "episode_context" in bundle
    assert "style_guide" in bundle
    assert bundle["style_guide"]["tone"] == DEFAULT_TONE
    assert bundle["scene_context"][0]["segment_ids"] == ["m-0001", "m-0002"]
