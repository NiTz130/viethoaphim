from vietdub.config import Settings
from vietdub.models import TimedSegment, millis_to_srt_time, srt_time_to_millis


def test_timed_segment_computes_duration():
    segment = TimedSegment(id="0001", start_ms=1000, end_ms=2600, text="\u4f60\u597d")
    assert segment.duration_ms == 1600


def test_timed_segment_rejects_negative_duration():
    try:
        TimedSegment(id="bad", start_ms=2000, end_ms=1000, text="x")
    except ValueError as exc:
        assert "end_ms must be greater than start_ms" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_srt_time_round_trip():
    assert millis_to_srt_time(3_723_456) == "01:02:03,456"
    assert srt_time_to_millis("01:02:03,456") == 3_723_456


def test_settings_defaults_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openai_base_url == ""
    assert settings.llm_model == ""
    assert settings.edge_voice == "vi-VN-HoaiMyNeural"
