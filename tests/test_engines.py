import json

from vietdub.ocr import FixtureOcrEngine
from vietdub.stt import FixtureSttEngine


def test_fixture_stt_engine_loads_segments(tmp_path):
    fixture = tmp_path / "stt.json"
    fixture.write_text(
        json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "stt"}]),
        encoding="utf-8",
    )
    result = FixtureSttEngine(fixture).transcribe(tmp_path / "audio.wav")
    assert result[0].text == "\u4f60\u597d"


def test_fixture_ocr_engine_loads_segments(tmp_path):
    fixture = tmp_path / "ocr.json"
    fixture.write_text(
        json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "\u90fd\u6000\u7591\u4f60", "source": "ocr"}]),
        encoding="utf-8",
    )
    result = FixtureOcrEngine(fixture).recognize(tmp_path / "video.mp4")
    assert result[0].source == "ocr"
