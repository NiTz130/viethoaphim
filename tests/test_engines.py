import json
import sys
import types

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


def test_paddle_ocr_engine_clears_stale_frames_before_extract(monkeypatch, tmp_path):
    from vietdub.ocr import PaddleSubtitleOcrEngine

    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake-video")
    frame_dir = tmp_path / "ocr_frames"
    frame_dir.mkdir()
    stale_frame = frame_dir / "frame_000001.jpg"
    stale_frame.write_bytes(b"old")
    keep_file = frame_dir / "notes.txt"
    keep_file.write_text("keep", encoding="utf-8")
    observed_before_extract = {}

    class _FakeOcrResult:
        # Mimic PaddleOCR 3.x predict() result shape: page.json["res"]["rec_texts"].
        def __init__(self, rec_texts):
            self.json = {"res": {"rec_texts": rec_texts}}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, frame):
            return [_FakeOcrResult(["\u4f60\u597d"])]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    def fake_run(command, **kwargs):
        observed_before_extract["stale_exists"] = stale_frame.exists()
        stale_frame.write_bytes(b"new")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("vietdub.ocr.paddle.subprocess.run", fake_run)
    monkeypatch.setattr("vietdub.ocr.paddle._guard_optional_torch_import", lambda: None)

    result = PaddleSubtitleOcrEngine(sample_every_seconds=0.5).recognize(video)

    assert observed_before_extract == {"stale_exists": False}
    assert keep_file.exists()
    assert result[0].text == "\u4f60\u597d"
