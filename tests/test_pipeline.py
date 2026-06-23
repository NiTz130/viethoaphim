import json

from vietdub.models import TimedSegment
from vietdub.pipeline import run_fixture_pipeline
from vietdub.srt import render_srt


def test_fixture_pipeline_stops_after_review_csv(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    stt = tmp_path / "stt.json"
    ocr = tmp_path / "ocr.json"
    stt.write_text(
        json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "stt"}]),
        encoding="utf-8",
    )
    ocr.write_text(
        json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )

    job = run_fixture_pipeline(video=video, jobs_dir=tmp_path / "jobs", stt_fixture=stt, ocr_fixture=ocr)

    assert (job.root / "translation" / "review.csv").exists()
    assert (job.root / "context" / "style_guide.json").exists()


def test_review_mode_does_not_create_preview_in_fixture_pipeline(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    stt = tmp_path / "stt.json"
    ocr = tmp_path / "ocr.json"
    stt.write_text(
        json.dumps([{"id": "s1", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "stt"}]),
        encoding="utf-8",
    )
    ocr.write_text(
        json.dumps([{"id": "o1", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )

    job = run_fixture_pipeline(video=video, jobs_dir=tmp_path / "jobs", stt_fixture=stt, ocr_fixture=ocr)

    assert not (job.root / "output" / "preview_vi.mp4").exists()


def test_render_srt_from_vietnamese_rows():
    segments = [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="Xin ch\u00e0o")]
    assert "Xin ch\u00e0o" in render_srt(segments)
