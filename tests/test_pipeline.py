import json

from vietdub.models import TimedSegment
from vietdub.jobs import Job
from vietdub.pipeline import resume_tts_and_render, run_fixture_pipeline
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


def test_resume_tts_and_render_writes_output_srt(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})

    async def fake_synthesize_segment(self, row, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp3")
        return output

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fake_synthesize_segment)

    srt_path = resume_tts_and_render(job, type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural"})())

    assert srt_path == tmp_path / "job" / "output" / "subtitles_vi.srt"
    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    assert (tmp_path / "job" / "tts" / "segments" / "m-0001.mp3").exists()


def test_resume_tts_and_render_keeps_srt_when_tts_segment_fails(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})

    async def fail_synthesize_segment(self, row, output):
        raise RuntimeError("tts failed")

    monkeypatch.setattr("vietdub.tts.EdgeTtsEngine.synthesize_segment", fail_synthesize_segment)

    srt_path = resume_tts_and_render(job, type("Settings", (), {"edge_voice": "vi-VN-HoaiMyNeural"})())

    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    report = (tmp_path / "job" / "tts" / "tts_warnings.json").read_text(encoding="utf-8")
    assert "m-0001" in report
