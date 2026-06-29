import json

import pytest
from pydub.generators import Sine

from vietdub.models import TimedSegment
from vietdub.jobs import Job
from vietdub.pipeline import _probe_video_duration_ms, resume_tts_and_render, run_fixture_pipeline
from vietdub.srt import render_srt


def _resume_settings():
    return type(
        "Settings",
        (),
        {
            "anthropic_api_key": "test-key",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "llm_model": "MiniMax-M3",
            "tts_voice_id": "vi-VN-HoaiMyNeural",
            "tts_api_key": "test-tts-key",
            "tts_base_url": "https://api.minimax.io/v1",
            "tts_model": "speech-2.8-hd",
            "sample_rate": 44_100,
            "ocr_use_gpu": False,
            "ocr_enable_mkldnn": False,
        },
    )()


def _ensure_merged_json(job, segment_ids=("m-0001",)) -> None:
    """Create a minimal transcript/merged.json so resume validation passes.

    M8 makes merged.json a hard requirement; tests that call resume_tts_and_render
    without a real pipeline run need this helper to fabricate the file.
    """
    merged = job.root / "transcript" / "merged.json"
    merged.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {
            "id": sid,
            "start_ms": 0,
            "end_ms": 1000,
            "text": "hi",
            "source": "ocr+stt",
        }
        for sid in segment_ids
    ]
    merged.write_text(json.dumps(segments), encoding="utf-8")


def _write_valid_mp3(path, duration_ms: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sine(440).to_audio_segment(duration=duration_ms).export(path, format="mp3")


def _patch_resume_media(monkeypatch):
    calls = []

    def fake_probe_media(video_path):
        return {"format": {"duration": "1.000"}, "streams": []}

    def fake_mux_preview(video, audio, subtitles, output):
        calls.append({"video": video, "audio": audio, "subtitles": subtitles, "output": output})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"preview")

    monkeypatch.setattr("vietdub.media.probe_media", fake_probe_media)
    monkeypatch.setattr("vietdub.media.mux_preview", fake_mux_preview)
    return calls


def _write_review_csv(job_root, rows: list[str]) -> None:
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n" + "".join(rows),
        encoding="utf-8-sig",
    )


def _assert_no_resume_render_artifacts(job: Job) -> None:
    assert not (job.root / "output" / "subtitles_vi.srt").exists()
    assert not (job.root / "tts" / "final_vi.wav").exists()
    assert not (job.root / "output" / "preview_vi.mp4").exists()
    assert "render" not in job.load_status()


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
    mux_calls = _patch_resume_media(monkeypatch)

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fake_synthesize_segment)

    _ensure_merged_json(job)
    srt_path = resume_tts_and_render(job, _resume_settings())

    final_audio = tmp_path / "job" / "tts" / "final_vi.wav"
    sync_report_path = tmp_path / "job" / "tts" / "sync_report.json"
    preview_path = tmp_path / "job" / "output" / "preview_vi.mp4"
    assert srt_path == tmp_path / "job" / "output" / "subtitles_vi.srt"
    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    assert (tmp_path / "job" / "tts" / "segments" / "m-0001.mp3").exists()
    assert final_audio.stat().st_size > 0
    assert preview_path.read_bytes() == b"preview"
    assert mux_calls == [{"video": job.input_video, "audio": final_audio, "subtitles": srt_path, "output": preview_path}]
    sync_report = json.loads(sync_report_path.read_text(encoding="utf-8"))
    assert sync_report["segments"][0]["segment_id"] == "m-0001"
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert status["render"]["details"]["final_audio"] == str(final_audio)
    assert status["render"]["details"]["preview"] == str(preview_path)


def test_resume_tts_and_render_rejects_duplicate_active_segment_id(tmp_path):
    job_root = tmp_path / "job"
    _write_review_csv(
        job_root,
        [
            "m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
            "m-0001,1000,2000,,\u4f60\u597d,Chao ban,,reviewed\n",
        ],
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})

    with pytest.raises(RuntimeError, match="Duplicate segment_id.*m-0001"):
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())

    _assert_no_resume_render_artifacts(job)
    assert not (job_root / "tts" / "segments" / "m-0001.mp3").exists()


def test_resume_tts_and_render_allows_duplicate_inactive_segment_id(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    _write_review_csv(
        job_root,
        [
            "m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
            "m-0001,1000,2000,,\u4f60\u597d,Bo qua,,skip\n",
            "m-0001,2000,3000,,\u4f60\u597d,,,reviewed\n",
        ],
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    _patch_resume_media(monkeypatch)

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fake_synthesize_segment)

    _ensure_merged_json(job)
    resume_tts_and_render(job, _resume_settings())

    status = job.load_status()
    assert status["render"]["details"]["segments"] == 1


def test_resume_tts_and_render_rejects_negative_active_timing(tmp_path):
    job_root = tmp_path / "job"
    _write_review_csv(
        job_root,
        ["m-0001,-500,500,,\u4f60\u597d,Xin chao,,reviewed\n"],
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})

    with pytest.raises(RuntimeError, match="Invalid timing.*m-0001"):
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())

    _assert_no_resume_render_artifacts(job)


@pytest.mark.parametrize("mux_output", ["missing", "empty"])
def test_resume_tts_and_render_rejects_missing_or_empty_preview_after_mux(monkeypatch, tmp_path, mux_output):
    job_root = tmp_path / "job"
    _write_review_csv(
        job_root,
        ["m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n"],
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    mux_calls = []

    def fake_probe_media(video_path):
        return {"format": {"duration": "1.000"}, "streams": []}

    def fake_mux_preview(video, audio, subtitles, output):
        mux_calls.append({"video": video, "audio": audio, "subtitles": subtitles, "output": output})
        if mux_output == "empty":
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"")

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.media.probe_media", fake_probe_media)
    monkeypatch.setattr("vietdub.media.mux_preview", fake_mux_preview)
    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fake_synthesize_segment)

    with pytest.raises(RuntimeError, match="Preview was not created"):
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())

    final_audio = job_root / "tts" / "final_vi.wav"
    sync_report = job_root / "tts" / "sync_report.json"
    preview = job_root / "output" / "preview_vi.mp4"
    assert final_audio.stat().st_size > 0
    assert sync_report.exists()
    assert not preview.exists()
    assert "render" not in job.load_status()
    assert mux_calls == [
        {
            "video": job.input_video,
            "audio": final_audio,
            "subtitles": job_root / "output" / "subtitles_vi.srt",
            "output": preview,
        }
    ]


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
    mux_calls = _patch_resume_media(monkeypatch)

    async def fail_synthesize_segment(self, row, output):
        raise RuntimeError("tts failed")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    try:
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())
    except RuntimeError as exc:
        assert "No valid TTS segment audio" in str(exc)
    else:
        raise AssertionError("expected final audio assembly failure")

    srt_path = tmp_path / "job" / "output" / "subtitles_vi.srt"
    assert "Xin ch\u00e0o" in srt_path.read_text(encoding="utf-8")
    report = (tmp_path / "job" / "tts" / "tts_warnings.json").read_text(encoding="utf-8")
    assert "m-0001" in report
    sync_report = json.loads((tmp_path / "job" / "tts" / "sync_report.json").read_text(encoding="utf-8"))
    assert "missing TTS audio" in sync_report["segments"][0]["warnings"][0]
    status = json.loads((tmp_path / "job" / "status.json").read_text(encoding="utf-8"))
    assert "render" not in status
    assert mux_calls == []


def test_resume_tts_and_render_removes_stale_segment_audio_when_retry_fails(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    segment_audio = job_root / "tts" / "segments" / "m-0001.mp3"
    _write_valid_mp3(segment_audio, duration_ms=400)
    final_audio = job_root / "tts" / "final_vi.wav"
    preview = job_root / "output" / "preview_vi.mp4"
    job = Job(root=job_root, config={})
    mux_calls = _patch_resume_media(monkeypatch)

    async def fail_synthesize_segment(self, row, output):
        output.write_bytes(b"partial retry audio")
        raise RuntimeError("tts failed")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    with pytest.raises(RuntimeError, match="No valid TTS segment audio"):
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())

    assert not segment_audio.exists()
    assert not final_audio.exists()
    assert not preview.exists()
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert "render" not in status
    assert mux_calls == []


def test_resume_tts_and_render_removes_stale_outputs_when_assembly_fails(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin ch\u00e0o,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    final_audio = job_root / "tts" / "final_vi.wav"
    final_audio.parent.mkdir(parents=True)
    final_audio.write_bytes(b"old-final-audio")
    preview = job_root / "output" / "preview_vi.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"old-preview")
    job = Job(root=job_root, config={})
    job.write_json(
        "status.json",
        {
            "render": {
                "state": "done",
                "details": {"final_audio": str(final_audio), "preview": str(preview)},
            }
        },
    )
    mux_calls = _patch_resume_media(monkeypatch)

    async def fail_synthesize_segment(self, row, output):
        raise RuntimeError("tts failed")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    try:
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())
    except RuntimeError as exc:
        assert "No valid TTS segment audio" in str(exc)
    else:
        raise AssertionError("expected final audio assembly failure")

    assert "Xin ch\u00e0o" in (job_root / "output" / "subtitles_vi.srt").read_text(encoding="utf-8")
    assert not final_audio.exists()
    assert not preview.exists()
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert "render" not in status
    assert mux_calls == []


@pytest.mark.parametrize("exception_type", [RuntimeError, OSError, ValueError])
def test_probe_video_duration_ms_returns_none_for_probe_exception(monkeypatch, tmp_path, exception_type):
    job_root = tmp_path / "job"
    job_root.mkdir()
    job = Job(root=job_root, config={})

    def fail_probe_media(video_path):
        raise exception_type("probe failed")

    monkeypatch.setattr("vietdub.media.probe_media", fail_probe_media)

    assert _probe_video_duration_ms(job) is None


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ("not a dict", None),
        ({"format": "not a dict"}, None),
        ({"format": {}}, None),
        ({"format": {"duration": "not-a-number"}}, None),
        ({"format": {"duration": "0"}}, None),
        ({"format": {"duration": "-1.25"}}, None),
        ({"format": {"duration": "1.234"}}, 1234),
    ],
)
def test_probe_video_duration_ms_handles_invalid_and_valid_duration_metadata(monkeypatch, tmp_path, metadata, expected):
    job_root = tmp_path / "job"
    job_root.mkdir()
    job = Job(root=job_root, config={})

    def fake_probe_media(video_path):
        return metadata

    monkeypatch.setattr("vietdub.media.probe_media", fake_probe_media)

    assert _probe_video_duration_ms(job) == expected


def test_resume_tts_rejects_path_traversal_segment_id(monkeypatch, tmp_path):
    review = tmp_path / "job" / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "..\\evil,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "job" / "input.mp4").write_bytes(b"fake")
    job = Job(root=tmp_path / "job", config={})

    async def fake_synthesize_segment(self, row, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp3")
        return output

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fake_synthesize_segment)

    try:
        _ensure_merged_json(job)
        resume_tts_and_render(job, type("Settings", (), {
    "anthropic_api_key": "test-key",
    "anthropic_base_url": "https://api.minimax.io/anthropic",
    "llm_model": "MiniMax-M3",
    "tts_voice_id": "vi-VN-HoaiMyNeural",
    "tts_api_key": "test-tts-key",
    "tts_base_url": "https://api.minimax.io/v1",
    "tts_model": "speech-2.8-hd",
    "sample_rate": 44_100,
})())
    except RuntimeError as exc:
        assert "Invalid segment_id" in str(exc)
    else:
        raise AssertionError("expected invalid segment_id failure")

    assert not (tmp_path / "job" / "evil.mp3").exists()
    assert not (tmp_path / "evil.mp3").exists()


def test_resume_tts_preserves_render_state_when_segment_validation_fails(tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "..\\evil,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    final_audio = job_root / "tts" / "final_vi.wav"
    final_audio.parent.mkdir(parents=True)
    final_audio.write_bytes(b"old-final-audio")
    preview = job_root / "output" / "preview_vi.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"old-preview")
    srt_path = job_root / "output" / "subtitles_vi.srt"
    srt_path.write_text("old-subtitles", encoding="utf-8")
    sync_report_path = job_root / "tts" / "sync_report.json"
    sync_report_path.write_text(json.dumps({"old": True}), encoding="utf-8")
    job = Job(root=job_root, config={})
    job.write_json(
        "status.json",
        {
            "render": {
                "state": "done",
                "details": {"final_audio": str(final_audio), "preview": str(preview)},
            }
        },
    )

    with pytest.raises(RuntimeError, match="Invalid segment_id"):
        _ensure_merged_json(job)
        resume_tts_and_render(job, _resume_settings())

    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert "render" in status
    assert final_audio.read_bytes() == b"old-final-audio"
    assert preview.read_bytes() == b"old-preview"
    assert srt_path.read_text(encoding="utf-8") == "old-subtitles"
    assert json.loads(sync_report_path.read_text(encoding="utf-8")) == {"old": True}


def test_resume_tts_rejects_unknown_transcript_segment_id(tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-9999,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": "m-0001", "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})

    try:
        resume_tts_and_render(job, type("Settings", (), {
    "anthropic_api_key": "test-key",
    "anthropic_base_url": "https://api.minimax.io/anthropic",
    "llm_model": "MiniMax-M3",
    "tts_voice_id": "vi-VN-HoaiMyNeural",
    "tts_api_key": "test-tts-key",
    "tts_base_url": "https://api.minimax.io/v1",
    "tts_model": "speech-2.8-hd",
    "sample_rate": 44_100,
})())
    except RuntimeError as exc:
        assert "m-9999" in str(exc)
    else:
        raise AssertionError("expected unknown segment_id failure")


def test_resume_tts_rejects_transcript_path_traversal_segment_id(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    unsafe_id = "..\\evil"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        f"{unsafe_id},0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": unsafe_id, "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    called = False

    async def fail_synthesize_segment(self, row, output):
        nonlocal called
        called = True
        raise AssertionError("synthesis should not run for unsafe segment_id")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    try:
        resume_tts_and_render(job, type("Settings", (), {
    "anthropic_api_key": "test-key",
    "anthropic_base_url": "https://api.minimax.io/anthropic",
    "llm_model": "MiniMax-M3",
    "tts_voice_id": "vi-VN-HoaiMyNeural",
    "tts_api_key": "test-tts-key",
    "tts_base_url": "https://api.minimax.io/v1",
    "tts_model": "speech-2.8-hd",
    "sample_rate": 44_100,
})())
    except RuntimeError as exc:
        assert "Invalid segment_id" in str(exc)
    else:
        raise AssertionError("expected invalid segment_id failure")

    assert not called
    assert not (job_root / "tts" / "evil.mp3").exists()


def test_resume_tts_rejects_duplicate_unsafe_segment_id_with_invalid_message(monkeypatch, tmp_path):
    """C2 regression: unsafe ID appearing twice must report 'Invalid segment_id'
    for the first occurrence, never 'Duplicate segment_id' which would mask the
    real issue (path traversal in the segment_id itself)."""
    job_root = tmp_path / "job"
    unsafe_id = "..\\evil"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        f"{unsafe_id},0,1000,,你好,Xin chao,,reviewed\n"
        f"{unsafe_id},1000,2000,,你好,Xin chao tiep,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": unsafe_id, "start_ms": 0, "end_ms": 1000, "text": "你好", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    called = False

    async def fail_synthesize_segment(self, row, output):
        nonlocal called
        called = True
        raise AssertionError("synthesis should not run for unsafe segment_id")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    with pytest.raises(RuntimeError) as exc_info:
        resume_tts_and_render(job, _resume_settings())

    message = str(exc_info.value)
    assert "Invalid segment_id" in message
    assert repr(unsafe_id) in message
    assert "Duplicate" not in message
    assert not called
    assert not (job_root / "tts" / "evil.mp3").exists()


def test_resume_tts_rejects_transcript_windows_absolute_segment_id(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    unsafe_id = "C:\\temp\\evil"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        f"{unsafe_id},0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    (job_root / "transcript").mkdir()
    (job_root / "transcript" / "merged.json").write_text(
        json.dumps([{"id": unsafe_id, "start_ms": 0, "end_ms": 1000, "text": "\u4f60\u597d", "source": "ocr"}]),
        encoding="utf-8",
    )
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    called = False

    async def fail_synthesize_segment(self, row, output):
        nonlocal called
        called = True
        raise AssertionError("synthesis should not run for unsafe segment_id")

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fail_synthesize_segment)

    try:
        resume_tts_and_render(job, type("Settings", (), {
    "anthropic_api_key": "test-key",
    "anthropic_base_url": "https://api.minimax.io/anthropic",
    "llm_model": "MiniMax-M3",
    "tts_voice_id": "vi-VN-HoaiMyNeural",
    "tts_api_key": "test-tts-key",
    "tts_base_url": "https://api.minimax.io/v1",
    "tts_model": "speech-2.8-hd",
    "sample_rate": 44_100,
})())
    except RuntimeError as exc:
        assert "Invalid segment_id" in str(exc)
    else:
        raise AssertionError("expected invalid segment_id failure")

    assert not called


def test_resume_tts_clears_stale_warning_count_after_success(monkeypatch, tmp_path):
    job_root = tmp_path / "job"
    review = job_root / "translation" / "review.csv"
    review.parent.mkdir(parents=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,\u4f60\u597d,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )
    warning_path = job_root / "tts" / "tts_warnings.json"
    warning_path.parent.mkdir(parents=True)
    warning_path.write_text(json.dumps([{"segment_id": "old", "error": "old failure"}]), encoding="utf-8")
    (job_root / "input.mp4").write_bytes(b"fake")
    job = Job(root=job_root, config={})
    _patch_resume_media(monkeypatch)

    async def fake_synthesize_segment(self, row, output):
        _write_valid_mp3(output, duration_ms=400)
        return output

    monkeypatch.setattr("vietdub.tts.MiniMaxTtsEngine.synthesize_segment", fake_synthesize_segment)

    _ensure_merged_json(job)
    resume_tts_and_render(job, _resume_settings())

    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["tts"]["details"]["warnings"] == 0
    assert status["render"]["details"]["preview"] == str(job_root / "output" / "preview_vi.mp4")
    assert not warning_path.exists()


def test_review_pipeline_writes_selected_system_memory(monkeypatch, tmp_path):
    from vietdub.memory import MemoryWarningItem, SystemMemory, TranslationExample
    from vietdub.pipeline import run_review_pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    captured_context = {}

    def fake_extract_audio(video_path, audio_path, sample_rate):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")

    class FakeSttEngine:
        def __init__(self, model, language):
            self.model = model
            self.language = language

        def transcribe(self, audio_path):
            return [
                TimedSegment(
                    id="s-0001",
                    start_ms=0,
                    end_ms=1000,
                    text="\u5c0f\u660e\u8bf4\u4f60\u597d",
                    source="stt",
                )
            ]

    class FakeOcrEngine:
        def __init__(self, sample_every_seconds=0.5, use_gpu=False, enable_mkldnn=False):
            pass
        def recognize(self, video_path):
            return [
                TimedSegment(
                    id="o-0001",
                    start_ms=0,
                    end_ms=1000,
                    text="\u5c0f\u660e\u8bf4\u4f60\u597d",
                    source="ocr",
                )
            ]

    def fake_translate(segments, context_bundle, *, settings, review=True):
        captured_context.update(context_bundle)
        return [
            type(
                "Row",
                (),
                {
                    "segment_id": "m-0001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker": None,
                    "text_cn": "\u5c0f\u660e\u8bf4\u4f60\u597d",
                    "text_vi": "Tieu Minh noi xin chao",
                    "context_note": "",
                    "status": "draft",
                    "model_dump": lambda self: {
                        "segment_id": self.segment_id,
                        "start_ms": self.start_ms,
                        "end_ms": self.end_ms,
                        "speaker": self.speaker,
                        "text_cn": self.text_cn,
                        "text_vi": self.text_vi,
                        "context_note": self.context_note,
                        "status": self.status,
                    },
                },
            )()
        ]

    def fake_collect_system_memory(jobs_dir, current_job=None):
        return SystemMemory(
            translation_examples=[
                TranslationExample(
                    text_cn="\u4f60\u597d",
                    text_vi="Xin chao",
                    source_job="old",
                    source="translation/review.csv",
                    confidence=0.95,
                )
            ],
            warnings=[MemoryWarningItem(path="old/context/characters.json", message="Invalid JSON: test")],
        )

    monkeypatch.setattr("vietdub.media.extract_audio", fake_extract_audio)
    monkeypatch.setattr("vietdub.stt.FasterWhisperSttEngine", FakeSttEngine)
    monkeypatch.setattr("vietdub.ocr.PaddleSubtitleOcrEngine", FakeOcrEngine)
    monkeypatch.setattr("vietdub.reference.build_reference_context", lambda segments, data_dir: {})
    monkeypatch.setattr("vietdub.translate.translate_with_llm", fake_translate)
    monkeypatch.setattr("vietdub.memory.collect_system_memory", fake_collect_system_memory)

    settings = type(
        "Settings",
        (),
        {
            "sample_rate": 44100,
            "stt_model": "tiny",
            "stt_language": "zh",
            "reference_data_dir": str(tmp_path / "data"),
            "openai_api_key": "key",
            "llm_model": "model",
            "openai_base_url": "",
            "ocr_use_gpu": False,
            "ocr_enable_mkldnn": False,
        },
    )()

    job = run_review_pipeline(video=video, jobs_dir=tmp_path / "jobs", series=None, settings=settings)

    assert captured_context["translation_examples"][0]["text_vi"] == "Xin chao"
    assert (job.root / "context" / "system_memory.json").exists()
    assert (job.root / "context" / "translation_examples.json").exists()
    warnings = json.loads((job.root / "context" / "system_memory_warnings.json").read_text(encoding="utf-8"))
    assert warnings == [{"path": "old/context/characters.json", "message": "Invalid JSON: test"}]


def test_review_pipeline_persists_artifacts_before_translate_failure(monkeypatch, tmp_path):
    from vietdub.pipeline import run_review_pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")

    def fake_extract_audio(video_path, audio_path, sample_rate):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")

    class FakeSttEngine:
        def __init__(self, model, language):
            self.model = model
            self.language = language

        def transcribe(self, audio_path):
            return [TimedSegment(id="s-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d", source="stt")]

    class FakeOcrEngine:
        def __init__(self, sample_every_seconds=0.5, use_gpu=False, enable_mkldnn=False):
            pass
        def recognize(self, video_path):
            return [TimedSegment(id="o-0001", start_ms=0, end_ms=1000, text="\u4f60\u597d", source="ocr")]

    def fail_translate(segments, context_bundle, *, settings, review=True):
        raise RuntimeError("translation failed")

    monkeypatch.setattr("vietdub.media.extract_audio", fake_extract_audio)
    monkeypatch.setattr("vietdub.stt.FasterWhisperSttEngine", FakeSttEngine)
    monkeypatch.setattr("vietdub.ocr.PaddleSubtitleOcrEngine", FakeOcrEngine)
    monkeypatch.setattr("vietdub.reference.build_reference_context", lambda segments, data_dir: {})
    monkeypatch.setattr("vietdub.translate.translate_with_llm", fail_translate)

    settings = type(
        "Settings",
        (),
        {
            "sample_rate": 44100,
            "stt_model": "tiny",
            "stt_language": "zh",
            "reference_data_dir": str(tmp_path / "data"),
            "openai_api_key": "key",
            "llm_model": "model",
            "openai_base_url": "",
            "ocr_use_gpu": False,
            "ocr_enable_mkldnn": False,
        },
    )()

    try:
        run_review_pipeline(video=video, jobs_dir=tmp_path / "jobs", series=None, settings=settings)
    except RuntimeError as exc:
        assert "translation failed" in str(exc)
    else:
        raise AssertionError("expected translation failure")

    job_root = tmp_path / "jobs" / "clip"
    assert (job_root / "stt" / "segments.json").exists()
    assert (job_root / "ocr" / "subtitles.json").exists()
    assert (job_root / "transcript" / "merged.json").exists()
    assert (job_root / "context" / "style_guide.json").exists()
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))
    assert status["extract"]["state"] == "done"
    assert status["stt"]["state"] == "done"
    assert status["ocr"]["state"] == "done"
    assert status["merge"]["state"] == "done"
    assert status["context"]["state"] == "done"
    assert "translate" not in status



def test_resume_fails_when_merged_json_missing(tmp_path):
    """M8: resume_tts_and_render raises a clear error when transcript/merged.json
    is missing (no silent fallback to m-N+ validation)."""
    from vietdub.pipeline import resume_tts_and_render
    from vietdub.jobs import JobManager

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    job_root = tmp_path / "jobs" / "clip"
    job_root.mkdir(parents=True)
    job = JobManager(tmp_path / "jobs").create(video, series=None)
    # NOTE: do NOT call run_review_pipeline; merged.json is intentionally absent.

    review = job.root / "translation" / "review.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(
        "segment_id,start_ms,end_ms,speaker,text_cn,text_vi,context_note,status\n"
        "m-0001,0,1000,,hi,Xin chao,,reviewed\n",
        encoding="utf-8-sig",
    )

    settings = _resume_settings()

    with pytest.raises(RuntimeError, match="Cannot resume"):
        resume_tts_and_render(job, settings)
