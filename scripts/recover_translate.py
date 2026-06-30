"""Recover a job whose translate step failed by re-running only the LLM step.

Uses the already-produced context bundle + merged transcript to call
translate_with_llm again, then writes translated.json + review.csv and marks
translate as done in status.json.

Falls back to copying text_cn into text_vi for any segment the LLM still
fails to translate, so the pipeline can always progress to TTS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vietdub.config import Settings  # noqa: E402
from vietdub.context import build_context_bundle  # noqa: E402
from vietdub.merge import merge_segments  # noqa: E402
from vietdub.models import StepName, TimedSegment, TranslationRow  # noqa: E402
from vietdub.reference import build_reference_context  # noqa: E402
from vietdub.memory import collect_system_memory, select_relevant_memory  # noqa: E402
from vietdub.stt import FasterWhisperSttEngine  # noqa: E402
from vietdub.ocr import PaddleSubtitleOcrEngine  # noqa: E402
from vietdub.translate import translate_with_llm, export_review_csv  # noqa: E402


def main(job_dir: Path) -> int:
    if not (job_dir / "stt" / "segments.json").exists():
        print(f"ERROR: missing {job_dir / 'stt' / 'segments.json'}", file=sys.stderr)
        return 2
    if not (job_dir / "ocr" / "subtitles.json").exists():
        print(f"ERROR: missing {job_dir / 'ocr' / 'subtitles.json'}", file=sys.stderr)
        return 2
    merged_path = job_dir / "transcript" / "merged.json"
    if not merged_path.exists():
        print(f"ERROR: missing {merged_path}", file=sys.stderr)
        return 2

    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing in env/.env", file=sys.stderr)
        return 2

    stt_segments = [TimedSegment.model_validate(s) for s in json.loads((job_dir / "stt" / "segments.json").read_text(encoding="utf-8"))]
    ocr_segments = [TimedSegment.model_validate(s) for s in json.loads((job_dir / "ocr" / "subtitles.json").read_text(encoding="utf-8"))]

    # Reuse merged.json from the failed run to avoid an extra merge call.
    merged = [TimedSegment.model_validate(s) for s in json.loads(merged_path.read_text(encoding="utf-8"))]
    if len(merged) != len(stt_segments):
        print(f"WARN: merged has {len(merged)} segs, STT has {len(stt_segments)}; using merged as-is", file=sys.stderr)

    reference_context = build_reference_context(merged, Path(settings.reference_data_dir))
    raw_system_memory = collect_system_memory(Path(settings.jobs_dir), current_job=job_dir)
    selected_system_memory = select_relevant_memory(raw_system_memory, merged)
    context_bundle = build_context_bundle(
        merged,
        series_context={},
        reference_context=reference_context,
        system_memory=selected_system_memory,
    )

    try:
        translations = translate_with_llm(merged, context_bundle, settings=settings, review=True)
    except Exception as exc:  # noqa: BLE001
        print(f"translate_with_llm failed: {exc!r}", file=sys.stderr)
        return 1

    if len(translations) != len(merged):
        print(f"ERROR: got {len(translations)} translations for {len(merged)} merged segs", file=sys.stderr)
        return 1

    fallback_count = 0
    for row, seg in zip(translations, merged):
        if not row.text_vi.strip():
            row.text_vi = f"[CHƯA DỊCH] {seg.text}"
            row.context_note = (row.context_note or "") + " [FALLBACK: LLM returned empty text_vi]"
            fallback_count += 1

    (job_dir / "translation").mkdir(parents=True, exist_ok=True)
    (job_dir / "translation" / "translated.json").write_text(
        json.dumps([row.model_dump() for row in translations], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_review_csv(job_dir / "translation" / "review.csv", merged, translations)

    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status[StepName.TRANSLATE.value] = {
        "state": "done",
        "finished_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "details": {
            "rows": len(translations),
            "review_csv": str(job_dir / "translation" / "review.csv"),
            "fallback_rows": fallback_count,
        },
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote {len(translations)} translations ({fallback_count} fallback). review.csv ready at:")
    print(f"  {job_dir / 'translation' / 'review.csv'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: recover_translate.py <job_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]).resolve()))