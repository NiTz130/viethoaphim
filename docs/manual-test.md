# Manual Test

## Environment

- Windows
- Python 3.11
- FFmpeg in PATH
- `OPENAI_API_KEY` set
- `LLM_MODEL` set

## Test Video

Use a 30-60 second `.mp4` clip with clear Chinese hard subtitles near the bottom of the frame.

## Commands

```powershell
python -m pip install -e ".[dev]"
vietdub run .\sample.mp4 --mode review --series sample-series
```

Open `jobs\sample\translation\review.csv`, fill at least three `text_vi` cells, save as UTF-8 CSV, then run:

```powershell
vietdub resume .\jobs\sample --from tts
```

## Acceptance Criteria

- A job folder is created under `jobs`.
- `stt/segments.json`, `ocr/subtitles.json`, `transcript/merged.json`, and `context/style_guide.json` exist after review run.
- `translation/review.csv` opens in Excel without mojibake.
- `output/subtitles_vi.srt` contains the reviewed Vietnamese lines.
- `tts/segments` contains one MP3 per reviewed non-empty row.
