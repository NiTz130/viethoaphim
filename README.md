# VietDub

Windows-first CLI for translating Chinese hard-subbed cartoon videos into Vietnamese dubbed previews.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Environment

```powershell
$env:OPENAI_API_KEY="your-key"
$env:LLM_MODEL="gpt-4.1-mini"
```

## Usage

```powershell
vietdub run input.mp4 --mode review --series my-series
vietdub resume jobs\input --from tts
vietdub inspect jobs\input
```

## MVP Flow

1. Put one `.mp4` video in the project folder.
2. Run review mode:

```powershell
vietdub run .\input.mp4 --mode review --series demo
```

3. Edit `jobs\<job-name>\translation\review.csv`.
4. Resume from TTS:

```powershell
vietdub resume .\jobs\<job-name> --from tts
```

5. Check:

- `jobs\<job-name>\translation\review.csv`
- `jobs\<job-name>\output\subtitles_vi.srt`
- `jobs\<job-name>\tts\segments\*.mp3`
- `jobs\<job-name>\output\preview_vi.mp4` when final audio muxing is available

## System Memory

Each review run scans previous job folders under `jobs/` before translation. It reuses:

- `translation/review.csv`
- `translation/translated.json`
- `context/characters.json`
- `context/glossary.json`
- `context/reference_context.json`

The current job is excluded from the scan. Reviewed rows in `review.csv` have the highest confidence, draft rows with `text_vi` are lower confidence, and generated `translated.json` rows are fallback examples.

New jobs write these memory artifacts under `jobs/<job-name>/context/`:

- `system_memory.json`: selected historical examples, characters, and glossary entries relevant to the current transcript
- `translation_examples.json`: high-confidence Chinese-to-Vietnamese examples included in the LLM prompt
- `characters.json`: names merged from reference data and relevant system memory
- `glossary.json`: terms merged from reference data and relevant system memory
- `system_memory_warnings.json`: scanner warnings when old job files are malformed
