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
