# VietDub

VietDub là CLI ưu tiên Windows để tạo bản lồng tiếng tiếng Việt từ video hoạt hình hoặc tiểu phẩm tiếng Trung có phụ đề cứng. Pipeline tách âm thanh, nhận diện lời thoại và phụ đề, trộn transcript, gọi LLM để dịch sang tiếng Việt, cho biên tập bằng CSV, sau đó tạo phụ đề `.srt`, file TTS và video preview.

## Tính Năng Chính

- Trích xuất audio từ `.mp4` bằng FFmpeg.
- STT tiếng Trung bằng `faster-whisper`.
- OCR phụ đề cứng bằng PaddleOCR trên vùng dưới khung hình.
- Trộn kết quả STT/OCR thành transcript theo mốc thời gian.
- Gọi OpenAI-compatible LLM để dịch sang tiếng Việt.
- Xuất `review.csv` UTF-8 BOM để mở bằng Excel.
- Resume từ bước TTS sau khi sửa bản dịch.
- Tạo MP3 từng câu bằng Edge TTS, ghép thành `final_vi.wav`.
- Render `subtitles_vi.srt` và mux video preview `preview_vi.mp4`.
- Tái sử dụng dữ liệu từ điển trong `data/` và bộ nhớ từ các job cũ trong `jobs/`.

## Yêu Cầu

- Windows PowerShell.
- Python 3.11 trở lên.
- FFmpeg và FFprobe có trong `PATH`.
- Internet để gọi LLM, Edge TTS và tải model OCR/STT lần đầu.
- OpenAI API key hoặc endpoint OpenAI-compatible.

Kiểm tra nhanh:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Nếu PaddleOCR hoặc faster-whisper cần tải model lần đầu, lần chạy đầu tiên có thể chậm hơn bình thường.

## Cấu Hình

Có thể set biến môi trường trực tiếp trong PowerShell hoặc tạo file `.env` ở thư mục gốc repo. `.env` không được commit.

Tối thiểu:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:LLM_MODEL="gpt-4.1-mini"
```

Ví dụ `.env`:

```dotenv
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=
EDGE_VOICE=vi-VN-HoaiMyNeural
STT_MODEL=medium
STT_LANGUAGE=zh
JOBS_DIR=jobs
REFERENCE_DATA_DIR=data
SAMPLE_RATE=44100
```

Ghi chú:

- `OPENAI_API_KEY` và `LLM_MODEL` bắt buộc cho bước dịch.
- `OPENAI_BASE_URL` để trống nếu dùng OpenAI mặc định. Nếu endpoint kết thúc bằng `/responses`, VietDub sẽ gọi Responses API; các endpoint khác dùng Chat Completions.
- `REFERENCE_DATA_DIR` mặc định là `data`. Tool tự động tìm thư mục con có `Dictionaries.config`.

## Workflow Review

Dùng chế độ review khi muốn sửa bản dịch trước khi tạo TTS.

```powershell
vietdub run .\sample.mp4 --mode review --series sample-series
```

Lệnh trên tạo job mới trong `jobs\sample` hoặc `jobs\sample-1` nếu tên đã tồn tại. Sau khi chạy xong, mở:

```text
jobs\<job-name>\translation\review.csv
```

Sửa cột `text_vi`, có thể đổi `status`:

- `draft`: bản dịch nháp hoặc do LLM tạo, vẫn có thể dùng để render.
- `reviewed`: dòng đã được biên tập và được ưu tiên khi dùng làm bộ nhớ cho job sau.
- `skip`: bỏ qua dòng này khi tạo SRT/TTS.

Sau khi sửa CSV, resume từ TTS:

```powershell
vietdub resume .\jobs\<job-name> --from tts
```

Kết quả chính:

```text
jobs\<job-name>\output\subtitles_vi.srt
jobs\<job-name>\tts\segments\*.mp3
jobs\<job-name>\tts\final_vi.wav
jobs\<job-name>\tts\sync_report.json
jobs\<job-name>\output\preview_vi.mp4
```

## Workflow Tự Động

Dùng `auto` nếu muốn pipeline chạy tiếp qua TTS/render ngay sau khi LLM dịch xong:

```powershell
vietdub run .\sample.mp4 --mode auto --series sample-series
```

Chế độ này vẫn ghi `translation\review.csv`, nhưng sẽ không dừng lại để chờ biên tập thủ công.

## Kiểm Tra Job

```powershell
vietdub inspect .\jobs\<job-name>
```

Lệnh này in đường dẫn job và nội dung `status.json`.

## Cấu Trúc Job

Mỗi job copy video đầu vào thành `input.mp4` và tạo các thư mục:

```text
audio\original.wav
stt\segments.json
ocr\subtitles.json
transcript\merged.json
context\*.json
translation\translated.json
translation\review.csv
tts\segments\*.mp3
tts\final_vi.wav
tts\sync_report.json
output\subtitles_vi.srt
output\preview_vi.mp4
status.json
job.json
```

`status.json` ghi lại các bước đã xong: `extract`, `stt`, `ocr`, `merge`, `context`, `translate`, `tts`, `render`.

## Dữ Liệu Tham Chiếu Và Bộ Nhớ

Repo có sẵn thư mục `data\Data của thtgiang (đọc README)` gồm các file từ điển như `Names.txt`, `Pronouns.txt`, `VietPhrase.txt` và `Dictionaries.config`. Khi dịch, VietDub:

- Đọc các mục từ điển có xuất hiện trong transcript hiện tại.
- Quét các job cũ trong `jobs/` để lấy ví dụ dịch, nhân vật và glossary.
- Ưu tiên dòng `reviewed` trong `review.csv`, sau đó đến dòng có `text_vi`, cuối cùng là `translated.json`.
- Ghi các file ngữ cảnh vào `jobs\<job-name>\context\`, gồm `system_memory.json`, `translation_examples.json`, `characters.json`, `glossary.json`, `reference_context.json`.

Job hiện tại được loại khỏi quá trình quét bộ nhớ để tránh tự học lại kết quả của chính nó.

## Lưu Ý Vận Hành

- OCR hiện cắt 28% phần dưới khung hình, phù hợp video có hard-sub nằm gần đáy màn hình.
- Nếu GPU/CUDA lỗi khi chạy faster-whisper, code tự fallback về CPU `int8`.
- Khi resume TTS, các artifact cũ của bước render sẽ được xóa trước để tránh preview/SRT cũ.
- `review.csv` phải giữ đủ các cột: `segment_id`, `start_ms`, `end_ms`, `speaker`, `text_cn`, `text_vi`, `context_note`, `status`.
- `segment_id` trong CSV phải tồn tại trong `transcript\merged.json` nếu file này có sẵn.

## Test

Chạy test từ thư mục gốc repo:

```powershell
pytest
```

Manual test nhanh:

```powershell
python -m pip install -e ".[dev]"
vietdub run .\sample.mp4 --mode review --series sample-series
vietdub resume .\jobs\sample --from tts
```

Chi tiết manual test nằm trong `docs\manual-test.md`.
