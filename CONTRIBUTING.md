# 🤝 Đóng góp cho VietDub

Cảm ơn bạn đã quan tâm đến việc đóng góp cho **VietDub**! 🎉

## 🐞 Báo lỗi

Trước khi tạo issue, vui lòng:

1. **Tìm kiếm** trong [Issues](https://github.com/NiTz130/viethoaphim/issues) xem lỗi đã được báo cáo chưa.
2. **Cập nhật** lên commit mới nhất (`git pull`) và thử lại.
3. **Thu thập log**:
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   vietdub run .\sample.mp4 --mode review --series "test-issue" 2>&1 | Tee-Object -FilePath debug.log
   ```
4. Tạo issue mới với:
   - Mô tả ngắn gọn vấn đề
   - Các bước tái hiện
   - Kết quả mong đợi vs thực tế
   - OS, Python version, GPU/CPU
   - Log liên quan

## 💡 Đề xuất tính năng

Mở issue với label `enhancement`, mô tả:

- Vấn đề đang giải quyết
- API / CLI bạn mong muốn
- Ví dụ input/output

## 🔀 Pull Request

### Quy trình

1. **Fork** repo và tạo branch từ `main`:
   ```bash
   git checkout -b feat/ten-tinh-nang
   ```
2. **Cài dev dependencies**:
   ```bash
   make setup
   pip install -e ".[dev]"
   ```
3. **Viết code + test** — đảm bảo:
   - Có test cho logic mới (`tests/`)
   - Không break test cũ: `pytest`
   - Không leak secret: `pre-commit run --all-files`
4. **Commit** với message rõ ràng (xem [Commit Convention](#-commit-convention)).
5. **Push** và mở Pull Request vào `main`.

### Quy tắc code

- **Style**: PEP 8 + type hints cho public API.
- **Docstring**: Google style cho mọi module / class / public function.
- **Import**: isort (stdlib → third-party → local).
- **Không commit**: `.env`, `jobs/`, `.venv/`, model weights lớn.
- **OCR changes**: chạy `make ocr-diff` để đảm bảo không regression.

## 📝 Commit Convention

Dùng [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `perf`.

**Ví dụ**:

```
feat(translate): add glossary priority for character names
fix(tts): dedupe segments before render to avoid repeated speech
docs(readme): add architecture diagram and quick start
test(ocr): add threshold test for paddleocr 3.x upgrade
```

## 🔒 Bảo mật

**Không commit API key, token, hay thông tin nhạy cảm.** Repo đã có `gitleaks` pre-commit hook — leak sẽ bị CI block.

Nếu phát hiện lỗ hổng bảo mật, **KHÔNG** tạo public issue. Email tác giả qua GitHub profile.

## 📜 License

Bằng việc đóng góp, bạn đồng ý rằng đóng góp sẽ được phát hành dưới [MIT License](LICENSE).

---

Cảm ơn bạn! 💙
