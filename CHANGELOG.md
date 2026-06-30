# Changelog

## Unreleased

### Changed
- Migrated to paddlepaddle 3.x + numpy 2.x + torch 2.6+ stack (hybrid pinning — see pyproject.toml comments).

### Added
- OCR regression harness (`make ocr-diff`, `make ocr-baseline`).
- Idempotent `setup.ps1` + cross-platform `make setup`.
- GitHub Actions CI: `test.yml`, `integration.yml`, `ocr-diff.yml`.

### Deprecated
- `requirements-ocr.txt` (shim only; deleted in next release after ≥3-day soak).