# OCR Setup

## Background

viethoaphimv4 uses PaddleOCR for Chinese subtitle extraction. The OCR stack
(paddlepaddle + paddleocr + their transitive deps) has known compatibility
constraints that conflict in a single Python environment:

- **paddlepaddle 2.6.2** requires `protobuf<=3.20.2`
- **onnxruntime ≥1.14** requires `protobuf>=4.25.8`
- **paddleocr's bundled torch** has DLL load issues on Windows 11 with newer versions

## Working combo

**Approach A: pinned old versions** (verified working on Windows 11 + Python 3.11)

| Package | Pinned Version | Reason |
|---|---|---|
| paddlepaddle | 2.6.2 | Current stable for paddleocr 2.x |
| paddleocr | 2.10.0 | Compatible with paddlepaddle 2.6.2 |
| protobuf | 3.20.2 | Highest version paddlepaddle 2.6.2 accepts |
| onnxruntime | 1.15.0 | Last version supporting protobuf 3.x |
| torch | 2.2.0 | Compatible with paddlepaddle 2.6.2's bundled torch (avoids DLL error) |

## Install

```bash
pip install -r requirements-ocr.txt
```

**Installation order matters:** if `torch` installs BEFORE `paddlepaddle`, paddlepaddle may override it. If you hit the Windows DLL error (`OSError: [WinError 127]` for `shm.dll`), run:

```bash
pip install torch==2.2.0 --force-reinstall --no-cache-dir
```

## Verify

After installation, run this smoke test:

```bash
python -c "from vietdub.ocr import PaddleSubtitleOcrEngine; PaddleSubtitleOcrEngine()"
```

If this exits 0 with no output, the OCR stack is compatible.

A scripted version of this check is at `C:\code\test\scripts\verify_ocr.py`.

## End-to-end test

To run the full pipeline including OCR (no bypass):

```bash
cd C:\code\viethoaphimv4
python C:\code\test\scripts\run_pipeline.py review
```

Output: 600+ segments with Chinese → Vietnamese translations in `jobs/<job-name>/translation/review.csv`. STT alone produces ~211 segments; OCR adds ~400 Chinese subtitle segments, bringing the total to ~608.

## Upgrading safely

If you later run `pip install --upgrade` (e.g., for unrelated packages), pip may upgrade paddlepaddle, paddleocr, or torch to incompatible versions. To restore the working combo:

```bash
pip install -r requirements-ocr.txt
```

Do NOT run `pip install paddlepaddle-gpu` — this is the GPU variant and has different deps. Stick with the CPU-only `paddlepaddle` package.

## Troubleshooting

### torch DLL error on Windows

If you see `OSError: [WinError 127] The specified procedure could not be found. Error loading "...\torch\lib\shm.dll"`:

```bash
pip install torch==2.2.0 --force-reinstall --no-cache-dir
```

### protobuf version conflict

If `pip install` fails with a protobuf version error:

```bash
pip install "protobuf==3.20.2"
pip install "onnxruntime==1.15.0"
```

### NumPy compatibility warnings

You may see `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` when importing torch. This is cosmetic and does not affect OCR functionality. To silence, downgrade NumPy: `pip install "numpy<2"`.

## Alternative: Docker

For production deployments where pip-based setup is fragile, paddlepaddle provides official Docker images with pre-resolved dependency graphs:

```dockerfile
FROM paddlepaddle/paddle:3.3.1
# ... (rest of viethoaphimv4 setup)
```

This avoids the protobuf conflict entirely (paddlepaddle 3.x uses newer protobuf). However, it does require modifying viethoaphimv4 to be compatible with paddlepaddle 3.x's API (currently only paddlepaddle 2.x is tested with viethoaphimv4).
