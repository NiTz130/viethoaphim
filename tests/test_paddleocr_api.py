"""Regression test for the PaddleOCR API contract used by vietdub.

PaddleOCR 3.x (paddlex-backed) changed the constructor signature and the
result format compared to 2.x. ``src/vietdub/ocr/paddle.py`` calls
``PaddleOCR(...)`` with a specific set of kwargs and reads
``result.json['res']['rec_texts']`` out of the prediction results. This
test pins both contracts so that any future paddleocr minor-version bump
that breaks them will fail at PR time, not in production.

The two tests are intentionally cheap to run on environments that have
paddleocr installed (most CI configurations). On environments without
paddleocr, both tests skip with a documented reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_paddleocr_constructor_signature() -> None:
    """PaddleOCR.__init__ must accept every kwarg vietdub passes to it.

    Behavior check (not signature inspection): actually try to construct
    a PaddleOCR instance with the 6 kwargs paddle.py uses. This catches
    the original cutover-3a bug class (2.x-only kwargs like use_angle_cls,
    use_gpu, show_log) AND future drift where paddleocr removes a kwarg
    we depend on. It is more robust than inspect.signature because
    PaddleOCR 3.x forwards some kwargs (device, enable_mkldnn) via
    **kwargs to the underlying paddlebase.PaddlePredictor, so they do
    not appear in inspect.signature but ARE accepted at construction.
    """
    pytest.importorskip("paddleocr")
    try:
        from paddleocr import PaddleOCR
    except (ImportError, OSError) as exc:
        pytest.skip(f"paddleocr import failed: {exc}")

    try:
        # Mirrors the production call in src/vietdub/ocr/paddle.py,
        # except use_textline_orientation=False (skip the angle model
        # to keep the static test fast — no inference happens here).
        PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"PaddleOCR.__init__ rejected vietdub's required kwargs: {exc}. "
            f"Có thể project đang dùng API cũ trên paddleocr mới, "
            f"hoặc paddleocr đã đổi signature."
        )


def test_paddleocr_predict_returns_ocrresult() -> None:
    """PaddleOCR.predict() must return objects whose .json['res']['rec_texts'] is a list.

    The smoke check exercises a real PaddleOCR 3.x inference on a synthetic
    image. It uses device='cpu' and enable_mkldnn=False to avoid the
    OneDNN crash observed on Windows with paddlepaddle 3.x. It also disables
    the angle classifier (use_textline_orientation=False) for speed: the
    synthetic image is upright so the angle model is not needed.
    """
    try:
        from paddleocr import PaddleOCR
    except (ImportError, OSError) as exc:
        pytest.skip(f"paddleocr not importable: {exc}")

    from PIL import Image, ImageDraw, ImageFont  # PIL ships with the venv

    # Build a synthetic 300x80 white image with the text "Hello".
    # We use ASCII here (not CJK) to avoid font-availability issues on
    # minimal Linux CI images. The test only checks result SHAPE, not
    # the recognized text, so the content of the image is irrelevant.
    img = Image.new("RGB", (300, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 25), "Hello", fill="black", font=ImageFont.load_default())
    tmp_path = _save_temp_png(img)

    try:
        try:
            ocr = PaddleOCR(
                lang="ch",
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except NotImplementedError as exc:
            # paddlepaddle 3.x raises this on Windows when OneDNN is requested
            # but the runtime lacks the required kernel. We forced
            # enable_mkldnn=False above, but paddle may still probe.
            pytest.skip(f"paddlepaddle OneDNN unavailable in this env: {exc}")

        try:
            results = list(ocr.predict(str(tmp_path)))
        except (RuntimeError, OSError) as exc:
            # Model download blocked, network down, or disk full.
            pytest.skip(f"PaddleOCR model unavailable: {exc}")

        assert len(results) >= 1, "PaddleOCR.predict() returned no results"
        page = results[0]
        page_json = getattr(page, "json", None)
        assert isinstance(page_json, dict), (
            f"Expected page.json to be dict, got {type(page_json).__name__}. "
            f"PaddleOCR result format may have changed."
        )
        inner = page_json.get("res")
        assert isinstance(inner, dict), (
            f"Expected page.json['res'] to be dict, got {type(inner).__name__}."
        )
        rec_texts = inner.get("rec_texts")
        assert isinstance(rec_texts, list), (
            f"Expected page.json['res']['rec_texts'] to be list, "
            f"got {type(rec_texts).__name__}."
        )
        # rec_texts can be empty (synthetic image may OCR as nothing).
        # We only assert shape, not content.
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_temp_png(img) -> Path:
    """Save a PIL image to a temporary PNG file and return the path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return Path(tmp.name)