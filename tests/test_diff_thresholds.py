import json
from pathlib import Path

import pytest

from vietdub.ocr.regression import build_report, passes_thresholds
from vietdub.ocr.schema import OcrSegment


FIXTURES = Path(__file__).parent / "fixtures"


def load_thresholds():
    return json.loads((FIXTURES / "diff_thresholds.json").read_text(encoding="utf-8"))


def test_thresholds_file_exists():
    assert (FIXTURES / "diff_thresholds.json").exists()


def test_thresholds_have_required_keys():
    t = load_thresholds()
    for k in ("segment_count_pct_max", "text_length_pct_max", "iou_min", "text_similarity_min"):
        assert k in t


def test_passes_thresholds_all_pass():
    metrics = {
        "segment_count": {"delta_pct": 2.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.90,
    }
    assert passes_thresholds(metrics, load_thresholds()) is True


def test_passes_thresholds_segment_count_fails():
    metrics = {
        "segment_count": {"delta_pct": 7.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.90,
    }
    assert passes_thresholds(metrics, load_thresholds()) is False


def test_passes_thresholds_similarity_fails():
    metrics = {
        "segment_count": {"delta_pct": 2.0},
        "mean_text_length": {"delta_pct": 5.0},
        "time_range_overlap": 0.97,
        "text_similarity": 0.80,
    }
    assert passes_thresholds(metrics, load_thresholds()) is False