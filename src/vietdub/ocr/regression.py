"""OCR regression diff: match segments across two runs and compute metrics."""
from difflib import SequenceMatcher
from typing import Optional

from .schema import OcrSegment


def _midpoint(seg: OcrSegment) -> int:
    return (seg.start_ms + seg.end_ms) // 2


def match_segments(
    baseline: list[OcrSegment],
    new: list[OcrSegment],
    threshold_ms: int = 200,
) -> list[tuple[str, str]]:
    """Greedy match by closest midpoint, threshold ±N ms. Tie-break by id asc, then distance."""
    # Filter zero/negative-duration segments
    base = sorted([s for s in baseline if s.end_ms > s.start_ms], key=lambda s: (_midpoint(s), s.id))
    new_filt = sorted([s for s in new if s.end_ms > s.start_ms], key=lambda s: (_midpoint(s), s.id))

    used_new: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for b in base:
        b_mid = _midpoint(b)
        candidates = [
            (abs(_midpoint(n) - b_mid), n.id, n)
            for n in new_filt
            if n.id not in used_new and abs(_midpoint(n) - b_mid) <= threshold_ms
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda c: (c[0], c[1]))  # distance, then id
        _, matched_id, _ = candidates[0]
        used_new.add(matched_id)
        pairs.append((b.id, matched_id))

    return pairs


def _iou(a: OcrSegment, b: OcrSegment) -> float:
    inter_start = max(a.start_ms, b.start_ms)
    inter_end = min(a.end_ms, b.end_ms)
    intersection = max(0, inter_end - inter_start)
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_metrics(
    baseline: list[OcrSegment],
    new: list[OcrSegment],
    matches: list[tuple[str, str]],
) -> dict:
    """Compute the 5 OCR regression metrics per spec §Data Flow."""
    base_by_id = {s.id: s for s in baseline}
    new_by_id = {s.id: s for s in new}
    matched_new_ids = {n for _, n in matches}

    # segment_count
    bc = len(baseline)
    nc = len(new)
    sc_delta = abs(nc - bc) / bc * 100 if bc else 0.0

    # mean_text_length
    b_mean = sum(len(s.text) for s in baseline) / bc if bc else 0.0
    n_mean = sum(len(s.text) for s in new) / nc if nc else 0.0
    tl_delta = abs(n_mean - b_mean) / b_mean * 100 if b_mean else 0.0

    # time_range_overlap (mean IoU of matched pairs)
    if matches:
        ious = [_iou(base_by_id[b], new_by_id[n]) for b, n in matches]
        iou_mean = sum(ious) / len(ious)
    else:
        iou_mean = 0.0

    # text_similarity (mean SequenceMatcher.ratio for matched pairs)
    if matches:
        sims = [
            SequenceMatcher(None, base_by_id[b].text, new_by_id[n].text).ratio()
            for b, n in matches
        ]
        sim_mean = sum(sims) / len(sims)
    else:
        sim_mean = 0.0

    # unmatched_pct (informational)
    unmatched_baseline = [b for b, _ in matches]  # missing pairs are unmatched
    unmatched_total = (bc - len(matches)) + (nc - len(matches))
    unmatched_pct = unmatched_total / max(bc, nc) * 100 if max(bc, nc) else 0.0

    return {
        "segment_count": {"baseline": bc, "new": nc, "delta_pct": round(sc_delta, 2)},
        "mean_text_length": {"baseline": round(b_mean, 2), "new": round(n_mean, 2), "delta_pct": round(tl_delta, 2)},
        "time_range_overlap": round(iou_mean, 4),
        "text_similarity": round(sim_mean, 4),
        "unmatched_pct": round(unmatched_pct, 2),
    }