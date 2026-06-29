from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from ..models import TimedSegment, TranslationRow
from ..reference import find_reference_data_dir, load_dictionary_entries


def _pkg_attr(name: str, default):
    """Resolve an attribute through the parent package, so test code can
    monkeypatch `vietdub.translate.X` and have the override take effect here."""
    pkg = sys.modules.get("vietdub.translate")
    if pkg is not None and hasattr(pkg, name):
        return getattr(pkg, name)
    return default


def _resolve_find_reference_data_dir():
    return _pkg_attr("find_reference_data_dir", find_reference_data_dir)


def _resolve_load_dictionary_entries():
    return _pkg_attr("load_dictionary_entries", load_dictionary_entries)


_CHINESE_NAME_RE = re.compile(r"[一-鿿]{2,4}")
_MAX_GLOSSARY_ENTRIES = 50


# Use the parent package's logger so test assertions on
# `r.name == "vietdub.translate"` continue to match.
logger = logging.getLogger("vietdub.translate")


def _load_initial_glossary() -> dict[str, str]:
    """Load initial name glossary from reference data (Names.txt).

    Uses the shared ``load_dictionary_entries`` helper, which parses the
    real ``cn_name=vi_name`` format. The reference directory is located via
    ``find_reference_data_dir`` so it works with both flat and nested
    layouts (e.g. ``data/Data của thtgiang (đọc README)/``). Returns an
    empty dict if the file is missing or unreadable.
    """
    glossary: dict[str, str] = {}
    try:
        ref_root = Path(os.environ.get("REFERENCE_DATA_DIR", "data"))
        ref_dir = _resolve_find_reference_data_dir()(ref_root)
        glossary = _resolve_load_dictionary_entries()(ref_dir / "Names.txt")
    except Exception as exc:
        logger.warning(
            f"Warning: failed to load initial glossary: {type(exc).__name__}: {exc}"
        )
        glossary = {}
    return glossary


def _load_pronouns(ref_dir: Path) -> list[dict[str, str]]:
    """Load pronoun guide from Pronouns.txt. Format: cn=vi per line.

    Uses the shared load_dictionary_entries helper (handles = separator).
    Returns empty list if file missing.
    """
    entries = _resolve_load_dictionary_entries()(ref_dir / "Pronouns.txt")
    return [{"source": cn, "target": vi} for cn, vi in entries.items()]


def _load_phrase_patterns(ref_dir: Path) -> list[dict[str, str]]:
    """Load phrase patterns from LuatNhan.txt. Format: cn{0}=vi{0} per line.

    Returns list of {source, target} dicts. Empty list if file missing.
    """
    patterns: list[dict[str, str]] = []
    path = ref_dir / "LuatNhan.txt"
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        patterns.append({"source": source.strip(), "target": target.strip()})
    return patterns


def _load_ignore_list(ref_dir: Path) -> list[str]:
    """Load Chinese boilerplate phrases to ignore from IgnoredChinesePhrases.txt.

    Each line is a long Chinese phrase (boilerplate from web novel scraping).
    Returns list of raw phrases. Empty list if file missing.
    """
    ignore_path = ref_dir / "IgnoredChinesePhrases.txt"
    if not ignore_path.exists():
        return []
    return [line.strip() for line in ignore_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _update_glossary(
    glossary: dict[str, str],
    rows: list[TranslationRow],
    batch: list[TimedSegment],
) -> None:
    """Extract name_cn → name_vi pairs from completed batch and merge into glossary.

    Heuristic: for each segment, find Chinese name runs (2-4 CJK chars) and
    align them positionally with capitalized Vietnamese words.
    """
    for row, seg in zip(rows, batch):
        cn_names = _CHINESE_NAME_RE.findall(seg.text)
        if not cn_names:
            continue
        vi_words = row.text_vi.split()
        for i, cn_name in enumerate(cn_names):
            if i >= len(vi_words):
                break
            vi_word = vi_words[i].strip(".,!?;:")
            if len(vi_word) >= 2 and vi_word[0].isupper():
                glossary.setdefault(cn_name, vi_word)


def _cap_glossary(
    glossary: dict[str, str],
    max_entries: int = _MAX_GLOSSARY_ENTRIES,
) -> None:
    """Cap glossary to max_entries by dropping oldest entries (FIFO)."""
    while len(glossary) > max_entries:
        oldest_key = next(iter(glossary))
        del glossary[oldest_key]