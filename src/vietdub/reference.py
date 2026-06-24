from __future__ import annotations

from pathlib import Path

from .models import TimedSegment


REFERENCE_FILES = ["Names.txt", "Pronouns.txt", "VietPhrase.txt"]


def find_reference_data_dir(data_dir: Path) -> Path:
    if (data_dir / "Dictionaries.config").exists():
        return data_dir
    if data_dir.exists():
        for child in data_dir.iterdir():
            if child.is_dir() and (child / "Dictionaries.config").exists():
                return child
    return data_dir


def load_dictionary_entries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        source, target = line.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            entries[source] = target
    return entries


def build_reference_context(
    segments: list[TimedSegment],
    data_dir: Path,
    max_entries_per_file: int = 40,
) -> dict[str, list[dict[str, str]]]:
    data_dir = find_reference_data_dir(data_dir)
    joined_text = "\n".join(segment.text for segment in segments)
    context: dict[str, list[dict[str, str]]] = {}

    for filename in REFERENCE_FILES:
        matches: list[dict[str, str]] = []
        entries = load_dictionary_entries(data_dir / filename)
        for source, target in sorted(entries.items(), key=lambda item: len(item[0]), reverse=True):
            if source in joined_text:
                matches.append({"source": source, "target": target})
            if len(matches) >= max_entries_per_file:
                break
        context[filename] = matches

    return context
