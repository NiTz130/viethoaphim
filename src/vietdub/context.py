from __future__ import annotations

from .models import DEFAULT_TONE, TimedSegment


EPISODE_SUMMARY = (
    "Tap phim hoat hinh Trung Quoc ngan, thoai nhanh, "
    "co yeu to hai sa dieu."
)
SCENE_SUMMARY = (
    "Canh mo dau hoac doan thoai lien tuc can dich theo cung ngu canh."
)


def derive_characters(reference_context: dict) -> list[dict[str, str]]:
    return [
        {
            "name_cn": entry["source"],
            "name_vi": entry["target"].split("/")[0],
            "source": "Names.txt",
        }
        for entry in reference_context.get("Names.txt", [])
    ]


def derive_glossary(reference_context: dict) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for filename, entries in reference_context.items():
        if filename == "Names.txt":
            continue
        for entry in entries:
            glossary[entry["source"]] = entry["target"]
    return glossary


def build_context_bundle(
    segments: list[TimedSegment],
    series_context: dict,
    reference_context: dict | None = None,
    system_memory: dict | None = None,
) -> dict:
    reference_context = reference_context or {}
    system_memory = system_memory or {"translation_examples": [], "characters": [], "glossary": []}
    joined = " ".join(segment.text for segment in segments[:20])
    scene_ids = [segment.id for segment in segments[:20]]
    characters = merge_characters(derive_characters(reference_context), system_memory.get("characters", []))
    glossary = merge_glossary(derive_glossary(reference_context), system_memory.get("glossary", []))
    return {
        "series_context": series_context,
        "reference_context": reference_context,
        "system_memory": system_memory,
        "translation_examples": system_memory.get("translation_examples", []),
        "episode_context": {
            "summary": EPISODE_SUMMARY,
            "source_excerpt": joined,
        },
        "characters": characters,
        "glossary": glossary,
        "style_guide": {
            "tone": DEFAULT_TONE,
            "translation_rules": [
                "Uu tien cau thoai tu nhien hon dich sat chu.",
                "Giu punchline ngan de hop timing TTS.",
                "Dich nhat quan ten rieng va cach xung ho trong toan bo job.",
                "Use reference_context for names, pronouns, and phrase hints, but keep Vietnamese dialogue natural.",
                "Prefer high-confidence system_memory examples when they match the current line, especially reviewed review.csv rows.",
            ],
        },
        "scene_context": [
            {
                "segment_ids": scene_ids,
                "summary": SCENE_SUMMARY,
                "characters": [character["name_vi"] for character in characters],
                "tone": DEFAULT_TONE,
            }
        ],
    }


def merge_characters(reference_characters: list[dict[str, str]], memory_characters: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for character in reference_characters:
        merged[character["name_cn"]] = character
    for character in memory_characters:
        name_cn = str(character.get("name_cn") or "").strip()
        name_vi = str(character.get("name_vi") or "").strip()
        if not name_cn or not name_vi or name_cn in merged:
            continue
        merged[name_cn] = {
            "name_cn": name_cn,
            "name_vi": name_vi,
            "source": "system_memory",
            "confidence": float(character.get("confidence") or 0.0),
        }
    return list(merged.values())


def merge_glossary(reference_glossary: dict[str, str], memory_glossary: list[dict]) -> dict[str, str]:
    merged = dict(reference_glossary)
    for entry in memory_glossary:
        source_text = str(entry.get("source_text") or "").strip()
        target = str(entry.get("target") or "").strip()
        if not source_text or not target or source_text in merged:
            continue
        merged[source_text] = target
    return merged
