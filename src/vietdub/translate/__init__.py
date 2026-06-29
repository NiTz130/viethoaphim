"""Vietnamese dubbing translation: subpackage.

Public API re-exported from sub-modules for backward compatibility:

- CSV I/O: `export_review_csv`, `import_review_csv` (csv.py)
- Prompt building: `build_translation_prompt`, `parse_translation_response` (prompt.py)
- LLM orchestration: `translate_with_llm` (llm.py)
- Module logger: `logger` (llm.py)
"""
from __future__ import annotations

from .csv import ALLOWED_REVIEW_STATUSES, CSV_FIELDS, export_review_csv, import_review_csv
from .glossary import (
    _load_ignore_list,
    _load_initial_glossary,
    _load_phrase_patterns,
    _load_pronouns,
    _update_glossary,
    _cap_glossary,
)
from .llm import (
    _review_batch_for_length,
    _translate_one_batch,
    _translate_one_batch_with_retry,
    _warned_unknown_caps,
    logger,
    translate_with_llm,
)
from .llm import LLM_BATCH_SIZE, LLM_MAX_TOKENS  # re-exported for test monkeypatching
from .prompt import build_translation_prompt, parse_translation_response

# Re-export the reference-data helpers at package scope so existing tests can
# keep using `monkeypatch.setattr(vietdub.translate, "find_reference_data_dir", ...)`.
# The sub-modules resolve these names dynamically through `sys.modules` so the
# patched values take effect inside translate_with_llm and _load_initial_glossary.
from ..reference import find_reference_data_dir, load_dictionary_entries

__all__ = [
    "CSV_FIELDS",
    "ALLOWED_REVIEW_STATUSES",
    "export_review_csv",
    "import_review_csv",
    "build_translation_prompt",
    "parse_translation_response",
    "translate_with_llm",
    "logger",
    "_update_glossary",
    "_cap_glossary",
    "_load_pronouns",
    "_load_phrase_patterns",
    "_load_ignore_list",
    "_load_initial_glossary",
    "_review_batch_for_length",
    "_translate_one_batch",
    "_translate_one_batch_with_retry",
    "_warned_unknown_caps",
    "LLM_BATCH_SIZE",
    "LLM_MAX_TOKENS",
    "find_reference_data_dir",
    "load_dictionary_entries",
]