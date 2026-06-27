from vietdub.models import TimedSegment
from vietdub.reference import build_reference_context, find_reference_data_dir, load_dictionary_entries


def test_load_dictionary_entries_reads_key_value_lines(tmp_path):
    path = tmp_path / "VietPhrase.txt"
    path.write_text(
        "# comment\n"
        "\u4f60\u597d=Xin ch\u00e0o/Ch\u00e0o\n"
        "invalid line\n"
        "\u795e\u79d8=th\u1ea7n b\u00ed\n",
        encoding="utf-8",
    )

    entries = load_dictionary_entries(path)

    assert entries == {"\u4f60\u597d": "Xin ch\u00e0o/Ch\u00e0o", "\u795e\u79d8": "th\u1ea7n b\u00ed"}


def test_build_reference_context_includes_only_matching_entries(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "Names.txt").write_text("\u5c0f\u660e=Ti\u1ec3u Minh\n", encoding="utf-8")
    (data_dir / "VietPhrase.txt").write_text(
        "\u795e\u79d8=th\u1ea7n b\u00ed\n"
        "\u4e0d\u5339\u914d=kh\u00f4ng kh\u1edbp\n",
        encoding="utf-8",
    )
    (data_dir / "Pronouns.txt").write_text("\u4f60=ng\u01b0\u01a1i\n", encoding="utf-8")

    context = build_reference_context(
        [TimedSegment(id="m-0001", start_ms=0, end_ms=1000, text="\u4f60\u6709\u795e\u79d8\u8fc7\u53bb")],
        data_dir,
    )

    assert context == {
        "Names.txt": [],
        "VietPhrase.txt": [{"source": "\u795e\u79d8", "target": "th\u1ea7n b\u00ed"}],
    }


def test_find_reference_data_dir_uses_child_with_dictionary_config(tmp_path):
    root = tmp_path / "data"
    child = root / "Data cua thtgiang"
    child.mkdir(parents=True)
    (child / "Dictionaries.config").write_text("Names=Names.txt\n", encoding="utf-8")

    assert find_reference_data_dir(root) == child
