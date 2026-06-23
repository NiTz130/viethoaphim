from vietdub.sync import atempo_filters, speed_factor_for_duration


def test_speed_factor_for_duration():
    assert speed_factor_for_duration(actual_ms=2000, target_ms=1000) == 2.0
    assert speed_factor_for_duration(actual_ms=1000, target_ms=2000) == 0.5


def test_atempo_filters_split_large_factor():
    assert atempo_filters(4.0) == ["atempo=2.0", "atempo=2.0"]
    assert atempo_filters(0.25) == ["atempo=0.5", "atempo=0.5"]
