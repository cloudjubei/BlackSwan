import pytest

from trainer.walk_forward import (
    DEFAULT_WALK_FORWARD_WINDOW,
    resolve_walk_forward_window,
    walk_forward_window_ids,
    windows_supported,
)


def test_2026_windows_exist_now_that_2026_data_is_on_disk():
    ids = walk_forward_window_ids()
    assert "2026" in ids and "alt-2026" in ids


def test_windows_supported_is_derived_from_the_asset_data_range():
    # An altcoin (2022-01 →) supports the alt-* windows but NOT the plain ones that train from 2020.
    alt = windows_supported("2022-01", "2026-06")
    assert "alt-2024" in alt
    assert "2024" not in alt
    # A deep-history asset (2017 →) supports everything, incl. the new 2026 window.
    full = windows_supported("2017-08", "2026-06")
    assert {"2024", "2026", "alt-2024"} <= set(full)
    # An asset whose data ends mid-2023 can't run a 2024 test window.
    short = windows_supported("2018-01", "2023-06")
    assert "2023" in short and "2024" not in short


def test_default_window_reproduces_the_legacy_2020_2023_vs_2024_split():
    train, test, meta = resolve_walk_forward_window({})
    assert train[0] == (2020, 1)
    assert train[-1] == (2023, 12)
    assert len(train) == 4 * 12
    assert test == [(2024, m) for m in range(1, 13)]
    assert meta["walk_forward_window"] == DEFAULT_WALK_FORWARD_WINDOW == "2024"


def test_absent_cfg_matches_empty_cfg_default():
    assert resolve_walk_forward_window() == resolve_walk_forward_window({})


def test_expanding_train_per_window():
    train22, test22, _ = resolve_walk_forward_window({"walk_forward_window": "2022"})
    assert train22[0] == (2020, 1)
    assert train22[-1] == (2021, 12)
    assert test22 == [(2022, m) for m in range(1, 13)]

    train23, test23, _ = resolve_walk_forward_window({"walk_forward_window": "2023"})
    assert train23[-1] == (2022, 12)
    assert test23 == [(2023, m) for m in range(1, 13)]


def test_windows_are_out_of_sample_last_train_month_before_first_test_month():
    for wid in walk_forward_window_ids():
        train, test, _ = resolve_walk_forward_window({"walk_forward_window": wid})
        assert max(train) < min(test)


def test_meta_carries_train_and_test_ranges():
    _, _, meta = resolve_walk_forward_window({"walk_forward_window": "2023"})
    assert meta["train_from"] == "2020-01"
    assert meta["train_to"] == "2022-12"
    assert meta["test_from"] == "2023-01"
    assert meta["test_to"] == "2023-12"


def test_unknown_window_fails_fast():
    with pytest.raises(SystemExit):
        resolve_walk_forward_window({"walk_forward_window": "1999"})


def test_2025_window_extends_the_expanding_train_through_2024():
    train, test, meta = resolve_walk_forward_window({"walk_forward_window": "2025"})
    assert train[0] == (2020, 1)
    assert train[-1] == (2024, 12)
    assert test == [(2025, m) for m in range(1, 13)]
    assert meta["test_to"] == "2025-12"


def test_alt_windows_train_from_2022_for_assets_whose_history_starts_there():
    train, test, meta = resolve_walk_forward_window({"walk_forward_window": "alt-2024"})
    assert train[0] == (2022, 1)
    assert train[-1] == (2023, 12)
    assert test == [(2024, m) for m in range(1, 13)]
    assert meta["train_from"] == "2022-01"

    train25, test25, _ = resolve_walk_forward_window({"walk_forward_window": "alt-2025"})
    assert train25[0] == (2022, 1)
    assert train25[-1] == (2024, 12)
    assert test25 == [(2025, m) for m in range(1, 13)]


def test_stk_windows_train_from_2018_for_deep_history_assets():
    # stk-* windows train from 2018 — deeper than the plain windows' 2020 start — for assets whose data
    # reaches back that far (stocks from 2018-01, BTC from 2017-08). The gate is DATA RANGE, not asset class.
    train, test, meta = resolve_walk_forward_window({"walk_forward_window": "stk-2024"})
    assert train[0] == (2018, 1)
    assert train[-1] == (2023, 12)
    assert test == [(2024, m) for m in range(1, 13)]
    assert meta["train_from"] == "2018-01"

    # windows_supported gates by on-disk range: a 2018-start asset gets stk-*, a 2022-start altcoin does not.
    deep = windows_supported("2018-01", "2026-06")
    assert "stk-2024" in deep
    assert "stk-2024" not in windows_supported("2022-01", "2026-06")  # altcoin can't train from 2018


def test_window_ids_include_the_fixed_and_open_ended_windows():
    ids = walk_forward_window_ids()
    for wid in ["2022", "2023", "2024", "2025", "alt-2024", "alt-2025"]:
        assert wid in ids
    for wid in ["oos-2024", "oos-2025", "alt-oos-2024", "alt-oos-2025"]:
        assert wid in ids


def test_open_ended_window_tests_from_cutoff_through_a_far_cap():
    # An oos-* window tests from the training cutoff to a far-future cap; config_builder's file-existence
    # filter truncates it to the latest data on disk, so the run tests ALL data after the cutoff.
    train, test, meta = resolve_walk_forward_window({"walk_forward_window": "oos-2024"})
    assert train[0] == (2020, 1)
    assert train[-1] == (2023, 12)
    assert test[0] == (2024, 1)
    assert test[-1][0] >= 2035  # runs to the far cap
    assert meta["test_from"] == "2024-01"
    assert meta["test_to"] == "latest"


def test_alt_open_ended_window_trains_from_2022():
    train, test, _ = resolve_walk_forward_window({"walk_forward_window": "alt-oos-2025"})
    assert train[0] == (2022, 1)
    assert train[-1] == (2024, 12)
    assert test[0] == (2025, 1)
    assert test[-1][0] >= 2035
