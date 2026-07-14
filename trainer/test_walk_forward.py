import pytest

from trainer.walk_forward import (
    DEFAULT_WALK_FORWARD_WINDOW,
    resolve_walk_forward_window,
    walk_forward_window_ids,
)


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


def test_window_ids_are_the_oos_years():
    assert walk_forward_window_ids() == ["2022", "2023", "2024", "2025", "alt-2024", "alt-2025"]
