import os

import pytest

from trainer import data_inventory
from trainer.data_inventory import available_assets, has_data, scan_inventory


def _touch(root, name):
    """Create an empty file `name` under `root` (a tmp dir standing in for binance/)."""
    path = os.path.join(root, name)
    with open(path, "w") as f:
        f.write("[]")
    return path


# --- scan_inventory ---


def test_scan_inventory_empty_dir_returns_empty(tmp_path):
    assert scan_inventory(str(tmp_path)) == {}


def test_scan_inventory_groups_timeframes_per_symbol(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1h-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1d-2020-09.json")
    _touch(tmp_path, "ETHUSDT-1m-2021-01.json")
    inv = scan_inventory(str(tmp_path))
    assert inv == {"BTCUSDT": ["1d", "1h", "1m"], "ETHUSDT": ["1m"]}


def test_scan_inventory_dedupes_repeated_timeframe_across_months(tmp_path):
    # The same timeframe appearing for several months collapses to a single entry (set-backed).
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1m-2020-10.json")
    _touch(tmp_path, "BTCUSDT-1m-2020-11.json")
    inv = scan_inventory(str(tmp_path))
    assert inv == {"BTCUSDT": ["1m"]}


def test_scan_inventory_timeframes_sorted(tmp_path):
    # Inserted out of lexical order; the returned list must be sorted.
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1d-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1h-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1w-2020-09.json")
    assert scan_inventory(str(tmp_path))["BTCUSDT"] == ["1d", "1h", "1m", "1w"]


def test_scan_inventory_symbols_sorted(tmp_path):
    _touch(tmp_path, "ZECUSDT-1m-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "ETHUSDT-1m-2020-09.json")
    assert list(scan_inventory(str(tmp_path)).keys()) == ["BTCUSDT", "ETHUSDT", "ZECUSDT"]


@pytest.mark.parametrize(
    "bad_name",
    [
        "btcusdt-1m-2020-09.json",  # lowercase symbol not allowed ([A-Z0-9]+)
        "BTC_USDT-1m-2020-09.json",  # underscore in symbol not allowed
        "BTCUSDT-1m-2020.json",  # missing -month segment
        "BTCUSDT-1y-2020-09.json",  # 'y' not in the m/h/d/w unit set
        "BTCUSDT-1m-2020-09.csv",  # wrong extension
        "README.json",  # not a kline filename at all
    ],
)
def test_scan_inventory_ignores_non_matching_files(tmp_path, bad_name):
    _touch(tmp_path, bad_name)
    assert scan_inventory(str(tmp_path)) == {}


def test_scan_inventory_accepts_multidigit_interval(tmp_path):
    # The regex allows any number of digits before the unit, e.g. 15m.
    _touch(tmp_path, "FOOUSDT-15m-2020-01.json")
    assert scan_inventory(str(tmp_path)) == {"FOOUSDT": ["15m"]}


def test_scan_inventory_mixes_good_and_bad_files(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "garbage.json")
    _touch(tmp_path, "BTCUSDT-1h-bad.json")
    assert scan_inventory(str(tmp_path)) == {"BTCUSDT": ["1m"]}


def test_scan_inventory_default_root_is_binance(monkeypatch):
    # When called with no root it globs the "binance" dir; assert that is the path globbed.
    captured = {}

    def fake_glob(pattern):
        captured["pattern"] = pattern
        return []

    monkeypatch.setattr(data_inventory.glob, "glob", fake_glob)
    scan_inventory()
    assert captured["pattern"] == os.path.join("binance", "*.json")


# --- available_assets ---


def test_available_assets_filters_by_timeframe(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    _touch(tmp_path, "BTCUSDT-1h-2020-09.json")
    _touch(tmp_path, "ETHUSDT-1m-2020-09.json")
    # Only BTCUSDT has 1h; both have 1m.
    assert available_assets("1h", str(tmp_path)) == ["BTCUSDT"]
    assert available_assets("1m", str(tmp_path)) == ["BTCUSDT", "ETHUSDT"]


def test_available_assets_empty_when_no_match(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    assert available_assets("1d", str(tmp_path)) == []


def test_available_assets_results_sorted(tmp_path):
    _touch(tmp_path, "ZECUSDT-1m-2020-09.json")
    _touch(tmp_path, "AAAUSDT-1m-2020-09.json")
    assert available_assets("1m", str(tmp_path)) == ["AAAUSDT", "ZECUSDT"]


# --- has_data ---


def test_has_data_true_when_present(tmp_path):
    _touch(tmp_path, "BTCUSDT-1d-2020-09.json")
    assert has_data("BTCUSDT", "1d", str(tmp_path)) is True


def test_has_data_false_for_missing_timeframe(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    assert has_data("BTCUSDT", "1h", str(tmp_path)) is False


def test_has_data_false_for_unknown_symbol(tmp_path):
    _touch(tmp_path, "BTCUSDT-1m-2020-09.json")
    assert has_data("DOGEUSDT", "1m", str(tmp_path)) is False
