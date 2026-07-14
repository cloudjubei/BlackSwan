import datetime

from scripts.backfill_klines import (
    check_rows,
    expected_bars,
    kline_from_csv_row,
    last_month_on_disk,
    latest_complete_month,
    missing_bars_between,
    month_filename,
    months_between,
    normalize_epoch_ms,
    validate_month,
)

_JUN1_2024 = 1717200000000
_CSV_ROW = [
    "1717200000000", "67540.01000000", "67900.00000000", "67428.44000000", "67766.85000000",
    "8837.66133000", "1717286399999", "598305207.99320460", "638484", "4235.27045000",
    "286752060.40289290", "0",
]


def _row(ts, interval_ms=60_000, price="100.00000000"):
    return {
        "timestamp": ts,
        "price_open": price,
        "price_high": price,
        "price_low": price,
        "price": price,
        "volume": "1.00000000",
        "timestamp_close": ts + interval_ms - 1,
        "asset_volume_quote": "100.00000000",
        "trades_number": 1,
        "asset_volume_taker_base": "0.50000000",
        "asset_volume_taker_quote": "50.00000000",
    }


# --- normalize_epoch_ms ---


def test_normalize_epoch_ms_passes_millisecond_stamps_through():
    assert normalize_epoch_ms("1717200000000") == 1717200000000
    assert normalize_epoch_ms(1717286399999) == 1717286399999


def test_normalize_epoch_ms_converts_microsecond_stamps():
    # Binance vision archives switched open/close times to MICROSECONDS from 2025-01-01.
    assert normalize_epoch_ms("1735689600000000") == 1735689600000
    assert normalize_epoch_ms(1735689599999999) == 1735689599999


# --- kline_from_csv_row ---


def test_kline_from_csv_row_1m_matches_raw_minute_shape():
    kline = kline_from_csv_row(_CSV_ROW, "BTCUSDT", "1m")
    assert kline == {
        "tokenPair": "BTCUSDT",
        "interval": "1m",
        "timestamp": 1717200000000,
        "price_open": "67540.01000000",
        "price_high": "67900.00000000",
        "price_low": "67428.44000000",
        "price": "67766.85000000",
        "volume": "8837.66133000",
        "timestamp_close": 1717286399999,
        "asset_volume_quote": "598305207.99320460",
        "trades_number": 638484,
        "asset_volume_taker_base": "4235.27045000",
        "asset_volume_taker_quote": "286752060.40289290",
    }


def test_kline_from_csv_row_coarse_intervals_omit_pair_fields():
    # Raw 1h/1d files on disk carry no tokenPair/interval keys — only the 1m files do.
    kline = kline_from_csv_row(_CSV_ROW, "BTCUSDT", "1d")
    assert "tokenPair" not in kline
    assert "interval" not in kline
    assert kline["timestamp"] == 1717200000000
    assert kline["trades_number"] == 638484


def test_kline_from_csv_row_normalizes_microsecond_timestamps():
    row = list(_CSV_ROW)
    row[0] = "1735689600000000"
    row[6] = "1735689659999999"
    kline = kline_from_csv_row(row, "BTCUSDT", "1m")
    assert kline["timestamp"] == 1735689600000
    assert kline["timestamp_close"] == 1735689659999


# --- month arithmetic ---


def test_months_between_spans_year_boundary():
    assert months_between((2024, 11), (2025, 2)) == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]


def test_months_between_single_month():
    assert months_between((2026, 6), (2026, 6)) == [(2026, 6)]


def test_months_between_empty_when_start_after_end():
    assert months_between((2026, 7), (2026, 6)) == []


def test_month_filename_is_not_zero_padded():
    # binance/ files use bare month numbers (BTCUSDT-1m-2022-1.json), never 01.
    assert month_filename("BTCUSDT", "1m", 2026, 6) == "BTCUSDT-1m-2026-6.json"
    assert month_filename("ETHUSDT", "1d", 2025, 12) == "ETHUSDT-1d-2025-12.json"


def test_latest_complete_month_is_previous_month():
    assert latest_complete_month(datetime.date(2026, 7, 14)) == (2026, 6)
    assert latest_complete_month(datetime.date(2026, 1, 3)) == (2025, 12)


def test_expected_bars_per_interval():
    assert expected_bars("1m", 2024, 2) == 29 * 1440
    assert expected_bars("1h", 2026, 6) == 30 * 24
    assert expected_bars("1d", 2026, 2) == 28


# --- last_month_on_disk ---


def test_last_month_on_disk_picks_numeric_max(tmp_path):
    for name in ["BTCUSDT-1m-2024-2.json", "BTCUSDT-1m-2024-10.json", "BTCUSDT-1h-2024-12.json"]:
        (tmp_path / name).write_text("[]")
    assert last_month_on_disk(str(tmp_path), "BTCUSDT", "1m") == (2024, 10)


def test_last_month_on_disk_none_when_absent(tmp_path):
    assert last_month_on_disk(str(tmp_path), "BTCUSDT", "1m") is None


# --- check_rows / validate_month ---


def test_check_rows_accepts_contiguous_month():
    rows = [_row(_JUN1_2024 + i * 60_000) for i in range(10)]
    assert check_rows(rows) == []


def test_check_rows_flags_duplicate_timestamps():
    rows = [_row(_JUN1_2024), _row(_JUN1_2024)]
    assert any("duplicate" in e for e in check_rows(rows))


def test_check_rows_flags_non_monotonic_timestamps():
    rows = [_row(_JUN1_2024 + 60_000), _row(_JUN1_2024)]
    assert any("monotonic" in e for e in check_rows(rows))


def test_check_rows_flags_unparseable_and_non_positive_prices():
    bad = _row(_JUN1_2024)
    bad["price"] = "not-a-number"
    assert any("numeric" in e for e in check_rows([bad]))
    zero = _row(_JUN1_2024 + 60_000)
    zero["price_low"] = "0.00000000"
    assert any("positive" in e for e in check_rows([zero]))


def test_check_rows_flags_empty_month():
    assert any("empty" in e for e in check_rows([]))


def test_validate_month_reports_gap_as_warning_not_error():
    rows = [_row(_JUN1_2024 + i * 60_000) for i in range(10)]
    del rows[5]
    errors, warnings = validate_month(rows, "1m", 2024, 6)
    assert errors == []
    assert any("missing" in w for w in warnings)


def test_validate_month_flags_bar_outside_month_as_error():
    rows = [_row(_JUN1_2024 - 60_000)]
    errors, _ = validate_month(rows, "1m", 2024, 6)
    assert any("outside" in e for e in errors)


def test_validate_month_clean_month_has_no_findings():
    rows = [_row(_JUN1_2024 + i * 60_000) for i in range(30 * 1440)]
    errors, warnings = validate_month(rows, "1m", 2024, 6)
    assert errors == []
    assert warnings == []


# --- missing_bars_between ---


def test_missing_bars_between_contiguous_months_is_zero():
    prev_last = _row(_JUN1_2024 - 60_000)
    first = _row(_JUN1_2024)
    assert missing_bars_between(prev_last, first, "1m") == 0


def test_missing_bars_between_counts_the_hole():
    prev_last = _row(_JUN1_2024 - 60_000)
    first = _row(_JUN1_2024 + 3 * 60_000)
    assert missing_bars_between(prev_last, first, "1m") == 3
