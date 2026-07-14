from scripts.backfill_stocks import (
    group_by_month,
    kline_from_daily_row,
    parse_stooq_csv,
    validate_stock_month,
)

_CSV = """Date,Open,High,Low,Close,Volume
2024-06-03,194.635,201.885,194.14,201.3,90180079
2024-06-04,199.98,203.09,198.85,202.65,68387094
"""


# --- parse_stooq_csv ---


def test_parse_stooq_csv_yields_field_rows():
    rows = parse_stooq_csv(_CSV)
    assert rows == [
        ["2024-06-03", "194.635", "201.885", "194.14", "201.3", "90180079"],
        ["2024-06-04", "199.98", "203.09", "198.85", "202.65", "68387094"],
    ]


def test_parse_stooq_csv_skips_blank_and_incomplete_lines():
    assert parse_stooq_csv("Date,Open,High,Low,Close,Volume\n\n2024-06-03,1,2\n") == []


def test_parse_stooq_csv_defaults_missing_volume_to_zero():
    rows = parse_stooq_csv("Date,Open,High,Low,Close,Volume\n2024-06-03,1,2,0.5,1.5,\n")
    assert rows == [["2024-06-03", "1", "2", "0.5", "1.5", "0"]]


# --- kline_from_daily_row ---


def test_kline_from_daily_row_maps_to_daily_kline_shape():
    kline = kline_from_daily_row(["2024-06-03", "194.635", "201.885", "194.14", "201.3", "90180079"])
    assert kline == {
        "timestamp": 1717372800000,
        "price_open": "194.63500000",
        "price_high": "201.88500000",
        "price_low": "194.14000000",
        "price": "201.30000000",
        "volume": "90180079.00000000",
        "timestamp_close": 1717372800000 + 86_400_000 - 1,
        "asset_volume_quote": "18153249902.70000076",
        "trades_number": 0,
        "asset_volume_taker_base": "45090039.50000000",
        "asset_volume_taker_quote": "9076624951.35000038",
    }


def test_kline_from_daily_row_neutral_taker_ratio_is_half():
    kline = kline_from_daily_row(["2024-06-03", "1", "1", "1", "2", "10"])
    assert float(kline["asset_volume_taker_base"]) / float(kline["volume"]) == 0.5


# --- group_by_month ---


def test_group_by_month_splits_on_calendar_month():
    klines = [
        kline_from_daily_row(["2024-06-28", "1", "2", "0.5", "1.5", "10"]),
        kline_from_daily_row(["2024-07-01", "1", "2", "0.5", "1.5", "10"]),
        kline_from_daily_row(["2024-07-02", "1", "2", "0.5", "1.5", "10"]),
    ]
    grouped = group_by_month(klines)
    assert list(grouped.keys()) == [(2024, 6), (2024, 7)]
    assert len(grouped[(2024, 7)]) == 2


# --- validate_stock_month ---


def _month_of_weekdays(year, month, days):
    return [
        kline_from_daily_row([f"{year}-{month:02d}-{d:02d}", "1", "2", "0.5", "1.5", "10"])
        for d in days
    ]


def test_validate_stock_month_weekend_holes_are_normal():
    # 2024-06: trading days only (3rd-7th, 10th-14th, ...) — weekend holes must not be findings.
    days = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 20, 21, 24, 25, 26, 27, 28]
    errors, warnings = validate_stock_month(_month_of_weekdays(2024, 6, days), 2024, 6)
    assert errors == []
    assert warnings == []


def test_validate_stock_month_flags_suspiciously_few_bars():
    errors, warnings = validate_stock_month(_month_of_weekdays(2024, 6, [3, 4, 5]), 2024, 6)
    assert errors == []
    assert any("bars" in w for w in warnings)


def test_validate_stock_month_flags_structural_errors():
    rows = _month_of_weekdays(2024, 6, [3, 3])
    errors, _ = validate_stock_month(rows, 2024, 6)
    assert any("duplicate" in e for e in errors)


def test_parse_stooq_csv_rejects_non_date_rows():
    # A bot-challenge page split on commas can still have >=6 fields; only date-shaped rows count.
    challenge = '(async()=>{const c="AAAA",d=4,t="0",e=1,f=2,g=3\n'
    assert parse_stooq_csv(challenge) == []
