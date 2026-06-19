import io
import json
import os

import pandas as pd
import pytest

from trainer.derive_cache import _OHLCV_SUM, _to_ms, derive_bars, ensure_derived

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000
# A 1m timestamp floored into an hour bucket; base % 3.6M = 1_600_000 so the bucket floors BELOW base.
_BASE = 1_600_000_000_000


def _raw_minutes(n, start=_BASE, step_ms=60_000, **overrides):
    """Build an n-row raw-1m frame shaped like the kline JSON the providers read (timestamp/
    timestamp_close as datetime64 — what pd.read_json produces — and float OHLCV)."""
    rows = []
    for i in range(n):
        ts = start + i * step_ms
        row = {
            "timestamp": ts,
            "timestamp_close": ts + step_ms - 1,
            "price_open": 100.0 + i,
            "price_high": 110.0 + i,
            "price_low": 90.0 - i,
            "price": 105.0 + i,
            "volume": 1.0,
            "asset_volume_quote": 2.0,
            "trades_number": 3,
            "asset_volume_taker_base": 0.5,
            "asset_volume_taker_quote": 0.6,
        }
        row.update(overrides)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp_close"] = pd.to_datetime(df["timestamp_close"], unit="ms")
    return df


# --- _to_ms ---


def test_to_ms_from_datetime64():
    # pd.read_json turns ms-int timestamps into datetime64; _to_ms must round-trip back to ms.
    s = pd.to_datetime(pd.Series([_BASE, _BASE + 60_000]), unit="ms")
    assert _to_ms(s).tolist() == [_BASE, _BASE + 60_000]


def test_to_ms_bare_ints_are_treated_as_ms():
    # Honouring the docstring's "or raw ms ints": a bare integer/numeric ms Series is treated as
    # milliseconds (not nanoseconds), so it round-trips back to the same ms value.
    s = pd.Series([_BASE])
    assert _to_ms(s).tolist() == [_BASE]
    # A multi-row numeric series also round-trips ms-for-ms.
    multi = pd.Series([_BASE, _BASE + 60_000])
    assert _to_ms(multi).tolist() == [_BASE, _BASE + 60_000]


# --- derive_bars: aggregation contract (open=first, high=max, low=min, close=last, sums) ---


def test_derive_bars_aggregates_one_bucket():
    df = _raw_minutes(4)  # 4 minutes, all inside one hour bucket
    out = derive_bars(df, _HOUR_MS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["price_open"] == 100.0  # first
    assert row["price_high"] == 113.0  # max over i=0..3 (110+3)
    assert row["price_low"] == 87.0  # min over i=0..3 (90-3)
    assert row["price"] == 108.0  # last close (105+3)


def test_derive_bars_sums_volume_columns():
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS).iloc[0]
    assert out["volume"] == pytest.approx(4.0)  # 1.0 * 4
    assert out["asset_volume_quote"] == pytest.approx(8.0)
    assert out["trades_number"] == 12  # 3 * 4
    assert out["asset_volume_taker_base"] == pytest.approx(2.0)
    assert out["asset_volume_taker_quote"] == pytest.approx(2.4)


def test_derive_bars_bucket_timestamp_is_floored_open():
    # The output timestamp is the floored bucket boundary, not the first row's raw timestamp.
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS)
    assert out["timestamp"].iloc[0] == (_BASE // _HOUR_MS) * _HOUR_MS
    assert out["timestamp"].iloc[0] < _BASE  # base is mid-bucket, so the boundary is earlier


def test_derive_bars_close_is_last_close_ms():
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS)
    # timestamp_close = the last minute's close ms (start of minute 3 + 59_999).
    assert out["timestamp_close"].iloc[0] == _BASE + 3 * 60_000 + 59_999


def test_derive_bars_splits_across_buckets():
    # 90 one-minute bars span 2 hour buckets (60 + 30). Expect exactly 2 derived rows.
    df = _raw_minutes(90, start=(_BASE // _HOUR_MS) * _HOUR_MS)
    out = derive_bars(df, _HOUR_MS)
    assert len(out) == 2
    assert out["timestamp"].iloc[1] - out["timestamp"].iloc[0] == _HOUR_MS
    # First bucket sums 60 minutes of volume, the second sums the remaining 30.
    assert out["volume"].iloc[0] == pytest.approx(60.0)
    assert out["volume"].iloc[1] == pytest.approx(30.0)


def test_derive_bars_daily_bucket():
    # One day's worth of buckets: two consecutive minutes still land in the same 1d bucket.
    df = _raw_minutes(2, start=(_BASE // _DAY_MS) * _DAY_MS)
    out = derive_bars(df, _DAY_MS)
    assert len(out) == 1
    assert out["timestamp"].iloc[0] == (_BASE // _DAY_MS) * _DAY_MS


def test_derive_bars_buckets_are_sorted():
    # Feed rows out of chronological order; groupby(sort=True) must order the buckets ascending.
    aligned = (_BASE // _HOUR_MS) * _HOUR_MS
    later = _raw_minutes(1, start=aligned + _HOUR_MS)
    earlier = _raw_minutes(1, start=aligned)
    df = pd.concat([later, earlier], ignore_index=True)
    out = derive_bars(df, _HOUR_MS)
    assert out["timestamp"].tolist() == sorted(out["timestamp"].tolist())


def test_derive_bars_carries_no_indicators_column():
    # Contract: derived bars carry NO indicators dict (indicators are a separate data-mine artifact).
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS)
    assert "indicators" not in out.columns


def test_derive_bars_output_columns_match_kline_shape():
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS)
    expected = {"timestamp", "price_open", "price_high", "price_low", "price", "timestamp_close", *_OHLCV_SUM}
    assert set(out.columns) == expected


def test_derive_bars_index_reset():
    df = _raw_minutes(90, start=(_BASE // _HOUR_MS) * _HOUR_MS)
    out = derive_bars(df, _HOUR_MS)
    assert list(out.index) == [0, 1]


def test_derive_bars_coerces_string_numerics():
    # Raw kline JSON stores OHLCV as decimal strings; derive_bars must coerce them via to_numeric.
    df = _raw_minutes(2, price_open="100.5", price_high="200.0", volume="2.5")
    out = derive_bars(df, _HOUR_MS).iloc[0]
    assert out["price_open"] == pytest.approx(100.5)
    assert out["price_high"] == pytest.approx(200.0)
    assert out["volume"] == pytest.approx(5.0)  # "2.5" summed twice


def test_derive_bars_coerces_garbage_to_nan():
    df = _raw_minutes(1, price_open="not-a-number")
    out = derive_bars(df, _HOUR_MS)
    assert pd.isna(out["price_open"].iloc[0])


def test_derive_bars_omits_absent_volume_columns():
    # Drop the taker columns; the `if c in d.columns` guard must skip them without error.
    df = _raw_minutes(2).drop(columns=["asset_volume_taker_base", "asset_volume_taker_quote"])
    out = derive_bars(df, _HOUR_MS)
    assert "asset_volume_taker_base" not in out.columns
    assert "volume" in out.columns


def test_derive_bars_empty_typed_frame_returns_empty_with_columns():
    # An empty frame that still carries the kline columns (typed) derives to an empty frame
    # that keeps the output columns — groupby over no rows yields zero buckets.
    cols = ["timestamp", "timestamp_close", "price_open", "price_high", "price_low", "price", *_OHLCV_SUM]
    df = pd.DataFrame({c: pd.Series([], dtype="float64") for c in cols})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_close"] = pd.to_datetime(df["timestamp_close"])
    out = derive_bars(df, _HOUR_MS)
    assert len(out) == 0
    assert "timestamp" in out.columns and "price" in out.columns


def test_derive_bars_truly_empty_frame_returns_empty_typed_frame():
    # A column-less empty frame (what pd.read_json("[]") of an empty month file produces) is
    # short-circuited to an empty frame with the kline output columns — no KeyError('timestamp').
    empty = pd.read_json(io.StringIO("[]"))
    out = derive_bars(empty, _HOUR_MS)
    assert len(out) == 0
    expected = {"timestamp", "price_open", "price_high", "price_low", "price", "timestamp_close", *_OHLCV_SUM}
    assert set(out.columns) == expected


def test_derive_bars_does_not_mutate_input():
    df = _raw_minutes(4)
    before = df.copy(deep=True)
    derive_bars(df, _HOUR_MS)
    # derive_bars copies the frame internally; the caller's df keeps its original columns/values.
    assert "_bucket" not in df.columns
    pd.testing.assert_frame_equal(df, before)


def test_derive_bars_roundtrips_through_json():
    # What a run actually reads back: to_json(records) then read_json keeps the kline columns.
    df = _raw_minutes(4)
    out = derive_bars(df, _HOUR_MS)
    back = pd.read_json(io.StringIO(out.to_json(orient="records")))
    assert set(back.columns) == set(out.columns)
    assert "indicators" not in back.columns


# --- ensure_derived: caching / freshness / content-addressing ---


def _write_source(binance_dir, symbol, year, month, df):
    """Persist a raw-1m frame as the binance/{symbol}-1m-{year}-{month}.json source file."""
    src = os.path.join(binance_dir, f"{symbol}-1m-{year}-{month}.json")
    # Store ms ints (round-tripped via to_json) so pd.read_json inside ensure_derived re-parses them.
    df.to_json(src, orient="records")
    return src


def _setup(tmp_path, monkeypatch):
    """ensure_derived resolves the source path RELATIVE to cwd (binance/...), so run inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    binance = tmp_path / "binance"
    binance.mkdir()
    cache = tmp_path / "derived"
    return str(binance), str(cache)


def test_ensure_derived_writes_cache_file(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(120))
    out = ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    assert out == [os.path.join(cache, "BTCUSDT-1h-2020-09.json")]
    assert os.path.exists(out[0])


def test_ensure_derived_creates_cache_dir(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    assert not os.path.exists(cache)
    _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(60))
    ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    assert os.path.isdir(cache)


def test_ensure_derived_skips_missing_source(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(60))
    # 2020-10 has no source file -> dropped from the returned paths.
    out = ensure_derived("BTCUSDT", [("2020", "09"), ("2020", "10")], "1h", cache_dir=cache)
    assert out == [os.path.join(cache, "BTCUSDT-1h-2020-09.json")]


def test_ensure_derived_returns_empty_when_no_sources(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    out = ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    assert out == []


def test_ensure_derived_reuses_fresh_cache(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(60))
    out = ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    dst = out[0]
    first_mtime = os.path.getmtime(dst)
    # Make the cache strictly newer than the source so the freshness check (dst_mtime >= src_mtime) holds.
    os.utime(dst, (first_mtime + 100, first_mtime + 100))
    fresh = os.path.getmtime(dst)
    ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    # No re-derivation: mtime is unchanged.
    assert os.path.getmtime(dst) == fresh


def test_ensure_derived_rederives_when_source_newer(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    src = _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(60))
    out = ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    dst = out[0]
    # Backdate the cache so its source is "newer" -> stale -> must re-derive.
    src_mtime = os.path.getmtime(src)
    os.utime(dst, (src_mtime - 100, src_mtime - 100))
    stale = os.path.getmtime(dst)
    ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    assert os.path.getmtime(dst) > stale


def test_ensure_derived_content_matches_derive_bars(tmp_path, monkeypatch):
    # The cached file content must equal derive_bars applied directly to the source.
    binance, cache = _setup(tmp_path, monkeypatch)
    df = _raw_minutes(120, start=(_BASE // _HOUR_MS) * _HOUR_MS)
    src = _write_source(binance, "BTCUSDT", "2020", "09", df)
    out = ensure_derived("BTCUSDT", [("2020", "09")], "1h", cache_dir=cache)
    # Re-reading the cached JSON turns the integer-ms timestamp back into datetime64 (what a run
    # sees); compare via _to_ms so we match ms-against-ms.
    cached = pd.read_json(io.StringIO(open(out[0]).read()))
    expected = derive_bars(pd.read_json(io.StringIO(open(src).read())), _HOUR_MS)
    assert _to_ms(cached["timestamp"]).tolist() == expected["timestamp"].tolist()
    assert cached["volume"].tolist() == pytest.approx(expected["volume"].tolist())
    assert cached["price_high"].tolist() == pytest.approx(expected["price_high"].tolist())


def test_ensure_derived_filename_encodes_symbol_fidelity_year_month(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    _write_source(binance, "BTCUSDT", "2024", "03", _raw_minutes(60))
    out = ensure_derived("BTCUSDT", [("2024", "03")], "1d", cache_dir=cache)
    assert os.path.basename(out[0]) == "BTCUSDT-1d-2024-03.json"


def test_ensure_derived_unknown_fidelity_raises(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        ensure_derived("BTCUSDT", [("2020", "09")], "5m", cache_dir=cache)


def test_ensure_derived_multiple_months(tmp_path, monkeypatch):
    binance, cache = _setup(tmp_path, monkeypatch)
    _write_source(binance, "BTCUSDT", "2020", "09", _raw_minutes(60))
    _write_source(binance, "BTCUSDT", "2020", "10", _raw_minutes(60))
    out = ensure_derived("BTCUSDT", [("2020", "09"), ("2020", "10")], "1h", cache_dir=cache)
    assert [os.path.basename(p) for p in out] == [
        "BTCUSDT-1h-2020-09.json",
        "BTCUSDT-1h-2020-10.json",
    ]
    assert all(os.path.exists(p) for p in out)
