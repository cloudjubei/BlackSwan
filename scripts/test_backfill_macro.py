import scripts.backfill_macro as bm


# --- extract_macro_observations: FRED output_type=4 (initial-release vintages) -> point-in-time ---


def test_extract_macro_observations_stamps_release_date_from_realtime_start():
    # output_type=4 returns each value AS INITIALLY RELEASED, with realtime_start = the date it became
    # public. That realtime_start IS the point-in-time release date (never the reference `date`).
    payload = {"observations": [
        {"date": "2024-01-01", "realtime_start": "2024-02-02", "realtime_end": "9999-12-31", "value": "3.7"},
    ]}
    obs = bm.extract_macro_observations(payload)
    assert obs == [{"refPeriod": "2024-01-01", "releaseDate": "2024-02-02", "value": 3.7, "vintage": "2024-02-02"}]


def test_extract_macro_observations_skips_fred_missing_marker():
    payload = {"observations": [{"date": "2024-01-01", "realtime_start": "2024-02-02", "value": "."}]}
    assert bm.extract_macro_observations(payload) == []


def test_extract_macro_observations_skips_non_numeric():
    payload = {"observations": [{"date": "2024-01-01", "realtime_start": "2024-02-02", "value": "N/A"}]}
    assert bm.extract_macro_observations(payload) == []


def test_extract_macro_observations_empty_or_malformed():
    assert bm.extract_macro_observations({}) == []
    assert bm.extract_macro_observations(None) == []


def test_extract_macro_observations_multiple_ordered():
    payload = {"observations": [
        {"date": "2024-01-01", "realtime_start": "2024-02-02", "value": "3.7"},
        {"date": "2024-02-01", "realtime_start": "2024-03-08", "value": "3.9"},
    ]}
    obs = bm.extract_macro_observations(payload)
    assert [o["value"] for o in obs] == [3.7, 3.9]
    assert [o["releaseDate"] for o in obs] == ["2024-02-02", "2024-03-08"]


# --- extract_macro_observations: daily non-revised rates (output_type=1, stamped as-of) ---


def test_extract_asof_daily_rate_stamps_release_at_reference_plus_publish_lag():
    # DFF (daily effective fed funds) is NEVER revised but exceeds the output_type=4 vintage cap, so it is
    # fetched as standard observations and stamped leak-safe: its value for day D is published the NEXT day.
    payload = {"observations": [{"date": "2024-03-15", "realtime_start": "2026-07-26", "value": "5.33"}]}
    obs = bm.extract_macro_observations(payload, "DFF")
    assert obs == [{"refPeriod": "2024-03-15", "releaseDate": "2024-03-16", "value": 5.33, "vintage": "2024-03-16"}]


def test_extract_asof_same_day_rate_uses_the_reference_date():
    # H.15 yields (DGS10/DFII10/T10Y2Y) print same-day after the close (pit_fusion places the intraday time).
    payload = {"observations": [{"date": "2024-03-15", "realtime_start": "2026-07-26", "value": "4.29"}]}
    obs = bm.extract_macro_observations(payload, "DGS10")
    assert obs[0]["releaseDate"] == "2024-03-15"


# --- observations_url is category-aware (vintage cap vs non-revised daily rate) ---


def test_observations_url_vintage_series_requests_initial_release_full_history():
    # Revised series -> initial-release vintages across ALL history (the default realtime window is TODAY,
    # which has no vintages and 400s — the bug this fixes).
    url = bm.observations_url("UNRATE", "KEY123")
    assert "series_id=UNRATE" in url
    assert "api_key=KEY123" in url
    assert "file_type=json" in url
    assert "output_type=4" in url  # initial release only — NOT the latest-revised series (a silent leak)
    assert "realtime_start=1776-07-04" in url
    assert "realtime_end=9999-12-31" in url


def test_observations_url_daily_rate_uses_standard_observations():
    # Daily market rates blow the output_type=4 vintage cap; they are not revised, so standard observations
    # (output_type=1) stamped as-of are equally leak-safe.
    url = bm.observations_url("DFF", "KEY123")
    assert "output_type=1" in url
    assert "output_type=4" not in url


# --- backfill_series_macro requires an API key ---


def test_backfill_series_macro_reports_missing_key(tmp_path):
    summary = bm.backfill_series_macro("UNRATE", str(tmp_path), api_key="")
    assert summary["written"] == 0
    assert any("FRED_API_KEY" in e for e in summary["errors"])
