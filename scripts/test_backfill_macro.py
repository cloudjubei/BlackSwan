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


# --- observations_url includes the point-in-time output type ---


def test_observations_url_requests_initial_release_vintages():
    url = bm.observations_url("UNRATE", "KEY123")
    assert "series_id=UNRATE" in url
    assert "api_key=KEY123" in url
    assert "file_type=json" in url
    assert "output_type=4" in url  # initial release only — NOT the latest-revised series (a silent leak)


# --- backfill_series_macro requires an API key ---


def test_backfill_series_macro_reports_missing_key(tmp_path):
    summary = bm.backfill_series_macro("UNRATE", str(tmp_path), api_key="")
    assert summary["written"] == 0
    assert any("FRED_API_KEY" in e for e in summary["errors"])
