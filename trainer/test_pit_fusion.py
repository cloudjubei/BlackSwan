import datetime
from zoneinfo import ZoneInfo

from trainer import pit_fusion


def _ms(year, month, day, hour=0, minute=0, tz="UTC"):
    dt = datetime.datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz))
    return int(dt.timestamp() * 1000)


# --- release_datetime_ms: a release DATE + publish wall-clock time -> epoch ms ---


def test_release_datetime_ms_applies_the_publish_time_in_the_given_tz():
    # 08:30 America/New_York on 2024-02-02 (EST, UTC-5) is 13:30 UTC.
    got = pit_fusion.release_datetime_ms("2024-02-02", "08:30", "America/New_York")
    assert got == _ms(2024, 2, 2, 13, 30)


def test_release_datetime_ms_respects_dst():
    # 08:30 America/New_York on 2024-07-01 (EDT, UTC-4) is 12:30 UTC.
    got = pit_fusion.release_datetime_ms("2024-07-01", "08:30", "America/New_York")
    assert got == _ms(2024, 7, 1, 12, 30)


def test_release_datetime_ms_later_time_is_larger():
    early = pit_fusion.release_datetime_ms("2024-02-02", "08:30", "America/New_York")
    late = pit_fusion.release_datetime_ms("2024-02-02", "16:15", "America/New_York")
    assert late > early


# --- asof_join: the core leakage-free as-of join ---


def test_asof_join_carries_the_latest_release_at_or_before_each_bar():
    bars = [10, 20, 30, 40]
    releases = [(15, "a"), (35, "b")]
    assert pit_fusion.asof_join(bars, releases) == [None, "a", "a", "b"]


def test_asof_join_is_inclusive_at_the_release_instant():
    assert pit_fusion.asof_join([15], [(15, "a")]) == ["a"]


def test_asof_join_before_first_release_is_none():
    assert pit_fusion.asof_join([1, 2], [(5, "a")]) == [None, None]


def test_asof_join_empty_releases_all_none():
    assert pit_fusion.asof_join([1, 2, 3], []) == [None, None, None]


def test_asof_join_empty_bars():
    assert pit_fusion.asof_join([], [(1, "a")]) == []


def test_asof_join_unsorted_releases_are_handled():
    # Releases passed out of order must still align correctly (sorted internally).
    assert pit_fusion.asof_join([20, 40], [(35, "b"), (15, "a")]) == ["a", "b"]


# --- fuse_series: THE leakage guard (release-time alignment, never reference-period) ---


def test_fuse_series_only_reveals_a_value_from_its_release_not_its_reference_period():
    # January's unemployment is published early February; it must be INVISIBLE during January and only
    # visible from its release. Aligning by reference period (the classic leak) would show it on Jan 15.
    observations = [
        {"refPeriod": "2024-01", "releaseDate": "2024-02-02", "value": 3.7},
        {"refPeriod": "2024-02", "releaseDate": "2024-03-08", "value": 3.9},
    ]
    bars = [
        _ms(2024, 1, 15, 14, 0),  # mid-January — Jan number NOT yet released
        _ms(2024, 1, 31, 14, 0),  # end of January — still not released
        _ms(2024, 2, 5, 14, 0),  # after the Feb 2 release
        _ms(2024, 3, 10, 14, 0),  # after the Mar 8 release of the Feb number
    ]
    fused = pit_fusion.fuse_series(bars, observations, publish_time="08:30", tz="America/New_York")
    assert fused == [None, None, 3.7, 3.9]


def test_fuse_series_forward_fills_between_releases():
    observations = [{"refPeriod": "2024-01", "releaseDate": "2024-02-02", "value": 3.7}]
    bars = [_ms(2024, 2, 5), _ms(2024, 5, 1), _ms(2024, 12, 31)]
    fused = pit_fusion.fuse_series(bars, observations, publish_time="08:30", tz="America/New_York")
    assert fused == [3.7, 3.7, 3.7]


def test_filing_series_use_a_next_session_safe_stamp_not_the_0830_macro_default():
    # SEC filings carry only a DATE + are often accepted after the equity close, so a filing is only safely
    # public the NEXT session. With FILING_PUBLISH_TIME_ET a same-day 16:00 ET equity-close bar must NOT see
    # it (the 08:30 macro default WOULD leak an after-hours filing intraday).
    obs = [{"refPeriod": "2023-09-30", "releaseDate": "2023-11-03", "value": 96995}]
    same_day_close = [_ms(2023, 11, 3, 16, 0, "America/New_York")]
    next_session = [_ms(2023, 11, 6, 9, 30, "America/New_York")]
    tz = "America/New_York"
    assert pit_fusion.fuse_series(same_day_close, obs, pit_fusion.FILING_PUBLISH_TIME_ET, tz) == [None]
    assert pit_fusion.fuse_series(next_session, obs, pit_fusion.FILING_PUBLISH_TIME_ET, tz) == [96995]


def test_fuse_series_release_day_before_publish_time_is_not_yet_visible():
    # A bar EARLIER in the day than the 08:30 release must not see the value released later that day.
    observations = [{"refPeriod": "2024-01", "releaseDate": "2024-02-02", "value": 3.7}]
    before = [_ms(2024, 2, 2, 8, 0, "America/New_York")]  # 08:00 ET, before the 08:30 release
    after = [_ms(2024, 2, 2, 9, 0, "America/New_York")]  # 09:00 ET, after
    assert pit_fusion.fuse_series(before, observations, "08:30", "America/New_York") == [None]
    assert pit_fusion.fuse_series(after, observations, "08:30", "America/New_York") == [3.7]
