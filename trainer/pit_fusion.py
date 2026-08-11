"""Point-in-time fusion: align a release-stamped series onto price bars WITHOUT look-ahead.

A macro/fundamental value is only knowable from the instant it was PUBLISHED (its release/filing
timestamp), never from the period it describes. Fusing by the reference period leaks the future
(January's unemployment "known" all January); fusing by the release timestamp is correct (January's
number is knowable only from its early-February release). This module is the leakage guard: an as-of
join that, for each price bar, carries the latest value released AT OR BEFORE that bar (forward-filled),
and nothing before the first release.

A release DATE (from FRED release dates / EDGAR filing dates) is turned into a release TIMESTAMP at the
series' actual publish wall-clock time — NOT a blanket 08:30 ET (FOMC 14:00, H.15 ~16:15, jobs 08:30 —
see PUBLISH_TIME_ET). A bar earlier in the release day than the publish time does not yet see the value.
"""

import bisect
import datetime
from zoneinfo import ZoneInfo

# Actual publish wall-clock (America/New_York) per FRED series id, so a release date maps to the right
# instant. Default 08:30 ET covers the 08:30 data drops (employment/CPI/claims/retail/GDP).
PUBLISH_TIME_ET = {
    "DFEDTARU": "14:00",  # FOMC decision
    "DFEDTARL": "14:00",
    "DFF": "16:00",       # NY Fed prior-day effective rate, next business day
    "DGS10": "16:15",     # H.15 release, after the equity close
    "DGS2": "16:15",
    "T10Y2Y": "16:15",
    "T10Y3M": "16:15",
    "DFII10": "16:15",
    "T10YIE": "16:15",    # H.15 breakeven, same release as its DGS10/DFII10 siblings
}
DEFAULT_PUBLISH_TIME_ET = "08:30"
DEFAULT_TZ = "America/New_York"
# SEC EDGAR filings carry only a filing DATE (no intraday acceptance time) and are frequently accepted
# after the equity close, so a filing is only safely public the NEXT session. Filing-based (fundamentals)
# series must stamp at end of the filing day — a same-day equity-close bar then never sees it — NOT the
# 08:30 macro default, which would leak an after-hours filing intraday. The fusion consumer passes this
# for fundamentals; when EDGAR's per-filing acceptanceDateTime is later mined, use that exact instant.
FILING_PUBLISH_TIME_ET = "23:59"


def publish_time_for(series_id):
    """The publish wall-clock (HH:MM, America/New_York) a FRED series' value becomes public."""
    return PUBLISH_TIME_ET.get(series_id, DEFAULT_PUBLISH_TIME_ET)


def release_datetime_ms(release_date, publish_time=DEFAULT_PUBLISH_TIME_ET, tz=DEFAULT_TZ):
    """Epoch ms for a release DATE (``YYYY-MM-DD``) at its publish wall-clock time in ``tz`` (DST-aware)."""
    year, month, day = (int(p) for p in release_date.split("-"))
    hour, minute = (int(p) for p in publish_time.split(":"))
    dt = datetime.datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz))
    return int(dt.timestamp() * 1000)


def asof_join(bar_ms, releases):
    """For each bar timestamp in ``bar_ms`` (epoch ms), the value of the latest release AT OR BEFORE it,
    else ``None`` (before the first release). ``releases`` is an iterable of ``(release_ms, value)``,
    sorted internally. The join is inclusive at the release instant and forward-fills between releases."""
    ordered = sorted(releases, key=lambda r: r[0])
    release_times = [r[0] for r in ordered]
    values = [r[1] for r in ordered]
    out = []
    for t in bar_ms:
        # Index of the last release with release_ms <= t.
        idx = bisect.bisect_right(release_times, t) - 1
        out.append(values[idx] if idx >= 0 else None)
    return out


def fuse_series(bar_ms, observations, publish_time=DEFAULT_PUBLISH_TIME_ET, tz=DEFAULT_TZ):
    """Fuse a point-in-time ``observations`` series (each ``{releaseDate, value, ...}``) onto ``bar_ms``:
    stamp each observation at its release datetime, then as-of-join. A bar sees an observation only from
    the instant it was released — the leakage guard."""
    releases = [
        (release_datetime_ms(obs["releaseDate"], publish_time, tz), obs["value"])
        for obs in observations
        if obs.get("releaseDate") is not None
    ]
    return asof_join(bar_ms, releases)
