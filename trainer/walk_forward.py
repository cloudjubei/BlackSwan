"""Walk-forward evaluation windows for the trading line.

Each window is one (train -> test) split that is trained and tested as its OWN run; the Model
Trainer sweeps ``walk_forward_window`` like a seed, so windows run in parallel and the viewer
groups them into an out-of-sample distribution. Honest OOS evaluation needs more than the single
2020-2023 / 2024 split: rolling the test across distinct regime-years (2022 bear, 2023 recovery,
2024 bull), each with an expanding train window, shows whether a config is robust across time
rather than tuned to one lucky period. Pure (no torch / no filesystem) so it stays trivially
testable; consumers (config_builder, summary) resolve paths and metrics from the returned pairs.
"""

DEFAULT_WALK_FORWARD_WINDOW = "2024"

# A far-future cap for OPEN-ENDED (oos-*) windows: test runs from the cutoff to this year, and
# config_builder's file-existence filter truncates it to the latest month actually on disk — so an
# open window tests ALL data after the training cutoff without the resolver touching the filesystem.
_OPEN_TEST_CAP = 2035

# id -> (first_train_year, first_test_year, last_test_year); train is the expanding window
# first_train .. first_test-1. Fixed windows test a single year (first_test == last_test); open-ended
# oos-* windows test first_test .. _OPEN_TEST_CAP (truncated to disk). Plain ids train from 2020 (the
# full BTC history); alt-* ids train from 2022 for assets whose on-disk history starts there (the
# altcoin 1m backfill begins 2022-01).
_WINDOWS = {
    # deep-history commodity/financial test years (train from 2006, where the extended yfinance price + CFTC COT +
    # macro all reach): a driver-conditioned or fixed-rule model needs only a short warm-up, so these give the
    # full 2008-2024 out-of-sample span (17 windows across the GFC, 2011, 2015-16, 2018, 2020) that the tighter
    # DSR confidence interval needs to reject a MODEST persistent edge, not just a large one.
    "2008": (2006, 2008, 2008),
    "2009": (2006, 2009, 2009),
    "2010": (2006, 2010, 2010),
    "2011": (2006, 2011, 2011),
    "2012": (2006, 2012, 2012),
    "2013": (2006, 2013, 2013),
    "2014": (2006, 2014, 2014),
    "2015": (2006, 2015, 2015),
    "2016": (2006, 2016, 2016),
    "2017": (2006, 2017, 2017),
    "2018": (2006, 2018, 2018),
    "2019": (2006, 2019, 2019),
    # 2020-21 kept training from 2018 (crypto/attention consumers share these ids); commodities warm fine either way.
    "2020": (2018, 2020, 2020),
    "2021": (2018, 2021, 2021),
    "2022": (2020, 2022, 2022),
    "2023": (2020, 2023, 2023),
    "2024": (2020, 2024, 2024),
    "2025": (2020, 2025, 2025),
    "2026": (2020, 2026, 2026),
    # cf-* (commodity-factor) DEEP-TRAIN windows: test 2020-2025 but train from 2006, so a long-formation factor
    # (e.g. the 5yr VALUE reversal, 1260 bars) always has its full lookback available in the loaded history —
    # the plain 2020-2024 ids train from 2018/2020 and would starve it. Same accounted test year, deeper warm-up.
    "cf-2020": (2006, 2020, 2020),
    "cf-2021": (2006, 2021, 2021),
    "cf-2022": (2006, 2022, 2022),
    "cf-2023": (2006, 2023, 2023),
    "cf-2024": (2006, 2024, 2024),
    "cf-2025": (2006, 2025, 2025),
    "alt-2024": (2022, 2024, 2024),
    "alt-2025": (2022, 2025, 2025),
    "alt-2026": (2022, 2026, 2026),
    # stk-* train from 2018 — a DEEPER expanding window than the plain 2020 start, for assets whose on-disk
    # history reaches that far (stocks from 2018-01, BTC from 2017-08). windows_supported gates by each
    # asset's data range, so a 2020/2022-start asset simply never offers them.
    "stk-2022": (2018, 2022, 2022),
    "stk-2023": (2018, 2023, 2023),
    "stk-2024": (2018, 2024, 2024),
    "stk-2025": (2018, 2025, 2025),
    "stk-2026": (2018, 2026, 2026),
    "stk-oos-2024": (2018, 2024, _OPEN_TEST_CAP),
    # Open-ended: test the full out-of-sample span from the cutoff through the latest data on disk.
    "oos-2024": (2020, 2024, _OPEN_TEST_CAP),
    "oos-2025": (2020, 2025, _OPEN_TEST_CAP),
    "alt-oos-2024": (2022, 2024, _OPEN_TEST_CAP),
    "alt-oos-2025": (2022, 2025, _OPEN_TEST_CAP),
}


def _months(year_lo, year_hi):
    return [(y, m) for y in range(year_lo, year_hi + 1) for m in range(1, 13)]


def walk_forward_window_ids():
    """The selectable window ids, full-history windows first, then the alt-* short-history ones."""
    return list(_WINDOWS.keys())


def windows_supported(start, end):
    """The window ids an asset can run given its on-disk data RANGE (``start``/``end`` = "YYYY-MM" of the
    asset's finest-timeframe coverage). A window is runnable when training can begin at/after the data
    start AND testing begins at/before the data end — so an altcoin (2022→) gets only the alt-* windows, a
    deep-history asset gets all, and an asset whose data ends early can't run a later test year. Pure — the
    inventory-driven per-asset capability the viewer renders on the asset card."""
    if not start or not end:
        return []
    out = []
    for wid, (first_train_year, first_test_year, _last) in _WINDOWS.items():
        train_from = f"{first_train_year:04d}-01"
        test_from = f"{first_test_year:04d}-01"
        if train_from >= start and test_from <= end:
            out.append(wid)
    return out


def resolve_walk_forward_window(cfg=None):
    """Return ``(train_pairs, test_pairs, meta)`` for the cfg's ``walk_forward_window``.

    Defaults to ``2024`` (train 2020-2023, test 2024) so a config without the lever reproduces the
    historical single split exactly. Train is the expanding window Jan (first train year) .. Dec
    (test_year - 1); test is the 12 months of the test year. Raises ``SystemExit`` on an unknown id
    so a typo fails fast instead of silently scoring the default window.
    """
    cfg = cfg or {}
    window = str(cfg.get("walk_forward_window", DEFAULT_WALK_FORWARD_WINDOW))
    if window not in _WINDOWS:
        raise SystemExit(
            f"unknown walk_forward_window {window!r} — choices: {walk_forward_window_ids()}"
        )
    first_train_year, first_test_year, last_test_year = _WINDOWS[window]
    open_ended = last_test_year >= _OPEN_TEST_CAP
    train_pairs = _months(first_train_year, first_test_year - 1)
    test_pairs = _months(first_test_year, last_test_year)
    meta = {
        "walk_forward_window": window,
        "train_from": f"{first_train_year}-01",
        "train_to": f"{first_test_year - 1}-12",
        "test_from": f"{first_test_year}-01",
        "test_to": "latest" if open_ended else f"{last_test_year}-12",
    }
    return train_pairs, test_pairs, meta
