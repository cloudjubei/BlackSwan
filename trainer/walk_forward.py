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

# id -> (first_train_year, test_year); train is the expanding window first .. test_year - 1. Plain
# year ids train from 2020 (the full BTC history); the alt-* ids train from 2022 for assets whose
# on-disk history starts there (the altcoin 1m backfill begins 2022-01).
_WINDOWS = {
    "2022": (2020, 2022),
    "2023": (2020, 2023),
    "2024": (2020, 2024),
    "2025": (2020, 2025),
    "alt-2024": (2022, 2024),
    "alt-2025": (2022, 2025),
}


def _months(year_lo, year_hi):
    return [(y, m) for y in range(year_lo, year_hi + 1) for m in range(1, 13)]


def walk_forward_window_ids():
    """The selectable window ids, full-history windows first, then the alt-* short-history ones."""
    return list(_WINDOWS.keys())


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
    first_train_year, test_year = _WINDOWS[window]
    train_pairs = _months(first_train_year, test_year - 1)
    test_pairs = _months(test_year, test_year)
    meta = {
        "walk_forward_window": window,
        "train_from": f"{first_train_year}-01",
        "train_to": f"{test_year - 1}-12",
        "test_from": f"{test_year}-01",
        "test_to": f"{test_year}-12",
    }
    return train_pairs, test_pairs, meta
