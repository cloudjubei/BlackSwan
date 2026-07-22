"""Inventory of which (symbol, timeframe) klines are actually on disk under binance/.

Used to capability-gate the trainer's ``asset`` lever: only assets whose files are
present for the requested timeframe are runnable. BTCUSDT carries 1m/1h/1d; the
altcoins carry 1m (the source of truth) plus 1d derived from it, with their 1h in
binance/derived/. stocks/ mirrors the same monthly naming for US daily klines and
is scanned via ``scan_stocks_inventory``.
"""

import glob
import os
import re

_FILENAME = re.compile(r"^([A-Z0-9]+)-([0-9]+[mhdw])-\d+-\d+\.json$")
_COVERAGE = re.compile(r"^([A-Z0-9]+)-([0-9]+[mhdw])-(\d+)-(\d+)\.json$")


def _month_label(year, month):
    return f"{year:04d}-{month:02d}"


def _months_inclusive(start, end):
    """Every (year, month) from ``start`` to ``end`` inclusive."""
    year, month = start
    out = []
    while (year, month) <= end:
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def scan_coverage(root="binance"):
    """Map ``symbol -> timeframe -> {start, end, months, gaps}`` from the (year, month) each
    filename encodes. ``start``/``end`` are zero-padded ``YYYY-MM`` labels of the earliest and
    latest month on disk; ``months`` counts distinct months present; ``gaps`` lists any missing
    ``YYYY-MM`` months between start and end (empty when the coverage is contiguous)."""
    present = {}
    for path in glob.glob(os.path.join(root, "*.json")):
        match = _COVERAGE.match(os.path.basename(path))
        if not match:
            continue
        symbol, timeframe, year, month = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        present.setdefault((symbol, timeframe), set()).add((year, month))
    coverage = {}
    for (symbol, timeframe), months in present.items():
        ordered = sorted(months)
        start, end = ordered[0], ordered[-1]
        gaps = [_month_label(y, m) for (y, m) in _months_inclusive(start, end) if (y, m) not in months]
        coverage.setdefault(symbol, {})[timeframe] = {
            "start": _month_label(*start),
            "end": _month_label(*end),
            "months": len(months),
            "gaps": gaps,
        }
    return {symbol: dict(sorted(tfs.items())) for symbol, tfs in sorted(coverage.items())}


def scan_inventory(root="binance"):
    """Map ``symbol -> sorted list of timeframes`` that have at least one file on disk."""
    found = {}
    for path in glob.glob(os.path.join(root, "*.json")):
        match = _FILENAME.match(os.path.basename(path))
        if not match:
            continue
        symbol, timeframe = match.group(1), match.group(2)
        found.setdefault(symbol, set()).add(timeframe)
    return {symbol: sorted(timeframes) for symbol, timeframes in sorted(found.items())}


def scan_stocks_inventory(root="stocks"):
    """``scan_inventory`` over the stocks/ daily-kline mirror (same filename convention)."""
    return scan_inventory(root)


def available_assets(timeframe, root="binance"):
    """The symbols that have at least one file for ``timeframe``."""
    return [s for s, tfs in scan_inventory(root).items() if timeframe in tfs]


def has_data(symbol, timeframe, root="binance"):
    """Whether ``symbol`` has at least one ``timeframe`` file on disk."""
    return timeframe in scan_inventory(root).get(symbol, [])
