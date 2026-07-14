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
