"""Backfill any yfinance daily instrument (commodities, FX, stocks) into its own directory.

The daily-kline pipeline is identical to the stocks backfill — fetch daily rows, map to the on-disk
kline shape, group by calendar month, validate, atomic-write — but parameterised on
``(symbol, source_symbol, out_dir)`` so commodities (``GC=F`` -> ``commodities/GOLD-1d-*.json``) and FX
(``EURUSD=X`` -> ``fx/EURUSD-1d-*.json``) reuse it. The catalog (``trainer.data_catalog``) is the source
of the symbol -> source-symbol map + the target directory; this module just executes one instrument.

FX carries no real volume (yfinance returns 0) — the neutral fill (quote/taker = 0) is intended, matching
how the stocks miner fills ``trades_number``; the correctness note is not to FEATURE that constant column.

Usage (from the repo root):
    .venv/bin/python -m scripts.backfill_market                 # all catalogued yfinance instruments
    .venv/bin/python -m scripts.backfill_market --class fx      # one asset class
    .venv/bin/python -m scripts.backfill_market --symbol GOLD   # one instrument
    .venv/bin/python -m scripts.backfill_market --dry-run
"""

import argparse
import os
import sys

from scripts.backfill_klines import latest_complete_month, month_filename, write_month
from scripts.backfill_stocks import (
    START_MONTH,
    group_by_month,
    kline_from_daily_row,
    rows_from_yfinance,
    validate_stock_month as validate_daily_month,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def backfill_yf_symbol(symbol, source_symbol, out_dir, start, through, dry_run=False):
    """Download every missing daily month for one yfinance instrument into ``out_dir``.

    ``symbol`` is the local filename stem; ``source_symbol`` is the yfinance ticker fetched. Skips
    months already on disk, validates each written month, and returns a summary dict."""
    summary = {
        "symbol": symbol,
        "source_symbol": source_symbol,
        "interval": "1d",
        "written": [],
        "skipped": 0,
        "gaps": [],
        "errors": [],
        "source": "yfinance",
    }
    rows = rows_from_yfinance(source_symbol, start, through)
    if not rows:
        summary["errors"].append(f"no rows from yfinance for {source_symbol}")
        return summary
    os.makedirs(out_dir, exist_ok=True)
    monthly = group_by_month([kline_from_daily_row(row) for row in rows])
    for (year, month), klines in monthly.items():
        path = os.path.join(out_dir, month_filename(symbol, "1d", year, month))
        if os.path.exists(path):
            summary["skipped"] += 1
            continue
        errors, warnings = validate_daily_month(klines, year, month)
        if errors:
            summary["errors"].append(f"{year}-{month}: {'; '.join(errors)}")
            continue
        for warning in warnings:
            summary["gaps"].append(f"{year}-{month}: {warning}")
        if not dry_run:
            write_month(path, klines)
        summary["written"].append((year, month))
    return summary


def _selected_instruments(class_id=None, symbol=None):
    from trainer import data_catalog

    yf = [inst for inst in data_catalog.instruments() if inst.source == data_catalog.YFINANCE]
    if symbol:
        return [inst for inst in yf if inst.symbol == symbol]
    if class_id:
        return [inst for inst in yf if inst.asset_class == class_id]
    return yf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--through", help="last month to fetch as YYYY-MM (default: last complete month)")
    parser.add_argument("--class", dest="class_id", help="restrict to one asset class (stocks/commodities/fx)")
    parser.add_argument("--symbol", help="restrict to one local symbol (e.g. GOLD)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + validate, write nothing")
    args = parser.parse_args(argv)
    if args.through:
        year, month = args.through.split("-")
        through = (int(year), int(month))
    else:
        through = latest_complete_month()
    os.chdir(_REPO_ROOT)

    instruments = _selected_instruments(args.class_id, args.symbol)
    if not instruments:
        print("no matching catalogued yfinance instruments")
        return 1
    print(f"Backfilling {len(instruments)} instrument(s) {START_MONTH[0]}-{START_MONTH[1]} through {through[0]}-{through[1]}")
    summaries = []
    for inst in instruments:
        summary = backfill_yf_symbol(inst.symbol, inst.source_symbol, inst.directory, START_MONTH, through, dry_run=args.dry_run)
        summaries.append(summary)
        span = f"{summary['written'][0][0]}-{summary['written'][0][1]} .. {summary['written'][-1][0]}-{summary['written'][-1][1]}" if summary["written"] else "-"
        print(f"  {inst.symbol:<8} [{inst.source_symbol:<9}] {len(summary['written'])} written, {summary['skipped']} skipped, {span}")
        for error in summary["errors"]:
            print(f"    ERROR: {error}")
        sys.stdout.flush()

    return 1 if any(s["errors"] for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
