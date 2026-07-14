"""Backfill stocks/ with daily kline JSONs for the top-10 US stocks (stooq CSV, yfinance fallback).

Writes monthly files mirroring the binance/ kline shape + naming ({TICKER}-1d-{YYYY}-{M}.json,
month not zero-padded) so the existing providers read them unchanged: pd.read_json coerces the
string prices exactly as it does for binance/ files. Fields the sources lack are filled with the
neutral values the reader expects: quote volume = close * volume, taker base/quote = half of
volume/quote (taker_buy_ratio = 0.5), trades_number = 0. Weekend/holiday holes are normal for
stocks, so validation only checks structure + a sane trading-day count per month. Prices are
split/dividend-adjusted (both sources), so a split never shows up as a fake overnight crash.

Usage (from the repo root):
    .venv/bin/python -m scripts.backfill_stocks             # 2018-01 through the last complete month
    .venv/bin/python -m scripts.backfill_stocks --dry-run
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

from scripts.backfill_klines import check_rows, latest_complete_month, month_filename, write_month

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCKS_DIR = os.path.join(_REPO_ROOT, "stocks")
STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}.us&i=d&d1={d1}&d2={d2}"

TICKERS = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT"]
START_MONTH = (2018, 1)

_DAY_MS = 86_400_000
_MIN_TRADING_DAYS = 15
_MAX_TRADING_DAYS = 23
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_stooq_csv(text):
    rows = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.strip().split(",")]
        if len(fields) < 5 or not _DATE.match(fields[0]):
            continue
        if len(fields) < 6 or not fields[5]:
            fields = fields[:5] + ["0"]
        rows.append(fields[:6])
    return rows


def kline_from_daily_row(row):
    date = datetime.datetime.strptime(row[0], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    timestamp = int(date.timestamp() * 1000)
    close, volume = float(row[4]), float(row[5])
    quote = close * volume
    return {
        "timestamp": timestamp,
        "price_open": "%.8f" % float(row[1]),
        "price_high": "%.8f" % float(row[2]),
        "price_low": "%.8f" % float(row[3]),
        "price": "%.8f" % close,
        "volume": "%.8f" % volume,
        "timestamp_close": timestamp + _DAY_MS - 1,
        "asset_volume_quote": "%.8f" % quote,
        "trades_number": 0,
        "asset_volume_taker_base": "%.8f" % (volume / 2.0),
        "asset_volume_taker_quote": "%.8f" % (quote / 2.0),
    }


def group_by_month(klines):
    grouped = {}
    for kline in klines:
        stamp = datetime.datetime.fromtimestamp(kline["timestamp"] / 1000, tz=datetime.timezone.utc)
        grouped.setdefault((stamp.year, stamp.month), []).append(kline)
    return dict(sorted(grouped.items()))


def validate_stock_month(rows, year, month):
    """(errors, warnings): structural checks only — non-trading days are expected, so the sole
    count check is a sane trading-day total for a full month."""
    errors = check_rows(rows)
    if errors:
        return errors, []
    warnings = []
    if not (_MIN_TRADING_DAYS <= len(rows) <= _MAX_TRADING_DAYS):
        warnings.append(f"{len(rows)} bars — unusual trading-day count for {year}-{month}")
    return errors, warnings


def _month_end(through):
    day = datetime.date(through[0], through[1], 28) + datetime.timedelta(days=4)
    return day - datetime.timedelta(days=day.day)


def fetch_stooq_history(ticker, start, through):
    last_day = _month_end(through)
    url = STOOQ_URL.format(
        ticker=ticker.lower(),
        d1=f"{start[0]}{start[1]:02d}01",
        d2=f"{last_day.year}{last_day.month:02d}{last_day.day:02d}",
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def rows_from_yfinance(ticker, start, through):
    import pandas as pd
    import yfinance as yf

    end = _month_end(through) + datetime.timedelta(days=1)
    df = yf.download(
        ticker,
        start=f"{start[0]}-{start[1]:02d}-01",
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return [
        [index.strftime("%Y-%m-%d"), str(row["Open"]), str(row["High"]), str(row["Low"]), str(row["Close"]), str(row.get("Volume", 0) or 0)]
        for index, row in df.iterrows()
    ]


def fetch_history_rows(ticker, start, through):
    """(rows, source): stooq CSV first; yfinance when stooq is unreachable or serves its
    JS bot-challenge page instead of CSV."""
    try:
        rows = parse_stooq_csv(fetch_stooq_history(ticker, start, through))
        if rows:
            return rows, "stooq"
    except (urllib.error.URLError, OSError):
        pass
    return rows_from_yfinance(ticker, start, through), "yfinance"


def backfill_ticker(ticker, start, through, dry_run=False):
    summary = {"symbol": ticker, "interval": "1d", "written": [], "skipped": 0, "gaps": [], "errors": [], "source": None}
    rows, summary["source"] = fetch_history_rows(ticker, start, through)
    if not rows:
        summary["errors"].append("no rows from stooq (bot-challenge or empty) nor yfinance")
        return summary
    monthly = group_by_month([kline_from_daily_row(row) for row in rows])
    for (year, month), klines in monthly.items():
        path = os.path.join(STOCKS_DIR, month_filename(ticker, "1d", year, month))
        if os.path.exists(path):
            summary["skipped"] += 1
            continue
        errors, warnings = validate_stock_month(klines, year, month)
        if errors:
            summary["errors"].append(f"{year}-{month}: {'; '.join(errors)}")
            continue
        for warning in warnings:
            summary["gaps"].append(f"{year}-{month}: {warning}")
        if not dry_run:
            write_month(path, klines)
        summary["written"].append((year, month))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--through", help="last month to fetch as YYYY-MM (default: last complete month)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + validate, write nothing")
    args = parser.parse_args(argv)
    if args.through:
        year, month = args.through.split("-")
        through = (int(year), int(month))
    else:
        through = latest_complete_month()
    os.makedirs(STOCKS_DIR, exist_ok=True)

    print(f"Backfilling stocks/ {START_MONTH[0]}-{START_MONTH[1]} through {through[0]}-{through[1]}")
    summaries = []
    for ticker in TICKERS:
        summary = backfill_ticker(ticker, START_MONTH, through, dry_run=args.dry_run)
        summaries.append(summary)
        span = f"{summary['written'][0][0]}-{summary['written'][0][1]} .. {summary['written'][-1][0]}-{summary['written'][-1][1]}" if summary["written"] else "-"
        print(f"  {ticker:<6} [{summary['source']}] {len(summary['written'])} months written, {summary['skipped']} skipped, {span}")
        for gap in summary["gaps"]:
            print(f"    note: {gap}")
        for error in summary["errors"]:
            print(f"    ERROR: {error}")
        sys.stdout.flush()

    failed = [s for s in summaries if s["errors"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
