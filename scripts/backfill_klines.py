"""Backfill binance/ monthly kline JSONs from Binance's public archives, then derive altcoin 1h/1d.

Downloads https://data.binance.vision monthly kline zips one month at a time (zip -> csv -> JSON ->
disk, nothing accumulates across months), converts each to the exact on-disk kline shape binance/
already uses (1m rows carry tokenPair/interval, coarser rows don't; prices/volumes stay verbatim
strings; timestamps are epoch ms — archives from 2025-01 use microseconds and are normalized),
validates every month (monotonic, no duplicates, numeric, cross-month continuity) and skips months
already on disk. After the download it materialises the altcoins' 1d files under binance/ (the
config_builder 1d path reads them raw) and their 1h derived cache via trainer.derive_cache.

Usage (from the repo root):
    .venv/bin/python -m scripts.backfill_klines             # download + derive through the last complete month
    .venv/bin/python -m scripts.backfill_klines --dry-run   # print the plan only
    .venv/bin/python -m scripts.backfill_klines --skip-derive
"""

import argparse
import calendar
import datetime
import glob
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINANCE_DIR = os.path.join(_REPO_ROOT, "binance")
ARCHIVE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"

ALTCOINS = ["ADAUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT", "SHIBUSDT", "SOLUSDT", "XRPUSDT"]
DOWNLOAD_TARGETS = [("BTCUSDT", "1m"), ("BTCUSDT", "1h"), ("BTCUSDT", "1d")] + [(s, "1m") for s in ALTCOINS]

_INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "1d": 86_400_000}
_MICROSECOND_THRESHOLD = 10**14
_FILE_PATTERN = "{symbol}-{interval}-([0-9]+)-([0-9]+)\\.json$"


def normalize_epoch_ms(value):
    stamp = int(value)
    return stamp // 1000 if stamp >= _MICROSECOND_THRESHOLD else stamp


def kline_from_csv_row(row, symbol, interval):
    kline = {}
    if interval == "1m":
        kline["tokenPair"] = symbol
        kline["interval"] = interval
    kline.update(
        {
            "timestamp": normalize_epoch_ms(row[0]),
            "price_open": row[1],
            "price_high": row[2],
            "price_low": row[3],
            "price": row[4],
            "volume": row[5],
            "timestamp_close": normalize_epoch_ms(row[6]),
            "asset_volume_quote": row[7],
            "trades_number": int(row[8]),
            "asset_volume_taker_base": row[9],
            "asset_volume_taker_quote": row[10],
        }
    )
    return kline


def months_between(start, end):
    months = []
    year, month = start
    while (year, month) <= end:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_filename(symbol, interval, year, month):
    return f"{symbol}-{interval}-{year}-{month}.json"


def latest_complete_month(today=None):
    today = today or datetime.date.today()
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)


def expected_bars(interval, year, month):
    days = calendar.monthrange(year, month)[1]
    return days * 86_400_000 // _INTERVAL_MS[interval]


def last_month_on_disk(directory, symbol, interval):
    pattern = re.compile(_FILE_PATTERN.format(symbol=symbol, interval=interval))
    months = []
    for path in glob.glob(os.path.join(directory, f"{symbol}-{interval}-*.json")):
        match = pattern.search(os.path.basename(path))
        if match:
            months.append((int(match.group(1)), int(match.group(2))))
    return max(months) if months else None


def check_rows(rows):
    """Structural errors shared by every kline series: emptiness, timestamp order/uniqueness,
    numeric + positive prices. Returns a list of error strings (empty = clean)."""
    if not rows:
        return ["empty month"]
    errors = []
    timestamps = [r["timestamp"] for r in rows]
    if len(set(timestamps)) != len(timestamps):
        errors.append("duplicate timestamps")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        errors.append("timestamps not strictly monotonic")
    for row in rows:
        try:
            values = [float(row[k]) for k in ("price_open", "price_high", "price_low", "price", "volume")]
        except (TypeError, ValueError):
            errors.append(f"non-numeric field at timestamp {row['timestamp']}")
            break
        if any(v != v for v in values):
            errors.append(f"NaN field at timestamp {row['timestamp']}")
            break
        if any(v <= 0 for v in values[:4]):
            errors.append(f"non-positive price at timestamp {row['timestamp']}")
            break
    return errors


def validate_month(rows, interval, year, month):
    """(errors, warnings) for one downloaded month. Gaps are warnings (Binance has real outages);
    structural problems and bars outside the month are errors."""
    errors = check_rows(rows)
    if errors:
        return errors, []
    month_start = int(datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    month_end = month_start + calendar.monthrange(year, month)[1] * 86_400_000
    if rows[0]["timestamp"] < month_start or rows[-1]["timestamp"] >= month_end:
        errors.append(f"bars outside {year}-{month}")
        return errors, []
    warnings = []
    missing = expected_bars(interval, year, month) - len(rows)
    if missing > 0:
        warnings.append(f"{missing} missing bars")
    return errors, warnings


def missing_bars_between(prev_last_row, first_row, interval):
    span = first_row["timestamp"] - (prev_last_row["timestamp"] + _INTERVAL_MS[interval])
    return max(0, span // _INTERVAL_MS[interval])


def fetch_month_rows(symbol, interval, year, month):
    """CSV rows for one archive month, or None when Binance has no archive (404)."""
    url = ARCHIVE_URL.format(symbol=symbol, interval=interval, year=year, month=month)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(archive.namelist()[0]) as csv_file:
            lines = io.TextIOWrapper(csv_file, encoding="utf-8")
            rows = [line.strip().split(",") for line in lines if line.strip()]
    if rows and rows[0][0] == "open_time":
        rows = rows[1:]
    return rows


def write_month(path, klines):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(klines, handle, separators=(",", ":"))
    os.replace(tmp_path, path)


def _last_kline_on_disk(path):
    with open(path) as handle:
        klines = json.load(handle)
    return klines[-1] if klines else None


def backfill_series(symbol, interval, through, dry_run=False):
    """Download every missing month for (symbol, interval) up to `through`. Returns a summary dict."""
    summary = {"symbol": symbol, "interval": interval, "written": [], "skipped": 0, "gaps": [], "errors": []}
    last = last_month_on_disk(BINANCE_DIR, symbol, interval)
    if last is None:
        summary["errors"].append("no existing months on disk — refusing to guess a start month")
        return summary
    start = months_between(last, through)[1:2]
    if not start:
        return summary
    prev_last_row = None
    prev_label = f"{last[0]}-{last[1]}"
    for (year, month) in months_between(start[0], through):
        path = os.path.join(BINANCE_DIR, month_filename(symbol, interval, year, month))
        if os.path.exists(path):
            summary["skipped"] += 1
            prev_last_row = None
            prev_label = f"{year}-{month}"
            continue
        if dry_run:
            summary["written"].append((year, month))
            continue
        rows = fetch_month_rows(symbol, interval, year, month)
        if rows is None:
            summary["errors"].append(f"{year}-{month}: no archive on data.binance.vision")
            continue
        klines = [kline_from_csv_row(row, symbol, interval) for row in rows]
        errors, warnings = validate_month(klines, interval, year, month)
        if errors:
            summary["errors"].append(f"{year}-{month}: {'; '.join(errors)}")
            continue
        if prev_last_row is None:
            prev_path = os.path.join(BINANCE_DIR, month_filename(symbol, interval, *last_month_before(year, month)))
            prev_last_row = _last_kline_on_disk(prev_path) if os.path.exists(prev_path) else None
        if prev_last_row is not None:
            hole = missing_bars_between(prev_last_row, klines[0], interval)
            if hole:
                summary["gaps"].append(f"{hole} bars between {prev_label} and {year}-{month}")
        for warning in warnings:
            summary["gaps"].append(f"{year}-{month}: {warning}")
        write_month(path, klines)
        summary["written"].append((year, month))
        prev_last_row = klines[-1]
        prev_label = f"{year}-{month}"
        print(f"  {symbol} {interval} {year}-{month}: {len(klines)} bars" + (f" ({'; '.join(warnings)})" if warnings else ""))
        sys.stdout.flush()
    return summary


def last_month_before(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def derive_altcoin_months(symbols=None, dry_run=False):
    """Materialise each altcoin's 1d files under binance/ (read raw by the config_builder 1d path)
    and its 1h derived cache under binance/derived/ (read by the 1h path), from the 1m source."""
    import pandas as pd

    from trainer.derive_cache import derive_bars, ensure_derived

    summaries = []
    for symbol in symbols or ALTCOINS:
        pattern = re.compile(_FILE_PATTERN.format(symbol=symbol, interval="1m"))
        months = sorted(
            (int(m.group(1)), int(m.group(2)))
            for m in (pattern.search(os.path.basename(p)) for p in glob.glob(os.path.join(BINANCE_DIR, f"{symbol}-1m-*.json")))
            if m
        )
        summary = {"symbol": symbol, "months": len(months), "derived_1d": 0, "skipped_1d": 0, "derived_1h": 0}
        for (year, month) in months:
            dst = os.path.join(BINANCE_DIR, month_filename(symbol, "1d", year, month))
            if os.path.exists(dst):
                summary["skipped_1d"] += 1
                continue
            if dry_run:
                summary["derived_1d"] += 1
                continue
            src = os.path.join(BINANCE_DIR, month_filename(symbol, "1m", year, month))
            derive_bars(pd.read_json(src), _INTERVAL_MS["1d"]).to_json(dst, orient="records")
            summary["derived_1d"] += 1
        if not dry_run:
            cached = ensure_derived(symbol, months, "1h", cache_dir=os.path.join(BINANCE_DIR, "derived"))
            summary["derived_1h"] = len(cached)
        summaries.append(summary)
        print(f"  {symbol}: 1d {summary['derived_1d']} derived / {summary['skipped_1d']} already present, 1h cache {summary['derived_1h']} months")
        sys.stdout.flush()
    return summaries


def _print_download_summary(summaries):
    print("\nsymbol     tf  written  skipped  range                 gaps")
    for s in summaries:
        span = f"{s['written'][0][0]}-{s['written'][0][1]} .. {s['written'][-1][0]}-{s['written'][-1][1]}" if s["written"] else "-"
        print(f"{s['symbol']:<10} {s['interval']:<3} {len(s['written']):>7} {s['skipped']:>8}  {span:<21} {len(s['gaps'])}")
        for gap in s["gaps"]:
            print(f"    gap: {gap}")
        for error in s["errors"]:
            print(f"    ERROR: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--through", help="last month to fetch as YYYY-MM (default: last complete month)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without downloading")
    parser.add_argument("--skip-derive", action="store_true", help="skip the altcoin 1h/1d derivation step")
    args = parser.parse_args(argv)
    if args.through:
        year, month = args.through.split("-")
        through = (int(year), int(month))
    else:
        through = latest_complete_month()
    os.chdir(_REPO_ROOT)

    print(f"Backfilling binance/ through {through[0]}-{through[1]}")
    summaries = []
    for (symbol, interval) in DOWNLOAD_TARGETS:
        print(f"{symbol} {interval}:")
        summaries.append(backfill_series(symbol, interval, through, dry_run=args.dry_run))
    _print_download_summary(summaries)

    if not args.skip_derive:
        print("\nDeriving altcoin 1d (binance/) + 1h (binance/derived/):")
        derive_altcoin_months(dry_run=args.dry_run)

    failed = [s for s in summaries if s["errors"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
