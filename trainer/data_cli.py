"""The data-mine CLI: emit the data catalog and mine a requested slice on demand.

Two subcommands, each writing a JSON summary to ``--out`` (the ``{summaryOut}`` the orchestrator reads):

    python -m trainer.data_cli catalog --out <path>
        -> {"assetClasses": [...]}  the static registry joined with on-disk coverage.

    python -m trainer.data_cli mine --request <path> --out <path> [--through YYYY-MM] [--dry-run]
        -> {"mined": [{symbol, source, written, skipped, errors, gaps}], "unknown": [...], "through": "Y-M"}
        request JSON = {"symbols": [...]} or {"class": "commodities"}, optional {"intervals": [...], "through": "Y-M"}.

Mining is idempotent (skips months already on disk) and dispatches by the catalogued source: crypto via
the Binance archive backfill (+ altcoin 1h/1d derivation from the 1m source), everything else via the
yfinance daily miner. This is the seam the model-trainer invokes through the manifest's ``dataCatalog`` /
``mineData`` command templates.
"""

import argparse
import json
import os
import sys

from scripts.backfill_fundamentals import backfill_ticker_fundamentals
from scripts.backfill_klines import backfill_series, derive_altcoin_months, latest_complete_month
from scripts.backfill_macro import backfill_series_macro
from scripts.backfill_market import backfill_yf_symbol
from scripts.backfill_stocks import START_MONTH
from trainer import data_catalog, data_inventory, data_linkage

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BTC = "BTCUSDT"
# Directories holding point-in-time RELEASE series (not OHLCV klines) — scanned differently for coverage.
_SERIES_DIRECTORIES = {"macro", "fundamentals"}


def build_full_catalog(root="."):
    """The catalog menu joined with the on-disk coverage under ``root``. Price classes scan kline months;
    the point-in-time release classes (macro/fundamentals) scan their single-file series."""
    coverage_by_directory = {}
    for cls in data_catalog.catalog():
        directory = cls["directory"]
        path = os.path.join(root, directory)
        coverage_by_directory[directory] = (
            data_inventory.scan_series_coverage(path)
            if directory in _SERIES_DIRECTORIES
            else data_inventory.scan_coverage(path)
        )
    return data_catalog.build_catalog(coverage_by_directory)


def plan_mine(request):
    """Resolve a mine ``request`` into ``([(instrument, intervals)], unknown_symbols)``.

    ``symbols`` are resolved within the requested ``class`` when one is given (so a fundamentals ticker
    that collides with a stock ticker is reachable). ``intervals`` default to the instrument's own and are
    always filtered to what it supports, so a request for an unsupported timeframe is quietly narrowed."""
    requested_intervals = request.get("intervals")
    symbols = request.get("symbols")
    asset_class = request.get("class")
    if symbols:
        pool = [
            inst for inst in data_catalog.instruments() if asset_class is None or inst.asset_class == asset_class
        ]
        by_symbol = {}
        for inst in pool:
            by_symbol.setdefault(inst.symbol, inst)
        chosen, unknown = [], []
        for symbol in symbols:
            inst = by_symbol.get(symbol)
            (chosen if inst else unknown).append(inst if inst else symbol)
        instruments = chosen
    elif asset_class:
        instruments = [inst for inst in data_catalog.instruments() if inst.asset_class == asset_class]
        # An unknown/misspelled class matches nothing — surface it as unknown so the mine reports a
        # failure instead of looking like a successful empty download.
        unknown = [] if instruments else [asset_class]
    else:
        return [], []
    targets = []
    for inst in instruments:
        intervals = requested_intervals or list(inst.intervals)
        intervals = [i for i in intervals if i in inst.intervals]
        targets.append((inst, intervals))
    return targets, unknown


def _series_result(inst, summary):
    return {
        "symbol": inst.symbol,
        "source": inst.source,
        "written": summary["written"],
        "skipped": 0,
        "errors": summary["errors"],
        "gaps": [],
        "observations": summary.get("observations", 0),
    }


def mine_one(inst, intervals, through, dry_run=False):
    """Mine one instrument, dispatching by source. Returns a normalised per-symbol result dict."""
    # A narrowed-to-empty interval set (an all-unsupported request that plan_mine quietly narrowed) is a
    # genuine no-op for EVERY source — not just Binance — so the same request shape behaves consistently.
    if not intervals:
        return {"symbol": inst.symbol, "source": inst.source, "written": 0, "skipped": 0, "errors": [], "gaps": []}
    if inst.source == data_catalog.FRED:
        return _series_result(inst, backfill_series_macro(inst.source_symbol, inst.directory, dry_run=dry_run))
    if inst.source == data_catalog.EDGAR:
        return _series_result(inst, backfill_ticker_fundamentals(inst.source_symbol, inst.directory, dry_run=dry_run))
    if inst.source == data_catalog.BINANCE:
        is_btc = inst.symbol == _BTC
        # Only BTC has native 1h/1d archives; altcoins are archived at 1m only and derive 1h/1d, so an
        # altcoin always mines 1m (whatever timeframe was asked) and then derives.
        native = ["1m", "1h", "1d"] if is_btc else ["1m"]
        # Fall back to `native` only when SOMETHING was asked (a coarse altcoin timeframe narrows to []
        # here but must still mine 1m to derive from). An empty `intervals` — a narrowed-to-empty
        # all-unsupported request — stays a genuine no-op instead of mining every native timeframe.
        to_fetch = [i for i in intervals if i in native] or (native if intervals else [])
        written = skipped = 0
        errors, gaps = [], []
        for interval in to_fetch:
            summary = backfill_series(inst.symbol, interval, through, dry_run=dry_run)
            written += len(summary["written"])
            skipped += summary["skipped"]
            errors += summary["errors"]
            gaps += summary["gaps"]
        if not is_btc and to_fetch:
            derive_altcoin_months([inst.symbol], dry_run=dry_run)
        return {"symbol": inst.symbol, "source": inst.source, "written": written, "skipped": skipped, "errors": errors, "gaps": gaps}
    summary = backfill_yf_symbol(inst.symbol, inst.source_symbol, inst.directory, START_MONTH, through, dry_run=dry_run)
    return {
        "symbol": inst.symbol,
        "source": inst.source,
        "written": len(summary["written"]),
        "skipped": summary["skipped"],
        "errors": summary["errors"],
        "gaps": summary["gaps"],
    }


def _through(value):
    if value:
        year, month = value.split("-")
        return (int(year), int(month))
    return latest_complete_month()


def _write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def _run_catalog(args):
    _write_json(args.out, {"assetClasses": build_full_catalog("."), "linkage": data_linkage.linkage()})
    return 0


def _run_mine(args):
    with open(args.request) as handle:
        request = json.load(handle)
    through = _through(args.through or request.get("through"))
    targets, unknown = plan_mine(request)
    mined = [mine_one(inst, intervals, through, dry_run=args.dry_run) for (inst, intervals) in targets]
    _write_json(args.out, {"mined": mined, "unknown": unknown, "through": f"{through[0]}-{through[1]}"})
    failed = any(r["errors"] for r in mined) or bool(unknown)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    catalog_parser = sub.add_parser("catalog", help="emit the data catalog joined with on-disk coverage")
    catalog_parser.add_argument("--out", required=True, help="write the catalog JSON here ({summaryOut})")
    catalog_parser.set_defaults(func=_run_catalog)

    mine_parser = sub.add_parser("mine", help="mine a requested slice on demand (idempotent)")
    mine_parser.add_argument("--request", required=True, help="read the mine request JSON here ({configPath})")
    mine_parser.add_argument("--out", required=True, help="write the mine result JSON here ({summaryOut})")
    mine_parser.add_argument("--through", help="last month to fetch as YYYY-MM (default: last complete month)")
    mine_parser.add_argument("--dry-run", action="store_true", help="plan + validate, write nothing")
    mine_parser.set_defaults(func=_run_mine)

    args = parser.parse_args(argv)
    if args.command != "catalog":
        os.chdir(_REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
