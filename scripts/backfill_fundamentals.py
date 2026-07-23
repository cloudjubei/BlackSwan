"""Backfill fundamentals/ with point-in-time company fundamentals from SEC EDGAR (free, no key).

EDGAR's `companyfacts` API is point-in-time BY CONSTRUCTION: every reported fact carries the actual
`filed` date it became public, so a fundamental is stamped at its FILING date (`releaseDate`), never the
fiscal-period end (which yfinance uses — a 45-90-day accounting look-ahead). Restatements are kept as
NEW dated rows, not overwrites. Output mirrors the macro release-series shape ({TICKER}.json = a list of
{concept, unit, refPeriod, releaseDate, value, form}) so the point-in-time fusion + coverage treat macro
and fundamentals uniformly.

Usage (from the repo root):
    .venv/bin/python -m scripts.backfill_fundamentals              # all catalogued fundamentals tickers
    .venv/bin/python -m scripts.backfill_fundamentals --symbol AAPL
    .venv/bin/python -m scripts.backfill_fundamentals --dry-run
"""

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

from scripts.backfill_klines import write_month as write_json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# EDGAR requires a descriptive User-Agent (it 403s a default urllib agent).
_USER_AGENT = "thefactory-datamine research (contact: data@thefactory.local)"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Highest-signal, universally-reported concepts. Revenues has two common tags; both are pulled and the
# fusion consumer picks whichever a company reports.
CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]


def cik_for_ticker(company_tickers, ticker):
    """The zero-padded 10-digit CIK for ``ticker`` (case-insensitive), or ``None``."""
    target = ticker.upper()
    for row in company_tickers.values():
        if str(row.get("ticker", "")).upper() == target:
            return f"{int(row['cik_str']):010d}"
    return None


def _duration_days(start, end):
    """Days between an ISO ``start`` and ``end`` date, or ``None`` when either is absent/malformed."""
    if not start or not end:
        return None
    try:
        return (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days
    except (TypeError, ValueError):
        return None


def extract_observations(facts, concepts):
    """Point-in-time observations for ``concepts`` from an EDGAR companyfacts payload, stamped at the FILING
    date (``releaseDate``).

    A flow concept (revenue/income/EPS) reports co-terminating DURATIONS — a 3-month quarter and the 6/9/
    12-month cumulative both ending the same day — so each observation carries its ``periodStart`` +
    ``durationDays`` and is keyed by ``(concept, unit, start, end, filed)``: the different durations stay
    DISTINCT (a consumer picks one consistent bucket) instead of collapsing into indistinguishable rows.
    Rows sharing that key (a same-day amendment) collapse to the latest accession — deterministic +
    revision-aware, never dependent on EDGAR's array order. Distinct filing DATES are kept (restatements).
    Rows missing a period end or filing date are dropped."""
    us_gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
    best = {}
    order = []
    for concept in concepts:
        node = us_gaap.get(concept)
        if not node:
            continue
        for unit, rows in (node.get("units") or {}).items():
            for row in rows or []:
                end, filed = row.get("end"), row.get("filed")
                if not end or not filed:
                    continue
                start = row.get("start")
                key = (concept, unit, start, end, filed)
                accn = str(row.get("accn") or "")
                if key not in best:
                    order.append(key)
                    best[key] = (accn, row, concept, unit, start, end, filed)
                elif accn >= best[key][0]:
                    best[key] = (accn, row, concept, unit, start, end, filed)
    out = []
    for key in order:
        _, row, concept, unit, start, end, filed = best[key]
        observation = {
            "concept": concept,
            "unit": unit,
            "refPeriod": end,
            "releaseDate": filed,
            "value": row.get("val"),
            "form": row.get("form"),
        }
        if start:
            observation["periodStart"] = start
        duration = _duration_days(start, end)
        if duration is not None:
            observation["durationDays"] = duration
        out.append(observation)
    return out


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_company_tickers():
    return _get_json(TICKERS_URL)


def fetch_company_facts(cik):
    return _get_json(FACTS_URL.format(cik=cik))


def backfill_ticker_fundamentals(ticker, out_dir, company_tickers=None, dry_run=False):
    """Fetch + write one ticker's point-in-time fundamentals. Returns a summary dict."""
    summary = {"symbol": ticker, "written": 0, "observations": 0, "errors": []}
    try:
        company_tickers = company_tickers if company_tickers is not None else fetch_company_tickers()
        cik = cik_for_ticker(company_tickers, ticker)
        if not cik:
            summary["errors"].append(f"no EDGAR CIK for {ticker}")
            return summary
        observations = extract_observations(fetch_company_facts(cik), CONCEPTS)
    except (urllib.error.URLError, OSError, ValueError) as error:
        summary["errors"].append(f"{ticker}: {error}")
        return summary
    summary["observations"] = len(observations)
    if not observations:
        summary["errors"].append(f"{ticker}: no observations for the tracked concepts")
        return summary
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, f"{ticker}.json"), observations)
        summary["written"] = 1
    return summary


def _tickers():
    from trainer import data_catalog

    return [inst.source_symbol for inst in data_catalog.instruments() if inst.asset_class == data_catalog.FUNDAMENTALS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", help="restrict to one ticker (e.g. AAPL)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + count, write nothing")
    args = parser.parse_args(argv)
    os.chdir(_REPO_ROOT)
    out_dir = os.path.join(_REPO_ROOT, "fundamentals")

    tickers = [args.symbol] if args.symbol else _tickers()
    company_tickers = fetch_company_tickers()
    print(f"Backfilling fundamentals/ for {len(tickers)} ticker(s)")
    summaries = []
    for ticker in tickers:
        summary = backfill_ticker_fundamentals(ticker, out_dir, company_tickers=company_tickers, dry_run=args.dry_run)
        summaries.append(summary)
        print(f"  {ticker:<6} {summary['observations']} observations, written={summary['written']}")
        for error in summary["errors"]:
            print(f"    ERROR: {error}")
        sys.stdout.flush()
    return 1 if any(s["errors"] for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
