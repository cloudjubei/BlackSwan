"""Backfill macro/ with point-in-time US macro series from FRED/ALFRED (free API key).

The point is POINT-IN-TIME correctness: default FRED returns the LATEST-REVISED value for every historical
date (a silent leak — a model would see a number that didn't exist yet). FRED `output_type=4` returns each
observation AS INITIALLY RELEASED, stamped with the `realtime_start` date it became public — exactly the
release date the leakage-guard fusion needs. Output mirrors the fundamentals/EDGAR shape ({SERIES}.json =
a list of {refPeriod, releaseDate, value, vintage}) so fusion + coverage treat every release series alike.

Needs a free FRED API key in the FRED_API_KEY environment variable (the spawned mine command inherits the
host env, so exporting it once is enough).

Usage (from the repo root):
    FRED_API_KEY=... .venv/bin/python -m scripts.backfill_macro           # all catalogued macro series
    FRED_API_KEY=... .venv/bin/python -m scripts.backfill_macro --symbol UNRATE
    .venv/bin/python -m scripts.backfill_macro --dry-run
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from scripts.backfill_klines import write_month as write_json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
_MISSING = (None, "", ".")


def observations_url(series_id, api_key):
    """The FRED observations endpoint for a series, requesting INITIAL-RELEASE vintages (output_type=4)."""
    query = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 4,
    })
    return f"{_OBSERVATIONS_URL}?{query}"


def extract_macro_observations(payload):
    """Point-in-time observations from a FRED output_type=4 payload: each row's `realtime_start` is the
    release/vintage date (never the reference `date`). Missing (`.`) and non-numeric values are dropped."""
    out = []
    for obs in (payload or {}).get("observations", []) or []:
        value = obs.get("value")
        if value in _MISSING:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        out.append({
            "refPeriod": obs.get("date"),
            "releaseDate": obs.get("realtime_start"),
            "value": numeric,
            "vintage": obs.get("realtime_start"),
        })
    return out


def fetch_series_observations(series_id, api_key):
    request = urllib.request.Request(observations_url(series_id, api_key), headers={"User-Agent": "thefactory-datamine"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def backfill_series_macro(series_id, out_dir, api_key=None, dry_run=False):
    """Fetch + write one FRED series' point-in-time observations. Returns a summary dict."""
    summary = {"symbol": series_id, "written": 0, "observations": 0, "errors": []}
    api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")
    if not api_key:
        summary["errors"].append("no FRED_API_KEY — set it to mine macro series")
        return summary
    try:
        observations = extract_macro_observations(fetch_series_observations(series_id, api_key))
    except (urllib.error.URLError, OSError, ValueError) as error:
        summary["errors"].append(f"{series_id}: {error}")
        return summary
    summary["observations"] = len(observations)
    if not observations:
        summary["errors"].append(f"{series_id}: no observations returned")
        return summary
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        write_json(os.path.join(out_dir, f"{series_id}.json"), observations)
        summary["written"] = 1
    return summary


def _series_ids():
    from trainer import data_catalog

    return [inst.source_symbol for inst in data_catalog.instruments() if inst.asset_class == data_catalog.MACRO]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", help="restrict to one FRED series id (e.g. UNRATE)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + count, write nothing")
    args = parser.parse_args(argv)
    os.chdir(_REPO_ROOT)
    out_dir = os.path.join(_REPO_ROOT, "macro")

    series = [args.symbol] if args.symbol else _series_ids()
    print(f"Backfilling macro/ for {len(series)} series")
    summaries = []
    for series_id in series:
        summary = backfill_series_macro(series_id, out_dir, dry_run=args.dry_run)
        summaries.append(summary)
        print(f"  {series_id:<10} {summary['observations']} observations, written={summary['written']}")
        for error in summary["errors"]:
            print(f"    ERROR: {error}")
        sys.stdout.flush()
    return 1 if any(s["errors"] for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
