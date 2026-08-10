"""Mine WTI crude futures settlements (M1..M4) for the commodity index-roll probe (trainer/roll.py).

EIA publishes the daily settlement of the front four NYMEX WTI (Cushing) contracts as free series
RCLC1..RCLC4 — the RAW per-contract settlements the roll trade needs (NOT a back-adjusted continuous
series, which would bake the roll in). Writes eia-wti/<asset>.json as
    { "YYYY-MM-DD": {"m1": .., "m2": .., "m3": .., "m4": ..}, ... }

Needs a free EIA API key (https://www.eia.gov/opendata/register.php). Provide it via EIA_API_KEY in the
environment (the backend env file already carries it) or --api-key. Reachability note: api.eia.gov must be
resolvable from wherever this runs — run it on the host, not a network-restricted sandbox.

    EIA_API_KEY=... python scripts/fetch_wti.py            # -> eia-wti/WTI.json
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

SERIES_TO_LEG = {"RCLC1": "m1", "RCLC2": "m2", "RCLC3": "m3", "RCLC4": "m4"}
EIA_ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
PAGE = 5000


def _fetch_page(api_key, offset, start):
    params = [
        ("api_key", api_key),
        ("frequency", "daily"),
        ("data[0]", "value"),
        ("start", start),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(PAGE)),
    ]
    for s in SERIES_TO_LEG:
        params.append(("facets[series][]", s))
    url = f"{EIA_ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["response"]


def fetch_settlements(api_key, start="2000-01-01"):
    """All RCLC1..RCLC4 daily settlements from ``start``, folded into {date: {m1,m2,m3,m4}}. Non-positive
    settlements are DROPPED (this excluded the 2020-04-20 negative WTI print — outside any roll trade window, so
    harmless to the roll probe, but recorded as a deliberate data choice). Paginates until the total is exhausted."""
    table = {}
    offset = 0
    total = None
    while total is None or offset < total:
        payload = _fetch_page(api_key, offset, start)
        total = int(payload.get("total", 0))
        rows = payload.get("data", [])
        if not rows:
            break
        for row in rows:
            leg = SERIES_TO_LEG.get(row.get("series"))
            val = row.get("value")
            if leg is None or val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v > 0:
                table.setdefault(row["period"], {})[leg] = v
        offset += len(rows)
        print(f"  fetched {offset}/{total}")
        time.sleep(0.2)
    return table


def main():
    parser = argparse.ArgumentParser(description="Mine WTI M1..M4 settlements from EIA for the roll probe")
    parser.add_argument("--asset", default="WTI")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--api-key", default=os.environ.get("EIA_API_KEY", ""))
    parser.add_argument("--out-dir", default="eia-wti")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("no EIA API key — set EIA_API_KEY in the environment or pass --api-key")

    table = fetch_settlements(args.api_key, start=args.start)
    both_legs = sum(1 for v in table.values() if "m1" in v and "m2" in v)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.asset}.json")
    with open(out_path, "w") as fh:
        json.dump(dict(sorted(table.items())), fh)
    days = sorted(table)
    print(f"wrote {out_path}: {len(table)} dates ({both_legs} with both M1+M2), {days[0] if days else '-'}..{days[-1] if days else '-'}")


if __name__ == "__main__":
    main()
