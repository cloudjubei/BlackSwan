"""Mine the EIA ENERGY complex M1/M2 futures settlements for the POOLED roll-basket probe (trainer/roll_basket.py).

The pooled probe trades the monthly M1-M2 roll spread across the four EIA energy legs as an equal-weight
portfolio, so diversification lifts the t-stat single-asset WTI lacked. Each leg's front two contracts are free
EIA series (RAW per-contract settlements, NOT a continuous series). Writes eia-energy/<SYMBOL>.json as
    { "YYYY-MM-DD": {"m1": .., "m2": ..}, ... }

Needs a free EIA_API_KEY (in the environment / the backend env file) and api.eia.gov reachable.

    EIA_API_KEY=... python scripts/fetch_energy.py            # -> eia-energy/{CRUDE,HEATOIL,RBOB,NATGAS}.json
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

# symbol -> (dataset path, {leg: EIA series id}). Front two contracts only (the spread needs M1 and M2).
LEGS = {
    "CRUDE": ("petroleum/pri/fut", {"m1": "RCLC1", "m2": "RCLC2"}),
    "HEATOIL": ("petroleum/pri/fut", {"m1": "EER_EPD2F_PE1_Y35NY_DPG", "m2": "EER_EPD2F_PE2_Y35NY_DPG"}),
    "RBOB": ("petroleum/pri/fut", {"m1": "EER_EPMRR_PE1_Y35NY_DPG", "m2": "EER_EPMRR_PE2_Y35NY_DPG"}),
    "NATGAS": ("natural-gas/pri/fut", {"m1": "RNGC1", "m2": "RNGC2"}),
}
PAGE = 5000


def _fetch_page(api_key, dataset, series_ids, offset, start):
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
    for sid in series_ids:
        params.append(("facets[series][]", sid))
    url = f"https://api.eia.gov/v2/{dataset}/data/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["response"]


def fetch_symbol(api_key, symbol, start="2000-01-01"):
    """{date: {m1, m2}} for one energy leg, positive settlements only, paginated to exhaustion."""
    dataset, leg_series = LEGS[symbol]
    series_to_leg = {sid: leg for leg, sid in leg_series.items()}
    table = {}
    offset = 0
    total = None
    while total is None or offset < total:
        payload = _fetch_page(api_key, dataset, list(series_to_leg), offset, start)
        total = int(payload.get("total", 0))
        rows = payload.get("data", [])
        if not rows:
            break
        for row in rows:
            leg = series_to_leg.get(row.get("series"))
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
        time.sleep(0.2)
    return table


def main():
    parser = argparse.ArgumentParser(description="Mine EIA energy-complex M1/M2 settlements for the roll-basket probe")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--api-key", default=os.environ.get("EIA_API_KEY", ""))
    parser.add_argument("--out-dir", default="eia-energy")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("no EIA API key — set EIA_API_KEY in the environment or pass --api-key")

    os.makedirs(args.out_dir, exist_ok=True)
    for symbol in LEGS:
        table = fetch_symbol(args.api_key, symbol, start=args.start)
        both = {d: v for d, v in table.items() if "m1" in v and "m2" in v}
        with open(os.path.join(args.out_dir, f"{symbol}.json"), "w") as fh:
            json.dump(dict(sorted(table.items())), fh)
        days = sorted(both)
        print(f"{symbol:8s} {len(both)} both-leg dates  {days[0] if days else '-'}..{days[-1] if days else '-'}")


if __name__ == "__main__":
    main()
