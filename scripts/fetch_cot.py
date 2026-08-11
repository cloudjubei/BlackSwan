"""Mine CFTC Commitments-of-Traders (COT) managed-money positioning for the metals — the positioning/flow probe.

The disaggregated COT report (free, weekly, via the CFTC Socrata API) gives MANAGED-MONEY (speculator) long/short
positions per futures market. The probe tests whether an EXTREME in speculative net positioning (crowded long ->
contrarian short, or momentum) times the metal. Writes cot/<ASSET>.json as
    { "YYYY-MM-DD": {"net": <m_money_long - m_money_short>, "oi": <open_interest_all>}, ... }
keyed by the REPORT date (the Tuesday the positions were measured). The publication lag (released the following
Friday) is applied at JOIN time in trainer/cot.py — this file stores only what was measured.

Keyless Socrata (moderate use). One request per metal (history ~1052 weeks fits under the 50000 row cap). For a
metal whose contract was renamed over time (e.g. copper), the max-open-interest row per report date is kept, so
the main contract is followed across the rename automatically (MICRO contracts are excluded).

    python scripts/fetch_cot.py            # -> cot/{GOLD,SILVER,COPPER}.json
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

DATASET = "72hh-3qpy"  # Disaggregated Futures-Only, combined
TFF_DATASET = "gpe5-46if"  # Traders in Financial Futures (leveraged funds = the hedge-fund speculators)
# CROSS-ASSET-CLASS: local symbol -> TFF market_and_exchange_names. The leveraged-funds cohort is the financial
# analogue of managed-money; equity/rate specs include real hedgers so the CTA-collinearity is weaker here.
FINANCIAL = {
    "SPY": "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "IEF": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "TLT": "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
}


def fetch_financial(market_name):
    """{report_date: {net, oi}} for one financial future from the TFF report — net = leveraged-funds long-short,
    oi = total open interest. One market, so no max-OI contract selection is needed."""
    params = {
        "$select": "report_date_as_yyyy_mm_dd,lev_money_positions_long,lev_money_positions_short,open_interest_all",
        "market_and_exchange_names": market_name,
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "50000",
    }
    url = f"https://publicreporting.cftc.gov/resource/{TFF_DATASET}.json?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        rows = json.load(resp)
    out = {}
    for r in rows:
        try:
            oi = float(r.get("open_interest_all"))
            net = float(r.get("lev_money_positions_long")) - float(r.get("lev_money_positions_short"))
        except (TypeError, ValueError):
            continue
        date = str(r.get("report_date_as_yyyy_mm_dd", ""))[:10]
        if date and oi > 0:
            out[date] = {"net": net, "oi": oi}
    return out
# local symbol -> CFTC commodity_name (the max-open-interest contract per report date is kept, so the main
# NYMEX/CBOT/COMEX contract is followed across the many minor basis/spread markets sharing a commodity_name).
# NB NATGAS is intentionally excluded: its "NATURAL GAS" commodity_name is dominated by ICE swap/basis contracts,
# so the max-OI heuristic does not cleanly select the NYMEX Henry Hub futures that the NG=F price tracks.
COMMODITY = {
    "GOLD": "GOLD", "SILVER": "SILVER", "COPPER": "COPPER",
    "WTI": "CRUDE OIL", "CORN": "CORN", "WHEAT": "WHEAT",
}


def fetch_commodity(commodity_name):
    """{report_date: {net, oi}} for one commodity — the MAX-open-interest (main) contract per report date,
    excluding MICRO contracts, so a mid-history contract rename is followed transparently."""
    params = {
        "$select": "report_date_as_yyyy_mm_dd,market_and_exchange_names,"
                   "m_money_positions_long_all,m_money_positions_short_all,open_interest_all",
        "commodity_name": commodity_name,
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "50000",
    }
    url = f"https://publicreporting.cftc.gov/resource/{DATASET}.json?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        rows = json.load(resp)
    best = {}
    for r in rows:
        name = r.get("market_and_exchange_names", "")
        if "MICRO" in name.upper():
            continue
        try:
            oi = float(r.get("open_interest_all"))
            long_ = float(r.get("m_money_positions_long_all"))
            short_ = float(r.get("m_money_positions_short_all"))
        except (TypeError, ValueError):
            continue
        date = str(r.get("report_date_as_yyyy_mm_dd", ""))[:10]
        if not date or oi <= 0:
            continue
        if date not in best or oi > best[date]["oi"]:
            best[date] = {"net": long_ - short_, "oi": oi}
    return best


def main():
    parser = argparse.ArgumentParser(description="Mine CFTC COT managed-money positioning for the metals")
    parser.add_argument("--out-dir", default="cot")
    parser.add_argument("--symbol", help="restrict to one local symbol (GOLD/SILVER/COPPER)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    symbols = [args.symbol] if args.symbol else list(COMMODITY)
    for sym in symbols:
        table = fetch_financial(FINANCIAL[sym]) if sym in FINANCIAL else fetch_commodity(COMMODITY[sym])
        with open(os.path.join(args.out_dir, f"{sym}.json"), "w") as fh:
            json.dump(dict(sorted(table.items())), fh)
        days = sorted(table)
        print(f"{sym:8s} {len(table)} weeks  {days[0] if days else '-'}..{days[-1] if days else '-'}")


if __name__ == "__main__":
    main()
