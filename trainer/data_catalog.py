"""Declarative catalog of the market data BlackSwan can mine, across asset classes.

The single static "menu" of WHAT data exists to acquire: asset class -> instruments, each carrying its
source + exact source symbol + native intervals + on-disk directory + a tier (rank within the class).
Consumed by BOTH the miner (which symbol to fetch from where) and the catalog emitter (what to show as
available-to-download). On-disk COVERAGE is layered on at emit time from ``data_inventory.scan_coverage``
-- this registry is the menu, the inventory is what's in the fridge.

Symbols are the local filename stem (``[A-Z0-9]+``, matching the kline filename grammar); ``source_symbol``
is the exact ticker at the source (e.g. ``GC=F`` gold futures, ``EURUSD=X`` FX). Prices for stocks /
commodities / FX are daily only; crypto is mined at 1m and 1h/1d are derived (``derive_cache``).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Asset-class ids.
CRYPTO = "crypto"
STOCKS = "stocks"
COMMODITIES = "commodities"
FX = "fx"

# Source ids -> which miner fetches the instrument.
BINANCE = "binance"      # data.binance.vision monthly archives (scripts.backfill_klines)
YFINANCE = "yfinance"    # yfinance daily bars (scripts.backfill_market / backfill_stocks)


@dataclass(frozen=True)
class Instrument:
    symbol: str            # local symbol / filename stem, [A-Z0-9]+
    label: str             # human label
    asset_class: str       # CRYPTO | STOCKS | COMMODITIES | FX
    source: str            # BINANCE | YFINANCE
    source_symbol: str     # exact ticker at the source
    directory: str         # on-disk dir under the repo root
    intervals: Tuple[str, ...]  # timeframes usable for training (crypto 1h/1d are derived from 1m)
    tier: int              # rank within the class (1 = top)
    # When a bar CLOSES, as "<tz> <HH:MM>" (or "UTC" for 24/7 crypto). Settlement/close differs per
    # market and NONE is 16:00 ET; a same-calendar-date join across classes without honouring this
    # imports later-session information (a 1-7h look-ahead). Used by the point-in-time fusion loader.
    bar_close_tz: str

# Equity regular-session and FX rollover closes (New York wall-clock).
_EQUITY_CLOSE = "America/New_York 16:00"
_FX_CLOSE = "America/New_York 17:00"


def _crypto(symbol, label, tier):
    # BTC has native 1m/1h/1d archives; altcoins are mined at 1m with 1h/1d derived. Either way the
    # three timeframes are usable for a run, so the catalog advertises all three. Crypto trades 24/7 —
    # bars align to UTC midnight.
    return Instrument(symbol, label, CRYPTO, BINANCE, symbol, "binance", ("1m", "1h", "1d"), tier, "UTC")


def _stock(symbol, label, tier):
    return Instrument(symbol, label, STOCKS, YFINANCE, symbol, "stocks", ("1d",), tier, _EQUITY_CLOSE)


def _commodity(symbol, label, source_symbol, tier, settle):
    return Instrument(symbol, label, COMMODITIES, YFINANCE, source_symbol, "commodities", ("1d",), tier, settle)


def _fx(symbol, label, source_symbol, tier):
    return Instrument(symbol, label, FX, YFINANCE, source_symbol, "fx", ("1d",), tier, _FX_CLOSE)


# Top-5 crypto (tiers 1-5) plus the remaining coins already on disk (tiers 6-9), so nothing acquired is
# hidden from the view. Ranked roughly by market cap among the coins Binance carries here.
_CRYPTO = [
    _crypto("BTCUSDT", "Bitcoin", 1),
    _crypto("ETHUSDT", "Ethereum", 2),
    _crypto("SOLUSDT", "Solana", 3),
    _crypto("XRPUSDT", "XRP", 4),
    _crypto("DOGEUSDT", "Dogecoin", 5),
    _crypto("ADAUSDT", "Cardano", 6),
    _crypto("DOTUSDT", "Polkadot", 7),
    _crypto("LTCUSDT", "Litecoin", 8),
    _crypto("SHIBUSDT", "Shiba Inu", 9),
]

# Top-10 US stocks by market cap (plain common-stock tickers).
_STOCKS = [
    _stock("NVDA", "NVIDIA", 1),
    _stock("MSFT", "Microsoft", 2),
    _stock("AAPL", "Apple", 3),
    _stock("GOOGL", "Alphabet", 4),
    _stock("AMZN", "Amazon", 5),
    _stock("META", "Meta Platforms", 6),
    _stock("AVGO", "Broadcom", 7),
    _stock("TSLA", "Tesla", 8),
    _stock("JPM", "JPMorgan Chase", 9),
    _stock("WMT", "Walmart", 10),
]

# Top commodities by liquidity, via yfinance continuous-front-month futures tickers. Each settles at
# its own exchange time (COMEX/NYMEX ~13:00-14:30 ET, ICE Brent ~19:30 London) — not the equity close.
_COMMODITIES = [
    _commodity("GOLD", "Gold", "GC=F", 1, "America/New_York 13:30"),
    _commodity("WTI", "WTI Crude Oil", "CL=F", 2, "America/New_York 14:30"),
    _commodity("BRENT", "Brent Crude Oil", "BZ=F", 3, "Europe/London 19:30"),
    _commodity("NATGAS", "Natural Gas", "NG=F", 4, "America/New_York 14:30"),
    _commodity("SILVER", "Silver", "SI=F", 5, "America/New_York 13:25"),
    _commodity("COPPER", "Copper", "HG=F", 6, "America/New_York 13:00"),
    _commodity("CORN", "Corn", "ZC=F", 7, "America/New_York 14:20"),
    _commodity("WHEAT", "Wheat", "ZW=F", 8, "America/New_York 14:20"),
]

# Top-5 FX majors by turnover, via yfinance `=X` pair tickers (no real volume -> neutral fill).
_FX = [
    _fx("EURUSD", "Euro / US Dollar", "EURUSD=X", 1),
    _fx("USDJPY", "US Dollar / Japanese Yen", "USDJPY=X", 2),
    _fx("GBPUSD", "British Pound / US Dollar", "GBPUSD=X", 3),
    _fx("AUDUSD", "Australian Dollar / US Dollar", "AUDUSD=X", 4),
    _fx("USDCAD", "US Dollar / Canadian Dollar", "USDCAD=X", 5),
]

_CLASSES = [
    (CRYPTO, "Crypto", "binance", _CRYPTO),
    (STOCKS, "US Stocks", "stocks", _STOCKS),
    (COMMODITIES, "Commodities", "commodities", _COMMODITIES),
    (FX, "FX", "fx", _FX),
]

_BY_SYMBOL = {inst.symbol: inst for _, _, _, insts in _CLASSES for inst in insts}


def _instrument_dict(inst: Instrument) -> dict:
    return {
        "symbol": inst.symbol,
        "label": inst.label,
        "assetClass": inst.asset_class,
        "source": inst.source,
        "sourceSymbol": inst.source_symbol,
        "intervals": list(inst.intervals),
        "directory": inst.directory,
        "tier": inst.tier,
        "barCloseTz": inst.bar_close_tz,
    }


def instruments() -> List[Instrument]:
    """Every catalogued instrument, flat."""
    return [inst for _, _, _, insts in _CLASSES for inst in insts]


def instrument(symbol: str) -> Optional[Instrument]:
    """The instrument for a local ``symbol``, or ``None`` if it isn't catalogued."""
    return _BY_SYMBOL.get(symbol)


def catalog() -> List[dict]:
    """The static menu: a list of asset-class dicts, each with its JSON-serializable instruments."""
    return [
        {
            "id": class_id,
            "label": label,
            "directory": directory,
            "instruments": [_instrument_dict(inst) for inst in insts],
        }
        for (class_id, label, directory, insts) in _CLASSES
    ]


def build_catalog(coverage_by_directory: dict) -> List[dict]:
    """The menu joined with on-disk coverage: each instrument gains an ``onDisk`` map (timeframe ->
    ``{start, end, months, gaps}``) looked up by its directory + symbol, or ``{}`` when nothing is on
    disk. ``coverage_by_directory`` maps a directory (``binance``/``stocks``/…) to that dir's
    ``scan_coverage`` result."""
    out = []
    for cls in catalog():
        instruments_with_coverage = []
        for inst in cls["instruments"]:
            on_disk = coverage_by_directory.get(inst["directory"], {}).get(inst["symbol"], {})
            instruments_with_coverage.append({**inst, "onDisk": on_disk})
        out.append({**cls, "instruments": instruments_with_coverage})
    return out
