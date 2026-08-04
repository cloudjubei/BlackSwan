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
ETFS = "etfs"                  # sector/thematic ETFs — tradeable + used as asset-linkage proxies
MACRO = "macro"                # US macro-economic series (point-in-time)
FUNDAMENTALS = "fundamentals"  # company fundamentals (point-in-time)

# Source ids -> which miner fetches the instrument.
BINANCE = "binance"      # data.binance.vision monthly archives (scripts.backfill_klines)
YFINANCE = "yfinance"    # yfinance daily bars (scripts.backfill_market / backfill_stocks)
FRED = "fred"            # FRED/ALFRED point-in-time vintages (scripts.backfill_macro)
EDGAR = "edgar"          # SEC EDGAR companyfacts, filing-date stamped (scripts.backfill_fundamentals)


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


def _etf(symbol, label, tier):
    return Instrument(symbol, label, ETFS, YFINANCE, symbol, "etfs", ("1d",), tier, _EQUITY_CLOSE)


def _macro(series_id, label, tier, publish_time):
    # Macro/fundamentals are RELEASE series, not klines — one file per series, timeframe "release". The
    # bar-close is the actual publish wall-clock (the point-in-time anchor the leakage-guard fusion uses).
    return Instrument(series_id, label, MACRO, FRED, series_id, "macro", ("release",), tier, "America/New_York " + publish_time)


def _fundamental(ticker, label, tier):
    return Instrument(ticker, label, FUNDAMENTALS, EDGAR, ticker, "fundamentals", ("release",), tier, "America/New_York (filing date)")


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

# US macro starter set (FRED series ids). ISM/PMI intentionally omitted (removed from FRED for licensing).
# Publish wall-clock per series: 08:30 ET data drops, 14:00 FOMC, 16:15 H.15 rates (see pit_fusion).
_MACRO = [
    _macro("UNRATE", "Unemployment rate", 1, "08:30"),
    _macro("PAYEMS", "Nonfarm payrolls", 2, "08:30"),
    _macro("CPIAUCNS", "CPI (NSA)", 3, "08:30"),
    _macro("CPIAUCSL", "CPI (SA)", 4, "08:30"),
    _macro("PCEPILFE", "Core PCE price index", 5, "08:30"),
    _macro("ICSA", "Initial jobless claims", 6, "08:30"),
    _macro("RSAFS", "Advance retail sales", 7, "08:30"),
    _macro("GDPC1", "Real GDP", 8, "08:30"),
    _macro("DFEDTARU", "Fed funds target (upper)", 9, "14:00"),
    _macro("DFF", "Fed funds effective", 10, "16:00"),
    _macro("DGS10", "10Y Treasury yield", 11, "16:15"),
    _macro("T10Y2Y", "10Y-2Y yield spread", 12, "16:15"),
    _macro("DFII10", "10Y real yield (TIPS)", 13, "16:15"),
]

# Company fundamentals (point-in-time via EDGAR) for the catalogued US stocks. Symbols are the tickers —
# they intentionally mirror the stocks class (fundamentals augment a stock), so lookups/mining scope by class.
_FUNDAMENTALS = [
    _fundamental("NVDA", "NVIDIA fundamentals", 1),
    _fundamental("MSFT", "Microsoft fundamentals", 2),
    _fundamental("AAPL", "Apple fundamentals", 3),
    _fundamental("GOOGL", "Alphabet fundamentals", 4),
    _fundamental("AMZN", "Amazon fundamentals", 5),
    _fundamental("META", "Meta Platforms fundamentals", 6),
    _fundamental("AVGO", "Broadcom fundamentals", 7),
    _fundamental("TSLA", "Tesla fundamentals", 8),
    _fundamental("JPM", "JPMorgan Chase fundamentals", 9),
    _fundamental("WMT", "Walmart fundamentals", 10),
]

# Sector/thematic ETFs — tradeable in their own right AND the mineable proxies the linkage graph points
# at (semiconductors, lithium, energy, financials, airlines, gold miners). yfinance daily, equity close.
_ETFS = [
    _etf("SOXX", "iShares Semiconductor ETF", 1),
    _etf("SMH", "VanEck Semiconductor ETF", 2),
    _etf("LIT", "Global X Lithium & Battery ETF", 3),
    _etf("XLE", "Energy Select Sector SPDR", 4),
    _etf("XLF", "Financial Select Sector SPDR", 5),
    _etf("JETS", "US Global Jets ETF", 6),
    _etf("GDX", "VanEck Gold Miners ETF", 7),
    _etf("SPY", "SPDR S&P 500 ETF (broad-equity risk proxy)", 8),
    _etf("UUP", "Invesco DB US Dollar Bullish ETF (dollar proxy)", 9),
]

_CLASSES = [
    (CRYPTO, "Crypto", "binance", _CRYPTO),
    (STOCKS, "US Stocks", "stocks", _STOCKS),
    (COMMODITIES, "Commodities", "commodities", _COMMODITIES),
    (FX, "FX", "fx", _FX),
    (ETFS, "ETFs", "etfs", _ETFS),
    (MACRO, "US Macro", "macro", _MACRO),
    (FUNDAMENTALS, "Fundamentals", "fundamentals", _FUNDAMENTALS),
]

# First-writer-wins so a bare symbol resolves to its PRICE instrument (stocks precede fundamentals), while
# a class-scoped lookup reaches the fundamentals/macro one.
_BY_SYMBOL = {}
for _, _, _, _insts in _CLASSES:
    for _inst in _insts:
        _BY_SYMBOL.setdefault(_inst.symbol, _inst)


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


def instrument(symbol: str, asset_class: Optional[str] = None) -> Optional[Instrument]:
    """The instrument for a local ``symbol``, or ``None`` if it isn't catalogued. When ``asset_class`` is
    given, resolve WITHIN that class (so a fundamentals ``AAPL`` is reachable distinctly from the stock
    ``AAPL``); otherwise a bare symbol resolves to its price instrument (first-writer-wins)."""
    if asset_class is not None:
        for inst in instruments():
            if inst.symbol == symbol and inst.asset_class == asset_class:
                return inst
        return None
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
