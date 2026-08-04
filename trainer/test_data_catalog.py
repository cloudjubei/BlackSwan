import re

from trainer import data_catalog
from trainer.data_catalog import (
    BINANCE,
    COMMODITIES,
    CRYPTO,
    EDGAR,
    ETFS,
    FRED,
    FUNDAMENTALS,
    FX,
    MACRO,
    STOCKS,
    YFINANCE,
)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+$")


def _all_instruments():
    return [inst for cls in data_catalog.catalog() for inst in cls["instruments"]]


def _class(class_id):
    return next(c for c in data_catalog.catalog() if c["id"] == class_id)


# --- catalog structure ---


def test_catalog_has_the_asset_classes_in_order():
    assert [c["id"] for c in data_catalog.catalog()] == [
        CRYPTO,
        STOCKS,
        COMMODITIES,
        FX,
        ETFS,
        MACRO,
        FUNDAMENTALS,
    ]


def test_etfs_via_yfinance_are_the_linkage_proxies():
    etfs = _class(ETFS)["instruments"]
    syms = {i["symbol"] for i in etfs}
    assert {"SOXX", "SMH", "LIT", "XLE", "XLF", "JETS", "GDX"} <= syms
    assert all(i["source"] == YFINANCE and i["intervals"] == ["1d"] for i in etfs)


def test_broad_market_proxies_are_catalogued():
    # SPY (equities) + UUP (dollar) are the keyless macro-risk context proxies (with GOLD) — tradeable ETFs.
    by_sym = {i["symbol"]: i for i in _class(ETFS)["instruments"]}
    assert by_sym["SPY"]["sourceSymbol"] == "SPY"
    assert by_sym["UUP"]["sourceSymbol"] == "UUP"


def test_each_asset_class_carries_a_label_and_directory():
    for cls in data_catalog.catalog():
        assert cls["label"]
        assert cls["directory"]
        assert cls["instruments"]


def test_every_instrument_has_the_required_fields():
    for inst in _all_instruments():
        for key in ("symbol", "label", "source", "sourceSymbol", "intervals", "directory", "tier"):
            assert key in inst, (inst["symbol"], key)
        assert inst["intervals"], inst["symbol"]


def test_symbols_obey_the_kline_filename_grammar():
    for inst in _all_instruments():
        assert _SYMBOL_RE.match(inst["symbol"]), inst["symbol"]


def test_sources_are_a_known_miner():
    for inst in _all_instruments():
        assert inst["source"] in (BINANCE, YFINANCE, FRED, EDGAR), inst["symbol"]


def test_tiers_are_unique_within_a_class():
    for cls in data_catalog.catalog():
        tiers = [inst["tier"] for inst in cls["instruments"]]
        assert len(tiers) == len(set(tiers)), cls["id"]


def test_symbols_are_unique_within_each_class():
    # Fundamentals intentionally reuse the stock tickers (they augment a stock), so uniqueness is per
    # class, not global.
    for cls in data_catalog.catalog():
        symbols = [inst["symbol"] for inst in cls["instruments"]]
        assert len(symbols) == len(set(symbols)), cls["id"]


def test_directory_matches_the_class_for_every_instrument():
    for cls in data_catalog.catalog():
        assert {inst["directory"] for inst in cls["instruments"]} == {cls["directory"]}


# --- the headline picks the user asked to view ---


def test_top_5_crypto_are_the_expected_coins_via_binance():
    crypto = _class(CRYPTO)
    top5 = [i["symbol"] for i in sorted(crypto["instruments"], key=lambda x: x["tier"])[:5]]
    assert top5 == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    assert all(i["source"] == BINANCE for i in crypto["instruments"])


def test_top_10_stocks_present_via_yfinance():
    stocks = _class(STOCKS)
    syms = {i["symbol"] for i in stocks["instruments"]}
    assert {"NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT"} <= syms
    assert all(i["source"] == YFINANCE for i in stocks["instruments"])


def test_commodities_use_yfinance_futures_tickers():
    by_sym = {i["symbol"]: i for i in _class(COMMODITIES)["instruments"]}
    assert by_sym["GOLD"]["sourceSymbol"] == "GC=F"
    assert by_sym["WTI"]["sourceSymbol"] == "CL=F"
    assert all(i["source"] == YFINANCE and i["intervals"] == ["1d"] for i in by_sym.values())


def test_fx_top_5_majors_via_yfinance_pair_tickers():
    fx = _class(FX)
    syms = [i["symbol"] for i in sorted(fx["instruments"], key=lambda x: x["tier"])]
    assert syms[:5] == ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]
    for inst in fx["instruments"]:
        assert inst["sourceSymbol"].endswith("=X")


# --- lookup ---


def test_instrument_lookup_by_symbol():
    assert data_catalog.instrument("GOLD").source_symbol == "GC=F"
    assert data_catalog.instrument("BTCUSDT").source == BINANCE
    assert data_catalog.instrument("NOPE") is None


# --- bar-close timezone (mandatory for point-in-time cross-asset fusion) ---


def _by_symbol(class_id):
    return {i["symbol"]: i for i in _class(class_id)["instruments"]}


def test_every_instrument_declares_a_bar_close_tz():
    for inst in _all_instruments():
        assert inst["barCloseTz"], inst["symbol"]
    # Crypto is 24/7 UTC; equities close 16:00 ET; FX rolls 17:00 NY; each commodity settles at its
    # own exchange time (none is 16:00 ET).
    assert _by_symbol(CRYPTO)["BTCUSDT"]["barCloseTz"] == "UTC"
    assert _by_symbol(STOCKS)["AAPL"]["barCloseTz"] == "America/New_York 16:00"
    assert _by_symbol(FX)["EURUSD"]["barCloseTz"] == "America/New_York 17:00"
    assert _by_symbol(COMMODITIES)["GOLD"]["barCloseTz"] == "America/New_York 13:30"
    assert _by_symbol(COMMODITIES)["BRENT"]["barCloseTz"] == "Europe/London 19:30"


# --- macro + fundamentals (point-in-time release series) ---


def test_macro_series_via_fred_with_the_publish_time_as_anchor():
    macro = _by_symbol(MACRO)
    assert macro["UNRATE"]["source"] == FRED
    assert macro["UNRATE"]["sourceSymbol"] == "UNRATE"
    assert macro["UNRATE"]["intervals"] == ["release"]
    # The bar-close is the actual publish wall-clock — 08:30 data drops, 14:00 FOMC, 16:15 H.15 rates.
    assert macro["UNRATE"]["barCloseTz"] == "America/New_York 08:30"
    assert macro["DFEDTARU"]["barCloseTz"] == "America/New_York 14:00"
    assert macro["DGS10"]["barCloseTz"] == "America/New_York 16:15"


def test_fundamentals_via_edgar_mirror_the_stock_tickers():
    fundamentals = _class(FUNDAMENTALS)["instruments"]
    assert {"AAPL", "NVDA", "JPM"} <= {i["symbol"] for i in fundamentals}
    assert all(i["source"] == EDGAR and i["intervals"] == ["release"] for i in fundamentals)


def test_instrument_lookup_is_class_scoped_for_colliding_tickers():
    # A bare AAPL resolves to the stock price; scoping to fundamentals reaches the EDGAR one.
    assert data_catalog.instrument("AAPL").asset_class == STOCKS
    assert data_catalog.instrument("AAPL", FUNDAMENTALS).asset_class == FUNDAMENTALS
    assert data_catalog.instrument("AAPL", MACRO) is None
    assert data_catalog.instrument("UNRATE").asset_class == MACRO


# --- build_catalog: registry menu joined with on-disk coverage ---


def test_build_catalog_attaches_on_disk_coverage_by_directory_and_symbol():
    coverage = {
        "binance": {"BTCUSDT": {"1d": {"start": "2017-08", "end": "2026-06", "months": 107, "gaps": []}}},
        "stocks": {},
        "commodities": {},
        "fx": {},
    }
    built = data_catalog.build_catalog(coverage)
    btc = next(i for c in built for i in c["instruments"] if i["symbol"] == "BTCUSDT")
    assert btc["onDisk"]["1d"]["months"] == 107
    # An instrument with nothing on disk reports an empty coverage map, never a crash.
    gold = next(i for c in built for i in c["instruments"] if i["symbol"] == "GOLD")
    assert gold["onDisk"] == {}
