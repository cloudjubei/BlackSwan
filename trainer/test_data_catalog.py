import re

from trainer import data_catalog
from trainer.data_catalog import BINANCE, COMMODITIES, CRYPTO, FX, STOCKS, YFINANCE

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+$")


def _all_instruments():
    return [inst for cls in data_catalog.catalog() for inst in cls["instruments"]]


def _class(class_id):
    return next(c for c in data_catalog.catalog() if c["id"] == class_id)


# --- catalog structure ---


def test_catalog_has_the_four_asset_classes_in_order():
    assert [c["id"] for c in data_catalog.catalog()] == [CRYPTO, STOCKS, COMMODITIES, FX]


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
        assert inst["source"] in (BINANCE, YFINANCE), inst["symbol"]


def test_tiers_are_unique_within_a_class():
    for cls in data_catalog.catalog():
        tiers = [inst["tier"] for inst in cls["instruments"]]
        assert len(tiers) == len(set(tiers)), cls["id"]


def test_symbols_are_unique_across_the_whole_catalog():
    symbols = [inst["symbol"] for inst in _all_instruments()]
    assert len(symbols) == len(set(symbols))


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


def test_every_instrument_declares_a_bar_close_tz():
    by_sym = {i["symbol"]: i for i in _all_instruments()}
    for inst in by_sym.values():
        assert inst["barCloseTz"], inst["symbol"]
    # Crypto is 24/7 UTC; equities close 16:00 ET; FX rolls 17:00 NY; each commodity settles at its
    # own exchange time (none is 16:00 ET).
    assert by_sym["BTCUSDT"]["barCloseTz"] == "UTC"
    assert by_sym["AAPL"]["barCloseTz"] == "America/New_York 16:00"
    assert by_sym["EURUSD"]["barCloseTz"] == "America/New_York 17:00"
    assert by_sym["GOLD"]["barCloseTz"] == "America/New_York 13:30"
    assert by_sym["BRENT"]["barCloseTz"] == "Europe/London 19:30"


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
