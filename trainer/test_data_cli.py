import json

import trainer.data_cli as cli
from trainer import data_catalog


# --- build_full_catalog: registry joined with on-disk coverage rooted at a dir ---


def test_build_full_catalog_merges_on_disk_coverage(tmp_path):
    com = tmp_path / "commodities"
    com.mkdir()
    (com / "GOLD-1d-2025-1.json").write_text("[]")
    (com / "GOLD-1d-2025-2.json").write_text("[]")
    cat = cli.build_full_catalog(str(tmp_path))
    gold = next(i for c in cat for i in c["instruments"] if i["symbol"] == "GOLD")
    assert gold["onDisk"]["1d"]["months"] == 2
    # An instrument with no file under this root reports empty coverage, never a crash.
    btc = next(i for c in cat for i in c["instruments"] if i["symbol"] == "BTCUSDT")
    assert btc["onDisk"] == {}


# --- plan_mine: resolve a request into concrete (instrument, intervals) targets ---


def test_plan_mine_by_symbols_preserves_order():
    targets, unknown = cli.plan_mine({"symbols": ["GOLD", "EURUSD"]})
    assert [t[0].symbol for t in targets] == ["GOLD", "EURUSD"]
    assert unknown == []


def test_plan_mine_by_class_selects_the_whole_class():
    targets, _ = cli.plan_mine({"class": "fx"})
    assert {t[0].symbol for t in targets} == {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"}


def test_plan_mine_reports_unknown_symbols_without_dropping_the_rest():
    targets, unknown = cli.plan_mine({"symbols": ["NOPE", "GOLD"]})
    assert [t[0].symbol for t in targets] == ["GOLD"]
    assert unknown == ["NOPE"]


def test_plan_mine_with_no_selection_is_empty():
    targets, unknown = cli.plan_mine({})
    assert targets == []
    assert unknown == []


def test_plan_mine_reports_an_unknown_class_instead_of_silently_no_opping():
    # A misspelled class must surface as `unknown` (so the mine reports failure), not look like a
    # successful empty mine.
    targets, unknown = cli.plan_mine({"class": "stcoks"})
    assert targets == []
    assert unknown == ["stcoks"]


def test_plan_mine_all_unsupported_intervals_narrows_to_empty():
    targets, _ = cli.plan_mine({"symbols": ["BTCUSDT"], "intervals": ["30m"]})
    assert targets[0][1] == []


def test_plan_mine_intervals_default_to_the_instruments_own():
    targets, _ = cli.plan_mine({"symbols": ["BTCUSDT"]})
    assert targets[0][1] == ["1m", "1h", "1d"]


def test_plan_mine_intervals_are_filtered_to_what_the_instrument_supports():
    targets, _ = cli.plan_mine({"symbols": ["GOLD"], "intervals": ["1m", "1d"]})
    assert targets[0][1] == ["1d"]  # a daily-only instrument keeps only 1d


# --- mine_one: dispatch to the right miner by source ---


def test_mine_one_dispatches_yfinance_with_local_and_source_symbol(monkeypatch):
    seen = {}

    def fake_yf(symbol, source_symbol, out_dir, start, through, dry_run=False):
        seen["args"] = (symbol, source_symbol, out_dir)
        return {"symbol": symbol, "written": [(2025, 1), (2025, 2)], "skipped": 1, "errors": [], "gaps": []}

    monkeypatch.setattr(cli, "backfill_yf_symbol", fake_yf)
    result = cli.mine_one(data_catalog.instrument("GOLD"), ["1d"], (2025, 2))
    assert seen["args"] == ("GOLD", "GC=F", "commodities")
    assert result["written"] == 2
    assert result["skipped"] == 1
    assert result["source"] == data_catalog.YFINANCE


def test_mine_one_with_no_intervals_mines_nothing(monkeypatch):
    # A narrowed-to-empty interval set (an all-unsupported request) must stay a no-op, not fall back to
    # mining every native timeframe. This holds for EVERY source, not just Binance.
    called = []
    monkeypatch.setattr(cli, "backfill_series", lambda *a, **k: called.append(a) or {"written": [], "skipped": 0, "errors": [], "gaps": []})
    res = cli.mine_one(data_catalog.instrument("BTCUSDT"), [], (2026, 1))
    assert called == []
    assert res["written"] == 0


def test_mine_one_empty_intervals_is_a_noop_for_yfinance_and_fred(monkeypatch):
    yf, fred = [], []
    monkeypatch.setattr(cli, "backfill_yf_symbol", lambda *a, **k: yf.append(a) or {"written": [(2025, 1)], "skipped": 0, "errors": [], "gaps": []})
    monkeypatch.setattr(cli, "backfill_series_macro", lambda *a, **k: fred.append(a) or {"written": 1, "observations": 5, "errors": []})
    assert cli.mine_one(data_catalog.instrument("GOLD"), [], (2025, 1))["written"] == 0
    assert cli.mine_one(data_catalog.instrument("UNRATE"), [], (2025, 1))["written"] == 0
    assert yf == [] and fred == []


def test_mine_one_dispatches_binance_per_interval_and_derives_for_altcoins(monkeypatch):
    fetched = []
    derived = []

    def fake_series(symbol, interval, through, dry_run=False):
        fetched.append((symbol, interval))
        return {"symbol": symbol, "interval": interval, "written": [(2026, 1)], "skipped": 0, "errors": [], "gaps": []}

    monkeypatch.setattr(cli, "backfill_series", fake_series)
    monkeypatch.setattr(cli, "derive_altcoin_months", lambda symbols=None, dry_run=False: derived.append(symbols))
    # BTC fetches every native interval, no derive.
    btc = cli.mine_one(data_catalog.instrument("BTCUSDT"), ["1m", "1h", "1d"], (2026, 1))
    assert set(fetched) == {("BTCUSDT", "1m"), ("BTCUSDT", "1h"), ("BTCUSDT", "1d")}
    assert btc["written"] == 3
    assert derived == []
    # An altcoin fetches ONLY 1m (1h/1d are derived), then derives.
    fetched.clear()
    eth = cli.mine_one(data_catalog.instrument("ETHUSDT"), ["1m", "1h", "1d"], (2026, 1))
    assert fetched == [("ETHUSDT", "1m")]
    assert derived == [["ETHUSDT"]]
    assert eth["written"] == 1


# --- main: the two subcommands write JSON summaries ---


def test_plan_mine_scopes_symbols_to_a_class_for_colliding_tickers():
    # A fundamentals AAPL (colliding with the stock AAPL) is reachable by scoping the class.
    targets, unknown = cli.plan_mine({"symbols": ["AAPL"], "class": "fundamentals"})
    assert targets[0][0].asset_class == "fundamentals"
    assert unknown == []
    # Without the class, a bare AAPL is the stock price.
    targets2, _ = cli.plan_mine({"symbols": ["AAPL"]})
    assert targets2[0][0].asset_class == "stocks"


def test_plan_mine_by_macro_class():
    targets, _ = cli.plan_mine({"class": "macro"})
    assert "UNRATE" in [t[0].symbol for t in targets]


def test_mine_one_dispatches_fred_for_macro(monkeypatch):
    seen = {}

    def fake_macro(series_id, out_dir, dry_run=False):
        seen["args"] = (series_id, out_dir)
        return {"symbol": series_id, "written": 1, "observations": 100, "errors": []}

    monkeypatch.setattr(cli, "backfill_series_macro", fake_macro)
    result = cli.mine_one(data_catalog.instrument("UNRATE"), ["release"], (2026, 1))
    assert seen["args"] == ("UNRATE", "macro")
    assert result["source"] == data_catalog.FRED
    assert result["written"] == 1
    assert result["observations"] == 100


def test_mine_one_dispatches_edgar_for_fundamentals(monkeypatch):
    seen = {}

    def fake_edgar(ticker, out_dir, dry_run=False):
        seen["args"] = (ticker, out_dir)
        return {"symbol": ticker, "written": 1, "observations": 50, "errors": []}

    monkeypatch.setattr(cli, "backfill_ticker_fundamentals", fake_edgar)
    result = cli.mine_one(data_catalog.instrument("AAPL", "fundamentals"), ["release"], (2026, 1))
    assert seen["args"] == ("AAPL", "fundamentals")
    assert result["source"] == data_catalog.EDGAR
    assert result["written"] == 1


def test_build_full_catalog_uses_series_coverage_for_macro(tmp_path):
    macro = tmp_path / "macro"
    macro.mkdir()
    (macro / "UNRATE.json").write_text(
        json.dumps([{"refPeriod": "2024-01", "releaseDate": "2024-02-02", "value": 3.7}])
    )
    cat = cli.build_full_catalog(str(tmp_path))
    unrate = next(i for c in cat for i in c["instruments"] if i["symbol"] == "UNRATE")
    assert unrate["onDisk"]["release"]["months"] == 1


def test_main_catalog_writes_the_catalog_summary(tmp_path, monkeypatch):
    out = tmp_path / "cat.json"
    monkeypatch.chdir(tmp_path)  # empty root => everything off-disk
    assert cli.main(["catalog", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [c["id"] for c in data["assetClasses"]] == [
        "crypto",
        "stocks",
        "commodities",
        "fx",
        "etfs",
        "macro",
        "fundamentals",
    ]
    # The catalog emit also carries the asset-linkage graph.
    assert "edges" in data["linkage"]
    assert data["linkage"]["edges"]


def test_main_mine_reads_request_and_writes_result(tmp_path, monkeypatch):
    req = tmp_path / "req.json"
    out = tmp_path / "out.json"
    req.write_text(json.dumps({"symbols": ["GOLD"], "through": "2025-02"}))
    monkeypatch.setattr(
        cli, "backfill_yf_symbol",
        lambda *a, **k: {"symbol": "GOLD", "written": [(2025, 1), (2025, 2)], "skipped": 0, "errors": [], "gaps": []},
    )
    assert cli.main(["mine", "--request", str(req), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mined"][0]["symbol"] == "GOLD"
    assert data["mined"][0]["written"] == 2
    assert data["unknown"] == []


def test_main_mine_reports_unknown_symbols(tmp_path, monkeypatch):
    req = tmp_path / "req.json"
    out = tmp_path / "out.json"
    req.write_text(json.dumps({"symbols": ["NOPE"]}))
    assert cli.main(["mine", "--request", str(req), "--out", str(out)]) == 1
    data = json.loads(out.read_text())
    assert data["unknown"] == ["NOPE"]
    assert data["mined"] == []
