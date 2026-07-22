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
    # mining every native timeframe.
    called = []
    monkeypatch.setattr(cli, "backfill_series", lambda *a, **k: called.append(a) or {"written": [], "skipped": 0, "errors": [], "gaps": []})
    res = cli.mine_one(data_catalog.instrument("BTCUSDT"), [], (2026, 1))
    assert called == []
    assert res["written"] == 0


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


def test_main_catalog_writes_the_catalog_summary(tmp_path, monkeypatch):
    out = tmp_path / "cat.json"
    monkeypatch.chdir(tmp_path)  # empty root => everything off-disk
    assert cli.main(["catalog", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [c["id"] for c in data["assetClasses"]] == ["crypto", "stocks", "commodities", "fx"]


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
