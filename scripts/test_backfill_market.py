import json
import os

import scripts.backfill_market as bm


def _rows(dates):
    """Minimal daily source rows [date, open, high, low, close, volume]."""
    return [[d, "1", "2", "0.5", "1.5", "10"] for d in dates]


def test_backfill_yf_symbol_writes_monthly_files_named_by_local_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda src, start, through: _rows(["2024-06-03", "2024-06-04", "2024-07-01"]))
    summary = bm.backfill_yf_symbol("GOLD", "GC=F", str(tmp_path), (2024, 6), (2024, 7))
    assert summary["symbol"] == "GOLD"
    assert summary["source_symbol"] == "GC=F"
    assert summary["written"] == [(2024, 6), (2024, 7)]
    assert os.path.exists(os.path.join(tmp_path, "GOLD-1d-2024-6.json"))
    assert os.path.exists(os.path.join(tmp_path, "GOLD-1d-2024-7.json"))
    june = json.load(open(os.path.join(tmp_path, "GOLD-1d-2024-6.json")))
    assert len(june) == 2
    assert june[0]["price"] == "1.50000000"  # daily close maps to the kline `price`


def test_backfill_yf_symbol_passes_source_symbol_to_the_fetch(tmp_path, monkeypatch):
    seen = {}

    def _capture(src, start, through):
        seen["src"] = src
        return _rows(["2024-06-03"])

    monkeypatch.setattr(bm, "rows_from_yfinance", _capture)
    bm.backfill_yf_symbol("EURUSD", "EURUSD=X", str(tmp_path), (2024, 6), (2024, 6))
    assert seen["src"] == "EURUSD=X"


def test_backfill_yf_symbol_skips_months_already_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda *a: _rows(["2024-06-03", "2024-07-01"]))
    bm.backfill_yf_symbol("WTI", "CL=F", str(tmp_path), (2024, 6), (2024, 7))
    again = bm.backfill_yf_symbol("WTI", "CL=F", str(tmp_path), (2024, 6), (2024, 7))
    assert again["written"] == []
    assert again["skipped"] == 2


def test_backfill_yf_symbol_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda *a: _rows(["2024-06-03"]))
    summary = bm.backfill_yf_symbol("EURUSD", "EURUSD=X", str(tmp_path), (2024, 6), (2024, 6), dry_run=True)
    assert summary["written"] == [(2024, 6)]
    assert os.listdir(tmp_path) == []


def test_backfill_yf_symbol_reports_empty_source_as_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda *a: [])
    summary = bm.backfill_yf_symbol("NOPE", "NO=F", str(tmp_path), (2024, 6), (2024, 6))
    assert summary["written"] == []
    assert summary["errors"]


def test_backfill_yf_symbol_creates_the_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda *a: _rows(["2024-06-03"]))
    out = os.path.join(str(tmp_path), "commodities")
    bm.backfill_yf_symbol("GOLD", "GC=F", out, (2024, 6), (2024, 6))
    assert os.path.exists(os.path.join(out, "GOLD-1d-2024-6.json"))


def test_backfill_yf_symbol_zero_volume_fx_is_neutral_not_an_error(tmp_path, monkeypatch):
    # FX carries no real volume (yfinance returns 0); the neutral fill must not fail validation.
    monkeypatch.setattr(bm, "rows_from_yfinance", lambda *a: [["2024-06-03", "1.1", "1.2", "1.0", "1.15", "0"]])
    summary = bm.backfill_yf_symbol("EURUSD", "EURUSD=X", str(tmp_path), (2024, 6), (2024, 6))
    assert not summary["errors"]
    row = json.load(open(os.path.join(tmp_path, "EURUSD-1d-2024-6.json")))[0]
    assert float(row["volume"]) == 0.0
    assert float(row["asset_volume_quote"]) == 0.0
