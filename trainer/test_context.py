import json

import pytest

from trainer import context as context_mod
from trainer.context import (
    CONTEXT_SERIES,
    DEFAULT_CONTEXT,
    context_columns,
    context_panel_ids,
    fuse_context_series,
    load_context_observations,
    load_price_observations,
    load_series_observations,
    resolve_context,
    _release_over_release_change,
)
from trainer.pit_fusion import release_datetime_ms


# ---- fuse_context_series: LEVEL ------------------------------------------------------------------

def test_level_is_forward_filled_raw_value_none_before_release():
    r = release_datetime_ms("2020-01-10", "08:30")
    bars = [r - 1000, r, r + 1000]
    obs = [{"releaseDate": "2020-01-10", "value": 4.0}]
    out = fuse_context_series(bars, obs, representation="level", publish_time="08:30")
    assert out == [None, 4.0, 4.0]


def test_level_applies_the_static_affine_scale():
    r = release_datetime_ms("2020-01-10", "08:30")
    bars = [r, r + 1]
    obs = [{"releaseDate": "2020-01-10", "value": 4.0}]
    out = fuse_context_series(bars, obs, representation="level", scale=0.1, publish_time="08:30")
    assert out == [pytest.approx(0.4), pytest.approx(0.4)]


def test_level_is_leakage_safe_at_the_publish_instant():
    # A bar earlier in the release day than the publish time must NOT see the value.
    r = release_datetime_ms("2020-01-10", "08:30")
    bars = [r - 60_000, r]  # one minute before the publish instant, and at it
    obs = [{"releaseDate": "2020-01-10", "value": 4.0}]
    out = fuse_context_series(bars, obs, representation="level", publish_time="08:30")
    assert out == [None, 4.0]


# ---- fuse_context_series: CHANGE -----------------------------------------------------------------

def test_change_is_release_over_release_percent_stamped_at_the_later_release():
    r1 = release_datetime_ms("2020-01-10", "08:30")
    r2 = release_datetime_ms("2020-02-10", "08:30")
    bars = [r1 + 1, r2 - 1, r2 + 1]
    obs = [{"releaseDate": "2020-01-10", "value": 4.0}, {"releaseDate": "2020-02-10", "value": 5.0}]
    out = fuse_context_series(bars, obs, representation="change", publish_time="08:30")
    # The change (5-4)/4 = 0.25 is knowable only from the SECOND release; before it, None.
    assert out == [None, None, pytest.approx(0.25)]


def test_release_over_release_change_skips_zero_prior_and_orders_by_date():
    obs = [
        {"releaseDate": "2020-03-10", "value": 6.0},
        {"releaseDate": "2020-01-10", "value": 4.0},
        {"releaseDate": "2020-02-10", "value": 5.0},
    ]
    changed = _release_over_release_change(obs)
    assert changed == [
        {"releaseDate": "2020-02-10", "value": pytest.approx(0.25)},
        {"releaseDate": "2020-03-10", "value": pytest.approx(0.2)},
    ]


def test_release_over_release_change_drops_zero_prior():
    obs = [{"releaseDate": "2020-01-10", "value": 0.0}, {"releaseDate": "2020-02-10", "value": 5.0}]
    assert _release_over_release_change(obs) == []


# ---- resolve_context -----------------------------------------------------------------------------

def test_default_context_is_none_and_yields_no_series():
    assert DEFAULT_CONTEXT == "none"
    panel_id, series = resolve_context()
    assert panel_id == "none"
    assert series == []


def test_rates_panel_resolves_to_level_rate_series():
    panel_id, series = resolve_context({"context_set": "rates"})
    assert panel_id == "rates"
    ids = [s["id"] for s in series]
    assert ids == ["DFF", "DGS10", "DFII10", "T10Y2Y"]
    assert all(s["representation"] == "level" for s in series)
    # Each carries its publish-time so the provider fuses leakage-safely.
    assert all("publish_time" in s and "tz" in s for s in series)


def test_unknown_context_set_fails_fast():
    with pytest.raises(SystemExit):
        resolve_context({"context_set": "does_not_exist"})


def test_context_panel_ids_include_none_and_curated_panels():
    ids = context_panel_ids()
    assert ids[0] == "none"
    assert "rates" in ids and "macro_core" in ids


def test_series_classification_follows_the_representation_rule():
    # Rates/percentages -> level; values/counts/indices -> change.
    assert CONTEXT_SERIES["DFF"]["representation"] == "level"
    assert CONTEXT_SERIES["UNRATE"]["representation"] == "level"
    assert CONTEXT_SERIES["T10Y2Y"]["representation"] == "level"
    assert CONTEXT_SERIES["CPIAUCNS"]["representation"] == "change"
    assert CONTEXT_SERIES["PAYEMS"]["representation"] == "change"


# ---- context_columns (pure; loader injected) -----------------------------------------------------

def test_context_columns_builds_named_columns_via_loader():
    r = release_datetime_ms("2020-01-10", "08:30")
    bars = [r - 1, r, r + 1]
    specs = [{
        "id": "DFF", "source": "fred", "representation": "level", "scale": 1.0,
        "publish_time": "08:30", "tz": "America/New_York",
    }]
    loaded = {"DFF": [{"releaseDate": "2020-01-10", "value": 4.0}]}
    cols = context_columns(bars, specs, lambda sid, src: loaded[sid])
    assert list(cols.keys()) == ["context_DFF"]
    assert cols["context_DFF"] == [None, 4.0, 4.0]


def test_context_columns_empty_specs_yields_no_columns():
    assert context_columns([1, 2, 3], [], lambda sid, src: []) == {}


# ---- load_context_observations (disk I/O) --------------------------------------------------------

def test_load_context_observations_reads_macro_json(tmp_path):
    macro = tmp_path / "macro"
    macro.mkdir()
    (macro / "DFF.json").write_text(
        json.dumps([{"refPeriod": "2020-01-01", "releaseDate": "2020-01-10", "value": 4.0, "vintage": "2020-01-10"}])
    )
    obs = load_context_observations("DFF", "fred", root=str(tmp_path))
    assert obs[0]["releaseDate"] == "2020-01-10"
    assert obs[0]["value"] == 4.0


def test_load_context_observations_fundamentals_use_the_fundamentals_dir(tmp_path):
    d = tmp_path / "fundamentals"
    d.mkdir()
    (d / "AAPL_Revenue.json").write_text(json.dumps([{"releaseDate": "2020-02-01", "value": 100}]))
    obs = load_context_observations("AAPL_Revenue", "edgar", root=str(tmp_path))
    assert obs[0]["value"] == 100


def test_load_context_observations_missing_file_fails_fast(tmp_path):
    # A context run REQUIRES its series mined to disk — never silently train on a zeroed channel.
    with pytest.raises(SystemExit):
        load_context_observations("NOPE", "fred", root=str(tmp_path))


# ---- price-source context (a linked/peer asset's return as a channel) ----------------------------

def test_load_price_observations_extracts_close_and_bar_close_date(tmp_path):
    b = tmp_path / "binance"
    b.mkdir()
    (b / "ETHUSDT-1d-2022-1.json").write_text(json.dumps([
        {"price": 3765.54, "timestamp_close": 1641081599999},   # 2022-01-01 23:59:59.999 UTC
        {"price": 3800.0, "timestamp_close": 1641167999999},    # 2022-01-02 23:59:59.999 UTC
    ]))
    obs = load_price_observations("ETHUSDT", root=str(tmp_path))
    by_value = {o["value"]: o["releaseDate"] for o in obs}
    assert by_value[3765.54] == "2022-01-01"
    assert by_value[3800.0] == "2022-01-02"


def test_load_price_observations_missing_fails_fast(tmp_path):
    with pytest.raises(SystemExit):
        load_price_observations("NOPE", root=str(tmp_path))


def test_load_series_observations_routes_price_vs_release_series(tmp_path):
    (tmp_path / "binance").mkdir()
    (tmp_path / "binance" / "ETHUSDT-1d-2022-1.json").write_text(
        json.dumps([{"price": 3765.54, "timestamp_close": 1641081599999}])
    )
    (tmp_path / "macro").mkdir()
    (tmp_path / "macro" / "DFF.json").write_text(json.dumps([{"releaseDate": "2020-01-10", "value": 4.0}]))
    price = load_series_observations("ETHUSDT", "price", root=str(tmp_path))
    fred = load_series_observations("DFF", "fred", root=str(tmp_path))
    assert price[0]["value"] == 3765.54
    assert fred[0]["value"] == 4.0


def test_majors_panel_is_price_source_change():
    panel_id, series = resolve_context({"context_set": "majors"})
    assert panel_id == "majors"
    assert [s["id"] for s in series] == ["BTCUSDT", "ETHUSDT"]
    assert all(s["source"] == "price" and s["representation"] == "change" for s in series)


def test_market_panel_is_the_keyless_macro_risk_proxies():
    panel_id, series = resolve_context({"context_set": "market"})
    assert panel_id == "market"
    assert [s["id"] for s in series] == ["GOLD", "SPY", "UUP"]
    assert all(s["source"] == "price" and s["representation"] == "change" for s in series)


def test_load_price_observations_resolves_directory_from_catalog(tmp_path):
    # A non-crypto price proxy (GOLD) lives in commodities/, not binance/ — the loader resolves its dir
    # from the catalog so the same code serves crypto peers AND market proxies.
    d = tmp_path / "commodities"
    d.mkdir()
    (d / "GOLD-1d-2024-1.json").write_text(json.dumps([{"price": 2000.0, "timestamp_close": 1704067199999}]))
    obs = load_price_observations("GOLD", root=str(tmp_path))
    assert obs[0]["value"] == 2000.0
