import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider, days_in_month

_COLUMNS = [
    "timestamp",
    "timestamp_close",
    "price",
    "price_open",
    "price_high",
    "price_low",
    "volume",
    "asset_volume_quote",
    "trades_number",
    "asset_volume_taker_base",
]


class _Provider(AbstractDataProvider):
    def get_timesteps(self):
        return 0

    def get_price(self, step):
        return 0.0

    def get_signal_buy_sell(self, step):
        return 0

    def get_signal_buy_profitable(self, step):
        return 0

    def get_signal_buy_drawdown(self, step):
        return 0

    def get_values(self, step):
        return None


def _provider(**cfg_kw):
    cfg = DataConfig(
        id="t",
        train_data_paths=[[]],
        test_data_paths=[[]],
        layers=["1d"],
        layers_test=["1d"],
        **cfg_kw,
    )
    return _Provider(cfg)


def _df(n=40):
    base = 1_600_000_000_000
    price = [100.0 * (1.02**i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": [base + i * 86400000 for i in range(n)],
            "timestamp_close": [base + i * 86400000 for i in range(n)],
            "price": price,
            "price_open": price,
            "price_high": [p * 1.01 for p in price],
            "price_low": [p * 0.99 for p in price],
            "volume": [1000.0 + i for i in range(n)],
            "asset_volume_quote": [50000.0 + i for i in range(n)],
            "trades_number": [10 + i for i in range(n)],
            "asset_volume_taker_base": [500.0 + i for i in range(n)],
        }
    )


def test_squash_features_clip():
    p = _provider(obs_squash="clip")
    out = p._squash_observation_features(pd.DataFrame({"a": [2.0, -3.0, 0.5], "b": [1.5, 0.0, -0.2]}))
    assert out["a"].tolist() == [1.0, -1.0, 0.5]
    assert out["b"].tolist() == [1.0, 0.0, -0.2]


def test_squash_features_tanh_bounds_everything():
    p = _provider(obs_squash="tanh")
    out = p._squash_observation_features(pd.DataFrame({"a": [5.0, -5.0, 0.0]}))
    assert out["a"].abs().le(1.0).all()
    assert out["a"].iloc[2] == 0.0


def test_squash_features_none_is_identity():
    p = _provider(obs_squash="none")
    out = p._squash_observation_features(pd.DataFrame({"a": [2.0, -3.0]}))
    assert out["a"].tolist() == [2.0, -3.0]


def test_curated_indicators_add_regime_and_trend_features():
    out = _provider(use_indicators=True)._add_curated_indicators(_df())
    assert "volRegime10" in out.columns
    assert "trendSlope10" in out.columns
    vals = pd.concat([out["volRegime10"], out["trendSlope10"]]).dropna()
    assert vals.between(-1.0, 1.0).all()


def test_curated_indicators_skipped_without_flag():
    out = _provider(use_indicators=False)._add_curated_indicators(_df())
    assert "volRegime10" not in out.columns


def test_process_df_simple_runs_and_clip_bounds_output():
    p = _provider(obs_squash="clip")
    out, prices, timestamps = p.process_df_simple(_df(), "day_of_week", list(_COLUMNS))
    nums = out.select_dtypes(include=[np.number])
    assert (nums.abs() <= 1.0 + 1e-9).all().all()
    assert len(prices) == 40


# ---------------------------------------------------------------------------
# Extra synthetic frames. `_df` above produces only the OHLCV columns used by
# process_df_simple / process_fidelity. The full `process_df` (non-simple) path
# also reads `asset_volume_taker_quote` (line 297 drop) and an `indicators`
# dict column (exploded per type/indicator), and `timestamp_close` arrives as a
# datetime64 (pd.read_json yields datetimes for ISO strings). `_df_full` mirrors
# that real shape so the branch-heavy process_df can be exercised.
# ---------------------------------------------------------------------------

# A minimal indicators dict carrying every key any process_df indicator branch
# explodes. Values are strings (the real data stores them as strings).
_INDICATORS = {
    "kallman15": "0.10",
    "timeseriesMomentum7": "0.20",
    "closenessTo1000": "0.30",
    "closenessTo10000": "0.40",
    "meanReversion10": "0.50",
    "meanReversion15": "0.60",
    "rsi5": "0.70",
    "rsi10": "0.80",
    "rsi15": "0.90",
    "choppiness30": "0.05",
    "cci5": "0.24",
    "cci7": "0.25",
    "cci10": "0.11",
    "disparityIndex7": "0.12",
    "disparityIndex10": "0.13",
    "sortinoRatio5": "0.23",
    "sortinoRatio30": "0.14",
    "volatilityVolume7": "0.16",
    "volatilityVolume30": "0.15",
    "williams5": "-0.50",
    "williams10": "-0.60",
    "stochasticOscillator5": "0.21",
    "stochasticOscillator10": "0.22",
    "turbulenceIndex10": "0.26",
}


def _df_full(n=12, close_datetime=True):
    """Synthetic frame with the full column set process_df consumes (taker_quote +
    indicators dict). timestamp_close defaults to datetime64 to match read_json."""
    base = pd.Timestamp("2021-06-01 00:00:00")
    price = [100.0 * (1.01**i) for i in range(n)]
    closes = [base + pd.Timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": [int((base + pd.Timedelta(days=i)).value // 10**6) for i in range(n)],
            "timestamp_close": closes if close_datetime else [int(c.value // 10**6) for c in closes],
            "price": price,
            "price_open": price,
            "price_high": [p * 1.01 for p in price],
            "price_low": [p * 0.99 for p in price],
            "volume": [1000.0 + i for i in range(n)],
            "asset_volume_quote": [50000.0 + i for i in range(n)],
            "trades_number": [10 + i for i in range(n)],
            "asset_volume_taker_base": [500.0 + i for i in range(n)],
            "asset_volume_taker_quote": [25000.0 + i for i in range(n)],
            "indicators": [dict(_INDICATORS) for _ in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Simple accessors / config-derived helpers
# ---------------------------------------------------------------------------


def test_get_id_encodes_dots_and_pipes():
    # '.' -> '~' and '|' -> ']' in the cache id; the layer/fidelity joins use '|'.
    p = _provider(
        type="only_price_percent",
        timestamp="day_of_week",
        indicator="none",
        lookback_window_size=32,
        buyreward_percent=0.004,
        buyreward_maxwait=20,
        fidelity_input="1m",
        fidelity_run="1h",
        fidelity_input_test="1m",
        fidelity_run_test="1m",
    )
    p.config.layers = ["1h", "1d"]
    p.config.layers_test = ["1d"]
    got = p.get_id(p.config)
    assert "." not in got
    assert "|" not in got
    # 0.004 -> 0~004 ; the layer join '1h|1d' -> '1h]1d'
    assert "0~004" in got
    assert "1h]1d" in got


def test_is_multilayered_true_and_false():
    assert _provider().is_multilayered() is False  # default layers == ["1d"]
    multi = _provider()
    multi.config.layers = ["1h", "1d"]
    assert multi.is_multilayered() is True


def test_get_lookback_window_returns_config_value():
    assert _provider(lookback_window_size=7).get_lookback_window() == 7


def test_get_start_index_is_zero():
    assert _provider().get_start_index() == 0


def test_get_timestamp_datetime_parses_iloc_row():
    p = _provider()
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-06-02 13:37:00"), "2021-06-03 00:00:00"]})
    assert p.get_timestamp_datetime(df, 0) == pd.Timestamp("2021-06-02 13:37:00")


# ---------------------------------------------------------------------------
# get_rewards_buy_sell : matched buy(+) / sell(-) runs
# ---------------------------------------------------------------------------


def test_rewards_buy_sell_alternating_runs():
    # Up-run starts a buy (count increments, label=count); down/flat starts a sell
    # (label=-count). A continued run repeats the bare action (+1 / -1). Last bar = 0.
    df = pd.DataFrame({"price": [1, 2, 3, 2, 1, 2, 3]})
    assert _provider().get_rewards_buy_sell(df) == [2, 1, -2, -1, 3, 1, 0]


def test_rewards_buy_sell_all_up():
    # First up transition opens buy #2 (count starts 1, ++ to 2); subsequent ups repeat +1.
    assert _provider().get_rewards_buy_sell(pd.DataFrame({"price": [1, 2, 3, 4]})) == [2, 1, 1, 0]


def test_rewards_buy_sell_all_down_stays_in_initial_sell():
    # action starts at -1, every step is a down/flat -> bare -1 repeated, never opening a -count.
    assert _provider().get_rewards_buy_sell(pd.DataFrame({"price": [4, 3, 2, 1]})) == [-1, -1, -1, 0]


def test_rewards_buy_sell_flat_counts_as_sell():
    # equal prices are NOT 'current < next', so they fall into the sell branch.
    assert _provider().get_rewards_buy_sell(pd.DataFrame({"price": [2, 2, 2]})) == [-1, -1, 0]


def test_rewards_buy_sell_single_row_is_just_terminal_zero():
    assert _provider().get_rewards_buy_sell(pd.DataFrame({"price": [5]})) == [0]


def test_rewards_buy_sell_matched_pairs_use_running_count():
    # Documented contract: each buy open gets an incrementing id and its matching sell mirrors it
    # (+2 paired with -2, +3 with -3). A strict zig-zag yields exactly those matched pairs.
    assert _provider().get_rewards_buy_sell(pd.DataFrame({"price": [1, 2, 1, 2, 1]})) == [2, -2, 3, -3, 0]


# ---------------------------------------------------------------------------
# get_rewards_buy : (profit_count, drawdown) per bar
# ---------------------------------------------------------------------------


def test_rewards_buy_immediately_profitable_count_one():
    # Each step gains 1% >= 0.4% threshold, so profit happens on the very next bar -> count 1.
    prof, dd = _provider().get_rewards_buy(
        pd.DataFrame({"price": [100.0, 101.0, 102.0, 103.0]}), percent_to_pass=0.004, max_buy_wait=20
    )
    assert prof == [1, 1, 1, 0]
    assert dd == [0, 0, 0, 0]


def test_rewards_buy_falling_never_profitable_tracks_drawdown():
    # Strictly falling: never reaches +0.4%, profit_count = 1 + bars examined; drawdown is the
    # min percent move seen in the window (most negative at the far bar).
    prof, dd = _provider().get_rewards_buy(
        pd.DataFrame({"price": [100.0, 99.0, 98.0, 97.0]}), percent_to_pass=0.004, max_buy_wait=20
    )
    assert prof == [4, 3, 2, 0]
    # drawdown is the most-negative percent move in the lookahead window for that bar.
    assert dd[0] == pytest.approx(97.0 / 100.0 - 1.0)  # min over {99,98,97}/100
    assert dd[1] == pytest.approx(97.0 / 99.0 - 1.0)  # min over {98,97}/99
    assert dd[2] == pytest.approx(97.0 / 98.0 - 1.0)  # only {97}/98
    assert dd[3] == 0


def test_rewards_buy_max_wait_caps_lookahead():
    # Flat prices never go profitable; the inner loop is capped by max_buy_wait, so profit_count
    # for early bars is 1 + min(remaining, max_buy_wait).
    prof, dd = _provider().get_rewards_buy(
        pd.DataFrame({"price": [100.0] * 5}), percent_to_pass=0.004, max_buy_wait=2
    )
    assert prof == [3, 3, 3, 2, 0]
    assert dd == [0, 0, 0, 0, 0]


def test_rewards_buy_single_row_terminal_only():
    prof, dd = _provider().get_rewards_buy(pd.DataFrame({"price": [100.0]}))
    assert prof == [0]
    assert dd == [0]


# ---------------------------------------------------------------------------
# _add_curated_indicators
# ---------------------------------------------------------------------------


def test_curated_indicators_missing_columns_returns_unchanged():
    # Guard: without price_high/price_low/volume the frame is returned untouched.
    p = _provider(use_indicators=True)
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
    out = p._add_curated_indicators(df)
    assert list(out.columns) == ["price"]


def test_curated_indicators_adds_full_set_when_enabled():
    p = _provider(use_indicators=True)
    df = pd.DataFrame(
        {
            "price": [100.0 + i for i in range(20)],
            "price_high": [101.0 + i for i in range(20)],
            "price_low": [99.0 + i for i in range(20)],
            "volume": [10.0 + i for i in range(20)],
        }
    )
    out = p._add_curated_indicators(df)
    for col in [
        "rsi10",
        "williams10",
        "stochasticOscillator10",
        "choppiness30",
        "meanReversion10",
        "turbulenceIndex10",
        "obv10",
        "volRegime10",
        "trendSlope10",
    ]:
        assert col in out.columns


# ---------------------------------------------------------------------------
# _squash_observation_features additional branches
# ---------------------------------------------------------------------------


def test_squash_features_empty_string_is_identity():
    # getattr fallback / falsy squash leaves the frame unchanged.
    p = _provider(obs_squash="")
    out = p._squash_observation_features(pd.DataFrame({"a": [4.0, -4.0]}))
    assert out["a"].tolist() == [4.0, -4.0]


def test_squash_features_only_touches_numeric_columns():
    p = _provider(obs_squash="clip")
    df = pd.DataFrame({"a": [5.0, -5.0], "label": ["x", "y"]})
    out = p._squash_observation_features(df)
    assert out["a"].tolist() == [1.0, -1.0]
    assert out["label"].tolist() == ["x", "y"]


# ---------------------------------------------------------------------------
# days_in_month module function
# ---------------------------------------------------------------------------


def test_days_in_month_february_leap_and_non_leap():
    assert days_in_month({"timestamp_close": pd.Timestamp("2020-02-15")}) == 29
    assert days_in_month({"timestamp_close": pd.Timestamp("2021-02-15")}) == 28
    assert days_in_month({"timestamp_close": pd.Timestamp("2021-04-10")}) == 30


# ---------------------------------------------------------------------------
# process_df : the type / indicator / timestamp branch matrix
# ---------------------------------------------------------------------------


def test_process_df_returns_six_aligned_outputs():
    p = _provider()
    df = _df_full()
    out, prices, timestamps, bs, bp, bd = p.process_df(
        df.copy(), "only_price_percent", "day_of_week", "none", 0.004, 20
    )
    n = len(df)
    # prices/timestamps are taken from the ORIGINAL frame, reward arrays are 1-per-row.
    assert len(prices) == n
    assert len(timestamps) == n
    assert len(bs) == n
    assert len(bp) == n
    assert len(bd) == n
    # final feature frame is entirely numeric and free of NaN/inf.
    assert out.select_dtypes(include=[np.number]).shape[1] == out.shape[1]
    assert not out.isna().any().any()


def test_process_df_standard_explodes_indicators_and_keeps_price():
    # "standard" keeps the raw OHLCV (not dropped) and concatenates the exploded indicators dict.
    out, *_ = _provider().process_df(_df_full().copy(), "standard", "none", "none", 0.004, 20)
    assert "price" in out.columns
    assert "rsi10" in out.columns  # came from the exploded indicators dict
    assert "kallman15" in out.columns
    assert "taker_buy_ratio" in out.columns


def test_process_df_all_percents_drops_price_and_takes_pct_change():
    # "all_percents" drops raw price/volume cols and pct_changes every exploded indicator.
    out, *_ = _provider().process_df(_df_full().copy(), "all_percents", "none", "none", 0.004, 20)
    assert "price" not in out.columns
    assert "price_percent" in out.columns
    assert "rsi10" in out.columns


@pytest.mark.parametrize(
    "ind,expected",
    [
        ("indicators1", ["kallman15", "rsi10", "rsi5", "rsi15"]),
        ("indicators2", ["rsi10", "choppiness30"]),
        ("indicators3", ["kallman15", "rsi10", "choppiness30"]),
        ("indicators4", ["kallman15", "choppiness30"]),
        ("indicators5", ["timeseriesMomentum7", "choppiness30"]),
        ("indicators6", ["cci10", "choppiness30"]),
        ("indicators7", ["disparityIndex7", "disparityIndex10"]),
        ("indicators8", ["sortinoRatio30"]),
        ("indicators9", ["volatilityVolume30", "volatilityVolume7"]),
        ("indicators10", ["cci5", "cci7", "cci10", "turbulenceIndex10"]),
        ("indicators11", ["williams5"]),
        ("indicators12", ["williams10"]),
        ("indicators13", ["stochasticOscillator5"]),
        ("indicators14", ["stochasticOscillator10"]),
        ("indicators15", ["sortinoRatio5"]),
    ],
)
def test_process_df_named_indicator_sets_pull_expected_columns(ind, expected):
    out, *_ = _provider().process_df(_df_full().copy(), "only_price_percent", "none", ind, 0.004, 20)
    for col in expected:
        assert col in out.columns


def test_process_df_single_named_indicator_branch():
    # An indicator that is not "none" and not one of the indicatorsN bundles pulls that single key.
    out, *_ = _provider().process_df(_df_full().copy(), "only_price_percent", "none", "rsi10", 0.004, 20)
    assert "rsi10" in out.columns


def test_process_df_solo_price_percent_drops_high_low_volume():
    # solo_price_percent adds price_percent but drops price/high/low/volume.
    out, *_ = _provider().process_df(_df_full().copy(), "solo_price_percent", "none", "none", 0.004, 20)
    assert "price_percent" in out.columns
    assert "price" not in out.columns
    assert "price_high" not in out.columns


def test_process_df_only_price_percent_sin_volume_skips_volume_percent():
    out, *_ = _provider().process_df(
        _df_full().copy(), "only_price_percent_sin_volume", "none", "none", 0.004, 20
    )
    # this type takes price_percent and the high/low percents but NOT volume_percent.
    assert "price_percent" in out.columns
    assert "price_high_percent" in out.columns
    assert "volume_percent" not in out.columns


@pytest.mark.parametrize(
    "type_,price_dropped",
    [
        ("solo_price", True),
        ("only_price", False),  # only_price keeps price but still drops timestamp_close
        ("only_price_percent", True),
        ("only_price_percent_change", True),
    ],
)
def test_process_df_type_variant_column_drops(type_, price_dropped):
    out, *_ = _provider().process_df(_df_full().copy(), type_, "none", "none", 0.004, 20)
    assert ("price" not in out.columns) is price_dropped
    assert out.select_dtypes(include=[np.number]).shape[1] == out.shape[1]


def test_process_df_timestamp_expanded_adds_calendar_features():
    out, *_ = _provider().process_df(
        _df_full().copy(), "standard", "expanded", "none", 0.004, 20
    )
    for col in ["month", "day", "time", "day_of_week"]:
        assert col in out.columns
    assert "days_in_month" not in out.columns  # intermediate dropped
    assert "timestamp" not in out.columns


# ---------------------------------------------------------------------------
# process_df_simple : windowing, scaling, taker guard, timestamp modes
# ---------------------------------------------------------------------------


def _simple_df(n=6, freq="1H"):
    base = pd.Timestamp("2021-06-01 00:00:00")
    closes = pd.date_range(base, periods=n, freq=freq)
    price = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": closes,
            "timestamp_close": closes,
            "price": price,
            "price_open": price,
            "price_high": [p + 1 for p in price],
            "price_low": [p - 1 for p in price],
            "volume": [10.0 + i for i in range(n)],
            "asset_volume_quote": [100.0 + i for i in range(n)],
            "trades_number": [5 + i for i in range(n)],
            "asset_volume_taker_base": [5.0 + i for i in range(n)],
        }
    )


def test_process_df_simple_drops_source_columns_and_adds_features():
    p = _provider()
    out, prices, timestamps = p.process_df_simple(_simple_df(), "none", list(_COLUMNS))
    # all of the raw source columns are dropped (columns_to_drop == columns + extras).
    for col in _COLUMNS:
        assert col not in out.columns
    # derived features survive.
    for col in ["price_z_score_1d", "price_to_max_1d", "price_percent", "taker_buy_ratio"]:
        assert col in out.columns
    assert list(prices) == [100.0 + i for i in range(6)]
    assert len(timestamps) == 6


def test_process_df_simple_taker_guard_when_column_absent():
    # If the caller omits asset_volume_taker_base from columns, taker_buy_ratio is a neutral 0.
    p = _provider()
    df = _simple_df().drop(columns=["asset_volume_taker_base"])
    cols = [c for c in _COLUMNS if c != "asset_volume_taker_base"]
    out, _, _ = p.process_df_simple(df, "none", cols)
    assert (out["taker_buy_ratio"] == 0.0).all()


def test_process_df_simple_taker_ratio_clipped_to_unit_interval():
    p = _provider()
    out, _, _ = p.process_df_simple(_simple_df(), "none", list(_COLUMNS))
    assert out["taker_buy_ratio"].between(0.0, 1.0).all()


def test_process_df_simple_normal_timestamp_adds_scaled_epoch_columns():
    p = _provider()
    out, _, _ = p.process_df_simple(_simple_df(), "normal", list(_COLUMNS))
    assert "timestamp_new" in out.columns
    assert "timestamp_close_new" in out.columns


def test_process_df_simple_expanded_timestamp_adds_calendar_features():
    p = _provider()
    out, _, _ = p.process_df_simple(_simple_df(freq="1D"), "expanded", list(_COLUMNS))
    for col in ["month", "day", "time", "day_of_week"]:
        assert col in out.columns
    assert "days_in_month" not in out.columns


def test_process_df_simple_window_scales_with_bar_spacing():
    # The 1d/1m/1y rolling windows are scaled by inferred minutes-per-bar. On 1m bars a 6-row
    # frame leaves price_z_score_1d entirely NaN (window 1440) -> filled with 0; on 1h bars the
    # 1d window is only 24 so it can populate. We assert the SHAPE (no NaN leaks) holds either way.
    p = _provider()
    out_1m, _, _ = p.process_df_simple(_simple_df(freq="1min"), "none", list(_COLUMNS))
    out_1h, _, _ = p.process_df_simple(_simple_df(freq="1H"), "none", list(_COLUMNS))
    assert not out_1m.isna().any().any()
    assert not out_1h.isna().any().any()
    # the short 1m frame can't fill the 1440-bar daily window -> all-zero z-score column.
    assert (out_1m["price_z_score_1d"] == 0.0).all()


def test_process_df_simple_squash_clip_bounds_all_numeric():
    p = _provider(obs_squash="clip")
    out, _, _ = p.process_df_simple(_simple_df(n=10), "none", list(_COLUMNS))
    nums = out.select_dtypes(include=[np.number])
    assert (nums.abs() <= 1.0 + 1e-9).all().all()


# ---------------------------------------------------------------------------
# get_current_mapping : fidelity x layer index assignment
# ---------------------------------------------------------------------------


def _ts_df(ts):
    return pd.DataFrame({"timestamp": [pd.Timestamp(ts)]})


@pytest.mark.parametrize(
    "fidelity,layer,expected",
    [
        # Wed 2021-06-02 13:37 (weekday=2, hour=13, minute=37)
        ("1m", "1w", 2 * 60 * 24 + 13 * 60 + 37),
        ("1m", "1d", 13 * 60 + 37),
        ("1m", "4h", (13 % 4) * 60 + 37),
        ("1m", "1h", 37),
        ("1m", "30m", 37 % 30),
        ("1m", "15m", 37 % 15),
        ("1m", "10m", 37 % 10),
        ("1m", "5m", 37 % 5),
        ("1m", "unknown", 0),
        ("30m", "1w", int((2 * 60 * 24 + 13 * 60 + 37) / 30 % (2 * 24 * 7))),
        ("30m", "1d", int((13 * 60 + 37) / 30 % (2 * 24))),
        ("30m", "4h", int((((13 % 4) * 60) + 37) / 30 % (2 * 4))),
        ("30m", "1h", int(37 / 30 % 2)),
        ("30m", "unknown", 0),
        ("15m", "1w", int((2 * 60 * 24 + 13 * 60 + 37) / 15 % (4 * 24 * 7))),
        ("15m", "1d", int((13 * 60 + 37) / 15 % (4 * 24))),
        ("15m", "4h", int((((13 % 4) * 60) + 37) / 15 % (4 * 4))),
        ("15m", "1h", int(37 / 15 % 4)),
        ("15m", "30m", int(37 / 15 % 2)),
        ("15m", "unknown", 0),
        ("10m", "1w", int((2 * 60 * 24 + 13 * 60 + 37) / 10 % (6 * 24 * 7))),
        ("10m", "1d", int((13 * 60 + 37) / 10 % (6 * 24))),
        ("10m", "4h", int((((13 % 4) * 60) + 37) / 10 % (6 * 4))),
        ("10m", "1h", int(37 / 10 % 6)),
        ("10m", "30m", int(37 / 10 % 3)),
        ("10m", "unknown", 0),
        ("5m", "1w", int((2 * 60 * 24 + 13 * 60 + 37) / 5 % (12 * 24 * 7))),
        ("5m", "1d", int((13 * 60 + 37) / 5 % (12 * 24))),
        ("5m", "4h", int((((13 % 4) * 60) + 37) / 5 % (12 * 4))),
        ("5m", "1h", int(37 / 5 % 12)),
        ("5m", "30m", int(37 / 5 % 6)),
        ("5m", "15m", int(37 / 5 % 3)),
        ("5m", "10m", int(37 / 5 % 2)),
        ("5m", "unknown", 0),
        # else-branch (any other fidelity, e.g. "1h"): hour-resolution mapping
        ("1h", "1w", 2 * 24 + 13),
        ("1h", "1d", 13 % 24),
        ("1h", "4h", 13 % 4),
        ("1h", "unknown", 0),
    ],
)
def test_get_current_mapping(fidelity, layer, expected):
    p = _provider()
    assert p.get_current_mapping(_ts_df("2021-06-02 13:37:00"), 0, layer, fidelity) == expected


# ---------------------------------------------------------------------------
# process_fidelity : OHLCV aggregation across multiplier_input bars
# ---------------------------------------------------------------------------


def _fidelity_df(n=12):
    ts = pd.to_datetime(["2021-06-01 %02d:00:00" % i for i in range(n)])
    tc = pd.to_datetime(["2021-06-01 %02d:59:59" % i for i in range(n)])
    price = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "timestamp_close": tc,
            "price": price,
            "price_open": [p - 0.5 for p in price],
            "price_high": [p + 1 for p in price],
            "price_low": [p - 1 for p in price],
            "volume": [10.0 + i for i in range(n)],
            "asset_volume_quote": [100.0 + i for i in range(n)],
            "trades_number": [5 + i for i in range(n)],
            "asset_volume_taker_base": [5.0 + i for i in range(n)],
        }
    )


def test_process_fidelity_single_run_aggregates_bars():
    # multiplier_input=2 collapses every 2 source bars: close=last, open=first, high=max, low=min,
    # volume/quote/trades/taker = sum. fidelity_steps = (12-0-0)//2 = 6.
    p = _provider()
    dfs, raw_dfs, prices = p.process_fidelity(
        _fidelity_df(12), "1h", 0, 2, "1h", 1, 0, "none"
    )
    assert len(dfs) == 1 and len(raw_dfs) == 1 and len(prices) == 1
    raw = raw_dfs[0]
    assert raw.shape[0] == 6
    assert raw["price"].tolist() == [101.0, 103.0, 105.0, 107.0, 109.0, 111.0]
    assert raw["price_open"].tolist() == [99.5, 101.5, 103.5, 105.5, 107.5, 109.5]
    assert raw["price_high"].tolist() == [102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
    assert raw["price_low"].tolist() == [99.0, 101.0, 103.0, 105.0, 107.0, 109.0]
    assert raw["volume"].tolist() == [21.0, 25.0, 29.0, 33.0, 37.0, 41.0]
    assert list(prices[0]) == [101.0, 103.0, 105.0, 107.0, 109.0, 111.0]


def test_process_fidelity_no_aggregation_passthrough():
    # multiplier_input=1 keeps every bar; closes equal the originals.
    p = _provider()
    dfs, raw_dfs, prices = p.process_fidelity(_fidelity_df(6), "1h", 0, 1, "1h", 1, 0, "none")
    assert raw_dfs[0].shape[0] == 6
    assert raw_dfs[0]["price"].tolist() == [100.0 + i for i in range(6)]


def test_process_fidelity_multi_run_places_by_mapping():
    # multiplier_run=2 produces two phase-shifted aggregations, each indexed into the output list by
    # get_current_mapping(layer="1h", fidelity else-branch -> hour%4) of its first bar.
    p = _provider()
    dfs, raw_dfs, prices = p.process_fidelity(
        _fidelity_df(12), "4h", 0, 2, "1h", 2, 1, "none"
    )
    assert len(dfs) == 2
    # both slots filled with non-empty frames.
    assert all(d.shape[0] > 0 for d in dfs)
    assert all(len(pr) > 0 for pr in prices)


# ---------------------------------------------------------------------------
# get_data / get_raw_data : JSON read + pipeline (tiny temp file, no market data)
# ---------------------------------------------------------------------------


def _write_temp_json(records):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(records, f)
    f.close()
    return f.name


def _json_records(n=6, with_indicators=False, with_taker_quote=False):
    base = pd.Timestamp("2021-06-01 00:00:00")
    recs = []
    for i in range(n):
        rec = {
            "timestamp": (base + pd.Timedelta(hours=i)).isoformat(),
            "timestamp_close": (base + pd.Timedelta(hours=i, minutes=59)).isoformat(),
            "price": 100.0 + i,
            "price_open": 100.0 + i,
            "price_high": 101.0 + i,
            "price_low": 99.0 + i,
            "volume": 10.0 + i,
            "asset_volume_quote": 100.0 + i,
            "trades_number": 5 + i,
            "asset_volume_taker_base": 5.0 + i,
        }
        if with_taker_quote:
            rec["asset_volume_taker_quote"] = 50.0 + i
        if with_indicators:
            rec["indicators"] = dict(_INDICATORS)
        recs.append(rec)
    return recs


def test_get_raw_data_reads_json_and_processes():
    p = _provider()
    path = _write_temp_json(_json_records(6))
    try:
        raw_df, result_df, prices, timestamps = p.get_raw_data([path], "none")
    finally:
        os.unlink(path)
    # get_raw_data selects exactly its default `columns` from the concatenated frame.
    assert list(raw_df.columns) == list(_COLUMNS)
    assert result_df.shape[0] == 6
    assert list(prices) == [100.0 + i for i in range(6)]
    assert len(timestamps) == 6


def test_get_raw_data_concats_multiple_paths():
    p = _provider()
    p1 = _write_temp_json(_json_records(3))
    p2 = _write_temp_json(_json_records(4))
    try:
        raw_df, result_df, prices, timestamps = p.get_raw_data([p1, p2], "none")
    finally:
        os.unlink(p1)
        os.unlink(p2)
    assert raw_df.shape[0] == 7
    assert result_df.shape[0] == 7


def test_get_data_reads_json_and_runs_full_process_df():
    p = _provider()
    path = _write_temp_json(_json_records(8, with_indicators=True, with_taker_quote=True))
    try:
        out, prices, timestamps, bs, bp, bd = p.get_data(
            [path], "only_price_percent", "day_of_week", "none", 0.004, 20
        )
    finally:
        os.unlink(path)
    assert out.shape[0] == 8
    assert len(prices) == 8
    assert len(bs) == 8
    assert out.select_dtypes(include=[np.number]).shape[1] == out.shape[1]
