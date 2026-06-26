"""Fast unit tests for MultiTimelineDataProvider.

The __init__ reads JSON files and runs the whole fidelity/feature pipeline, so every test bypasses it
via ``Cls.__new__`` and wires only the attributes the method-under-test reads (synthetic in-memory
DataFrames / lists / a fake timestamp frame for the mapping math).

Focus areas:
  * the windowing/lookback grid construction in get_values, in BOTH branches
    (is_resolved_from_fidelity True/False);
  * the [n_layers, lookback, per_bar] shaping that base_crypto_env._concat_layer_grids consumes;
  * the get_current_mapping fidelity->layer bucket arithmetic;
  * the trivial accessors (timesteps / start_index / price / signals).
"""

import types

import numpy as np
import pandas as pd
import pytest

from src.data.multitimeline_dataprovider import MultiTimelineDataProvider, _period_ratio


def test_process_fidelity_cache_skips_the_expensive_resample_on_warm(tmp_path, monkeypatch):
    # SPEEDUP PROOF (A6): process_fidelity is ~97% of a multi-year provider build. The cache must make the
    # SECOND build (same window) skip the resample entirely — proven here by counting _process_fidelity_
    # uncached invocations: it runs once (cold) and zero times (warm), so the dominant cost is genuinely
    # eliminated, not merely re-paid faster.
    import types as _types

    from src.data import feature_cache

    monkeypatch.setattr(feature_cache, "_CACHE_DIR", str(tmp_path / "fc"))
    monkeypatch.setenv("BS_FEATURE_CACHE", "1")
    prov = _bare()
    prov.config = _types.SimpleNamespace(use_indicators=False, obs_squash="none")
    df = _grid_df(64, 4)
    calls = []

    def fake_uncached(*args, **kwargs):
        calls.append(1)
        return (["DFS"], ["RAW"], ["PRICES"])

    monkeypatch.setattr(prov, "_process_fidelity_uncached", fake_uncached)
    r_cold = prov.process_fidelity(df, "1d", 0, 24, "1h", 1, 24, "day_of_week")
    r_warm = prov.process_fidelity(df, "1d", 0, 24, "1h", 1, 24, "day_of_week")
    assert r_cold == r_warm == (["DFS"], ["RAW"], ["PRICES"])
    assert len(calls) == 1  # warm build did NOT recompute the resample


def test_process_fidelity_cache_misses_when_input_frame_changes(tmp_path, monkeypatch):
    # The cache key is the input frame's CONTENT hash, so a different frame must rebuild (no stale reuse).
    import types as _types

    from src.data import feature_cache

    monkeypatch.setattr(feature_cache, "_CACHE_DIR", str(tmp_path / "fc"))
    monkeypatch.setenv("BS_FEATURE_CACHE", "1")
    prov = _bare()
    prov.config = _types.SimpleNamespace(use_indicators=False, obs_squash="none")
    calls = []
    monkeypatch.setattr(prov, "_process_fidelity_uncached", lambda *a, **k: (calls.append(1), ([], [], []))[1])
    prov.process_fidelity(_grid_df(64, 4, base=0.0), "1d", 0, 24, "1h", 1, 24, "day_of_week")
    prov.process_fidelity(_grid_df(64, 4, base=99.0), "1d", 0, 24, "1h", 1, 24, "day_of_week")
    assert len(calls) == 2  # different frame content -> distinct key -> rebuilt


def test_period_ratio_base_to_target_bars():
    # How many BASE bars per one TARGET bar — drives both the run STEP size (divider_run) and each
    # observed layer's resample multiplier. Same granularity -> 1; the daily-step gap 1h->1d -> 24.
    assert _period_ratio("1h", "1h") == 1
    assert _period_ratio("1h", "1d") == 24  # the previously-missing daily-step mapping
    assert _period_ratio("1h", "1w") == 168
    assert _period_ratio("1d", "1d") == 1
    assert _period_ratio("1d", "1w") == 7  # the previously-missing 1d-base 1w multiplier
    assert _period_ratio("1m", "1h") == 60
    assert _period_ratio("1m", "5m") == 5


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def _bare():
    """A MultiTimelineDataProvider with no __init__ run; caller sets the needed attributes."""
    return MultiTimelineDataProvider.__new__(MultiTimelineDataProvider)


def _grid_df(n, cols, base=0.0):
    data = {}
    for c in range(cols):
        data[f"f{c}"] = [float(base + c * 1000 + i) for i in range(n)]
    return pd.DataFrame(data)


def _ts_df(minutes):
    """A frame whose 'timestamp' column carries the given per-row datetimes (for get_current_mapping)."""
    return pd.DataFrame({"timestamp": [pd.Timestamp(m) for m in minutes]})


# --------------------------------------------------------------------------------------------------
# trivial accessors
# --------------------------------------------------------------------------------------------------

def test_get_timesteps_returns_steps_field():
    p = _bare()
    p.steps = 42
    assert p.get_timesteps() == 42


def test_get_start_index_returns_starting_index_field():
    p = _bare()
    p.starting_index = 7
    assert p.get_start_index() == 7


def test_get_price_indexes_prices_directly():
    p = _bare()
    p.prices = [9.0, 8.0, 7.0]
    assert p.get_price(0) == 9.0
    assert p.get_price(2) == 7.0


def test_get_raw_df_for_plotting_returns_field():
    p = _bare()
    sentinel = pd.DataFrame({"a": [1]})
    p.raw_df_for_plotting = sentinel
    assert p.get_raw_df_for_plotting() is sentinel


def test_signal_getters_index_directly_no_offset():
    p = _bare()
    p.signals_buy_sell = [1, 2, 3]
    p.signals_buy_profitable = [10, 20, 30]
    p.signals_buy_drawdown = [100, 200, 300]
    assert p.get_signal_buy_sell(1) == 2
    assert p.get_signal_buy_profitable(2) == 30
    assert p.get_signal_buy_drawdown(0) == 100


# --------------------------------------------------------------------------------------------------
# get_values: non-fidelity branch (is_resolved_from_fidelity == False)
# --------------------------------------------------------------------------------------------------

def _nonfid(lookback, dfs, index_mappings, starting_index):
    p = _bare()
    p.is_resolved_from_fidelity = False
    p.config = types.SimpleNamespace(lookback_window_size=lookback)
    p.dfs = dfs
    p.index_mappings = index_mappings
    p.starting_index = starting_index
    return p


def test_nonfidelity_single_layer_lookback_grid():
    # No higher layers (index_mappings empty): one [lookback, per_bar] grid wrapped in a list -> 3-D
    # array of one layer.
    base = _grid_df(6, 2)
    p = _nonfid(lookback=2, dfs=[base], index_mappings=[], starting_index=1)
    v = p.get_values(0)  # offset = 0 + 1 = 1 -> base rows [0,1]
    assert v.shape == (1, 2, 2)  # [n_layers=1, lookback=2, per_bar=2]
    assert np.array_equal(v[0], base.loc[0:1].values)


def test_nonfidelity_two_layers_stacks_on_layer_axis():
    # Base layer + one higher layer -> [n_layers=2, lookback, per_bar]. This is the 3-D shape the env's
    # _concat_layer_grids later merges onto the feature axis.
    base = _grid_df(6, 2, base=0.0)
    hi = _grid_df(4, 2, base=900.0)
    # index_mappings[0][offset] = higher-layer index for that base offset.
    p = _nonfid(lookback=2, dfs=[base, hi], index_mappings=[[0, 1, 2, 3, 3, 3]], starting_index=1)
    v = p.get_values(0)  # offset 1 -> base rows[0,1]; hi index = mappings[0][1] = 1 -> hi rows[0,1]
    assert v.shape == (2, 2, 2)
    assert np.array_equal(v[0], base.loc[0:1].values)
    assert np.array_equal(v[1], hi.loc[0:1].values)


def test_nonfidelity_window_slides_and_higher_layer_follows_mapping():
    base = _grid_df(8, 1, base=0.0)
    hi = _grid_df(6, 1, base=500.0)
    mappings = [[0, 0, 1, 2, 3, 4, 5, 5]]
    p = _nonfid(lookback=3, dfs=[base, hi], index_mappings=mappings, starting_index=1)
    v = p.get_values(3)  # offset = 3 + 1 = 4 -> base rows [2,3,4]; hi index = mappings[0][4] = 3 -> hi [1,2,3]
    assert v.shape == (2, 3, 1)
    assert v[0][:, 0].tolist() == [2.0, 3.0, 4.0]
    assert v[1][:, 0].tolist() == [501.0, 502.0, 503.0]


def test_nonfidelity_lookback_one_single_layer_flattens():
    base = _grid_df(5, 3, base=0.0)
    p = _nonfid(lookback=1, dfs=[base], index_mappings=[], starting_index=0)
    v = p.get_values(2)  # offset 2 -> single row flattened
    assert v.ndim == 1
    assert v.tolist() == [2.0, 1002.0, 2002.0]


def test_nonfidelity_lookback_one_two_layers_concatenates_features():
    # CONTRACT: with multiple layers each layer's features are laid side-by-side on the feature axis
    # (that is what the 3-D path does and what _concat_layer_grids/the design comment describe).
    # Concatenated width = base_features + hi_features, NOT an element-wise sum.
    base = _grid_df(5, 2, base=0.0)
    hi = _grid_df(4, 2, base=900.0)
    p = _nonfid(lookback=1, dfs=[base, hi], index_mappings=[[0, 1, 2, 3, 3]], starting_index=0)
    v = p.get_values(1)  # offset 1 -> base row1 [1,1001]; hi idx mappings[0][1]=1 -> hi row1 [901,1901]
    assert v.tolist() == [1.0, 1001.0, 901.0, 1901.0]


def test_nonfidelity_lookback_one_two_layers_does_not_sum():
    # Guard against the old element-wise-add regression: layers must NOT be summed.
    base = _grid_df(5, 2, base=0.0)
    hi = _grid_df(4, 2, base=900.0)
    p = _nonfid(lookback=1, dfs=[base, hi], index_mappings=[[0, 1, 2, 3, 3]], starting_index=0)
    v = p.get_values(1)
    # base row1 = [1, 1001], hi row1 = [901, 1901] -> concatenated, not [902.0, 2902.0].
    assert v.tolist() != [902.0, 2902.0]
    assert len(v) == 4


# --------------------------------------------------------------------------------------------------
# get_values: fidelity branch (is_resolved_from_fidelity == True)
# --------------------------------------------------------------------------------------------------

def _fid(lookback, layers, fidelity_run, multipliers, fidelity_dfs, raw_df, divider_run=1, starting_index=0):
    p = _bare()
    p.is_resolved_from_fidelity = True
    p.config = types.SimpleNamespace(lookback_window_size=lookback)
    p.layers = layers
    p.fidelity_run = fidelity_run
    p.multipliers = multipliers
    p.fidelity_dfs = fidelity_dfs
    p.raw_df = raw_df
    p.divider_run = divider_run
    # The observation offset = starting_index + step*divider_run (the warmup-anchored decision bar that
    # get_price(step) is also sliced from). Real providers set this = fidelity_offset-1; tests that don't
    # care pass 0 (offset == step*divider_run).
    p.starting_index = starting_index
    return p


def test_fidelity_single_layer_grid_shape():
    layer_df = _grid_df(6, 2)
    # layer '1h' under fidelity '1h' -> get_current_mapping hits the default branch, layer '1h' -> 0.
    raw = _ts_df(["2021-01-01 00:%02d:00" % i for i in range(6)])
    p = _fid(lookback=2, layers=["1h"], fidelity_run="1h", multipliers=[1],
             fidelity_dfs=[[layer_df]], raw_df=raw)
    # step 3: run-fidelity layer's bar IS the decision bar, so the window ENDS at it (backward) -> rows [2,3].
    v = p.get_values(3)
    assert v.shape == (1, 2, 2)
    assert np.array_equal(v[0], layer_df.loc[2:3].values)


def test_fidelity_mapping_routes_to_minute_bucket():
    # fidelity '1m', layer '1h' (multiplier 60) -> mapping == minute, routing to that minute's bucket.
    # The '1h' layer is COARSER than the '1m' step, so its current hour is still forming: the window
    # ends at index-1 (the last CLOSED hour bar), never the future.
    buckets = [pd.DataFrame({"f0": [float(b * 100 + j) for j in range(10)]}) for b in range(60)]
    raw = _ts_df([pd.Timestamp("2021-01-01 00:00:00") + pd.Timedelta(minutes=i) for i in range(200)])
    p = _fid(lookback=2, layers=["1h"], fidelity_run="1m", multipliers=[60],
             fidelity_dfs=[buckets], raw_df=raw)
    # step 123 -> minute 3 -> bucket 3; index = (123-3)/60 = 2; coarser -> top = 1 -> bucket3 rows [0,1].
    v = p.get_values(123)
    assert v.shape == (1, 2, 1)
    assert v[0][:, 0].tolist() == [300.0, 301.0]


def test_fidelity_index_advances_within_bucket():
    # offset 180 (minute 0 of the 4th hour), layer '1h', multiplier 60: mapping = minute 0, index =
    # (180-0)/60 = 3, coarser -> top = 2 -> the window ending at the 3rd CLOSED hour inside bucket 0.
    buckets = [pd.DataFrame({"f0": [float(b * 100 + j) for j in range(20)]}) for b in range(60)]
    raw = _ts_df([pd.Timestamp("2021-01-01 00:00:00") + pd.Timedelta(minutes=i) for i in range(200)])
    p = _fid(lookback=2, layers=["1h"], fidelity_run="1m", multipliers=[60],
             fidelity_dfs=[buckets], raw_df=raw)
    v = p.get_values(180)  # mapping = minute(of row180)=0, index=3, top=2 -> bucket0 rows [1,2]
    assert v[0][:, 0].tolist() == [1.0, 2.0]


def test_fidelity_two_layers_stack_on_layer_axis():
    layer_a = _grid_df(6, 2, base=0.0)
    layer_b = _grid_df(6, 2, base=900.0)
    raw = _ts_df(["2021-01-01 00:%02d:00" % i for i in range(6)])
    # Both layers under default-branch fidelity '1h' map to 0 (layers not handled there -> 0); both
    # multiplier 1, so each window ENDS at the decision bar (step 3) -> rows [2,3], stacked on layer axis.
    p = _fid(lookback=2, layers=["1h", "1d"], fidelity_run="1h", multipliers=[1, 1],
             fidelity_dfs=[[layer_a], [layer_b]], raw_df=raw)
    v = p.get_values(3)
    assert v.shape == (2, 2, 2)
    assert np.array_equal(v[0], layer_a.loc[2:3].values)
    assert np.array_equal(v[1], layer_b.loc[2:3].values)


def test_fidelity_divider_run_scales_offset():
    # offset = starting_index + step*divider_run. With starting_index 0 and divider_run 5, step 26 ->
    # offset 130 -> layer '1h' under fidelity '1m' -> mapping = minute 10; index = (130-10)/60 = 2; coarser
    # -> top = 1 -> bucket10 rows [0,1]. (The end-of-period close anchoring lives in starting_index for the
    # real provider, which is an end-of-period row — so it is NOT re-added here as a +(divider-1) term.)
    buckets = [pd.DataFrame({"f0": [float(b * 100 + j) for j in range(10)]}) for b in range(60)]
    raw = _ts_df([pd.Timestamp("2021-01-01 00:00:00") + pd.Timedelta(minutes=i) for i in range(200)])
    p = _fid(lookback=2, layers=["1h"], fidelity_run="1m", multipliers=[60],
             fidelity_dfs=[buckets], raw_df=raw, divider_run=5)
    v = p.get_values(26)
    assert v[0][:, 0].tolist() == [1000.0, 1001.0]


def _daily_asof_layers(n_hours):
    """fidelity_dfs for a DAILY step over a 1h base observing [1h, 1d], each bar's feature = its as-of
    close (1h index): the 1h base bar j closes at index j; the single-phase 1d bar for day k closes at
    k*24+23 (end of day k)."""
    base = pd.DataFrame({"asof": [float(i) for i in range(n_hours)]})
    daily = pd.DataFrame({"asof": [float(k * 24 + 23) for k in range(n_hours // 24)]})
    return [[base], [daily]]


def test_resolved_daily_step_offset_is_start_index_aligned_and_never_future():
    # DAILY step (divider_run=24) over a 1h base observing [1h, 1d]. The decision bar = starting_index +
    # step*24 (the SAME base bar get_price(step) is sliced from — start_index is an end-of-day row, so the
    # decision is the day's CLOSE). The observation must end EXACTLY there: the 1h layer the end-of-day hour,
    # the 1d layer the just-closed daily bar — never a future bar (leak), never behind it (the desync the
    # offset-omits-start_index bug caused, which left the observation ~warmup days behind get_price).
    n, start_index = 12 * 24, 47  # start_index 47 = end of day 1 (a 2-day warmup); offset lands end-of-day
    p = _fid(lookback=2, layers=["1h", "1d"], fidelity_run="1d", multipliers=[1, 24],
             fidelity_dfs=_daily_asof_layers(n), raw_df=_hourly_raw(n), divider_run=24,
             starting_index=start_index)
    for step in range(0, (n - start_index) // 24):
        decision = start_index + step * 24  # == the base row get_price(step) is anchored to
        observed = np.asarray(p.get_values(step))
        assert observed.max() <= decision, (
            f"LOOK-AHEAD LEAK: daily step {step} observes a bar closing at 1h-index {observed.max()} "
            f"(future > decision {decision})")
        assert observed.max() == decision, (
            f"DESYNC/STALE: daily step {step} observation ends at {observed.max()}, not the priced "
            f"decision bar {decision} (= start_index {start_index} + step*24)")


def test_fidelity_lookback_one_single_layer_flattens():
    layer_df = _grid_df(6, 3)
    raw = _ts_df(["2021-01-01 00:%02d:00" % i for i in range(6)])
    p = _fid(lookback=1, layers=["1h"], fidelity_run="1h", multipliers=[1],
             fidelity_dfs=[[layer_df]], raw_df=raw)
    v = p.get_values(0)  # mapping 0, index 0 -> single row flattened
    assert v.ndim == 1
    assert v.tolist() == [0.0, 1000.0, 2000.0]


def test_fidelity_lookback_one_two_layers_concatenates_features():
    # Same contract as the non-fidelity lookback-one path: layers are concatenated onto the feature
    # axis, NOT element-wise added.
    layer_a = _grid_df(4, 2, base=0.0)
    layer_b = _grid_df(4, 2, base=900.0)
    raw = _ts_df(["2021-01-01 00:%02d:00" % i for i in range(4)])
    p = _fid(lookback=1, layers=["1h", "1d"], fidelity_run="1h", multipliers=[1, 1],
             fidelity_dfs=[[layer_a], [layer_b]], raw_df=raw)
    v = p.get_values(0)  # layer_a row0 [0,1000]; layer_b row0 [900,1900]
    assert v.tolist() == [0.0, 1000.0, 900.0, 1900.0]


def test_fidelity_lookback_one_two_layers_does_not_sum():
    # Guard against the old element-wise-add regression in the fidelity branch.
    layer_a = _grid_df(4, 2, base=0.0)
    layer_b = _grid_df(4, 2, base=900.0)
    raw = _ts_df(["2021-01-01 00:%02d:00" % i for i in range(4)])
    p = _fid(lookback=1, layers=["1h", "1d"], fidelity_run="1h", multipliers=[1, 1],
             fidelity_dfs=[[layer_a], [layer_b]], raw_df=raw)
    v = p.get_values(0)
    # layer_a row0 [0,1000], layer_b row0 [900,1900] -> concatenated, not [900.0, 2900.0].
    assert v.tolist() != [900.0, 2900.0]
    assert len(v) == 4


# --------------------------------------------------------------------------------------------------
# get_values LOOK-AHEAD GUARD (regression: RETURN_ENGINE_AUDIT.md — fidelity_set='1d' @ timeframe='1h')
#
# At an hourly step s the observation must contain NO bar whose close is in the future. Encodes each
# resampled bar's feature value as its OWN as-of 1h close index; asserts every observed value <= s.
# Before the fix, the resolved-fidelity window sliced df.loc[index : index+lb-1] FORWARD, so a coarser
# (e.g. daily) layer observed bars closing up to ~767h ahead — manufacturing the audited +261% return.
# --------------------------------------------------------------------------------------------------

def _hourly_raw(n_hours):
    base = pd.Timestamp("2024-01-01 00:00:00")
    return _ts_df([base + pd.Timedelta(hours=i) for i in range(n_hours)])


def _daily_substreams_asof(n_hours):
    """24 phase-shifted daily substreams resampled from an hourly base, exactly as process_fidelity
    lays them out: substream h, bar k = 1h block [h+k*24 .. h+k*24+23], whose CLOSE is 1h index
    h+k*24+23. Each bar's single feature carries that close index (its as-of time)."""
    subs = []
    for h in range(24):
        closes, k = [], 0
        while h + k * 24 + 23 < n_hours:
            closes.append(float(h + k * 24 + 23))
            k += 1
        subs.append(pd.DataFrame({"asof": closes}))
    return subs


def test_resolved_fidelity_coarser_layer_never_observes_future():
    lookback, n = 4, 24 * 12
    p = _fid(lookback=lookback, layers=["1d"], fidelity_run="1h", multipliers=[24],
             fidelity_dfs=[_daily_substreams_asof(n)], raw_df=_hourly_raw(n))
    for step in range(0, n - 24):
        observed = np.asarray(p.get_values(step))
        assert observed.size == 0 or observed.max() <= step, (
            f"LOOK-AHEAD LEAK: hourly step {step} observes a daily bar closing at 1h-index "
            f"{observed.max()} (future > {step})")


def test_resolved_fidelity_run_layer_never_observes_future():
    lookback, n = 4, 50
    base = pd.DataFrame({"asof": [float(i) for i in range(n)]})  # run-fidelity bar i closes at index i
    p = _fid(lookback=lookback, layers=["1h"], fidelity_run="1h", multipliers=[1],
             fidelity_dfs=[[base]], raw_df=_hourly_raw(n))
    for step in range(0, n):
        observed = np.asarray(p.get_values(step))
        assert observed.max() <= step, (
            f"LOOK-AHEAD LEAK: run-fidelity step {step} observes bar at index {observed.max()} (future)")


# --------------------------------------------------------------------------------------------------
# get_current_mapping: fidelity -> layer bucket arithmetic
# --------------------------------------------------------------------------------------------------

# Wed 2021-01-06 13:25:00 -> weekday()=2 (Mon=0), hour=13, minute=25.
_WED = _ts_df(["2021-01-06 13:25:00"])


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("1w", 2 * 1440 + 13 * 60 + 25),  # 3685
        ("1d", 13 * 60 + 25),             # 805
        ("4h", (13 % 4) * 60 + 25),       # 85
        ("1h", 25),                       # minute
        ("30m", 25 % 30),                 # 25
        ("15m", 25 % 15),                 # 10
        ("10m", 25 % 10),                 # 5
        ("5m", 25 % 5),                   # 0
        ("2h", 0),                        # unrecognised layer -> 0
    ],
)
def test_get_current_mapping_fidelity_1m(layer, expected):
    p = _bare()
    assert p.get_current_mapping(_WED, 0, layer, "1m") == expected


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("1w", 2 * 24 + 13),  # weekday*24 + hour = 61
        ("1d", 13 % 24),      # 13
        ("4h", 13 % 4),       # 1
        ("1h", 0),            # unrecognised in the default (hourly) branch -> 0
    ],
)
def test_get_current_mapping_default_hourly_branch(layer, expected):
    # Any fidelity not in {1m,30m,15m,10m,5m,1d} falls through to the else (hourly) branch.
    p = _bare()
    assert p.get_current_mapping(_WED, 0, layer, "1h") == expected


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("1w", 2),  # daily step buckets by DAY -> weekday (Wed -> 2 of 0..6)
        ("1d", 0),  # the 1d layer at a daily step is a single phase
        ("1h", 0),  # the 1h base collapses to its single df
    ],
)
def test_get_current_mapping_daily_step(layer, expected):
    # A DAILY step (fidelity='1d') over the 1h base buckets by DAY, not hour — the hourly else-branch
    # would return out-of-range hour phases (e.g. weekday*24+hour for 1w), indexing past the substreams.
    p = _bare()
    assert p.get_current_mapping(_WED, 0, layer, "1d") == expected


def test_get_current_mapping_daily_step_weekday_full_cycle():
    # The 1w layer at a daily step selects one of EXACTLY 7 weekday-phase substreams (multiplier_run =
    # 168/24 or 7/1 = 7). Pin the full Mon..Sun cycle -> 0..6 (the existing test pins only Wed->2) and that
    # the phase NEVER overruns 0..6 — the old hourly else-branch returned weekday*24+hour (up to 167),
    # which would index far past the 7 substreams. '1d'/'1h' collapse to the single phase 0.
    week = _ts_df([f"2021-01-{4 + i:02d} 09:30:00" for i in range(7)])  # 2021-01-04 is a Monday
    p = _bare()
    for i in range(7):
        m = p.get_current_mapping(week, i, "1w", "1d")
        assert m == i, f"weekday phase for day-offset {i} should be {i}, got {m}"
        assert 0 <= m <= 6
        assert p.get_current_mapping(week, i, "1d", "1d") == 0
        assert p.get_current_mapping(week, i, "1h", "1d") == 0


@pytest.mark.parametrize(
    "fidelity,layer,expected",
    [
        ("30m", "1h", int(25 / 30) % 2),   # 0
        ("15m", "1h", int(25 / 15) % 4),   # 1
        ("10m", "1h", int(25 / 10) % 6),   # 2
        ("5m", "1h", int(25 / 5) % 12),    # 5
        ("30m", "9x", 0),                  # unrecognised layer -> 0
    ],
)
def test_get_current_mapping_sub_hour_fidelities(fidelity, layer, expected):
    p = _bare()
    assert p.get_current_mapping(_WED, 0, layer, fidelity) == expected


def test_get_current_mapping_uses_step_row():
    # The mapping is computed from the timestamp at the given step row, not row 0.
    df = _ts_df(["2021-01-06 13:00:00", "2021-01-06 13:25:00"])
    p = _bare()
    assert p.get_current_mapping(df, 1, "1h", "1m") == 25
    assert p.get_current_mapping(df, 0, "1h", "1m") == 0


def test_get_timestamp_datetime_parses_step_row():
    df = _ts_df(["2021-01-06 13:25:00", "2021-01-07 09:10:00"])
    p = _bare()
    dt = p.get_timestamp_datetime(df, 1)
    assert dt.hour == 9 and dt.minute == 10
