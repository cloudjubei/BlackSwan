"""End-to-end look-ahead REGRESSION guard for the daily-step multi-fidelity path.

The fast unit guards in test_multitimeline_dataprovider.py (test_resolved_*_never_observes_future,
test_resolved_daily_step_observes_end_of_day_never_future) build the provider via the `_fid` helper,
which hand-wires fidelity_dfs and BYPASSES process_fidelity. The +261% look-ahead leak in
RETURN_ENGINE_AUDIT.md lived in the get_values offset/index/top arithmetic AND the process_fidelity
resampling layout — so the only test that can catch a recurrence is one that runs the WHOLE real pipeline
(build_data_config -> real ensure_derived resampling -> create_provider) and asserts, for every step and
every observed layer, that the newest observed bar's CLOSE never reaches past the decision moment.

The invariant (mirrors the unit guards, on real timestamps): at a daily step the decision bar is
`raw_df row step*divider_run + get_start_index()` (the warmup-anchored bar get_price(step) is also sliced
from; start_index is an end-of-period row, so this is the day's CLOSE). No observed bar in ANY layer may
close after that timestamp. (It MAY close before — a coarser still-forming layer observes the previous
closed bar via the top=index-1 guard; that is conservative-safe, not a leak.) A second test pins the
ALIGNMENT: the observation must end on exactly that bar, not behind it (the omits-start_index desync).

Needs the real BTCUSDT 1h+1d klines under binance/; skipped when absent so a data-less checkout stays green.
"""

import os

import numpy as np
import pandas as pd
import pytest

import trainer.config_builder as cb
import trainer.walk_forward as wf
from src.data.data_factory import create_provider

_HAVE_DATA = os.path.exists("binance/BTCUSDT-1h-2020-1.json") and os.path.exists(
    "binance/BTCUSDT-1d-2020-1.json"
)
pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason="needs binance/ BTCUSDT 1h+1d klines for the real-resampling e2e guard"
)


def _tiny_pairs(cfg=None):
    # Three months is enough for daily/weekly resampling + the lookback warmup, and keeps the derived
    # cache build fast. Test window is irrelevant here (we only step the train provider).
    return [(2020, 1), (2020, 2), (2020, 3)], [(2020, 4)], {"walk_forward_window": "tiny"}


def _ts_col(df):
    for c in ("timestamp_close", "timestamp"):
        if c in df.columns:
            return c
    raise AssertionError(f"no timestamp column in {list(df.columns)}")


def _build(timeframe, fidelity_set, monkeypatch):
    monkeypatch.setattr(wf, "resolve_walk_forward_window", _tiny_pairs)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", _tiny_pairs)
    dc = cb.build_data_config({"timeframe": timeframe, "fidelity_set": fidelity_set})
    return create_provider(
        dc,
        dc.train_data_paths,
        dc.fidelity_input,
        dc.fidelity_run,
        list(dc.layers),
        dc.buyreward_maxwait,
        dc.buyreward_percent,
    )


def _observed_close(p, raw, raw_ts, offset, i, layer):
    """The CLOSE timestamp of the newest bar layer `i` observes at base offset `offset`, re-deriving
    get_values' own offset/mapping/index/top arithmetic. Returns None when no bar has closed yet
    (top < 0 -> the zero-pad sentinel, nothing observed)."""
    mapping = p.get_current_mapping(raw, offset, layer, p.fidelity_run)
    index = int((offset - mapping) / p.multipliers[i])
    top = index if p.multipliers[i] <= p.divider_run else index - 1
    if top < 0:
        return None
    if layer == p.fidelity_input:
        src = raw  # the base layer reads the strided raw frame, not fidelity_raw_dfs
    else:
        # fidelity_raw_dfs holds ONLY the resampled (non-base) layers, in layer order — map i past the base.
        j = sum(1 for prev in p.layers[:i] if prev != p.fidelity_input)
        src = p.fidelity_raw_dfs[j][mapping]
    src_ts = _ts_col(src)
    return pd.to_datetime(src[src_ts].iloc[min(top, len(src) - 1)])


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h", "1d+1w"])
def test_daily_step_no_observed_bar_closes_after_decision_real_provider(fidelity_set, monkeypatch):
    p = _build("1d", fidelity_set, monkeypatch)
    raw = p.raw_df
    raw_ts = _ts_col(raw)
    steps = p.get_timesteps()
    assert steps > 5, f"{fidelity_set}: too few steps ({steps}) to be a meaningful guard"

    checks = 0
    for step in range(steps):
        offset = step * p.divider_run + p.get_start_index()
        decision = pd.to_datetime(raw[raw_ts].iloc[offset])
        for i, layer in enumerate(p.layers):
            observed = _observed_close(p, raw, raw_ts, offset, i, layer)
            if observed is None:
                continue
            checks += 1
            assert observed <= decision, (
                f"LOOK-AHEAD LEAK [{fidelity_set}] step {step} layer '{layer}': observed close {observed} "
                f"> decision close {decision} (margin {observed - decision})"
            )
    assert checks > 0, f"{fidelity_set}: no layer/step was actually checked"


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h"])
def test_daily_step_observation_offset_matches_price_anchor_real_provider(fidelity_set, monkeypatch):
    """The observation must end on the SAME base bar get_price(step) is sliced from
    (start_index + step*divider_run). As-of-encode the BASE layer (each row -> its own base-row index) so
    its newest observed value reveals the offset get_values actually used; assert that == the price anchor
    for every step. RED before the get_start_index() offset fix (the observation was anchored at
    divider_run-1, a full warmup ~31 days BEHIND the priced/rewarded bar — the desync)."""
    p = _build("1d", fidelity_set, monkeypatch)
    base_idx = list(p.layers).index(p.fidelity_input)
    # Single-column as-of encoding so the per-layer grids stack cleanly: base row r -> r (reveals the
    # offset); other layers -> 0 (ignored by the anchor assertion, and trivially <= offset so this also
    # re-confirms nothing is observed past the decision bar).
    for i in range(len(p.fidelity_dfs)):
        for m in range(len(p.fidelity_dfs[i])):
            ln = len(p.fidelity_dfs[i][m])
            vals = [float(r) for r in range(ln)] if i == base_idx else [0.0] * ln
            p.fidelity_dfs[i][m] = pd.DataFrame({"asof": vals})
    si = p.get_start_index()
    for step in range(p.get_timesteps()):
        anchor = si + step * p.divider_run
        result = np.asarray(p.get_values(step))
        assert result[base_idx].max() == anchor, (
            f"DESYNC [{fidelity_set}] step {step}: observation ends at base row {result[base_idx].max()} "
            f"but get_price(step) is anchored at {anchor} (start_index {si} + step*{p.divider_run})"
        )
        assert result.max() <= anchor


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h", "1d+1w"])
def test_daily_step_boundary_get_values_finite_and_no_indexerror(fidelity_set, monkeypatch):
    # The classic off-by-one boundaries: step 0 (lookback front-pad / top<0 zero-grid) and the last step
    # (offset near len(raw_df), top=min(top,len-1) clamp). Both must produce finite, correctly shaped
    # observations with no IndexError.
    p = _build("1d", fidelity_set, monkeypatch)
    steps = p.get_timesteps()
    for step in (0, 1, steps - 1):
        v = np.asarray(p.get_values(step))
        assert np.all(np.isfinite(v)), f"{fidelity_set} step {step}: non-finite observation"
