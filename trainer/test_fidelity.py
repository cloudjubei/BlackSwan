import pytest

from trainer import fidelity as fidelity_mod
from trainer.fidelity import DEFAULT_FIDELITY_SET, fidelity_set_ids, resolve_fidelity


def test_run_frequency_comes_from_timeframe_not_the_set():
    _, spec = resolve_fidelity({"timeframe": "1h", "fidelity_set": "1h+1d+1w"})
    assert spec["fidelity_run"] == "1h"
    assert spec["fidelity_input"] == "1h"
    assert spec["layers"] == ["1h", "1d", "1w"]
    assert spec["lookback"] == 32


def test_auto_resolves_to_the_concrete_set_id():
    # "auto" is a launch-form convenience that resolves to the ACTUAL layer-set id, so stored runs carry
    # the real value (never the synonym) and group cleanly with explicit picks.
    assert resolve_fidelity({"timeframe": "1h"})[1]["layers"] == ["1h", "1d"]
    assert resolve_fidelity({"timeframe": "1d"})[1]["layers"] == ["1d"]
    assert resolve_fidelity({"timeframe": "1h", "fidelity_set": "auto"})[0] == "1h+1d"
    assert resolve_fidelity({"timeframe": "1d", "fidelity_set": "auto"})[0] == "1d"
    assert DEFAULT_FIDELITY_SET == "auto"  # still the default INPUT in the launch form


def test_default_cfg_is_single_daily():
    fid, spec = resolve_fidelity()
    assert spec["fidelity_run"] == "1d"
    assert spec["layers"] == ["1d"]
    assert spec["lookback"] == 1
    assert spec["fidelity_input"] == "1d"


def test_single_daily_set_at_daily_timeframe():
    _, spec = resolve_fidelity({"timeframe": "1d", "fidelity_set": "1d"})
    assert spec["layers"] == ["1d"]
    assert spec["fidelity_run"] == "1d"


def test_coarser_only_stack_at_hourly_step():
    # An hourly-stepping agent fed only coarser layers (1d, 1w) every hour.
    _, spec = resolve_fidelity({"timeframe": "1h", "fidelity_set": "1d+1w"})
    assert spec["layers"] == ["1d", "1w"]
    assert spec["fidelity_run"] == "1h"
    assert spec["fidelity_input"] == "1h"


def test_daily_step_with_finer_layers_uses_the_1h_base():
    # A daily-stepping agent CAN observe finer (1h) layers now: load the 1h base and step it day by day
    # (the symmetric counterpart of 1h-step observing a resampled 1d). The base is the finest layer.
    _, spec = resolve_fidelity({"timeframe": "1d", "fidelity_set": "1h+1d"})
    assert spec["layers"] == ["1h", "1d"]
    assert spec["fidelity_run"] == "1d"
    assert spec["fidelity_input"] == "1h"  # finest layer drives the base


def test_daily_step_with_single_1h_uses_the_1h_base():
    _, spec = resolve_fidelity({"timeframe": "1d", "fidelity_set": "1h"})
    assert spec["layers"] == ["1h"]
    assert spec["fidelity_run"] == "1d"
    assert spec["fidelity_input"] == "1h"


def test_daily_step_coarser_only_stack_stays_on_1d_base():
    # No 1h layer -> the 1d base is the finest; the 1w layer is resampled from it (step stays daily).
    _, spec = resolve_fidelity({"timeframe": "1d", "fidelity_set": "1d+1w"})
    assert spec["layers"] == ["1d", "1w"]
    assert spec["fidelity_run"] == "1d"
    assert spec["fidelity_input"] == "1d"


def test_hourly_step_with_single_daily_resolves():
    # A single coarser '1d' layer at an hourly step is '1h+1d' with the '1h' dropped: the agent acts
    # hourly but observes only the 1d layer (resampled from the 1h base every hour).
    _, spec = resolve_fidelity({"timeframe": "1h", "fidelity_set": "1d"})
    assert spec["layers"] == ["1d"]
    assert spec["fidelity_run"] == "1h"
    assert spec["fidelity_input"] == "1h"


def test_minute_step_auto_resolves_to_1m_plus_1h():
    # B0: a minute-stepping agent on the 1m base, observing 1m micro-structure + an hourly context layer.
    _, spec = resolve_fidelity({"timeframe": "1m"})
    assert spec["fidelity_run"] == "1m"
    assert spec["fidelity_input"] == "1m"
    assert spec["layers"] == ["1m", "1h"]
    assert spec["lookback"] == 32


def test_minute_base_observed_at_hourly_step_decouples_cadence():
    # B2: observe 1m micro-structure but DECIDE hourly — fidelity_input=1m (base, stepped 60 bars/decision
    # via divider_run), fidelity_run=1h (the decision cadence). This is the cadence/data decoupling.
    _, spec = resolve_fidelity({"timeframe": "1h", "fidelity_set": "1m+1h"})
    assert spec["fidelity_run"] == "1h"
    assert spec["fidelity_input"] == "1m"
    assert spec["layers"] == ["1m", "1h"]
    assert spec["lookback"] == 32


def test_minute_step_with_coarse_layers_uses_1m_base():
    _, spec = resolve_fidelity({"timeframe": "1m", "fidelity_set": "1m+1h+1d"})
    assert spec["fidelity_input"] == "1m"
    assert spec["fidelity_run"] == "1m"
    assert spec["layers"] == ["1m", "1h", "1d"]


def test_validate_accepts_a_minute_run_and_minute_base_layers():
    assert fidelity_mod._validate("1m", ["1m", "1h"], "1m+1h") is None
    assert fidelity_mod._validate("1h", ["1m", "1h"], "1m+1h") is None


def test_minute_set_ids_are_offered():
    ids = fidelity_set_ids()
    assert "1m" in ids and "1m+1h" in ids and "1m+1h+1d" in ids


def test_unknown_fidelity_set_fails_fast():
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": "1h", "fidelity_set": "5m+1h"})


def test_ids_include_auto_and_coarser_only():
    ids = fidelity_set_ids()
    assert "auto" in ids and "1d+1w" in ids and "1h+1d+1w" in ids


def test_unsupported_timeframe_fails_fast_via_public_api():
    # A timeframe that is neither 1h nor 1d (e.g. weekly stepping) is not provider-supported — `auto`
    # picks the ['1d'] stack, then _validate rejects the unknown run. Covers the line-88 error path.
    with pytest.raises(SystemExit, match="unsupported timeframe"):
        resolve_fidelity({"timeframe": "1w"})


@pytest.mark.parametrize("run", ["1w", "5m", ""])
def test_unsupported_timeframes_all_raise(run):
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": run})


def test_validate_hourly_step_rejects_finer_than_1h_layer():
    # Defensive guard (line 82): an hourly-step agent cannot observe a sub-hourly layer because the 1h
    # base is the finest and can't be upsampled. Unreachable through the public sets (every _LAYER_SETS
    # entry is within {1h,1d,1w}), so exercised directly to pin the fail-fast contract.
    with pytest.raises(SystemExit, match="incompatible"):
        fidelity_mod._validate("1h", ["5m"], "5m+1h")


def test_validate_hourly_step_accepts_allowed_layers_returns_none():
    # All-allowed layers at an hourly step pass validation (returns implicitly None, raises nothing).
    assert fidelity_mod._validate("1h", ["1h", "1d", "1w"], "1h+1d+1w") is None


def test_validate_daily_step_single_1d_passes():
    assert fidelity_mod._validate("1d", ["1d"], "1d") is None
