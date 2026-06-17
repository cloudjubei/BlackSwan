import pytest

from trainer.fidelity import DEFAULT_FIDELITY_SET, fidelity_set_ids, resolve_fidelity


def test_run_frequency_comes_from_timeframe_not_the_set():
    _, spec = resolve_fidelity({"timeframe": "1h", "fidelity_set": "1h+1d+1w"})
    assert spec["fidelity_run"] == "1h"
    assert spec["fidelity_input"] == "1h"
    assert spec["layers"] == ["1h", "1d", "1w"]
    assert spec["lookback"] == 32


def test_auto_follows_timeframe():
    assert resolve_fidelity({"timeframe": "1h"})[1]["layers"] == ["1h", "1d"]
    assert resolve_fidelity({"timeframe": "1d"})[1]["layers"] == ["1d"]
    assert resolve_fidelity({"timeframe": "1h", "fidelity_set": "auto"})[0] == "auto"
    assert DEFAULT_FIDELITY_SET == "auto"


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


def test_daily_step_with_finer_layers_fails_fast():
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": "1d", "fidelity_set": "1h+1d"})


def test_daily_step_with_single_1h_fails_fast():
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": "1d", "fidelity_set": "1h"})


def test_hourly_step_with_single_daily_fails_fast():
    # Single '1d' at an hourly step is incoherent — use '1h' or a multi-layer set like '1d+1w'.
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": "1h", "fidelity_set": "1d"})


def test_unknown_fidelity_set_fails_fast():
    with pytest.raises(SystemExit):
        resolve_fidelity({"timeframe": "1h", "fidelity_set": "5m+1h"})


def test_ids_include_auto_and_coarser_only():
    ids = fidelity_set_ids()
    assert "auto" in ids and "1d+1w" in ids and "1h+1d+1w" in ids
