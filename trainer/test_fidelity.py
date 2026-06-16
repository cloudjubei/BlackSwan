import pytest

from trainer.fidelity import DEFAULT_INTRADAY_SET, fidelity_set_ids, resolve_fidelity


def test_explicit_fidelity_set_wins():
    fid, spec = resolve_fidelity({"fidelity_set": "1h+1d+1w"})
    assert fid == "1h+1d+1w"
    assert spec["layers"] == ["1h", "1d", "1w"]
    assert spec["fidelity_run"] == "1h"
    assert spec["lookback"] == 32


def test_single_daily_set():
    fid, spec = resolve_fidelity({"fidelity_set": "1d"})
    assert fid == "1d"
    assert spec["layers"] == ["1d"]
    assert spec["fidelity_run"] == "1d"
    assert spec["lookback"] == 1


def test_derives_from_timeframe_when_absent_1h():
    fid, spec = resolve_fidelity({"timeframe": "1h"})
    assert fid == DEFAULT_INTRADAY_SET == "1h+1d"
    assert spec["layers"] == ["1h", "1d"]
    assert spec["lookback"] == 32


def test_derives_from_timeframe_when_absent_1d():
    fid, spec = resolve_fidelity({"timeframe": "1d"})
    assert fid == "1d"
    assert spec["layers"] == ["1d"]
    assert spec["lookback"] == 1


def test_default_cfg_is_daily():
    fid, _ = resolve_fidelity()
    assert fid == "1d"


def test_auto_derives_from_timeframe_so_the_lever_default_is_safe():
    assert resolve_fidelity({"fidelity_set": "auto", "timeframe": "1h"})[0] == "1h+1d"
    assert resolve_fidelity({"fidelity_set": "auto", "timeframe": "1d"})[0] == "1d"
    assert resolve_fidelity({"fidelity_set": "", "timeframe": "1h"})[0] == "1h+1d"


def test_unknown_fidelity_set_fails_fast():
    with pytest.raises(SystemExit):
        resolve_fidelity({"fidelity_set": "5m+1h"})


def test_ids_listed_narrow_to_wide():
    assert fidelity_set_ids() == ["1d", "1h", "1h+1d", "1h+1d+1w"]
