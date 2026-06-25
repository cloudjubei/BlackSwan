import pytest
from omegaconf import OmegaConf

from src.conf.data_config import DataConfig
from src.data import data_factory


def _config(layers):
    return OmegaConf.structured(
        DataConfig(
            id="t",
            train_data_paths=[[]],
            test_data_paths=[[]],
            layers=list(layers),
            layers_test=list(layers),
        )
    )


def _stub_providers(monkeypatch):
    monkeypatch.setattr(data_factory, "SingleDataProvider", lambda config, paths: ("single", paths))
    monkeypatch.setattr(
        data_factory, "MultiTimelineDataProvider", lambda config, paths, *a, **k: ("multi", paths)
    )


def _create(layers, fidelity_input, fidelity_run):
    return data_factory.create_provider(_config(layers), [["f"]], fidelity_input, fidelity_run, list(layers), 0.0, 0.0)


def test_single_daily_layer_at_daily_step_uses_single_provider(monkeypatch):
    _stub_providers(monkeypatch)
    assert _create(["1d"], "1d", "1d")[0] == "single"


def test_single_hourly_layer_at_hourly_step_uses_single_provider(monkeypatch):
    _stub_providers(monkeypatch)
    assert _create(["1h"], "1h", "1h")[0] == "single"


def test_single_coarse_layer_at_finer_step_uses_multi_provider(monkeypatch):
    # 'fidelity_set=1d' at 'timeframe=1h' — one coarser layer resampled from the 1h base every hour.
    _stub_providers(monkeypatch)
    assert _create(["1d"], "1h", "1h")[0] == "multi"


def test_multi_layer_uses_multi_provider(monkeypatch):
    _stub_providers(monkeypatch)
    assert _create(["1d", "1w"], "1h", "1h")[0] == "multi"


def test_single_base_layer_at_coarser_step_uses_multi_provider(monkeypatch):
    # 'fidelity_set=1h' at 'timeframe=1d' — even a lone layer that IS the base needs the MULTI provider
    # when the STEP is coarser than the base (divider_run > 1): the single path can't step day by day.
    _stub_providers(monkeypatch)
    assert _create(["1h"], "1h", "1d")[0] == "multi"


def test_no_layers_is_unsupported(monkeypatch):
    _stub_providers(monkeypatch)
    with pytest.raises(ValueError):
        _create([], "1h", "1h")
