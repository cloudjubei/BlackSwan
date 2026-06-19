"""Direct tests for the env dispatch in ``create_environment``.

This is the highest-value test in this batch: it pins that each ``EnvConfig.type`` key maps to the
right env class. Constructing the real env classes runs a heavy ``__init__`` (data-provider IO + an
initial ``reset``), so we monkeypatch the factory's class references with cheap recording stubs and
assert which stub was instantiated. Only the dispatch table is under test, not the env bodies.
"""

import pytest

import src.environment.env_factory as env_factory
from src.conf.env_config import EnvConfig


class _Recorder:
    """Stand-in env: records the (config, data_provider, device) it was built with."""

    last = None

    def __init__(self, config, data_provider, device):
        self.config = config
        self.data_provider = data_provider
        self.device = device
        type(self).last = self


def _patch_all(monkeypatch):
    # Distinct subclasses so each dispatch branch can be told apart by identity.
    classes = {}
    for name in (
        "TradeAllCryptoEnv",
        "SwapCryptoEnv",
        "TrendPredictEnv",
        "DipPredictEnv",
        "RegressionPredictEnv",
    ):
        cls = type(name, (_Recorder,), {})
        monkeypatch.setattr(env_factory, name, cls)
        classes[name] = cls
    return classes


_TYPE_TO_CLASS = {
    "trade_all": "TradeAllCryptoEnv",
    "swap": "SwapCryptoEnv",
    "trend_predict": "TrendPredictEnv",
    "dip_predict": "DipPredictEnv",
    "regression_predict": "RegressionPredictEnv",
}


@pytest.mark.parametrize("env_type,expected_cls_name", sorted(_TYPE_TO_CLASS.items()))
def test_dispatch_maps_type_to_correct_class(monkeypatch, env_type, expected_cls_name):
    classes = _patch_all(monkeypatch)
    config = EnvConfig(type=env_type)
    sentinel_provider = object()

    out = env_factory.create_environment(config, sentinel_provider, "cpu")

    expected_cls = classes[expected_cls_name]
    assert isinstance(out, expected_cls)
    # Every other stub class must be untouched, proving an exact 1:1 mapping (no fall-through).
    for name, cls in classes.items():
        if name != expected_cls_name:
            assert cls.last is None


def test_dispatch_forwards_arguments_verbatim(monkeypatch):
    classes = _patch_all(monkeypatch)
    config = EnvConfig(type="trade_all")
    provider = object()

    out = env_factory.create_environment(config, provider, "cuda:0")

    assert out.config is config
    assert out.data_provider is provider
    assert out.device == "cuda:0"


def test_unknown_type_raises_value_error(monkeypatch):
    _patch_all(monkeypatch)
    config = EnvConfig(type="does_not_exist")
    with pytest.raises(ValueError):
        env_factory.create_environment(config, object(), "cpu")


@pytest.mark.xfail(reason="BUG: error string is missing the f-prefix so {config.type} is not interpolated", strict=False)
def test_unknown_type_error_message_interpolates_type(monkeypatch):
    # CONTRACT: the error should name the offending type. The source uses a plain (non-f) string
    # literal "{config.type} - env not supported", so the type is NOT interpolated -> documents the bug.
    _patch_all(monkeypatch)
    config = EnvConfig(type="totally_bogus")
    with pytest.raises(ValueError) as exc:
        env_factory.create_environment(config, object(), "cpu")
    assert "totally_bogus" in str(exc.value)
