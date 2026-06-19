from omegaconf import OmegaConf

from src.util.dict import dict_config_to_dict, dict_config_to_params


# ---------------------------------------------------------------------------
# dict_config_to_dict : DictConfig -> plain {str(key): value}
# ---------------------------------------------------------------------------


def test_dict_config_to_dict_flat():
    cfg = OmegaConf.create({"a": 1, "b": "hi", "c": 2.5})
    assert dict_config_to_dict(cfg) == {"a": 1, "b": "hi", "c": 2.5}


def test_dict_config_to_dict_keys_are_strings():
    # keys are coerced via str(); the result mapping is keyed by plain strings.
    cfg = OmegaConf.create({"x": 10})
    out = dict_config_to_dict(cfg)
    assert all(isinstance(k, str) for k in out.keys())


def test_dict_config_to_dict_preserves_nested_value_as_is():
    # only the TOP level is unpacked; nested values stay as DictConfig (not recursed here).
    cfg = OmegaConf.create({"a": 1, "b": {"x": 2}})
    out = dict_config_to_dict(cfg)
    assert out["a"] == 1
    assert out["b"]["x"] == 2


def test_dict_config_to_dict_empty():
    assert dict_config_to_dict(OmegaConf.create({})) == {}


# ---------------------------------------------------------------------------
# dict_config_to_params : flatten nested config (dot reducer) + prefix every key
# ---------------------------------------------------------------------------


def test_dict_config_to_params_flat_keys_get_prefix():
    cfg = OmegaConf.create({"a": 1, "b": 2})
    assert dict_config_to_params(cfg, "p.") == {"p.a": 1, "p.b": 2}


def test_dict_config_to_params_nested_uses_dot_reducer():
    # flatten_dict with reducer="dot" joins nested keys with '.', then the prefix is prepended.
    cfg = OmegaConf.create({"a": 1, "b": {"x": 2, "y": 3}})
    out = dict_config_to_params(cfg, "run.")
    assert out == {"run.a": 1, "run.b.x": 2, "run.b.y": 3}


def test_dict_config_to_params_empty_prefix():
    cfg = OmegaConf.create({"a": 1, "b": {"x": 2}})
    assert dict_config_to_params(cfg, "") == {"a": 1, "b.x": 2}


def test_dict_config_to_params_deeply_nested():
    cfg = OmegaConf.create({"a": {"b": {"c": 5}}})
    assert dict_config_to_params(cfg, "k_") == {"k_a.b.c": 5}
