import math

import pytest

from trainer import dip


# --- tiny fakes for build_summary (env + model collaborators) -----------------


class _FakeEnv:
    """Carries only the attributes build_summary reads off the test env."""

    def __init__(self, n_positive=0, n_negative=0, predictions=None):
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.predictions = predictions


class _FakeModel:
    def __init__(self, model_id=None):
        self.id = model_id


# A regression run_state is [f1, simple_ratio, accuracy, precision, recall, negative_recall, ...]
def _state(f1=0.0, simple_ratio=0.0, accuracy=0.0, precision=0.0, recall=0.0, negative_recall=0.0):
    return [f1, simple_ratio, accuracy, precision, recall, negative_recall]


# --- build_data_config -------------------------------------------------------


def test_build_data_config_defaults_to_btc_1d_dip():
    cfg = dip.build_data_config({})
    assert cfg.id == "BTCUSDT-1d-dip"
    assert cfg.lookback_window_size == 1
    assert cfg.timestamp == "none"
    assert cfg.type == "only_price_percent"
    assert cfg.fidelity_input == "1d" and cfg.fidelity_run == "1d"
    assert list(cfg.layers) == ["1d"]
    assert cfg.fidelity_input_test == "1d" and cfg.fidelity_run_test == "1d"
    assert list(cfg.layers_test) == ["1d"]
    # buyreward defaults
    assert cfg.buyreward_maxwait == 5
    assert cfg.buyreward_percent == pytest.approx(0.02)


def test_build_data_config_honours_asset_and_buyreward_levers():
    cfg = dip.build_data_config(
        {"asset": "ETHUSDT", "buyreward_maxwait": 7, "buyreward_percent": 0.03, "data_type": "only_price"}
    )
    assert cfg.id == "ETHUSDT-1d-dip"
    assert cfg.type == "only_price"
    assert cfg.buyreward_maxwait == 7
    assert cfg.buyreward_percent == pytest.approx(0.03)
    # the test-side maxwait/percent mirror the train-side values
    assert cfg.buyreward_maxwait_test == 7
    assert cfg.buyreward_percent_test == pytest.approx(0.03)


def test_build_data_config_paths_are_window_pairs(monkeypatch):
    # echo the daily-file resolver so we can assert which pairs were used without touching disk.
    from trainer import config_builder

    monkeypatch.setattr(
        config_builder, "_daily_files", lambda pairs, symbol="BTCUSDT": [f"{symbol}:{y}-{m}" for (y, m) in pairs]
    )
    cfg = dip.build_data_config({"asset": "SOLUSDT"})
    train = list(cfg.train_data_paths[0])
    test = list(cfg.test_data_paths[0])
    # the dip line uses the canonical 2020-2023 train / 2024 test single split.
    assert train[0] == "SOLUSDT:2020-1"
    assert train[-1] == "SOLUSDT:2023-12"
    assert test == [f"SOLUSDT:2024-{m}" for m in range(1, 13)]


# --- build_env_config --------------------------------------------------------


def test_build_env_config_is_regression_predict_with_empty_observations():
    cfg = dip.build_env_config({})
    assert cfg.type == "regression_predict"
    assert list(cfg.observations_contain) == []
    assert cfg.batch_size == 32


def test_build_env_config_honours_batch_size_lever():
    cfg = dip.build_env_config({"batch_size": 128})
    assert cfg.batch_size == 128


# --- build_model_config ------------------------------------------------------


def test_build_model_config_defaults():
    config = dip.build_model_config({})
    assert config.model_type == "regression"
    reg = config.model_regression
    assert reg is not None
    assert reg.learning_rate == pytest.approx(0.0001)
    assert reg.loss_fn == "bcelogits"
    assert reg.episodes == 1
    assert reg.seed is None
    assert reg.pos_weight == pytest.approx(0.0)
    assert reg.decision_threshold == pytest.approx(0.5)
    # the dip line always trains exactly one model (no inner best-of-N pick).
    assert config.iterations_to_pick_best == 1


def test_build_model_config_maps_all_scalar_levers():
    config = dip.build_model_config(
        {
            "learning_rate": 0.0005,
            "loss_fn": "bce",
            "episodes": 4,
            "seed": 11,
            "pos_weight": 2.5,
            "decision_threshold": 0.7,
        }
    )
    reg = config.model_regression
    assert reg.learning_rate == pytest.approx(0.0005)
    assert reg.loss_fn == "bce"
    assert reg.episodes == 4
    assert reg.seed == 11
    assert reg.pos_weight == pytest.approx(2.5)
    assert reg.decision_threshold == pytest.approx(0.7)


def test_build_model_config_seed_none_when_absent_or_explicit_none():
    assert dip.build_model_config({}).model_regression.seed is None
    assert dip.build_model_config({"seed": None}).model_regression.seed is None


def test_build_model_config_checkpoint_only_when_truthy():
    assert dip.build_model_config({}).model_regression.checkpoint_to_load is None
    # an empty-string checkpoint is falsy and must NOT be applied.
    assert dip.build_model_config({"checkpoint_to_load": ""}).model_regression.checkpoint_to_load is None
    cfg = dip.build_model_config({"checkpoint_to_load": "run-123"})
    assert cfg.model_regression.checkpoint_to_load == "run-123"


def test_build_model_config_optional_net_arch_and_activation():
    config = dip.build_model_config({"net_arch": "64,32", "activation_fn": "Tanh"})
    assert list(config.model_regression.net_arch) == [64, 32]
    assert config.model_regression.activation_fn == "Tanh"


def test_build_model_config_net_arch_accepts_list():
    config = dip.build_model_config({"net_arch": [128, 64]})
    assert list(config.model_regression.net_arch) == [128, 64]


def test_build_model_config_does_not_mutate_the_shared_template():
    # build_model_config must deepcopy model_regression_dip; two calls with different levers
    # should not bleed into each other.
    from src.conf.model_config import model_regression_dip

    before = list(model_regression_dip.model_regression.learning_rate)
    dip.build_model_config({"learning_rate": 0.999})
    after = list(model_regression_dip.model_regression.learning_rate)
    assert before == after


# --- _finite -----------------------------------------------------------------


def test_finite_passes_through_finite_numbers():
    assert dip._finite(5) == 5.0
    assert dip._finite(2.5) == pytest.approx(2.5)
    assert dip._finite(0) == 0.0


def test_finite_replaces_nan_and_inf_with_default():
    assert dip._finite(float("nan")) == 0.0
    assert dip._finite(float("inf"), default=9.0) == 9.0
    assert dip._finite(float("-inf"), default=-1.0) == -1.0


def test_finite_replaces_non_numeric_with_default():
    assert dip._finite("oops", default=3.0) == 3.0
    assert dip._finite(None, default=-2.0) == -2.0


def test_finite_default_is_zero():
    assert dip._finite([1, 2]) == 0.0


# --- _health -----------------------------------------------------------------


def test_health_ok_with_positive_signal():
    state = _state(precision=0.4, recall=0.6)
    assert dip._health(state) == {"status": "ok", "flags": []}


def test_health_degenerate_when_no_positive_signal():
    state = _state(precision=0.0, recall=0.0)
    health = dip._health(state)
    assert health["status"] == "degenerate"
    assert health["flags"] == ["no_positive_signal"]


def test_health_short_state_defaults_to_degenerate():
    # a state shorter than index 5 yields precision=recall=0 -> no positive signal.
    assert dip._health([0.0, 0.0])["status"] == "degenerate"


def test_health_treats_nan_precision_recall_as_zero():
    state = _state(precision=float("nan"), recall=float("nan"))
    assert dip._health(state)["status"] == "degenerate"


def test_health_ok_when_only_precision_positive():
    # precision + recall > 0 is enough to clear the degenerate flag.
    assert dip._health(_state(precision=0.3, recall=0.0))["status"] == "ok"


# --- build_summary -----------------------------------------------------------


def test_build_summary_objective_is_f1_and_metrics_map_state():
    state = _state(f1=0.8, simple_ratio=0.55, accuracy=0.7, precision=0.6, recall=0.9, negative_recall=0.4)
    env = _FakeEnv(n_positive=10, n_negative=30, predictions=[1] * 40)
    out = dip.build_summary(env, state, {"asset": "ETHUSDT"}, _FakeModel(), "2026-01-01T00:00:00Z")
    assert out["objective"] == pytest.approx(0.8)
    m = out["metrics"]
    assert m["f1"] == pytest.approx(0.8)
    assert m["simple_ratio"] == pytest.approx(0.55)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(0.6)
    assert m["recall"] == pytest.approx(0.9)
    assert m["negative_recall"] == pytest.approx(0.4)
    # positive_rate = n_positive / (n_positive + n_negative)
    assert m["positive_rate"] == pytest.approx(10 / 40)


def test_build_summary_positive_rate_zero_when_no_samples():
    out = dip.build_summary(_FakeEnv(0, 0), _state(f1=0.5), {}, _FakeModel(), "ts")
    assert out["metrics"]["positive_rate"] == 0.0


def test_build_summary_dataset_stamps_asset_timeframe_and_candle_count():
    env = _FakeEnv(predictions=[0, 1, 0, 1])
    out = dip.build_summary(env, _state(), {"asset": "SOLUSDT"}, _FakeModel(), "ts")
    assert out["dataset"]["asset"] == "SOLUSDT"
    assert out["dataset"]["timeframe"] == "1d"
    assert out["dataset"]["candles"] == 4


def test_build_summary_dataset_asset_defaults_to_btc():
    out = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel(), "ts")
    assert out["dataset"]["asset"] == "BTCUSDT"


def test_build_summary_candles_zero_for_missing_or_none_predictions():
    # env without a `predictions` attribute -> getattr default [] -> 0 candles.
    out = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel(), "ts")
    assert out["dataset"]["candles"] == 0
    # explicit None predictions also collapse to 0 (the `or []` guard).
    out2 = dip.build_summary(_FakeEnv(predictions=None), _state(), {}, _FakeModel(), "ts")
    assert out2["dataset"]["candles"] == 0


def test_build_summary_config_is_copied_verbatim():
    cfg = {"asset": "BTCUSDT", "loss_fn": "bce", "episodes": 3}
    out = dip.build_summary(_FakeEnv(), _state(), cfg, _FakeModel(), "ts")
    assert out["config"] == cfg


def test_build_summary_provenance_records_ran_at():
    out = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel(), "2026-06-19T12:00:00Z")
    assert out["provenance"]["ranAt"] == "2026-06-19T12:00:00Z"


def test_build_summary_health_block_reflects_state():
    degen = dip.build_summary(_FakeEnv(), _state(precision=0.0, recall=0.0), {}, _FakeModel(), "ts")
    assert degen["health"]["status"] == "degenerate"
    ok = dip.build_summary(_FakeEnv(), _state(precision=0.5, recall=0.5), {}, _FakeModel(), "ts")
    assert ok["health"]["status"] == "ok"


def test_build_summary_artifacts_only_when_model_has_id():
    with_ck = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel("chk-1"), "ts")
    assert with_ck["artifacts"] == {"checkpoint": "checkpoints/chk-1", "best": False}
    without = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel(None), "ts")
    assert "artifacts" not in without


def test_build_summary_artifacts_absent_when_model_lacks_id_attr():
    class _NoId:
        pass

    out = dip.build_summary(_FakeEnv(), _state(), {}, _NoId(), "ts")
    assert "artifacts" not in out


def test_build_summary_seed_stamped_into_summary_and_provenance():
    out = dip.build_summary(_FakeEnv(), _state(), {"seed": 7}, _FakeModel(), "ts")
    assert out["seed"] == 7
    assert out["provenance"]["seed"] == 7


def test_build_summary_no_seed_keys_when_seed_absent():
    out = dip.build_summary(_FakeEnv(), _state(), {}, _FakeModel(), "ts")
    assert "seed" not in out
    assert "seed" not in out["provenance"]


def test_build_summary_empty_state_collapses_all_metrics_to_zero():
    out = dip.build_summary(_FakeEnv(), [], {}, _FakeModel(), "ts")
    assert out["objective"] == 0.0
    assert all(v == 0.0 for v in out["metrics"].values())


def test_build_summary_nan_state_values_are_finite_zero():
    nan = float("nan")
    out = dip.build_summary(_FakeEnv(), [nan, nan, nan, nan, nan, nan], {}, _FakeModel(), "ts")
    assert out["objective"] == 0.0
    assert math.isfinite(out["metrics"]["accuracy"])
    assert out["metrics"]["accuracy"] == 0.0


# --- require_data_present (delegation) ---------------------------------------


def test_require_data_present_delegates_to_config_builder(monkeypatch):
    from trainer import config_builder

    seen = []
    monkeypatch.setattr(config_builder, "require_data_present", lambda cfg=None: seen.append(cfg))
    sentinel = {"asset": "BTCUSDT"}
    dip.require_data_present(sentinel)
    assert seen == [sentinel]


def test_require_data_present_defaults_cfg_to_none(monkeypatch):
    from trainer import config_builder

    seen = []
    monkeypatch.setattr(config_builder, "require_data_present", lambda cfg=None: seen.append(cfg))
    dip.require_data_present()
    assert seen == [None]
