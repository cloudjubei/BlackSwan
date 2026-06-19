"""Direct tests for RegressionModel in regression_model.py.

RegressionModel wraps a torch model + optimizer + loss. We bypass __init__ with ``__new__`` and use
tiny torch modules / fake envs so the id-building, config plumbing, train backward loop, and the
sigmoid->threshold decision in test() are exercised quickly on CPU with no real data files.
"""

import os
from types import SimpleNamespace

import torch

from src.conf.model_config import ModelConfig, ModelRegressionConfig
from src.model.regression_model import RegressionModel


# --- fakes -----------------------------------------------------------------

class _FixedLogitsModel:
    """A callable that ignores its input and returns fixed logits; .view(-1) flattens them."""

    def __init__(self, logits):
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, X):
        return self._logits


class _DataloaderEnv:
    def __init__(self, batches):
        self._batches = batches
        self.stored = []
        self.total_reward = None

    def get_dataloader(self):
        return self._batches

    def store_result(self, outputs, y):
        self.stored.append((outputs, y))


def _reg_config(**kw):
    defaults = dict(model_name="mlp", net_arch=[8], custom_net_arch=[""])
    defaults.update(kw)
    return ModelConfig(model_type="regression", model_regression=ModelRegressionConfig(**defaults))


def _reg_model(config=None, model=None, optimizer=None, loss_fn=None):
    m = RegressionModel.__new__(RegressionModel)
    m.config = config if config is not None else _reg_config()
    m.model = model
    m.optimizer = optimizer
    m.loss_fn = loss_fn
    return m


# --- id / config plumbing --------------------------------------------------

def test_get_id_is_sanitised_and_carries_key_fields():
    cfg = _reg_config(learning_rate=0.0001, loss_fn="mse")
    m = _reg_model(cfg)
    the_id = m.get_id(cfg)
    assert the_id.startswith("regression_mlp_")
    assert "." not in the_id
    assert "|" not in the_id
    assert "0~0001" in the_id  # learning_rate sanitised
    assert "_mse_" in the_id


def test_get_id_abbreviates_custom_net_arch():
    cfg = _reg_config(net_arch=[8, 4], custom_net_arch=["Dropout"])
    m = _reg_model(cfg)
    the_id = m.get_id(cfg)
    assert "8]4" in the_id  # net_arch joined with sanitised pipe
    assert "Dropt" in the_id  # 'Dropout' -> 'Drop' + 't'


def test_get_id_empty_custom_net_arch_no_abbrev():
    cfg = _reg_config(net_arch=[8], custom_net_arch=[""])
    the_id = _reg_model(cfg).get_id(cfg)
    assert "Dropt" not in the_id


def test_get_episodes_reads_config():
    m = _reg_model(_reg_config(episodes=7))
    assert m.get_episodes() == 7


def test_is_pretrained_follows_checkpoint_to_load():
    assert _reg_model(_reg_config(checkpoint_to_load=None)).is_pretrained() is False
    assert _reg_model(_reg_config(checkpoint_to_load="some/ckpt")).is_pretrained() is True


# --- test(): sigmoid -> threshold decision ---------------------------------

def test_test_thresholds_sigmoid_at_default_half():
    # logits 3,-3,0 -> sigmoid ~0.95, ~0.047, 0.5. With threshold 0.5 (strict >), 0.5 is NOT selected.
    model = _FixedLogitsModel([3.0, -3.0, 0.0])
    env = _DataloaderEnv([(torch.zeros(3, 2), torch.tensor([1, 0, 1]))])
    m = _reg_model(_reg_config(decision_threshold=0.5), model=model)
    m.test(env)
    assert model.eval_called is True  # test() puts the model in eval mode
    outputs, _ = env.stored[0]
    assert outputs.tolist() == [1, 0, 0]
    assert outputs.dtype == torch.int32  # decisions are integer 0/1


def test_test_high_threshold_suppresses_all():
    model = _FixedLogitsModel([3.0, -3.0, 0.0])
    env = _DataloaderEnv([(torch.zeros(3, 2), torch.tensor([1, 0, 1]))])
    m = _reg_model(_reg_config(decision_threshold=0.99), model=model)
    m.test(env)
    assert env.stored[0][0].tolist() == [0, 0, 0]


def test_test_low_threshold_selects_all():
    model = _FixedLogitsModel([3.0, -3.0, 0.0])
    env = _DataloaderEnv([(torch.zeros(3, 2), torch.tensor([1, 0, 1]))])
    m = _reg_model(_reg_config(decision_threshold=0.0), model=model)
    m.test(env)
    # sigmoid is always strictly > 0, so threshold 0.0 selects every sample.
    assert env.stored[0][0].tolist() == [1, 1, 1]


def test_test_threshold_falls_back_to_half_when_attr_missing():
    # a config object lacking decision_threshold falls back to 0.5 via getattr.
    model = _FixedLogitsModel([3.0, -3.0, 0.0])
    env = _DataloaderEnv([(torch.zeros(3, 2), torch.tensor([1, 0, 1]))])
    m = _reg_model(SimpleNamespace(model_regression=SimpleNamespace()), model=model)
    m.test(env)
    assert env.stored[0][0].tolist() == [1, 0, 0]


def test_test_stores_result_per_batch():
    model = _FixedLogitsModel([0.0, 0.0])
    batches = [
        (torch.zeros(2, 2), torch.tensor([1, 0])),
        (torch.zeros(2, 2), torch.tensor([0, 1])),
    ]
    env = _DataloaderEnv(batches)
    m = _reg_model(_reg_config(), model=model)
    m.test(env)
    assert len(env.stored) == 2  # one store_result per dataloader batch


# --- train(): backward loop, env.total_reward, checkpoint save -------------

def test_train_runs_backward_loop_and_records_loss(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    cfg = _reg_config(episodes=2, checkpoints_folder=str(tmp_path))
    m = _reg_model(cfg, model=model, optimizer=optimizer, loss_fn=loss_fn)
    m.id = "reg-train-id"
    batches = [
        (torch.randn(3, 2), torch.randn(3)),
        (torch.randn(4, 2), torch.randn(4)),
    ]
    env = _DataloaderEnv(batches)
    m.train(env)
    # the final episode's accumulated loss is written onto the env as a python float.
    assert isinstance(env.total_reward, float)
    # train saves a checkpoint at checkpoints_folder/id.
    assert os.path.exists(os.path.join(str(tmp_path), "reg-train-id"))


def test_train_zero_episodes_does_not_touch_env_or_save(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    cfg = _reg_config(episodes=0, checkpoints_folder=str(tmp_path))
    m = _reg_model(cfg, model=model, optimizer=optimizer, loss_fn=loss_fn)
    m.id = "unused-id"
    env = _DataloaderEnv([(torch.randn(2, 2), torch.randn(2))])
    m.train(env)
    # with no episodes the loop body never runs: no reward recorded, no checkpoint written.
    assert env.total_reward is None
    assert not os.path.exists(os.path.join(str(tmp_path), "unused-id"))
