"""Direct unit tests for the deterministic HodlModel baseline.

HodlModel just drives the env through a buy-and-hold trajectory: one BUY, hold for the rest, then
SELL on the last step (with TP/SL temporarily disabled so they cannot force an early exit). We
exercise it with a tiny recording fake env via ``__new__`` so no real data provider is needed.
"""

import types

from src.model.hodl_model import HodlModel


class _RecordingEnv:
    """Records the exact sequence of actions stepped, plus reset/TP-SL bookkeeping."""

    def __init__(self, timesteps, take_profit=0.1, stop_loss=0.1):
        self._timesteps = timesteps
        self.env_config = types.SimpleNamespace(take_profit=take_profit, stop_loss=stop_loss)
        self.reset_calls = 0
        self.steps = []
        # Snapshot of (take_profit, stop_loss) seen on each step, to prove they're disabled mid-run.
        self.tpsl_during_steps = []

    def reset(self):
        self.reset_calls += 1

    def get_timesteps(self):
        return self._timesteps

    def step(self, action):
        self.steps.append(action)
        self.tpsl_during_steps.append(
            (self.env_config.take_profit, self.env_config.stop_loss)
        )


def _model(model_type="hodl"):
    m = HodlModel.__new__(HodlModel)
    m.config = types.SimpleNamespace(model_type=model_type)
    return m


def test_get_id_uses_model_type():
    assert _model("hodl").get_id(config=None) == "hodl"


def test_get_id_reads_self_config_not_arg():
    # get_id formats self.config.model_type, ignoring the passed-in config argument entirely.
    m = _model("hodl")
    other = types.SimpleNamespace(model_type="rl")
    assert m.get_id(config=other) == "hodl"


def test_show_train_render_true():
    assert _model().show_train_render() is True


def test_is_pretrained_true():
    assert _model().is_pretrained() is True


def test_get_action_returns_none():
    assert _model().get_action(env=None, obs=None) is None


def test_train_action_sequence_buy_hold_sell():
    # timesteps=5 -> BUY(1), then 5-2=3 HOLDs(0), then SELL(2).
    env = _RecordingEnv(timesteps=5)
    _model().train(env)
    assert env.steps == [1, 0, 0, 0, 2]
    assert env.reset_calls == 1


def test_train_min_length_two_has_no_holds():
    # timesteps=2 -> BUY then SELL with zero holds (range(0) is empty).
    env = _RecordingEnv(timesteps=2)
    _model().train(env)
    assert env.steps == [1, 2]


def test_train_disables_take_profit_and_stop_loss_during_run():
    # Every recorded step must have seen TP/SL set to None so they can't force an early close.
    env = _RecordingEnv(timesteps=4, take_profit=0.1, stop_loss=0.05)
    _model().train(env)
    assert env.tpsl_during_steps == [(None, None)] * len(env.steps)


def test_train_restores_take_profit_and_stop_loss_after_run():
    env = _RecordingEnv(timesteps=4, take_profit=0.1, stop_loss=0.05)
    _model().train(env)
    assert env.env_config.take_profit == 0.1
    assert env.env_config.stop_loss == 0.05


def test_train_restores_none_tpsl_when_originally_none():
    env = _RecordingEnv(timesteps=3, take_profit=None, stop_loss=None)
    _model().train(env)
    assert env.env_config.take_profit is None
    assert env.env_config.stop_loss is None


def test_test_delegates_to_train_same_sequence():
    env = _RecordingEnv(timesteps=4)
    _model().test(env, deterministic=True)
    assert env.steps == [1, 0, 0, 2]
    assert env.reset_calls == 1


def test_train_with_three_timesteps_single_hold():
    env = _RecordingEnv(timesteps=3)
    _model().train(env)
    assert env.steps == [1, 0, 2]
