"""Unit tests for the extractable pure logic in src/main.py.

src/main.py is mostly a Hydra-driven sweep entrypoint (the `main` function drives
the full data→env→model→train→test pipeline and is not unit-testable without real
market data + a training loop). Two pieces ARE pure decision logic and are covered
here by faking the heavy collaborators:

  * get_model_configs_count — sums combination counts over the config's model searches.
  * run_model               — the pretrained-skip + deterministic-test gate + the exact
                              shape of the result row appended for the leaderboard.

The heavy bits (run_training / run_testing render+train, and main's nested sweep)
are deliberately not unit-tested — see the agent notes.
"""

import types

import src.main as main_mod


# --- get_model_configs_count -------------------------------------------------


def test_get_model_configs_count_sums_combinations(monkeypatch):
    # Each search expands to a different number of concrete configs; the count is their sum.
    searches = ["searchA", "searchB"]
    expansions = {"searchA": [1, 2, 3], "searchB": [1, 2]}
    monkeypatch.setattr(main_mod, "get_model_combinations", lambda s: expansions[s])
    config = types.SimpleNamespace(model_configs=searches)
    assert main_mod.get_model_configs_count(config) == 5


def test_get_model_configs_count_empty_is_zero(monkeypatch):
    monkeypatch.setattr(main_mod, "get_model_combinations", lambda s: [])
    config = types.SimpleNamespace(model_configs=[])
    assert main_mod.get_model_configs_count(config) == 0


def test_get_model_configs_count_single_search(monkeypatch):
    monkeypatch.setattr(main_mod, "get_model_combinations", lambda s: ["a", "b", "c", "d"])
    config = types.SimpleNamespace(model_configs=["only"])
    assert main_mod.get_model_configs_count(config) == 4


# --- run_model fakes ---------------------------------------------------------


class _FakeModel:
    def __init__(self, pretrained=False, deterministic=True, mid="m1", state=None):
        self._pretrained = pretrained
        self._deterministic = deterministic
        self.id = mid
        self._state = state if state is not None else [0.1, 0.2, 0.3]
        self.trained = False

    def is_pretrained(self):
        return self._pretrained

    def has_deterministic_test(self):
        return self._deterministic

    # collaborators touched by run_training / run_testing (both go through env.setup + model.x):
    def get_reward_model(self):
        return "reward"

    def get_reward_multipliers(self):
        return {}

    def train(self, env):
        self.trained = True

    def test(self, env, deterministic):
        self.tested = (env, deterministic)

    def show_train_render(self):
        return False


class _FakeEnv:
    def __init__(self, eid="e1", state=None):
        self.id = eid
        self._state = state if state is not None else [0.1, 0.2, 0.3]
        self.total_reward = 0.0
        self.did_setup = False

    def setup(self, reward_model, multipliers):
        self.did_setup = True

    def get_run_state(self):
        return list(self._state)

    def render(self):
        raise AssertionError("render must not be called when show_render is False")


def _fake_config(device="cpu", show_render=False):
    return types.SimpleNamespace(device=device, show_render=show_render, local_only=True)


# --- run_model ---------------------------------------------------------------


def test_run_model_appends_provider_env_model_id_then_state(monkeypatch):
    model = _FakeModel(mid="model-x", state=[9.0, 8.0])
    monkeypatch.setattr(main_mod, "create_model", lambda mc, env, device: model)
    provider = types.SimpleNamespace(id="prov-1")
    env_train = _FakeEnv(eid="train-env")
    env_test = _FakeEnv(eid="test-env", state=[9.0, 8.0])
    states = []
    main_mod.run_model(_fake_config(), object(), provider, env_train, env_test, states)
    assert len(states) == 1
    # row = [provider.id, env_test.id, model.id, *test_state]
    assert states[0] == ["prov-1", "test-env", "model-x", 9.0, 8.0]


def test_run_model_trains_when_not_pretrained(monkeypatch):
    model = _FakeModel(pretrained=False)
    monkeypatch.setattr(main_mod, "create_model", lambda mc, env, device: model)
    provider = types.SimpleNamespace(id="p")
    main_mod.run_model(_fake_config(), object(), provider, _FakeEnv(), _FakeEnv(), [])
    assert model.trained is True


def test_run_model_skips_training_when_pretrained(monkeypatch):
    model = _FakeModel(pretrained=True)
    monkeypatch.setattr(main_mod, "create_model", lambda mc, env, device: model)
    provider = types.SimpleNamespace(id="p")
    main_mod.run_model(_fake_config(), object(), provider, _FakeEnv(), _FakeEnv(), [])
    # a pretrained model is loaded, not retrained.
    assert model.trained is False


def test_run_model_without_deterministic_test_appends_nothing(monkeypatch):
    model = _FakeModel(deterministic=False)
    monkeypatch.setattr(main_mod, "create_model", lambda mc, env, device: model)
    provider = types.SimpleNamespace(id="p")
    states = []
    main_mod.run_model(_fake_config(), object(), provider, _FakeEnv(), _FakeEnv(), states)
    assert states == []


def test_run_model_pretrained_still_tests_and_records(monkeypatch):
    # pretrained + deterministic-test: no training, but it IS evaluated and recorded.
    model = _FakeModel(pretrained=True, deterministic=True, mid="pre", state=[1.0])
    monkeypatch.setattr(main_mod, "create_model", lambda mc, env, device: model)
    provider = types.SimpleNamespace(id="p")
    env_test = _FakeEnv(eid="te", state=[1.0])
    states = []
    main_mod.run_model(_fake_config(), object(), provider, _FakeEnv(), env_test, states)
    assert model.trained is False
    assert states == [["p", "te", "pre", 1.0]]


# --- run_testing / run_training (deterministic glue, no render) --------------


def test_run_testing_sets_up_env_and_returns_state(monkeypatch):
    model = _FakeModel(mid="m", state=[5.0, 6.0])
    env = _FakeEnv(state=[5.0, 6.0])
    state = main_mod.run_testing(_fake_config(), model, env, True)
    assert env.did_setup is True
    assert state == [5.0, 6.0]
    # deterministic flag is threaded into model.test.
    assert model.tested == (env, True)


def test_run_training_sets_up_env_trains_and_returns_state(monkeypatch):
    model = _FakeModel(state=[3.0])
    env = _FakeEnv(state=[3.0])
    state = main_mod.run_training(_fake_config(), model, env, types.SimpleNamespace(id="d"))
    assert env.did_setup is True
    assert model.trained is True
    assert state == [3.0]
