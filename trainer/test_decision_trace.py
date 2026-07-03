import json
import types

import numpy as np
import torch

from trainer import decision_trace as dt


# --- fakes -------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, lookback=0, layers=None):
        self._lookback = lookback
        self.config = types.SimpleNamespace(layers=list(layers) if layers is not None else [])

    def get_lookback_window(self):
        return self._lookback


class _BaseEnv:
    """Holds the post-test per-step arrays `build_base_steps` reads (no replay needed)."""

    def __init__(self, actions, forced=None, rewards=None, lookback=0):
        self.actions = list(actions)
        self.forced_actions = list(forced) if forced is not None else [0] * len(actions)
        self.rewards_history = list(rewards) if rewards is not None else [0.0] * len(actions)
        self.actions_made = [True] * len(actions)
        self.data_provider = _FakeProvider(lookback)


class _QPolicy:
    def __init__(self, q_net):
        self.q_net = q_net

    def obs_to_tensor(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1), False


class _DistPolicy:
    def __init__(self, weight):
        self._w = torch.as_tensor(weight, dtype=torch.float32)

    def obs_to_tensor(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1), False

    def get_distribution(self, obs_tensor):
        logits = obs_tensor @ self._w
        probs = torch.softmax(logits, dim=1)
        inner = types.SimpleNamespace(probs=probs.reshape(-1), logits=logits.reshape(-1))
        return types.SimpleNamespace(distribution=inner)


class _RL:
    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic=True, **kwargs):
        with torch.no_grad():
            q = self.policy.obs_to_tensor(obs)[0]
            if hasattr(self.policy, "q_net"):
                out = self.policy.q_net(q)
            else:
                out = self.policy.get_distribution(q).distribution.logits.reshape(1, -1)
        return np.array([int(torch.argmax(out.reshape(-1)))]), None


class _Model:
    def __init__(self, policy, model_name="dqn", checkpoint=True, model_id="fakeid"):
        self.rl_model = _RL(policy)
        self.rl_config = types.SimpleNamespace(model_name=model_name)
        self.id = model_id
        self._checkpoint = checkpoint

    def produces_checkpoint(self):
        return self._checkpoint


class _ReplayEnv:
    """A deterministic, steppable env that replays a fixed observation sequence and records
    actions_made / forced_actions (so attribution gating is exercised)."""

    def __init__(self, obs_seq, made=None, forced=None, lookback=0, layers=None, env_config=None):
        self._obs_seq = [np.asarray(o, dtype=np.float32) for o in obs_seq]
        self._made = list(made) if made is not None else [True] * len(obs_seq)
        self._forced = list(forced) if forced is not None else [0] * len(obs_seq)
        self.data_provider = _FakeProvider(lookback, layers)
        if env_config is not None:
            self.env_config = env_config
        self.actions = list(range(len(obs_seq)))
        self.forced_actions = list(self._forced)
        self.rewards_history = [0.0] * len(obs_seq)
        self.actions_made = list(self._made)

    def reset(self):
        self._i = 0
        self.actions_made = []
        self.forced_actions = []
        return self._obs_seq[0], {}

    def step(self, action):
        i = self._i
        self.actions_made.append(self._made[i])
        self.forced_actions.append(self._forced[i])
        self._i += 1
        done = self._i >= len(self._obs_seq)
        nxt = self._obs_seq[self._i] if not done else self._obs_seq[-1]
        return nxt, 0.0, done, False, {}


def _q_net(weight, bias):
    net = torch.nn.Linear(len(weight[0]), len(weight), bias=True)
    with torch.no_grad():
        net.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        net.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return net


# buy(1) fires on feature 0; sell(2) is suppressed — a 3-action (hold/buy/sell) value net.
_W = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
_B = [0.0, 0.0, -5.0]


# --- build_base_steps --------------------------------------------------------


def test_build_base_steps_labels_forced_and_reward():
    env = _BaseEnv(actions=[0, 1, 2], forced=[0, 0, 2], rewards=[0.0, 0.5, -0.3])
    steps = dt.build_base_steps(env, 0)
    assert steps[0] == {"step": 0, "action": "hold", "reward": 0.0}
    assert steps[1] == {"step": 1, "action": "buy", "reward": 0.5}
    assert steps[2] == {"step": 2, "action": "sell", "forced": True, "reward": -0.3}


def test_build_base_steps_respects_lookback_slice():
    env = _BaseEnv(actions=[9, 9, 1, 2], forced=[0, 0, 0, 0], rewards=[0, 0, 1, 2])
    steps = dt.build_base_steps(env, 2)
    assert [s["action"] for s in steps] == ["buy", "sell"]
    assert steps[0]["step"] == 0


def test_build_base_steps_empty_returns_empty():
    assert dt.build_base_steps(_BaseEnv(actions=[]), 0) == []


def test_label_out_of_range():
    env = _BaseEnv(actions=[7])
    assert dt.build_base_steps(env, 0)[0]["action"] == "action_7"


# --- confidence / runner_up --------------------------------------------------


def test_confidence_of_prob_is_direct():
    assert dt.confidence_of([0.1, 0.7, 0.2], 1, "prob") == 0.7


def test_confidence_of_q_is_softmax():
    c = dt.confidence_of([1.0, 3.0, 1.0], 1, "q")
    assert 0.5 < c < 1.0


def test_confidence_of_out_of_range_is_zero():
    assert dt.confidence_of([0.5, 0.5], 9, "prob") == 0.0


def test_runner_up_picks_second_best():
    assert dt.runner_up([0.1, 0.6, 0.3], 1) == 2


def test_runner_up_single_action_is_none():
    assert dt.runner_up([1.0], 0) is None


# --- observation_layout (pure) ----------------------------------------------


def test_observation_layout_single_layer_no_extras():
    layout = dt.observation_layout(obs_dim=4, lookback=1, layer_names=["1d"], active_extras=[])
    assert layout == {
        "lookback": 1,
        "per_bar": 4,
        "layers": [{"name": "1d", "start": 0, "width": 4}],
        "extras": [],
    }


def test_observation_layout_multi_layer_with_extras():
    # lookback 2 × per_bar 5 = 10; per bar = [1h(2), 1d(2), drawdown(1)]
    layout = dt.observation_layout(
        obs_dim=10, lookback=2, layer_names=["1h", "1d"], active_extras=["drawdown"]
    )
    assert layout["per_bar"] == 5
    assert layout["layers"] == [
        {"name": "1h", "start": 0, "width": 2},
        {"name": "1d", "start": 2, "width": 2},
    ]
    assert layout["extras"] == [{"name": "drawdown", "start": 4, "width": 1}]


def test_observation_layout_unnamed_layer_fallback():
    layout = dt.observation_layout(obs_dim=6, lookback=1, layer_names=[], active_extras=[])
    assert layout["layers"] == [{"name": "layer0", "start": 0, "width": 6}]


def test_observation_layout_not_divisible_by_lookback_is_none():
    assert dt.observation_layout(obs_dim=9, lookback=2, layer_names=["1d"], active_extras=[]) is None


def test_observation_layout_layers_dont_divide_is_none():
    # 7 layer columns can't split evenly across 2 layers → bail rather than mislabel.
    assert (
        dt.observation_layout(obs_dim=8, lookback=1, layer_names=["1h", "1d"], active_extras=["drawdown"])
        is None
    )


def test_observation_layout_zero_lookback_is_none():
    assert dt.observation_layout(obs_dim=4, lookback=0, layer_names=["1d"], active_extras=[]) is None


# --- _active_observation_extras ----------------------------------------------


def test_active_observation_extras_order_and_gating():
    cfg = types.SimpleNamespace(
        observations_contain=["in_position", "drawdown", "networth_percent_this_trade"],
        take_profit=0.02,
        stop_loss=None,
    )
    env = types.SimpleNamespace(env_config=cfg)
    # emission order: networth, drawdown, take_profit, stop_loss(off), in_position
    assert dt._active_observation_extras(env) == [
        "networth_percent_this_trade",
        "drawdown",
        "take_profit",
        "in_position",
    ]


def test_active_observation_extras_no_config():
    assert dt._active_observation_extras(types.SimpleNamespace()) == []


# --- _group_attribution ------------------------------------------------------


def test_group_attribution_by_layer_and_engineered():
    env = _ReplayEnv(
        [[0, 0, 0, 0, 0]],
        lookback=1,
        layers=["1h", "1d"],
        env_config=types.SimpleNamespace(
            observations_contain=["drawdown"], take_profit=None, stop_loss=None
        ),
    )
    # per_bar 5 = [1h(2), 1d(2), drawdown(1)]; saliency sums |.| per group over the single bar.
    groups = dt._group_attribution([1.0, 1.0, 3.0, 0.0, 5.0], env)
    assert groups == {"layer:1h": 2.0, "layer:1d": 3.0, "engineered:drawdown": 5.0}


def test_group_attribution_sums_over_lookback_bars():
    env = _ReplayEnv([[0, 0]], lookback=2, layers=["1d"])
    # [lookback=2, per_bar=2]; bar0=[1,2], bar1=[3,4] → layer:1d = |1|+|2|+|3|+|4| = 10
    groups = dt._group_attribution([1.0, 2.0, 3.0, 4.0], env)
    assert groups == {"layer:1d": 10.0}


def test_group_attribution_unreconcilable_is_none():
    env = _ReplayEnv([[0, 0]], lookback=2, layers=["1d"])
    assert dt._group_attribution([1.0, 2.0, 3.0], env) is None


# --- _policy_action_values ---------------------------------------------------


def test_policy_action_values_q_net_with_saliency():
    policy = _QPolicy(_q_net(_W, _B))
    values, kind, saliency = dt._policy_action_values(policy_holder(policy), [1.0, 0, 0, 0], 1, True)
    assert kind == "q"
    assert len(values) == 3
    assert saliency is not None and len(saliency) == 4
    # feature 0 drives the buy Q-value, so it carries all the saliency.
    assert saliency[0] > 0 and float(np.sum(saliency[1:])) == 0.0


def test_policy_action_values_distribution_path():
    policy = _DistPolicy([[0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    values, kind, saliency = dt._policy_action_values(policy_holder(policy), [2.0, 0, 0, 0], 1, True)
    assert kind == "prob"
    assert abs(float(np.sum(values)) - 1.0) < 1e-5
    assert saliency is not None


def test_policy_action_values_no_policy_is_none():
    assert dt._policy_action_values(types.SimpleNamespace(policy=None), [0], 0, False) == (
        None,
        None,
        None,
    )


def test_policy_action_values_swallows_errors():
    class _Boom:
        policy = types.SimpleNamespace(obs_to_tensor=lambda self, o: (_ for _ in ()).throw(ValueError()))

    assert dt._policy_action_values(_Boom(), [0], 0, False) == (None, None, None)


def policy_holder(policy):
    return types.SimpleNamespace(policy=policy)


# --- replay_enrichment -------------------------------------------------------


def test_replay_enrichment_non_rl_model_is_none():
    assert dt.replay_enrichment(_ReplayEnv([[0, 0]]), types.SimpleNamespace()) is None


def test_replay_enrichment_captures_confidence_and_alternative():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]])
    model = _Model(_QPolicy(_q_net(_W, _B)))
    enrichment, attribution = dt.replay_enrichment(env, model)
    assert len(enrichment) == 2
    assert set(enrichment[0]["actionValues"]) == {"hold", "buy", "sell"}
    assert 0.0 <= enrichment[0]["confidence"] <= 1.0
    assert enrichment[0]["alternativeAction"] != "buy"
    assert attribution["method"] == "gradient-saliency"
    assert attribution["samples"] == 1  # only the executed buy step is attributed
    assert len(attribution["perFeature"]) == 4


def test_replay_enrichment_groups_attribution_by_layer():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))))
    # the buy step's saliency is [1,0,0,0] (feature 0 drives buy); the single "1d" layer sums it.
    assert attribution["byGroup"] == {"layer:1d": 1.0}


def test_replay_enrichment_attaches_per_step_group_saliency():
    # buy(1) fires step 0 (attributed); hold(0) step 1 is not attributed → carries no per-step saliency.
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    enrichment, _ = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))))
    assert enrichment[0]["saliencyByGroup"] == {"layer:1d": 1.0}
    assert "saliencyByGroup" not in enrichment[1]


def test_replay_enrichment_omits_per_step_group_saliency_when_layout_unreconcilable():
    # obs_dim 4 with lookback 3 can't reconcile → the run-level perFeature still ships, but no per-step group.
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=3, layers=["1d"])
    enrichment, attribution = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))))
    assert "saliencyByGroup" not in enrichment[0]
    assert len(attribution["perFeature"]) == 4


def test_replay_enrichment_skips_unexecuted_for_attribution():
    # both steps choose buy, but the second was not executed (a no-op) → not attributed.
    env = _ReplayEnv([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], made=[True, False])
    _, attribution = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))))
    assert attribution["samples"] == 1


def test_replay_enrichment_without_attribution_has_no_saliency():
    env = _ReplayEnv([[1.0, 0, 0, 0]])
    _, attribution = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))), want_attribution=False)
    assert attribution is None


class _RecordingRL:
    """Records the kwargs each predict() call receives — to prove the recurrent branch threads lstm
    state. Returns hold (0) so the attribution path (which needs a real policy) is skipped."""

    def __init__(self):
        self.calls = []

    def predict(self, obs, deterministic=True, **kwargs):
        self.calls.append(kwargs)
        return np.array([0]), kwargs.get("state")


def test_replay_enrichment_threads_lstm_state_for_reppo_custom():
    rl = _RecordingRL()
    model = types.SimpleNamespace(rl_model=rl, rl_config=types.SimpleNamespace(model_name="reppo-custom"))
    dt.replay_enrichment(_ReplayEnv([[0, 0], [0, 0], [0, 0]]), model, want_attribution=False)
    assert len(rl.calls) == 3
    assert all("state" in c and "episode_start" in c for c in rl.calls)


def test_replay_enrichment_omits_lstm_state_for_non_recurrent():
    rl = _RecordingRL()
    model = types.SimpleNamespace(rl_model=rl, rl_config=types.SimpleNamespace(model_name="dqn"))
    dt.replay_enrichment(_ReplayEnv([[0, 0], [0, 0]]), model, want_attribution=False)
    assert all("state" not in c for c in rl.calls)


# --- Adebayo model-randomization sanity check --------------------------------


class _TorchPolicy(torch.nn.Module):
    """A policy whose weights CAN be randomized (a real nn.Module) — to exercise the sanity check."""

    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def obs_to_tensor(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1), False


def test_rank_correlation_identical_reversed_and_degenerate():
    assert abs(dt._rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(dt._rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert dt._rank_correlation([5, 5, 5], [1, 2, 3]) == 0.0  # constant ⇒ undefined ⇒ 0
    assert dt._rank_correlation([1, 2, 3], [1, 2]) == 0.0  # length mismatch


def test_randomized_saliency_is_none_when_policy_cannot_randomize():
    # _QPolicy is a plain object (no .parameters()) → best-effort returns None, no sanity check.
    model = _Model(_QPolicy(_q_net(_W, _B)))
    assert dt._randomized_saliency(model.rl_model, [np.array([1.0, 0, 0, 0])]) is None


def test_sanity_check_emitted_for_a_randomizable_policy():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(env, _Model(_TorchPolicy(_q_net(_W, _B))))
    sanity = attribution["sanityCheck"]
    assert sanity["method"] == "model-randomization"
    assert -1.0 <= sanity["rankCorrelation"] <= 1.0
    assert isinstance(sanity["passed"], bool)


def test_sanity_check_absent_when_policy_cannot_randomize():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(env, _Model(_QPolicy(_q_net(_W, _B))))
    assert "sanityCheck" not in attribution


# --- Integrated Gradients ----------------------------------------------------


def test_integrated_gradients_linear_net_equals_input_times_gradient():
    # For a linear value net IG = obs ⊙ grad; buy(1) row = [1,0,0,0] and obs feature0=1 → ig=[1,0,0,0].
    ig = dt._integrated_gradients(_Model(_QPolicy(_q_net(_W, _B))).rl_model, np.array([1.0, 0, 0, 0]), 1)
    assert ig is not None
    assert abs(ig[0] - 1.0) < 1e-5
    assert all(abs(v) < 1e-5 for v in ig[1:])


def test_integrated_gradients_none_without_a_value_net():
    policy = types.SimpleNamespace(obs_to_tensor=lambda o: (torch.zeros(1, 4), None))
    assert dt._integrated_gradients(_Model(policy).rl_model, np.array([1.0, 0, 0, 0]), 0) is None


def test_replay_enrichment_uses_integrated_gradients_when_requested():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(
        env, _Model(_TorchPolicy(_q_net(_W, _B))), method="integrated-gradients"
    )
    assert attribution["method"] == "integrated-gradients"
    assert attribution["byGroup"] == {"layer:1d": 1.0}


# --- Occlusion importance ----------------------------------------------------


def test_occlusion_importance_flags_the_decisive_feature():
    # buy(1) value = feature 0; occluding feature 0 (1→0) drops its Q by 1, the rest are no-ops.
    imp = dt._occlusion_importance(_Model(_QPolicy(_q_net(_W, _B))).rl_model, np.array([1.0, 2.0, 3.0, 4.0]), 1)
    assert imp is not None
    assert abs(imp[0] - 1.0) < 1e-5
    assert all(abs(v) < 1e-5 for v in imp[1:])


def test_replay_enrichment_uses_occlusion_when_requested():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(env, _Model(_TorchPolicy(_q_net(_W, _B))), method="occlusion")
    assert attribution["method"] == "occlusion"
    assert attribution["byGroup"] == {"layer:1d": 1.0}


# --- Tabular SHAP (permutation-sampled Shapley values) -----------------------


def test_shap_values_linear_net_equals_input_contribution():
    # For a linear value net SHAP φ_j = w_j·obs_j (baseline 0); buy(1) row = [1,0,0,0], obs feature0=1 → [1,0,0,0].
    shap = dt._shap_values(_Model(_QPolicy(_q_net(_W, _B))).rl_model, np.array([1.0, 2.0, 3.0, 4.0]), 1)
    assert shap is not None
    assert abs(shap[0] - 1.0) < 1e-6
    assert all(abs(v) < 1e-6 for v in shap[1:])


def test_shap_values_none_without_a_value_net():
    policy = types.SimpleNamespace(obs_to_tensor=lambda o: (torch.zeros(1, 4), None))
    assert dt._shap_values(_Model(policy).rl_model, np.array([1.0, 0, 0, 0]), 0) is None


def test_shap_values_deterministic_across_calls():
    # A nonlinear (softmax) policy so the permutation sampling actually matters; seeded ⇒ reproducible.
    model = _Model(_DistPolicy([[0.0, 1.0, 0.0], [0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.2]]))
    a = dt._shap_values(model.rl_model, np.array([1.0, 2.0, 3.0, 4.0]), 1)
    b = dt._shap_values(model.rl_model, np.array([1.0, 2.0, 3.0, 4.0]), 1)
    assert a is not None
    assert np.array_equal(a, b)


def test_replay_enrichment_uses_tabular_shap_when_requested():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 0, 0, 0]], lookback=1, layers=["1d"])
    _, attribution = dt.replay_enrichment(
        env, _Model(_TorchPolicy(_q_net(_W, _B))), method="tabular-shap"
    )
    assert attribution["method"] == "tabular-shap"
    assert attribution["byGroup"] == {"layer:1d": 1.0}


# --- reward breakdown --------------------------------------------------------


def test_reward_breakdown_aggregates_components_and_totals():
    env = types.SimpleNamespace(
        reward_components=[
            {"base": 1.0, "turnover_penalty": -0.1, "noop_penalty": 0.0},
            {"base": 2.0, "turnover_penalty": -0.2, "noop_penalty": -0.5},
        ]
    )
    bd = dt._reward_breakdown(env, 0)
    assert bd["base"] == 3.0
    assert abs(bd["turnover_penalty"] + 0.3) < 1e-9
    assert bd["noop_penalty"] == -0.5
    assert abs(bd["total"] - 2.2) < 1e-9  # total == summed reward


def test_reward_breakdown_drops_the_lookback_prefix():
    env = types.SimpleNamespace(reward_components=[{"base": 99.0}, {"base": 2.0}])
    assert dt._reward_breakdown(env, 1)["base"] == 2.0


def test_reward_breakdown_none_without_components():
    assert dt._reward_breakdown(types.SimpleNamespace(), 0) is None


# --- latent state map (Phase 5) ----------------------------------------------


def test_latent_map_projects_penultimate_activations_to_2d():
    env = _ReplayEnv(
        [[1.0, 0, 0, 0], [0.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]],
        lookback=1,
        layers=["1d"],
    )
    lm = dt._latent_map(env, _Model(_TorchPolicy(_q_net(_W, _B))))
    assert lm["method"] == "pca"
    assert lm["dim"] == 4
    assert 0.0 <= lm["varianceExplained"] <= 1.0
    assert len(lm["points"]) == 5
    assert set(lm["points"][0]) == {"x", "y", "action"}


def test_latent_map_none_without_a_value_or_action_net():
    policy = types.SimpleNamespace(obs_to_tensor=lambda o: (torch.zeros(1, 4), None))
    env = _ReplayEnv([[1.0, 0, 0, 0]] * 4, lookback=1, layers=["1d"])
    assert dt._latent_map(env, _Model(policy)) is None


def test_latent_map_none_with_too_few_steps():
    env = _ReplayEnv([[1.0, 0, 0, 0], [0.0, 1, 0, 0]], lookback=1, layers=["1d"])
    assert dt._latent_map(env, _Model(_TorchPolicy(_q_net(_W, _B)))) is None


# --- linear probe (Alain & Bengio) -------------------------------------------


def test_linear_probe_high_accuracy_when_the_latent_separates_actions():
    feats = [[i, 0.0] for i in range(-9, 0)] + [[i, 0.0] for i in range(1, 10)]
    labels = ["sell"] * 9 + ["buy"] * 9  # feature 0 sign perfectly separates them
    probe = dt._linear_probe(feats, labels)
    assert probe["classes"] == 2
    assert probe["method"] == "ridge-linear"
    assert probe["accuracy"] >= 0.9
    assert probe["accuracy"] >= probe["baseline"]
    assert probe["testSize"] == 6


def test_linear_probe_none_for_a_single_class():
    assert dt._linear_probe([[float(i)] for i in range(6)], ["hold"] * 6) is None


def test_linear_probe_none_for_too_few_samples():
    assert dt._linear_probe([[0.0], [1.0]], ["a", "b"]) is None


def test_replay_enrichment_collect_features():
    env = _ReplayEnv([[1.0, 2.0, 0, 0]])
    enrichment, _ = dt.replay_enrichment(
        env, _Model(_QPolicy(_q_net(_W, _B))), want_attribution=False, collect_features=True
    )
    assert enrichment[0]["features"] == [1.0, 2.0, 0.0, 0.0]


# --- compact / counts / downsample ------------------------------------------


def test_action_counts():
    assert dt._action_counts([{"action": "hold"}, {"action": "buy"}, {"action": "hold"}]) == {
        "hold": 2,
        "buy": 1,
    }


def test_downsample_indices_empty():
    assert dt._downsample_indices(0) == []


def test_downsample_indices_caps_and_aligns_with_summary():
    n = 1000
    idx = dt._downsample_indices(n)
    assert len(idx) <= dt._COMPACT_MAX_STEPS
    assert idx[0] == 0 and idx[-1] == n - 1


def test_compact_trace_drops_features_and_keeps_counts():
    steps = [{"step": i, "action": "hold", "features": [0.1]} for i in range(5)]
    trace = dt._compact_trace(steps, {"hold": 5}, {"perFeature": [0.2]})
    assert trace["totalSteps"] == 5
    assert trace["actionCounts"] == {"hold": 5}
    assert all("features" not in s for s in trace["steps"])
    assert trace["featureAttribution"] == {"perFeature": [0.2]}


def test_compact_trace_omits_attribution_when_absent():
    trace = dt._compact_trace([{"step": 0, "action": "hold"}], {"hold": 1}, None)
    assert "featureAttribution" not in trace


# --- _write_full_trace -------------------------------------------------------


def test_write_full_trace_next_to_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    rel = dt._write_full_trace([{"step": 0, "action": "hold"}], _Model(_QPolicy(_q_net(_W, _B))), None)
    assert rel == "checkpoints/fakeid.traces.jsonl"
    assert (tmp_path / rel).read_text().strip() == json.dumps({"step": 0, "action": "hold"})


def test_write_full_trace_summary_out_fallback(tmp_path):
    out = str(tmp_path / "s.json")
    model = _Model(_QPolicy(_q_net(_W, _B)), checkpoint=False)
    rel = dt._write_full_trace([{"step": 0, "action": "hold"}], model, out)
    assert rel == out + ".traces.jsonl"


def test_write_full_trace_no_target_is_none():
    model = _Model(_QPolicy(_q_net(_W, _B)), checkpoint=False)
    assert dt._write_full_trace([{"step": 0, "action": "hold"}], model, None) is None


# --- attach_decision_trace ---------------------------------------------------


class _FullEnv(_ReplayEnv):
    """Replay env that also carries the post-test arrays so a single object covers base + replay."""

    def __init__(self, obs_seq, actions, forced=None, rewards=None, made=None, lookback=0):
        super().__init__(obs_seq, made=made, forced=forced, lookback=lookback)
        self.actions = list(actions)
        self.forced_actions = list(forced) if forced is not None else [0] * len(actions)
        self.rewards_history = list(rewards) if rewards is not None else [0.0] * len(actions)


def test_attach_decision_trace_merges_base_and_enrichment():
    env = _FullEnv(obs_seq=[[1.0, 0, 0, 0], [0.0, 0, 0, 0]], actions=[1, 0], rewards=[0.1, -0.1])
    summary = {}
    dt.attach_decision_trace(summary, env, _Model(_QPolicy(_q_net(_W, _B))), {}, None, True)
    trace = summary["artifacts"]["decisionTrace"]
    assert trace["totalSteps"] == 2
    assert trace["actionCounts"] == {"buy": 1, "hold": 1}
    first = trace["steps"][0]
    assert first["action"] == "buy" and first["reward"] == 0.1
    assert "confidence" in first and "actionValues" in first


def test_attach_decision_trace_disabled_is_noop():
    env = _FullEnv(obs_seq=[[1.0, 0, 0, 0]], actions=[1])
    summary = {}
    dt.attach_decision_trace(summary, env, _Model(_QPolicy(_q_net(_W, _B))), {"emit_decision_trace": False}, None, True)
    assert summary == {}


def test_attach_decision_trace_empty_steps_is_noop():
    env = _FullEnv(obs_seq=[[0, 0, 0, 0]], actions=[])
    summary = {}
    dt.attach_decision_trace(summary, env, _Model(_QPolicy(_q_net(_W, _B))), {}, None, True)
    assert summary == {}


def test_attach_decision_trace_non_rl_is_base_only():
    env = _BaseEnv(actions=[1, 0], rewards=[0.1, 0.0])
    summary = {}
    dt.attach_decision_trace(summary, env, types.SimpleNamespace(), {}, None, False)
    steps = summary["artifacts"]["decisionTrace"]["steps"]
    assert all("confidence" not in s for s in steps)
    assert [s["action"] for s in steps] == ["buy", "hold"]


def test_attach_decision_trace_full_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    env = _FullEnv(obs_seq=[[1.0, 0, 0, 0], [0.0, 0, 0, 0]], actions=[1, 0])
    summary = {}
    dt.attach_decision_trace(
        summary, env, _Model(_QPolicy(_q_net(_W, _B))), {"decision_trace_full": True}, None, True
    )
    rel = summary["artifacts"]["decisionTraceFile"]
    assert rel == "checkpoints/fakeid.traces.jsonl"
    assert len((tmp_path / rel).read_text().strip().splitlines()) == 2
