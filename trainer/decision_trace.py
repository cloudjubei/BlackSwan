"""Decision-trace emission for a BlackSwan test run — the xAI "explain WHY it acted" layer.

After ``model.test()`` the env already holds the full per-step record of what the agent DID
(``actions`` / ``actions_made`` / ``forced_actions`` / ``rewards_history`` …). This module turns that
into a generic, domain-oblivious decision trace the Model Trainer's Explain view renders:

  * ``build_base_steps`` — per-step {action label, forced?, reward} reconstructed from the env arrays
    alone (works for ANY model, no replay), and
  * ``replay_enrichment`` — a second DETERMINISTIC pass over the test window that re-runs the policy to
    capture, per step, the model's CONFIDENCE and its per-action values (DQN Q-values, PPO/TRPO action
    probabilities) — so "why so few sells?" can be answered with sell-value-vs-hold-value over time —
    plus optional gradient saliency over the observation (which inputs drove the decision).

The compact trace is embedded at ``summary['artifacts']['decisionTrace']``; the full per-step trace
(with raw observations) is an opt-in sidecar (``decision_trace_full``). Everything is best-effort: a
model that exposes no introspectable policy (sbx/JAX, hodl) still gets the base action trace, and any
failure degrades to "no enrichment" rather than failing the run — a missing trace is not an error.
"""

import json
import os

import numpy as np

from src.model.rl_model import is_recurrent_model_name
from trainer import summary as summary_mod

# Action integer → generic label. The env encodes 0=hold, 1=buy(open long), 2=sell(close long),
# 3=short, 4=cover; the Model Trainer side treats these as arbitrary strings (no trading vocabulary).
_ACTION_LABELS = ["hold", "buy", "sell", "short", "cover"]
# The embedded trace is downsampled to this many steps so the Explain timeline shares the x-axis with
# the run's price/equity charts (which downsample to the same cap with the same stride).
_COMPACT_MAX_STEPS = summary_mod._MAX_SERIES_POINTS
# Gradient saliency is bounded to this many executed (non-forced) decisions so attribution stays a
# small constant cost (~one backprop each) regardless of how long the test window is.
_ATTRIBUTION_MAX_SAMPLES = 256
# The attention-heatmap aggregate (opt-in, `decision_trace_attention`) degrades to None when either axis of
# a captured attention matrix exceeds this — so the embedded grid stays small (≤ this² cells), bounding the
# inline summary size the way the saliency sample budget bounds attribution cost.
_MAX_ATTN_DIM = 64
# Riemann-sum steps for Integrated Gradients (opt-in, `decision_trace_ig`). More steps = a tighter
# completeness approximation at linear extra cost (one backprop each); 32 is the common default.
_IG_STEPS = 32
# Sampled feature orderings for permutation SHAP (opt-in, `decision_trace_method="tabular-shap"`). Each
# ordering costs one forward pass per feature; more orderings = a tighter Shapley estimate. Exact for a
# linear value net regardless of count; 16 is a modest default for this heavier, model-agnostic method.
_SHAP_PERMUTATIONS = 16
# Adebayo et al. "Sanity Checks for Saliency Maps": a FAITHFUL attribution should change when the model's
# weights are randomized. We recompute saliency on the SAME observations with a randomly re-initialised
# copy of the policy; if its rank-correlation with the real saliency stays AT/ABOVE this, the saliency is
# insensitive to the learned weights (it reflects the input/architecture, not what the model learned) and
# FAILS the check — so the viewer can warn instead of letting a plausible-but-unfaithful map mislead.
_SANITY_RANK_CORR_MAX = 0.5


def _label(action_int):
    return _ACTION_LABELS[action_int] if 0 <= action_int < len(_ACTION_LABELS) else f"action_{action_int}"


def build_base_steps(env, lookback):
    """Per-step decision records from the env's OWN retained arrays — no replay, any model type.

    Each step carries the POLICY's chosen action label, whether the environment FORCED an exit that
    step (an automatic TP/SL/trailing close, not the policy's choice), and the step reward. Sliced to
    the live window the same way ``summary.py`` slices its series, so the indices line up."""
    actions = [summary_mod._action_int(a) for a in summary_mod._live(getattr(env, "actions", []), lookback)]
    n = len(actions)
    if n == 0:
        return []
    forced = [summary_mod._action_int(f) for f in summary_mod._live(getattr(env, "forced_actions", []), lookback)]
    rewards = [summary_mod._finite(r) for r in summary_mod._live(getattr(env, "rewards_history", []), lookback)]
    steps = []
    for i in range(n):
        step = {"step": i, "action": _label(actions[i])}
        if i < len(forced) and forced[i] != 0:
            step["forced"] = True
        if i < len(rewards):
            step["reward"] = rewards[i]
        steps.append(step)
    return steps


def _softmax(values):
    v = np.asarray(values, dtype=float).reshape(-1)
    m = np.max(v)
    e = np.exp(v - m)
    s = float(e.sum())
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def confidence_of(values, action, kind):
    """The chosen action's confidence in [0,1]: the probability directly (policy nets) or a softmax over
    the Q-values (value nets), so confidence is comparable across model families."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if action < 0 or action >= len(v):
        return 0.0
    if kind == "prob":
        return float(v[action])
    return float(_softmax(v)[action])


def runner_up(values, action):
    """Index of the second-best action (what the model nearly did instead), or ``None`` if there is no
    distinct alternative — the basis for the per-step ``alternativeAction``."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if len(v) < 2:
        return None
    for j in np.argsort(v)[::-1]:
        if int(j) != action:
            return int(j)
    return None


def _values_map(values):
    return {_label(j): float(values[j]) for j in range(len(values))}


def _policy_action_values(rl_model, obs, action, want_grad):
    """Best-effort per-action values for ``obs`` (and, when ``want_grad``, |∂value/∂obs| saliency for the
    chosen ``action``). Returns ``(values, kind, saliency)`` where ``kind`` is ``'prob'`` (policy
    distribution) or ``'q'`` (value net); ``(None, None, None)`` if the model exposes neither (e.g. a
    JAX/sbx policy)."""
    import torch

    policy = getattr(rl_model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor"):
        return None, None, None
    try:
        obs_tensor, _ = policy.obs_to_tensor(obs)
        obs_tensor = obs_tensor.clone().detach().float().requires_grad_(bool(want_grad))
        values, kind, scalar = None, None, None
        if hasattr(policy, "get_distribution"):
            dist = policy.get_distribution(obs_tensor)
            inner = getattr(dist, "distribution", None)
            probs = getattr(inner, "probs", None)
            logits = getattr(inner, "logits", None)
            if probs is not None:
                values, kind = probs.reshape(-1), "prob"
                scalar = logits.reshape(-1)[action] if logits is not None else torch.log(values[action])
        if values is None:
            net = (
                getattr(policy, "q_net", None)
                or getattr(rl_model, "q_net", None)
                or getattr(policy, "quantile_net", None)
            )
            if net is None:
                return None, None, None
            q = net(obs_tensor)
            if q.dim() == 3:  # distributional heads (QR-DQN / IQN): average the quantiles per action
                q = q.mean(dim=1)
            values, kind, scalar = q.reshape(-1), "q", q.reshape(-1)[action]
        saliency = None
        if want_grad and scalar is not None and 0 <= action < values.shape[0]:
            scalar.backward()
            if obs_tensor.grad is not None:
                saliency = obs_tensor.grad.detach().abs().reshape(-1).cpu().numpy()
        return values.detach().cpu().numpy(), kind, saliency
    except Exception:
        return None, None, None


def _action_scalar(policy, rl_model, obs_tensor, action):
    """The differentiable scalar value for ``action`` (policy logit/log-prob, else value-net Q) at a
    grad-enabled ``obs_tensor`` — the integrand for Integrated Gradients. ``None`` if neither is exposed."""
    import torch

    if hasattr(policy, "get_distribution"):
        inner = getattr(policy.get_distribution(obs_tensor), "distribution", None)
        logits = getattr(inner, "logits", None)
        probs = getattr(inner, "probs", None)
        if logits is not None:
            return logits.reshape(-1)[action]
        if probs is not None:
            return torch.log(probs.reshape(-1)[action])
    net = (
        getattr(policy, "q_net", None)
        or getattr(rl_model, "q_net", None)
        or getattr(policy, "quantile_net", None)
    )
    if net is None:
        return None
    q = net(obs_tensor)
    if q.dim() == 3:  # distributional heads: average quantiles per action
        q = q.mean(dim=1)
    return q.reshape(-1)[action]


def _integrated_gradients(rl_model, obs, action, steps=_IG_STEPS):
    """Integrated Gradients (Sundararajan et al. 2017): integrate ∂value[action]/∂obs along a
    zero-baseline→obs path and scale by (obs − baseline). Axiomatic (sensitivity + implementation
    invariance) and robust to gradient saturation that one-shot saliency suffers. Returns per-feature
    |attribution|, or ``None`` when the policy exposes no differentiable value."""
    import torch

    policy = getattr(rl_model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor"):
        return None
    try:
        base, _ = policy.obs_to_tensor(obs)
        base = base.clone().detach().float()
        baseline = torch.zeros_like(base)
        total = torch.zeros_like(base)
        for k in range(1, steps + 1):
            x = (baseline + (k / steps) * (base - baseline)).clone().detach().requires_grad_(True)
            scalar = _action_scalar(policy, rl_model, x, action)
            if scalar is None:
                return None
            scalar.backward()
            if x.grad is None:
                return None
            total = total + x.grad.detach()
        ig = ((base - baseline) * (total / steps)).detach().abs().reshape(-1).cpu().numpy()
        return ig
    except Exception:
        return None


def _rank_correlation(a, b):
    """Spearman rank correlation of two equal-length vectors (no scipy); 0.0 when undefined/constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return 0.0
    if a.std() == 0 or b.std() == 0:  # a constant vector has no rank ordering — correlation is undefined
        return 0.0
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


class _PolicyShim:
    """Minimal stand-in so ``_policy_action_values`` can run against a randomized policy copy."""

    def __init__(self, policy):
        self.policy = policy
        self.q_net = getattr(policy, "q_net", None)


def _occlusion_importance(rl_model, obs, action, baseline=0.0):
    """Model-agnostic occlusion importance: set each feature to ``baseline`` in turn and measure
    |Δ value[action]|. No gradients — so it sidesteps the saturation + weight-randomization failure modes
    that bite saliency/IG, at the cost of one forward pass per feature. Returns per-feature importance."""
    values, _, _ = _policy_action_values(rl_model, obs, action, False)
    if values is None or action >= len(values):
        return None
    base_v = float(values[action])
    flat = np.asarray(obs, dtype=float).reshape(-1).copy()
    out = np.zeros(len(flat))
    for j in range(len(flat)):
        original = flat[j]
        if original == baseline:
            continue  # already at baseline ⇒ occluding it changes nothing
        flat[j] = baseline
        v, _, _ = _policy_action_values(rl_model, flat, action, False)
        flat[j] = original
        if v is not None and action < len(v):
            out[j] = abs(base_v - float(v[action]))
    return out


def _shap_values(rl_model, obs, action, baseline=0.0, permutations=_SHAP_PERMUTATIONS):
    """Permutation-sampled SHAP (Štrumbelj & Kononenko 2014): the mean marginal contribution each feature
    makes to value[action] as it is added, over random feature orderings, from a zero baseline — a
    model-agnostic Shapley attribution. No gradients (like occlusion), so it works for any introspectable
    policy; deterministic (seeded orderings). Exact for a linear value net and → exact Shapley as
    ``permutations`` → n!. Returns per-feature |attribution|, or ``None`` when the policy exposes no value."""
    values, _, _ = _policy_action_values(rl_model, obs, action, False)
    if values is None or action >= len(values):
        return None
    flat = np.asarray(obs, dtype=float).reshape(-1)
    n = len(flat)
    if n == 0:
        return None
    rng = np.random.default_rng(0)  # fixed so the attribution is reproducible across re-runs

    def value_at(mask):
        v, _, _ = _policy_action_values(rl_model, np.where(mask, flat, baseline), action, False)
        return None if v is None or action >= len(v) else float(v[action])

    base_v = value_at(np.zeros(n, dtype=bool))
    if base_v is None:
        return None
    phi = np.zeros(n)
    for _ in range(permutations):
        mask = np.zeros(n, dtype=bool)
        prev = base_v
        for j in rng.permutation(n):
            mask[j] = True
            cur = value_at(mask)
            if cur is None:
                return None
            phi[j] += cur - prev
            prev = cur
    return np.abs(phi / permutations)


def _attribution_of(rl_model, obs, action, method):
    """One per-feature |attribution| for ``obs``/``action`` by the chosen ``method``."""
    if method == "integrated-gradients":
        return _integrated_gradients(rl_model, obs, action)
    if method == "occlusion":
        return _occlusion_importance(rl_model, obs, action)
    if method == "tabular-shap":
        return _shap_values(rl_model, obs, action)
    return _policy_action_values(rl_model, obs, action, True)[2]


def _randomized_saliency(rl_model, obs_list, method="gradient-saliency"):
    """Aggregated attribution over ``obs_list`` for a WEIGHT-RANDOMIZED copy of the policy (each obs
    attributed to the randomized policy's own argmax, by the SAME ``method`` as the real map) — the
    control map for the Adebayo sanity check. Best-effort: ``None`` if the policy can't be copied."""
    import copy

    import torch

    policy = getattr(rl_model, "policy", None)
    if policy is None or not obs_list:
        return None
    try:
        rand = copy.deepcopy(policy)
        torch.manual_seed(0)  # fixed re-init so the sanity verdict is reproducible across re-runs
        with torch.no_grad():
            for p in rand.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)
        shim = _PolicyShim(rand)
        total = None
        count = 0
        for obs in obs_list:
            values, _, _ = _policy_action_values(shim, obs, 0, False)
            if values is None or len(values) == 0:
                continue
            attribution = _attribution_of(shim, obs, int(np.argmax(values)), method)
            if attribution is None:
                continue
            total = attribution if total is None else total + attribution
            count += 1
        return (total / count) if count else None
    except Exception:
        return None


def _sanity_check(rl_model, per_feature, attributed_obs, method="gradient-saliency"):
    """The Adebayo model-randomization sanity check for the real ``per_feature`` map: low rank correlation
    with the randomized-weights map ⇒ the attribution reflects the LEARNED function (passes). ``None`` when
    it can't be computed (so a missing check is never mistaken for a failed one)."""
    rand = _randomized_saliency(rl_model, attributed_obs, method)
    if rand is None or len(rand) != len(per_feature):
        return None
    corr = _rank_correlation(per_feature, rand)
    return {
        "method": "model-randomization",
        "rankCorrelation": corr,
        "passed": bool(abs(corr) < _SANITY_RANK_CORR_MAX),
    }


# Engineered per-bar extra columns base_crypto_env.get_next_observation appends — in THIS order —
# after the fidelity-layer columns within each bar. MUST stay in sync with that method's if-chain:
# attribution grouping is best-effort, but a stale count here shifts the layer/extra boundary and
# mislabels the last columns (the width check only catches counts that no longer divide the layers).
_OBSERVATION_EXTRA_ORDER = (
    "networth_percent_this_trade",
    "drawdown",
    "take_profit",
    "stop_loss",
    "in_position",
)


def _active_observation_extras(env):
    """Which engineered extras the env appended to each bar, in emission order — read from env_config
    the same way get_next_observation gates them (membership for the named flags, not-None for tp/sl)."""
    cfg = getattr(env, "env_config", None)
    if cfg is None:
        return []
    contains = set(getattr(cfg, "observations_contain", None) or [])
    active = []
    for name in _OBSERVATION_EXTRA_ORDER:
        if name == "take_profit":
            present = getattr(cfg, "take_profit", None) is not None
        elif name == "stop_loss":
            present = getattr(cfg, "stop_loss", None) is not None
        else:
            present = name in contains
        if present:
            active.append(name)
    return active


def observation_layout(obs_dim, lookback, layer_names, active_extras):
    """Map a flat observation into named column groups. The observation is a TIME-MAJOR
    ``[lookback, per_bar]`` grid; within each bar the fidelity layers' columns come first (in
    ``layer_names`` order, equal width) then one column per active engineered extra. Returns
    ``{lookback, per_bar, layers, extras}`` or ``None`` when the counts don't reconcile, so attribution
    degrades to per-feature-only rather than mislabel. Pure."""
    lookback = int(lookback)
    if lookback < 1 or obs_dim < 1 or obs_dim % lookback != 0:
        return None
    per_bar = obs_dim // lookback
    n_layers = max(1, len(layer_names))
    layer_portion = per_bar - len(active_extras)
    if layer_portion < n_layers or layer_portion % n_layers != 0:
        return None
    per_bar_layer = layer_portion // n_layers
    layers = [
        {"name": layer_names[i] if i < len(layer_names) else f"layer{i}",
         "start": i * per_bar_layer, "width": per_bar_layer}
        for i in range(n_layers)
    ]
    extras = [
        {"name": name, "start": n_layers * per_bar_layer + k, "width": 1}
        for k, name in enumerate(active_extras)
    ]
    return {"lookback": lookback, "per_bar": per_bar, "layers": layers, "extras": extras}


def _env_observation_layout(env, obs_dim):
    provider = getattr(env, "data_provider", None)
    if provider is None:
        return None
    try:
        lookback = int(provider.get_lookback_window())
    except Exception:
        return None
    layer_names = list(getattr(getattr(provider, "config", None), "layers", None) or [])
    return observation_layout(obs_dim, lookback, layer_names, _active_observation_extras(env))


def _group_attribution(per_feature, env):
    """Aggregate per-observation-feature saliency into named groups — by fidelity LAYER (`layer:1h`,
    `layer:1d`) and by engineered extra (`engineered:drawdown`) — summing |saliency| over the lookback
    window so the Explain view shows which INPUT GROUP drove the decisions. ``None`` when the layout
    can't be reconciled (the per-feature vector still ships)."""
    layout = _env_observation_layout(env, len(per_feature))
    if not layout:
        return None
    grid = np.abs(np.asarray(per_feature, dtype=float).reshape(layout["lookback"], layout["per_bar"]))
    groups = {}
    for layer in layout["layers"]:
        groups[f"layer:{layer['name']}"] = float(grid[:, layer["start"] : layer["start"] + layer["width"]].sum())
    for extra in layout["extras"]:
        groups[f"engineered:{extra['name']}"] = float(grid[:, extra["start"] : extra["start"] + extra["width"]].sum())
    return groups or None


def replay_enrichment(
    env, model, want_attribution=True, collect_features=False, method="gradient-saliency"
):
    """Re-run the deterministic test rollout, capturing per-step confidence / per-action values (and
    optional saliency + raw observations). Mirrors ``RLModel.test`` so it reproduces the SAME actions.
    Returns ``(per_step_enrichment, attribution)`` or ``None`` when the model is not an introspectable RL
    model. Resets the env, so call it AFTER the summary has been built from the post-test arrays."""
    rl_model = getattr(model, "rl_model", None)
    if rl_model is None or not hasattr(rl_model, "predict"):
        return None

    recurrent = is_recurrent_model_name(getattr(getattr(model, "rl_config", None), "model_name", None))
    obs, _ = env.reset()
    enrichment = []
    saliency_sum = None
    saliency_count = 0
    attributed_obs = []
    budget = _ATTRIBUTION_MAX_SAMPLES if want_attribution else 0
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while True:
        if recurrent:
            action, lstm_states = rl_model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
        else:
            action, _ = rl_model.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).reshape(-1)[0])

        want_grad = budget > 0 and action_int != 0
        if want_grad and method in ("integrated-gradients", "occlusion", "tabular-shap"):
            values, kind, _ = _policy_action_values(rl_model, obs, action_int, False)
            saliency = _attribution_of(rl_model, obs, action_int, method)
        else:
            values, kind, saliency = _policy_action_values(rl_model, obs, action_int, want_grad)
        enrich = {}
        if values is not None and len(values) > 0:
            enrich["actionValues"] = _values_map(values)
            enrich["confidence"] = confidence_of(values, action_int, kind)
            alt = runner_up(values, action_int)
            if alt is not None:
                enrich["alternativeAction"] = _label(alt)
        if collect_features:
            enrich["features"] = [summary_mod._finite(x) for x in np.asarray(obs, dtype=float).reshape(-1)]
        enrichment.append(enrich)

        obs_prev = obs
        obs, _, done, _, _ = env.step(action)
        episode_starts = np.array([bool(done)])

        made = bool(env.actions_made[-1]) if getattr(env, "actions_made", None) else False
        forced = summary_mod._action_int(env.forced_actions[-1]) if getattr(env, "forced_actions", None) else 0
        if want_grad and saliency is not None and made and forced == 0:
            saliency_sum = saliency if saliency_sum is None else saliency_sum + saliency
            saliency_count += 1
            attributed_obs.append(obs_prev)
            budget -= 1
            step_groups = _group_attribution(saliency, env)
            if step_groups:
                enrich["saliencyByGroup"] = step_groups
        if done:
            break

    attribution = None
    if saliency_count > 0:
        per_feature = (saliency_sum / saliency_count).tolist()
        attribution = {
            "perFeature": per_feature,
            "method": method,
            "samples": saliency_count,
        }
        by_group = _group_attribution(per_feature, env)
        if by_group:
            attribution["byGroup"] = by_group
        sanity = _sanity_check(rl_model, per_feature, attributed_obs, method)
        if sanity:
            attribution["sanityCheck"] = sanity
    return enrichment, attribution


def _action_counts(steps):
    counts = {}
    for s in steps:
        counts[s["action"]] = counts.get(s["action"], 0) + 1
    return counts


def _downsample_indices(n, cap=_COMPACT_MAX_STEPS):
    if n <= 0:
        return []
    return summary_mod._downsample_indexed(list(range(n)), cap)[1]


def _reward_breakdown(env, lookback):
    """Aggregate the env's per-step reward components over the LIVE window into named contributions — the
    "why this reward" view: what the base earned vs what each penalty dragged off, plus the total (which
    equals the summed reward). ``None`` when the env records no components."""
    components = getattr(env, "reward_components", None)
    if not components:
        return None
    live = summary_mod._live(components, lookback)
    totals = {}
    for c in live:
        if not isinstance(c, dict):
            continue
        for k, v in c.items():
            totals[k] = totals.get(k, 0.0) + summary_mod._finite(v)
    if not totals:
        return None
    totals["total"] = sum(totals.values())
    return {k: summary_mod._finite(v) for k, v in totals.items()}


def _linear_probe(features, labels):
    """Linear probe (Alain & Bengio): how well a LINEAR classifier on the latent predicts the action —
    high held-out accuracy vs the majority-class baseline ⇒ the representation linearly encodes the
    decision. Ridge one-vs-rest (closed-form + deterministic), a 1-in-3 held-out split. ``None`` when
    there's only one class or too few samples."""
    classes = sorted(set(labels))
    n = len(labels)
    if len(classes) < 2 or n < 6:
        return None
    test_idx = [i for i in range(n) if i % 3 == 0]
    train_idx = [i for i in range(n) if i % 3 != 0]
    if len(train_idx) < len(classes) or not test_idx:
        return None
    cls = {c: j for j, c in enumerate(classes)}
    feats = np.asarray(features, dtype=float)
    biased = np.hstack([feats, np.ones((n, 1))])
    onehot = np.zeros((n, len(classes)))
    for i, lab in enumerate(labels):
        onehot[i, cls[lab]] = 1.0
    x_train = biased[train_idx]
    try:
        weights = np.linalg.solve(
            x_train.T @ x_train + np.eye(biased.shape[1]), x_train.T @ onehot[train_idx]
        )
    except Exception:
        return None
    predicted = (biased[test_idx] @ weights).argmax(axis=1)
    truth = np.array([cls[labels[i]] for i in test_idx])
    counts = np.bincount(truth, minlength=len(classes))
    return {
        "accuracy": float((predicted == truth).mean()),
        "baseline": float(counts.max() / len(truth)),
        "classes": len(classes),
        "method": "ridge-linear",
        "testSize": len(test_idx),
    }


def _latent_map(env, model):
    """Project the policy's PENULTIMATE-layer activations over the test rollout to 2D (deterministic PCA),
    so the viewer can show how the model ORGANISES states internally — clusters by decision reveal learned
    structure. A forward pre-hook on the final value/action Linear captures its input (the penultimate
    representation). Best-effort: ``None`` when the layer can't be located or there are too few steps."""
    import torch

    rl_model = getattr(model, "rl_model", None)
    policy = getattr(rl_model, "policy", None) if rl_model is not None else None
    if policy is None or not hasattr(rl_model, "predict"):
        return None
    net = getattr(policy, "q_net", None) or getattr(policy, "action_net", None)
    if net is None:
        return None
    last_linear = None
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    if last_linear is None:
        return None

    captured = {}

    def hook(_module, inputs):
        captured["act"] = inputs[0].detach()

    handle = last_linear.register_forward_pre_hook(hook)
    rows, actions = [], []
    try:
        recurrent = is_recurrent_model_name(getattr(getattr(model, "rl_config", None), "model_name", None))
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        while True:
            captured.clear()
            if recurrent:
                action, lstm_states = rl_model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _ = rl_model.predict(obs, deterministic=True)
            if "act" in captured:
                rows.append(captured["act"].reshape(-1).cpu().numpy())
                actions.append(_label(int(np.asarray(action).reshape(-1)[0])))
            obs, _, done, _, _ = env.step(action)
            episode_starts = np.array([bool(done)])
            if done:
                break
    except Exception:
        return None
    finally:
        handle.remove()

    if len(rows) < 3:
        return None
    idx = _downsample_indices(len(rows))
    try:
        X = np.array([rows[i] for i in idx], dtype=float)
        acts = [actions[i] for i in idx]
        centered = X - X.mean(axis=0)
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ components[:2].T
    except Exception:
        return None
    total_var = float((singular**2).sum())
    variance_explained = float((singular[:2] ** 2).sum() / total_var) if total_var > 0 else 0.0
    points = [
        {"x": float(proj[i, 0]), "y": float(proj[i, 1]), "action": acts[i]} for i in range(len(acts))
    ]
    result = {
        "points": points,
        "varianceExplained": variance_explained,
        "dim": int(X.shape[1]),
        "method": "pca",
    }
    probe = _linear_probe(X, acts)
    if probe:
        result["probe"] = probe
    return result


def _compact_trace(steps, action_counts, attribution):
    """The embedded trace: steps downsampled to the chart axis (raw ``features`` dropped to stay light),
    full-rollout action counts, and any attribution."""
    kept = [dict(steps[i]) for i in _downsample_indices(len(steps))]
    for s in kept:
        s.pop("features", None)
    trace = {"steps": kept, "actionCounts": action_counts, "totalSteps": len(steps)}
    if attribution:
        trace["featureAttribution"] = attribution
    return trace


def _write_full_trace(steps, model, summary_out):
    """Write the full per-step trace (incl. raw observations) as JSONL next to the checkpoint, so it
    survives the run; returns the project-relative path or ``None``. Opt-in (``decision_trace_full``)."""
    checkpoint_id = getattr(model, "id", None)
    produces = getattr(model, "produces_checkpoint", lambda: False)
    if produces() and checkpoint_id:
        rel_path = f"checkpoints/{checkpoint_id}.traces.jsonl"
    elif summary_out:
        rel_path = f"{summary_out}.traces.jsonl"
    else:
        return None
    try:
        with open(rel_path, "w") as f:
            for s in steps:
                f.write(json.dumps(s) + "\n")
        return rel_path
    except Exception:
        return None


def write_snapshot_traces(summary, snapshots, trace_fn, sidecar_path):
    """Generate a decision trace for each mid-training snapshot (via the injected ``trace_fn(checkpointRef)``)
    and STREAM them to a JSONL sidecar — one ``{step, trace}`` per line, so only one trace is ever held in
    RAM at a time (A6 memory safety). Attaches a lightweight index on
    ``summary['artifacts']['snapshotTraces']``: ``[{step, checkpointRef, traceFile, keyMetrics}]``.
    Best-effort per snapshot (a failed load is skipped, never fatal); no-op (no key) for empty snapshots."""
    if not snapshots:
        return
    # If the sidecar can't be opened, emit nothing — an index whose entries point at a file that was never
    # written is worse than no index, and there's no point generating (expensive) traces we can't persist.
    try:
        handle = open(sidecar_path, "w")
    except Exception:
        return
    index = []
    try:
        for snap in snapshots:
            step = snap.get("step")
            ref = os.path.basename(str(snap.get("path", "")))
            try:
                trace = trace_fn(ref)
            except Exception:
                trace = None
            if not trace:
                continue
            handle.write(json.dumps({"step": step, "trace": trace}) + "\n")
            index.append(
                {
                    "step": step,
                    "checkpointRef": ref,
                    "traceFile": sidecar_path,
                    "keyMetrics": {
                        "actionCounts": trace.get("actionCounts", {}),
                        "totalSteps": trace.get("totalSteps"),
                    },
                }
            )
    finally:
        handle.close()
    if index:
        summary.setdefault("artifacts", {})["snapshotTraces"] = index
    else:
        try:
            os.remove(sidecar_path)  # every snapshot failed — never leave an empty, unreferenced sidecar
        except OSError:
            pass


def _collect_attention_matrices(model):
    """Every policy submodule that stashed a ``last_attn`` this forward, as ``(label, tensor)`` — the
    attention weight matrices the custom blocks / sequence extractor compute (see src/model/custom). A
    recipe may wire zero, one, or many attention blocks, so this enumerates ALL of them. Best-effort:
    ``[]`` on a non-introspectable model or any error."""
    try:
        rl_model = getattr(model, "rl_model", None)
        policy = getattr(rl_model, "policy", None) if rl_model is not None else None
        if policy is None or not hasattr(policy, "modules"):
            return []
        out = []
        for i, module in enumerate(policy.modules()):
            last_attn = getattr(module, "last_attn", None)
            if last_attn is not None:
                out.append((f"{type(module).__name__}[{i}]", last_attn))
        return out
    except Exception:
        return []


def _square_2d(tensor):
    """Reduce a captured attention tensor to a 2-D [rows, cols] grid: mean over a leading batch axis, then
    require a real matrix (both dims > 1). ``None`` for degenerate shapes (e.g. AdditiveAttention's
    [batch, seq, 1] or GlobalContextAttention's [batch, 1, 1]), which carry no usable heatmap."""
    try:
        arr = np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor, dtype=float)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return arr


def _attention_matrix(env, model):
    """Aggregate the policy's captured attention weights over the deterministic test rollout into ONE
    mean [rows, cols] matrix (domain-oblivious: query positions × key positions). Its own bounded replay
    (mirrors ``_latent_map``), best-effort. ``None`` when the model exposes no usable attention, or the
    matrix exceeds ``_MAX_ATTN_DIM`` (degrade rather than emit a huge inline grid)."""
    rl_model = getattr(model, "rl_model", None)
    if rl_model is None or not hasattr(rl_model, "predict"):
        return None
    recurrent = is_recurrent_model_name(getattr(getattr(model, "rl_config", None), "model_name", None))
    try:
        obs, _ = env.reset()
    except Exception:
        return None
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    acc = None
    count = 0
    budget = _ATTRIBUTION_MAX_SAMPLES
    try:
        while True:
            if recurrent:
                action, lstm_states = rl_model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _ = rl_model.predict(obs, deterministic=True)
            if budget > 0:
                grid = None
                for _, tensor in _collect_attention_matrices(model):
                    grid = _square_2d(tensor)
                    if grid is not None:
                        break
                if grid is not None:
                    if grid.shape[0] > _MAX_ATTN_DIM or grid.shape[1] > _MAX_ATTN_DIM:
                        return None  # degrade: too large to embed as an aggregate grid
                    if acc is None:
                        acc = grid
                        count = 1
                        budget -= 1
                    elif grid.shape == acc.shape:
                        acc = acc + grid
                        count += 1
                        budget -= 1
            obs, _, done, _, _ = env.step(action)
            episode_starts = np.array([bool(done)])
            if done or budget <= 0:
                break
    except Exception:
        return None
    if acc is None or count == 0:
        return None
    mean = acc / count
    rows, cols = mean.shape
    return {
        "rows": [f"q{i}" for i in range(rows)],
        "cols": [f"k{j}" for j in range(cols)],
        "grid": [[summary_mod._finite(mean[i, j]) for j in range(cols)] for i in range(rows)],
        "method": "attention-weights",
    }


def _attn_sidecar_path(model, summary_out):
    """Where the per-step attention JSONL lives — next to the checkpoint (survives the run) or beside the
    summary. Mirrors ``_write_full_trace``'s scheme. ``None`` when there's nowhere durable to write it."""
    checkpoint_id = getattr(model, "id", None)
    produces = getattr(model, "produces_checkpoint", lambda: False)
    if produces() and checkpoint_id:
        return f"checkpoints/{checkpoint_id}.attn.jsonl"
    if summary_out:
        return f"{summary_out}.attn.jsonl"
    return None


def write_per_step_attention(env, model, sidecar_path, cfg=None):
    """Stream ONE per-step attention matrix per line to a JSONL sidecar (never inline, never >1 matrix in
    RAM) — the temporal companion to the run-aggregate ``_attention_matrix``. Each line is
    ``{step, rows, cols, grid}`` for a step whose policy exposes a usable ``[<=cap, <=cap]`` matrix; a step
    whose grid exceeds ``_MAX_ATTN_DIM`` is skipped (degraded, not truncated), and the whole pass is bounded
    to ``_ATTRIBUTION_MAX_SAMPLES`` steps so the sidecar can't grow unbounded. Best-effort: returns the
    written path, or ``None`` (no attention / unwritable / nothing written — an empty sidecar is removed)."""
    rl_model = getattr(model, "rl_model", None)
    if rl_model is None or not hasattr(rl_model, "predict"):
        return None
    try:
        handle = open(sidecar_path, "w")
    except Exception:
        return None
    recurrent = is_recurrent_model_name(getattr(getattr(model, "rl_config", None), "model_name", None))
    written = 0
    try:
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        step_idx = 0
        budget = _ATTRIBUTION_MAX_SAMPLES
        while True:
            if recurrent:
                action, lstm_states = rl_model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _ = rl_model.predict(obs, deterministic=True)
            if budget > 0:
                grid = None
                for _, tensor in _collect_attention_matrices(model):
                    grid = _square_2d(tensor)
                    if grid is not None:
                        break
                if grid is not None and grid.shape[0] <= _MAX_ATTN_DIM and grid.shape[1] <= _MAX_ATTN_DIM:
                    rows, cols = grid.shape
                    handle.write(
                        json.dumps(
                            {
                                "step": step_idx,
                                "rows": [f"q{i}" for i in range(rows)],
                                "cols": [f"k{j}" for j in range(cols)],
                                "grid": [[summary_mod._finite(grid[i, j]) for j in range(cols)] for i in range(rows)],
                            }
                        )
                        + "\n"
                    )
                    written += 1
                    budget -= 1
            obs, _, done, _, _ = env.step(action)
            episode_starts = np.array([bool(done)])
            step_idx += 1
            if done or budget <= 0:
                break
    except Exception:
        pass  # a partial sidecar is fine — `written` decides whether we keep it
    finally:
        handle.close()
    if written == 0:
        try:
            os.remove(sidecar_path)  # never leave an empty sidecar / a bogus artifact reference
        except OSError:
            pass
        return None
    return sidecar_path


def attach_decision_trace(summary, env, model, cfg, summary_out, is_rl):
    """Build the decision trace and attach it to ``summary['artifacts']``. Best-effort and additive: a
    missing/empty trace leaves the summary untouched. Resets ``env`` when it replays, so the summary must
    already be built from the env's post-test arrays before this is called."""
    if not cfg.get("emit_decision_trace", True):
        return
    lookback = summary_mod._lookback(env, cfg)
    steps = build_base_steps(env, lookback)
    if not steps:
        return

    attribution = None
    want_full = bool(cfg.get("decision_trace_full", False))
    if is_rl:
        try:
            result = replay_enrichment(
                env,
                model,
                want_attribution=bool(cfg.get("decision_trace_attribution", True)),
                collect_features=want_full,
                method=str(cfg.get("decision_trace_method", "gradient-saliency")),
            )
        except Exception:
            result = None
        if result is not None:
            enrichment, attribution = result
            if len(enrichment) == len(steps):
                for step, enrich in zip(steps, enrichment):
                    step.update(enrich)

    action_counts = _action_counts(steps)
    trace_file = _write_full_trace(steps, model, summary_out) if want_full else None

    artifacts = summary.setdefault("artifacts", {})
    trace = _compact_trace(steps, action_counts, attribution)
    breakdown = _reward_breakdown(env, lookback)
    if breakdown:
        trace["rewardBreakdown"] = breakdown
    if is_rl and cfg.get("decision_trace_latent", False):
        try:
            latent_map = _latent_map(env, model)
        except Exception:
            latent_map = None
        if latent_map:
            trace["latentMap"] = latent_map
    if is_rl and cfg.get("decision_trace_attention", False):
        try:
            attention_matrix = _attention_matrix(env, model)
        except Exception:
            attention_matrix = None
        if attention_matrix:
            trace["attentionMatrix"] = attention_matrix
    # Per-step attention is a separate, opt-in SIDECAR (never inline — per-step grids over a rollout are
    # unbounded) referenced by `attentionMatrixFile`; distinct flag from the inline aggregate above.
    if is_rl and cfg.get("decision_trace_attention_per_step", False):
        try:
            attn_path = _attn_sidecar_path(model, summary_out)
            written = write_per_step_attention(env, model, attn_path, cfg) if attn_path else None
        except Exception:
            written = None
        if written:
            trace["attentionMatrixFile"] = written
    artifacts["decisionTrace"] = trace
    if trace_file:
        artifacts["decisionTraceFile"] = trace_file
