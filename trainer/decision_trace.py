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

import numpy as np

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


def _group_attribution(per_feature, lookback):
    """Aggregate per-feature saliency into named groups. When the observation is a clean
    ``[lookback, per_bar]`` grid, group by lookback BAR (newest→oldest) so the view can show whether the
    model weights recent bars; otherwise omit grouping (the per-feature vector still ships)."""
    n = len(per_feature)
    if lookback and lookback > 1 and n % lookback == 0:
        per_bar = n // lookback
        groups = {}
        for bar in range(lookback):
            seg = per_feature[bar * per_bar : (bar + 1) * per_bar]
            groups[f"bar[-{lookback - 1 - bar}]"] = float(sum(abs(x) for x in seg))
        return groups
    return None


def replay_enrichment(env, model, want_attribution=True, collect_features=False):
    """Re-run the deterministic test rollout, capturing per-step confidence / per-action values (and
    optional saliency + raw observations). Mirrors ``RLModel.test`` so it reproduces the SAME actions.
    Returns ``(per_step_enrichment, attribution)`` or ``None`` when the model is not an introspectable RL
    model. Resets the env, so call it AFTER the summary has been built from the post-test arrays."""
    rl_model = getattr(model, "rl_model", None)
    if rl_model is None or not hasattr(rl_model, "predict"):
        return None

    recurrent = getattr(getattr(model, "rl_config", None), "model_name", None) == "reppo"
    obs, _ = env.reset()
    enrichment = []
    saliency_sum = None
    saliency_count = 0
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

        obs, _, done, _, _ = env.step(action)
        episode_starts = np.array([bool(done)])

        made = bool(env.actions_made[-1]) if getattr(env, "actions_made", None) else False
        forced = summary_mod._action_int(env.forced_actions[-1]) if getattr(env, "forced_actions", None) else 0
        if want_grad and saliency is not None and made and forced == 0:
            saliency_sum = saliency if saliency_sum is None else saliency_sum + saliency
            saliency_count += 1
            budget -= 1
        if done:
            break

    attribution = None
    if saliency_count > 0:
        per_feature = (saliency_sum / saliency_count).tolist()
        attribution = {
            "perFeature": per_feature,
            "method": "gradient-saliency",
            "samples": saliency_count,
        }
        by_group = _group_attribution(per_feature, summary_mod._lookback(env, {}))
        if by_group:
            attribution["byGroup"] = by_group
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
    artifacts["decisionTrace"] = _compact_trace(steps, action_counts, attribution)
    if trace_file:
        artifacts["decisionTraceFile"] = trace_file
