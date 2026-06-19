"""Fidelity = two SEPARATE things the user kept conflating:

  • the STEP — how often the agent acts — comes from the `timeframe` lever (`fidelity_run`).
  • the LAYERS — which timeframes it OBSERVES at each step — come from the `fidelity_set` dataset
    lever (e.g. 1h+1d).

They are independent: an hourly-stepping agent can observe coarser layers (1d, 1w) fed every hour;
a daily-stepping agent could observe finer ones. But the loaded base data can't be UPSAMPLED, so the
multi-timeline provider only implements a subset of (step × layers) combos. This module derives the
run from `timeframe`, the layers from `fidelity_set`, and FAILS FAST on any combo the provider can't
serve — so an incompatible experiment errors immediately instead of silently mis-running. Pure (no
torch / no filesystem); consumers (config_builder, summary) read the resolved spec.
"""

DEFAULT_FIDELITY_SET = "auto"

# fidelity_set label -> the OBSERVED layer stack (the run/step is separate, from `timeframe`).
# "auto" derives the stack from the run.
_LAYER_SETS = {
    "1d": ["1d"],
    "1h": ["1h"],
    "1h+1d": ["1h", "1d"],
    "1h+1d+1w": ["1h", "1d", "1w"],
    "1d+1w": ["1d", "1w"],
}


def fidelity_set_ids():
    """Selectable fidelity-set ids (the observed layer stacks), `auto` first."""
    return ["auto", *_LAYER_SETS.keys()]


def resolve_fidelity(cfg=None):
    """Return ``(set_id, spec)`` where spec = {layers, fidelity_run, fidelity_input, lookback}.

    The run/step is the ``timeframe`` lever; the layers are the ``fidelity_set`` (``auto`` = follow the
    run). Raises ``SystemExit`` on a combo the multi-timeline provider can't serve — fail fast, never
    silently mis-run.
    """
    cfg = cfg or {}
    run = str(cfg.get("timeframe", "1d"))
    raw = cfg.get("fidelity_set")
    fset = str(raw) if raw not in (None, "", "auto") else "auto"
    if fset == "auto":
        layers = ["1h", "1d"] if run == "1h" else ["1d"]
        set_id = "auto"
    else:
        if fset not in _LAYER_SETS:
            raise SystemExit(f"unknown fidelity_set {fset!r} — choices: {fidelity_set_ids()}")
        layers = list(_LAYER_SETS[fset])
        set_id = fset
    _validate(run, layers, set_id)
    return set_id, {
        "layers": layers,
        "fidelity_run": run,
        "fidelity_input": "1h" if run == "1h" else "1d",
        "lookback": 32 if run == "1h" else 1,
    }


def _validate(run, layers, set_id):
    label = "+".join(layers)
    if run == "1d":
        # Daily step: only the single 1d timeline (SingleDataProvider on the raw 1d files). A finer or
        # multi-layer stack at a daily step needs a sub-daily base the provider can't yet step daily.
        if layers != ["1d"]:
            raise SystemExit(
                f"incompatible timeframe × fidelity_set: a daily-step agent (timeframe=1d) only "
                f"supports the single '1d' dataset, not {set_id!r} ({label}). Finer/multi layers need "
                f"an hourly step (timeframe=1h); daily-step multi-fidelity is a planned data-provider "
                f"enhancement."
            )
        return
    if run == "1h":
        # Hourly step on a 1h base: any layer in {1h, 1d, 1w} is fine — the 1h base IS the finest, and
        # the multi-timeline provider resamples coarser layers (1d, 1w) from it. A single coarser layer
        # (e.g. '1d') is just the multi stack with the 1h dropped: act hourly, observe only the 1d layer.
        # Nothing FINER than 1h (no upsample).
        allowed = {"1h", "1d", "1w"}
        bad = [layer for layer in layers if layer not in allowed]
        if bad:
            raise SystemExit(
                f"incompatible timeframe × fidelity_set: an hourly-step agent (timeframe=1h) can only "
                f"observe layers in {sorted(allowed)} (the 1h base resamples coarser layers), not "
                f"{bad} — a finer layer would need a sub-hourly base."
            )
        return
    raise SystemExit(f"unsupported timeframe {run!r} — use '1h' or '1d'.")
