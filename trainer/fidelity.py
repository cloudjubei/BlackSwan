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
    "1m": ["1m"],
    "1m+1h": ["1m", "1h"],
    "1m+1h+1d": ["1m", "1h", "1d"],
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
        # Resolve "auto" to its CONCRETE layer-set id (e.g. 1h -> "1h+1d", 1d -> "1d") so stored runs
        # carry the actual value, not the synonym — "auto" is only a convenience INPUT in the launch form.
        # A minute step mirrors the hourly default (finest base + one coarser context layer): 1m -> 1m+1h.
        if run == "1m":
            layers = ["1m", "1h"]
        elif run == "1h":
            layers = ["1h", "1d"]
        else:
            layers = ["1d"]
        set_id = "+".join(layers)
    else:
        if fset not in _LAYER_SETS:
            raise SystemExit(f"unknown fidelity_set {fset!r} — choices: {fidelity_set_ids()}")
        layers = list(_LAYER_SETS[fset])
        set_id = fset
    _validate(run, layers, set_id)
    # The BASE that's loaded = the FINEST granularity among the step and the observed layers, so the
    # provider can both STEP at the run cadence and RESAMPLE every (coarser-or-equal) layer from it. An
    # hourly step always needs the 1h base; a DAILY step needs the 1h base too whenever it observes a 1h
    # layer (step it day by day, divider 24), else the 1d base. Lookback follows the base granularity.
    # The BASE granularity is the FINEST of the step and the observed layers: 1m if any 1m is involved,
    # else 1h if any 1h, else 1d. A coarser step over a finer base (e.g. an hourly step over a 1m base)
    # is served by the provider's divider_run (decision cadence decoupled from the observed micro-data).
    if run == "1m" or "1m" in layers:
        fidelity_input = "1m"
    elif run == "1h" or "1h" in layers:
        fidelity_input = "1h"
    else:
        fidelity_input = "1d"
    return set_id, {
        "layers": layers,
        "fidelity_run": run,
        "fidelity_input": fidelity_input,
        "lookback": 32 if fidelity_input in ("1h", "1m") else 1,
    }


def _validate(run, layers, set_id):
    # The step must be a provider-supported cadence, and every observed layer must be one the finest base
    # (now 1m, the canonical source) can serve — {1m, 1h, 1d, 1w}. A coarser step over a finer base
    # (e.g. an hourly step over a 1m base) is served by the provider's divider_run, the same machinery
    # that lets a daily step observe a resampled 1h layer.
    if run not in ("1m", "1h", "1d"):
        raise SystemExit(f"unsupported timeframe {run!r} — use '1m', '1h' or '1d'.")
    allowed = {"1m", "1h", "1d", "1w"}
    bad = [layer for layer in layers if layer not in allowed]
    if bad:
        raise SystemExit(
            f"incompatible timeframe × fidelity_set: layers {bad} aren't supported (allowed "
            f"{sorted(allowed)}) — the finest base is 1m, so nothing sub-minute can be observed."
        )
