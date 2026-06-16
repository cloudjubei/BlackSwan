"""Multi-fidelity observation layer sets for the trading line.

The agent can observe several timeframes at once — the multi-timeline provider stacks one layer per
entry in ``layers`` (resampling the higher layers from the loaded ``fidelity_run`` bars), so wider
sets give more temporal context. This was previously implicit in the ``timeframe`` lever (1h silently
meant the 1h+1d stack); ``fidelity_set`` makes the stack an explicit, sweepable, self-documenting
choice. Pure (no torch / no filesystem) so both ``config_builder`` (path/lever wiring) and ``summary``
(self-documenting the run) resolve from the same source.
"""

DEFAULT_INTRADAY_SET = "1h+1d"

# id -> the observation stack it selects. ``fidelity_run`` is the bar size the agent steps on (the
# base file loaded); ``layers`` are resampled from it; ``lookback`` is the per-layer window.
_FIDELITY_SETS = {
    "1d": {"layers": ["1d"], "fidelity_run": "1d", "lookback": 1},
    "1h": {"layers": ["1h"], "fidelity_run": "1h", "lookback": 32},
    "1h+1d": {"layers": ["1h", "1d"], "fidelity_run": "1h", "lookback": 32},
    "1h+1d+1w": {"layers": ["1h", "1d", "1w"], "fidelity_run": "1h", "lookback": 32},
}


def fidelity_set_ids():
    """The selectable fidelity-set ids, ordered narrowest first."""
    return list(_FIDELITY_SETS)


def resolve_fidelity(cfg=None):
    """Return ``(set_id, spec)`` for the cfg's fidelity stack.

    An explicit ``fidelity_set`` lever wins; ``auto`` (or absent) derives from ``timeframe`` so
    existing configs and the lever's default are unchanged (``1h`` -> the ``1h+1d`` multi-layer path,
    anything else -> single daily). Raises ``SystemExit`` on an unknown id so a typo fails fast.
    """
    cfg = cfg or {}
    fset = cfg.get("fidelity_set")
    if fset and str(fset) not in ("", "auto"):
        fset = str(fset)
        if fset not in _FIDELITY_SETS:
            raise SystemExit(f"unknown fidelity_set {fset!r} — choices: {fidelity_set_ids()}")
        return fset, _FIDELITY_SETS[fset]
    fset = DEFAULT_INTRADAY_SET if str(cfg.get("timeframe", "1d")) == "1h" else "1d"
    return fset, _FIDELITY_SETS[fset]
