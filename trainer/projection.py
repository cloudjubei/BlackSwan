"""Projection = how much of an asset's OWN data the model observes (its "data exposure").

A per-asset ladder, coarsest -> richest:

  • minimal          — the single own-return column (the bare microstructure floor)
  • standard         — own return + high/low/volume percents (no curated indicators)
  • with_indicators  — standard + the curated indicator set
  • with_extra_data  — with_indicators + this asset's fused point-in-time context series (later step)

It is PER ASSET so a mixed-asset dataset can expose each asset differently (a richly-observed
tradeable alongside a minimally-observed context asset). Today one tradeable is populated; the
`projections` map (asset -> rung) overrides the scalar `projection` lever for named assets, so the
multi-asset levels reuse this resolver unchanged. Pure (no torch / no filesystem); config_builder
reads the resolved spec and sets the low-level DataConfig fields (type, use_indicators) from it.
"""

DEFAULT_PROJECTION = "standard"

# rung -> the low-level DataConfig feature knobs it resolves to. The `type` values are the provider's
# existing feature-set encodings (src/data/abstract_dataprovider.py process_df): solo_price_percent =
# one own-return column; only_price_percent = + high/low/volume percents. `use_indicators` gates the
# curated indicator columns. History-preserving: pre-projection runs all used only_price_percent (with
# the use_indicators bool varying), so they map exactly onto standard / with_indicators.
_PROJECTIONS = {
    "minimal": {"type": "solo_price_percent", "use_indicators": False},
    "standard": {"type": "only_price_percent", "use_indicators": False},
    "with_indicators": {"type": "only_price_percent", "use_indicators": True},
}


def projection_ids():
    """Selectable projection rungs, coarsest first."""
    return list(_PROJECTIONS.keys())


def resolve_projection(cfg=None, asset=None):
    """Return ``(rung, spec)`` where spec = {type, use_indicators} for ``asset``.

    A per-asset ``projections`` map (asset -> rung) wins over the scalar ``projection`` lever, which
    wins over the default. Raises ``SystemExit`` on an unknown rung — fail fast, never silently
    mis-expose.
    """
    cfg = cfg or {}
    per_asset = cfg.get("projections") or {}
    raw = per_asset.get(asset) if asset is not None else None
    if raw in (None, ""):
        raw = cfg.get("projection")
    rung = str(raw) if raw not in (None, "") else DEFAULT_PROJECTION
    if rung not in _PROJECTIONS:
        raise SystemExit(f"unknown projection {rung!r} — choices: {projection_ids()}")
    return rung, dict(_PROJECTIONS[rung])
