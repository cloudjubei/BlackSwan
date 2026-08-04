import pytest

from trainer import projection as projection_mod
from trainer.projection import (
    DEFAULT_PROJECTION,
    projection_ids,
    resolve_projection,
)


def test_default_is_standard():
    # No projection lever -> the standard exposure, which is exactly today's default feature set
    # (own return + high/low/volume percents, no curated indicators) so pre-projection runs are
    # unchanged.
    rung, spec = resolve_projection()
    assert rung == "standard"
    assert spec == {"type": "only_price_percent", "use_indicators": False}
    assert DEFAULT_PROJECTION == "standard"


def test_minimal_is_the_single_own_return():
    rung, spec = resolve_projection({"projection": "minimal"})
    assert rung == "minimal"
    assert spec == {"type": "solo_price_percent", "use_indicators": False}


def test_with_indicators_enables_the_curated_set():
    # with_indicators == today's paper setup (only_price_percent + use_indicators), so migrated
    # use_indicators:true runs land here.
    _, spec = resolve_projection({"projection": "with_indicators"})
    assert spec == {"type": "only_price_percent", "use_indicators": True}


def test_unknown_projection_fails_fast():
    with pytest.raises(SystemExit):
        resolve_projection({"projection": "does_not_exist"})


def test_projection_ids_are_coarsest_first():
    assert projection_ids() == ["minimal", "standard", "with_indicators"]


def test_per_asset_map_overrides_the_scalar_lever():
    # Per-asset shaping: a mixed-asset dataset can expose each asset differently. The projections map
    # (asset -> rung) wins over the scalar projection for named assets.
    cfg = {"projection": "minimal", "projections": {"ETHUSDT": "with_indicators"}}
    assert resolve_projection(cfg, asset="ETHUSDT")[0] == "with_indicators"


def test_per_asset_falls_back_to_scalar_for_unlisted_asset():
    cfg = {"projection": "minimal", "projections": {"ETHUSDT": "with_indicators"}}
    assert resolve_projection(cfg, asset="BTCUSDT")[0] == "minimal"


def test_scalar_projection_applies_when_no_per_asset_entry():
    assert resolve_projection({"projection": "with_indicators"}, asset="BTCUSDT")[0] == "with_indicators"


def test_returned_spec_is_a_copy_not_shared_state():
    _, spec = resolve_projection({"projection": "minimal"})
    spec["use_indicators"] = True
    _, spec2 = resolve_projection({"projection": "minimal"})
    assert spec2["use_indicators"] is False
