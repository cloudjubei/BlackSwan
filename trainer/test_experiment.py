import json

from trainer import experiment as ex


def _rec(**kw):
    base = dict(
        exp_id="demo-2020-01", name="Demo scan", thesis="X beats Y net of fees",
        purpose="probe", matrix={"asset": ["GOLD", "SPY"]},
        cells=[{"config": {"asset": "GOLD"}, "metrics": {"return_vs_hold_pct": -4.8}}],
        aggregate={"pass_rate": 0.0}, verdict="X did not beat Y", status="refutes",
        created_at="2020-01-01T00:00:00", hypothesis_id="hyp-x-beats-y",
        provenance={"gitCommit": "abc123", "gitDirty": True},
    )
    base.update(kw)
    return ex.build_record(**base)


def test_record_is_stamped_as_non_rl_and_links_to_the_one_thesis():
    r = _rec()
    # a side-experiment is a SOURCE for the one thesis concept, and unmistakably NOT an RL model run
    assert r["kind"] == ex.KIND and "NOT an RL model run" in r["kind"]
    assert r["hypothesisId"] == "hyp-x-beats-y"  # links to the ONE thesis it feeds
    assert r["thesis"] == "X beats Y net of fees"
    assert r["status"] == "refutes"
    assert "asset" in r["matrix"] and len(r["cells"]) == 1


def test_hypothesis_id_defaults_to_none_for_standalone_experiments():
    assert ex.build_record(
        exp_id="e", name="n", thesis="t", purpose="p", matrix={}, cells=[], aggregate={},
        verdict="v", status="inconclusive", created_at="2020-01-01",
    )["hypothesisId"] is None


def test_save_and_load_round_trip(tmp_path):
    r = _rec()
    d = ex.save_record(r, root=str(tmp_path))
    assert (tmp_path / "demo-2020-01" / "record.json").exists()
    assert (tmp_path / "demo-2020-01" / "report.md").exists()
    assert ex.load_record("demo-2020-01", root=str(tmp_path)) == r
    assert d.endswith("demo-2020-01")


def test_report_surfaces_thesis_status_link_and_non_rl_kind():
    md = ex.format_report(_rec(status="supports"))
    assert "# Side-experiment: Demo scan" in md
    assert "NOT an RL model run" in md
    assert "X beats Y net of fees" in md
    assert "**SUPPORTS**" in md
    assert "hyp-x-beats-y" in md  # the linked thesis is shown


def test_saved_json_is_valid_and_complete(tmp_path):
    ex.save_record(_rec(), root=str(tmp_path))
    d = json.load(open(tmp_path / "demo-2020-01" / "record.json"))
    for k in ("id", "kind", "thesis", "status", "matrix", "cells", "aggregate", "verdict", "provenance"):
        assert k in d
