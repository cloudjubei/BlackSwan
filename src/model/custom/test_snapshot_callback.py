"""Direct tests for SnapshotCheckpointCallback (A6 mid-training snapshots).

No real training: the callback's `num_timesteps` is set directly (as SB3's `on_step` would) and `_on_step`
is driven across interval crossings, so the save-at-every-N + snapshot-recording logic is exercised with a
fake save_fn and no model.
"""

import os

from src.model.custom.snapshot_callback import SnapshotCheckpointCallback


def _drive(cb, timesteps):
    for t in timesteps:
        cb.num_timesteps = t
        assert cb._on_step() is True  # never halts training


def test_fires_once_per_interval_crossing_with_distinct_step_paths():
    saved = []
    cb = SnapshotCheckpointCallback(interval=10, save_fn=saved.append, base_id="m~1", folder="ckpts")
    _drive(cb, [5, 10, 15, 20, 25, 30])
    assert saved == [
        os.path.join("ckpts", "m~1.step10"),
        os.path.join("ckpts", "m~1.step20"),
        os.path.join("ckpts", "m~1.step30"),
    ]
    assert cb.snapshots == [
        {"step": 10, "path": os.path.join("ckpts", "m~1.step10")},
        {"step": 20, "path": os.path.join("ckpts", "m~1.step20")},
        {"step": 30, "path": os.path.join("ckpts", "m~1.step30")},
    ]


def test_catches_up_multiple_crossings_after_baseline():
    # After the baseline first step, a single _on_step that jumps several intervals snapshots each crossed one.
    saved = []
    cb = SnapshotCheckpointCallback(interval=10, save_fn=saved.append, base_id="m", folder="")
    cb.num_timesteps = 1
    cb._on_step()  # baseline: seeds _next at the next multiple (10), fires nothing yet
    cb.num_timesteps = 35
    cb._on_step()
    assert [s["step"] for s in cb.snapshots] == [10, 20, 30]


def test_does_not_backfill_history_on_a_high_starting_timestep():
    # A continue_from run loads a parent whose num_timesteps is already high; the callback must snapshot only
    # NEW thresholds ahead of that start, never backfill the parent's historical steps in a burst at step 0.
    saved = []
    cb = SnapshotCheckpointCallback(interval=100, save_fn=saved.append, base_id="m", folder="")
    cb.num_timesteps = 1000  # loaded parent already at 1000
    cb._on_step()
    assert cb.snapshots == []  # no backfill of 100..1000
    cb.num_timesteps = 1100
    cb._on_step()
    assert [s["step"] for s in cb.snapshots] == [1100]  # first threshold AHEAD of the start


def test_no_op_when_interval_is_zero_or_none():
    for interval in (0, None):
        saved = []
        cb = SnapshotCheckpointCallback(interval=interval, save_fn=saved.append, base_id="m", folder="")
        _drive(cb, [10, 20, 30])
        assert saved == []
        assert cb.snapshots == []


def test_max_snapshots_ring_buffers_and_deletes_the_evicted_zip(tmp_path):
    # A cap keeps only the last N snapshots (the recent learning trajectory) and DELETES the superseded .zip
    # so disk doesn't grow unbounded. save_fn writes a real {path}.zip; eviction removes exactly that file.
    def save_fn(path):
        open(path + ".zip", "w").close()  # SB3 writes {path}.zip

    cb = SnapshotCheckpointCallback(
        interval=10, save_fn=save_fn, base_id="m", folder=str(tmp_path), max_snapshots=2
    )
    _drive(cb, [5, 10, 20, 30])  # start below interval so 10/20/30 all fire (lazy seed skips only step0)
    # only the last 2 retained in the index
    assert [s["step"] for s in cb.snapshots] == [20, 30]
    # the evicted step10 .zip is gone; the kept ones remain
    assert not os.path.exists(os.path.join(str(tmp_path), "m.step10.zip"))
    assert os.path.exists(os.path.join(str(tmp_path), "m.step20.zip"))
    assert os.path.exists(os.path.join(str(tmp_path), "m.step30.zip"))


def test_max_snapshots_falsy_is_unbounded_and_deletes_nothing(tmp_path):
    def save_fn(path):
        open(path + ".zip", "w").close()

    for cap in (0, None):
        cb = SnapshotCheckpointCallback(
            interval=10, save_fn=save_fn, base_id=f"m{cap}", folder=str(tmp_path), max_snapshots=cap
        )
        _drive(cb, [5, 10, 20, 30])
        assert [s["step"] for s in cb.snapshots] == [10, 20, 30]  # all kept
        assert os.path.exists(os.path.join(str(tmp_path), f"m{cap}.step10.zip"))  # nothing deleted


def test_eviction_is_best_effort_when_the_zip_is_already_gone(tmp_path):
    # A missing .zip (e.g. save_fn that doesn't write, or an external cleanup) must never raise.
    cb = SnapshotCheckpointCallback(
        interval=10, save_fn=lambda p: None, base_id="m", folder=str(tmp_path), max_snapshots=1
    )
    _drive(cb, [5, 10, 20])  # step10 evicted though its .zip was never written
    assert [s["step"] for s in cb.snapshots] == [20]
