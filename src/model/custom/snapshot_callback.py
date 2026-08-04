import os

from stable_baselines3.common.callbacks import BaseCallback


class SnapshotCheckpointCallback(BaseCallback):
    """Save a RETAINED checkpoint every ``interval`` SB3 timesteps during training, so a decision trace can
    be generated from each snapshot afterwards and the learning trajectory diffed over time (A6).

    Native ``EveryNTimesteps`` semantics on ``self.num_timesteps`` (which SB3 updates before each
    ``_on_step``): every crossed multiple of ``interval`` is snapshotted to ``{folder}/{base_id}.step{n}``
    via ``save_fn`` and recorded in ``self.snapshots``. A single step may cross several intervals (a large
    rollout) — each is saved. ``interval`` falsy (0/None) makes it inert, so the default training path
    passes no callback and is byte-identical. Never halts training (``_on_step`` returns True).

    ``max_snapshots`` (falsy = unbounded) RING-BUFFERS the retained snapshots: once more than that many exist
    the OLDEST is dropped from the index and its ``{path}.zip`` best-effort deleted, so a long run keeps only
    the recent learning trajectory instead of one full checkpoint per interval. Only the ``.zip`` this
    callback itself wrote is ever removed (never the base path or the final checkpoint)."""

    def __init__(self, interval, save_fn, base_id, folder="", max_snapshots=None, verbose=0):
        super().__init__(verbose)
        self.interval = int(interval) if interval else 0
        self.save_fn = save_fn
        self.base_id = base_id
        self.folder = folder
        self.max_snapshots = int(max_snapshots) if max_snapshots else 0
        self.snapshots = []
        # Seeded lazily on the first _on_step from the RUN's starting num_timesteps — so a continue_from run
        # (whose loaded parent already sits at a high step count) snapshots only NEW thresholds AHEAD of that
        # start, never backfilling the parent's history in a burst at step 0.
        self._next = None

    def _on_step(self) -> bool:
        if self.interval <= 0:
            return True
        if self._next is None:
            self._next = ((self.num_timesteps // self.interval) + 1) * self.interval
        while self.num_timesteps >= self._next:
            step = self._next
            path = os.path.join(self.folder, f"{self.base_id}.step{step}")
            self.save_fn(path)
            self.snapshots.append({"step": step, "path": path})
            self._next += self.interval
            while self.max_snapshots > 0 and len(self.snapshots) > self.max_snapshots:
                evicted = self.snapshots.pop(0)
                try:
                    zip_path = str(evicted["path"]) + ".zip"
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                except OSError:
                    pass  # best-effort eviction — never fail training over a stale snapshot file
        return True
