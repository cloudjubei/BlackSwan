"""Disk cache for the deterministic, expensive feature-frame build.

Every seed/algorithm run on the SAME (data window, processing config) rebuilds an IDENTICAL processed
feature frame — rolling z-scores, Pi-cycle SMAs/EMAs, per-fidelity resamples — costing seconds to tens
of seconds. That build is a pure function of (source file CONTENTS, processing config), so we memoize its
result keyed by a hash of both: each source path with its size + mtime_ns (so a regenerated/derived file
invalidates the entry, mirroring derive_cache's freshness guard) plus every feature-affecting config
field. NOTE the key is built here, NOT from the provider's get_id(), because get_id() omits
feature-affecting fields like use_indicators / obs_squash — keying on it would silently serve the wrong
features. A miss rebuilds and writes atomically; a hit unpickles. Set BS_FEATURE_CACHE=0 to disable.
"""

import hashlib
import os
import pickle
import tempfile

_CACHE_DIR = os.path.join("binance", "derived", "feature_cache")


def _enabled():
    return os.environ.get("BS_FEATURE_CACHE", "1") != "0"


def _flatten(paths):
    out = []
    for p in paths:
        if isinstance(p, (list, tuple)):
            out.extend(_flatten(p))
        else:
            out.append(p)
    return out


def _source_signature(paths):
    """(path, size, mtime_ns) per source file, sorted — a content-change fingerprint independent of
    the order the caller lists the files. A missing file signs as (-1, -1) so it can't alias a present one."""
    sig = []
    for p in sorted(_flatten(paths)):
        try:
            st = os.stat(p)
            sig.append((p, st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append((p, -1, -1))
    return sig


def compute_key(paths, params):
    payload = repr((_source_signature(paths), sorted((str(k), repr(v)) for k, v in params.items())))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _write_atomic(path, result, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_or_build_key(key, build, cache_dir=None):
    """Like load_or_build but with a CALLER-COMPUTED key — for inputs that aren't source files (e.g. an
    in-memory frame keyed by its content hash). ``build`` is a zero-arg callable producing picklable
    artifacts; a corrupt/old-format entry is treated as a miss and rebuilt over. BS_FEATURE_CACHE=0
    bypasses entirely (no disk touched). cache_dir defaults to _CACHE_DIR resolved at call time."""
    if not _enabled():
        return build()
    cache_dir = cache_dir or _CACHE_DIR
    path = os.path.join(cache_dir, key + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    result = build()
    _write_atomic(path, result, cache_dir)
    return result


def load_or_build(paths, params, build, cache_dir=None):
    """Return the cached build for (source files, params), computing + persisting it on a miss.

    ``build`` is a zero-arg callable that produces the (picklable) artifacts. The cache is transparent:
    with BS_FEATURE_CACHE=0 it just calls build() and writes nothing."""
    if not _enabled():
        return build()
    return load_or_build_key(compute_key(paths, params), build, cache_dir)
