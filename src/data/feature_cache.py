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


def load_or_build(paths, params, build, cache_dir=_CACHE_DIR):
    """Return the cached build for (source files, params), computing + persisting it on a miss.

    ``build`` is a zero-arg callable that produces the (picklable) artifacts. The cache is transparent:
    with BS_FEATURE_CACHE=0 it just calls build() and writes nothing. A corrupt/old-format entry is
    treated as a miss and rebuilt over."""
    if not _enabled():
        return build()
    key = compute_key(paths, params)
    path = os.path.join(cache_dir, key + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    result = build()
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
    return result
