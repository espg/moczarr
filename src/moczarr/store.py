"""Store layer: obstore-backed access to hive store roots and leaves.

One backend (obstore) by design — the walk's termination rule ("no digit
children ⇒ nothing finer exists") and the debris model both lean on
strongly-consistent, uncached delimiter-LIST semantics reaching this code
unmediated, so a second backend would be a second correctness surface
(rationale on the plan thread, espg/moczarr#1).

Leaf functions take ``(store_root, leaf)`` with ``leaf`` a store-relative
path (from :func:`moczarr.convention.leaf_path` or :func:`walk_leaves`), so
every access — local or S3 — goes through one store handle and a missing
leaf is uniformly a clean GET miss, never a backend-dependent error.

Postures, per the design's D9 discipline:

- The manifest is the bootstrap: absent reads ``None``, malformed raises —
  there is no degraded mode without it.
- Cache tiers (root MOC) are tolerant: garbage reads as absent with a debug
  log, and the caller degrades to :func:`walk_leaves` — never wrong answers.
- The commit stamp gates completeness: an unstamped or malformed-stamp leaf
  is debris and reads ``None``. Presence requires the stamp; absence (a
  clean LIST/GET miss) is trustworthy on its own.
- A PRESENT-but-corrupt bitmap sidecar raises (see ``moczarr.coverage``).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from moczarr.convention import (
    COMMIT_ATTR,
    MANIFEST_NAME,
    ROOT_COVERAGE_NAME,
    is_base_component,
    morton_word,
    parse_manifest,
    split_leaf_name,
)
from moczarr.coverage import (
    decode_bitmap,
    parse_leaf_coverage,
    parse_root_coverage,
    ranges_contain,
)

logger = logging.getLogger(__name__)

#: Stores already warned about a stale root MOC in this process (O7 lean:
#: warn once per store per stale episode, trust silently otherwise, never
#: auto-walk on the hot path).
_stale_warned: set[str] = set()


def open_object_store(path: str, *, anonymous: bool = False, **kwargs: Any):
    """Open an obstore store at ``path`` (``s3://...`` or a local directory).

    Read-only posture: a missing local directory raises ``FileNotFoundError``
    (the writer creates stores; the reader never does). ``anonymous=True``
    skips request signing for public buckets (the source.coop case);
    ``kwargs`` pass through to ``S3Store.from_url`` (``region=...`` etc.).
    """
    if path.startswith("s3://"):
        from obstore.store import S3Store

        if anonymous:
            kwargs.setdefault("skip_signature", True)
        return S3Store.from_url(path, **kwargs)
    from obstore.store import LocalStore

    local = Path(path).expanduser().resolve()
    if not local.is_dir():
        raise FileNotFoundError(f"{path!r} is not a directory — not a readable store root")
    return LocalStore(local)


def read_json(store, key: str):
    """GET+parse one small JSON object; ``None`` when it does not exist.

    Parse errors propagate — tolerant callers wrap this per their tier's
    posture.
    """
    import obstore
    from obstore.exceptions import NotFoundError

    try:
        data = obstore.get(store, key).bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    return json.loads(bytes(data))


def read_manifest(store_root: str, **store_kwargs: Any) -> dict | None:
    """The store's validated ``morton_hive.json``; ``None`` when absent.

    Loud on malformed content (bad JSON or a failed :func:`parse_manifest`):
    the manifest is the reader's bootstrap, so garbage here is an error, not
    a degradable cache.
    """
    payload = read_json(open_object_store(store_root, **store_kwargs), MANIFEST_NAME)
    return None if payload is None else parse_manifest(payload)


def load_root_coverage(store_root: str, **store_kwargs: Any) -> dict | None:
    """The store-root coverage envelope, or ``None`` when unusable.

    Tolerant (the root MOC is a regenerable cache): a missing object,
    unparsable JSON, or an unknown spec/encoding all read as absent — with a
    debug log so the degradation is discoverable — and the caller falls back
    to :func:`walk_leaves`.
    """
    store = open_object_store(store_root, **store_kwargs)
    try:
        payload = read_json(store, ROOT_COVERAGE_NAME)
    except ValueError as e:
        logger.debug(f"unparsable {ROOT_COVERAGE_NAME} at {store_root} ({e}); ignoring")
        return None
    envelope = parse_root_coverage(payload)
    if payload is not None and envelope is None:
        logger.debug(f"{ROOT_COVERAGE_NAME} at {store_root} has an unknown spec/encoding; ignoring")
    return envelope


def read_commit(store_root: str, leaf: str, **store_kwargs: Any) -> dict | None:
    """A leaf's commit stamp, or ``None`` for debris / absent leaves.

    One GET of the leaf's root ``zarr.json`` (the leaf is vanilla zarr v3 by
    convention, so the root group metadata is that one object; no zarr
    machinery needed to check completeness). A missing object, a missing
    stamp, and a malformed (non-mapping) stamp are the same answer: not
    complete. A present-but-unparsable ``zarr.json`` raises — that leaf
    claims to exist and cannot be half-trusted.
    """
    store = open_object_store(store_root, **store_kwargs)
    meta = read_json(store, f"{leaf.strip('/')}/zarr.json")
    if not isinstance(meta, dict):
        return None
    attrs = meta.get("attributes")
    stamp = attrs.get(COMMIT_ATTR) if isinstance(attrs, dict) else None
    return dict(stamp) if isinstance(stamp, dict) else None


def read_leaf_coverage(store_root: str, leaf: str, **store_kwargs: Any) -> dict | None:
    """A leaf's coverage envelope off its commit stamp, or ``None``."""
    return parse_leaf_coverage(read_commit(store_root, leaf, **store_kwargs))


def read_coverage_bitmap(
    store_root: str, leaf: str, *, coverage: dict | None = None, **store_kwargs: Any
) -> np.ndarray | None:
    """A leaf's exact occupied cell words from its bitmap sidecar, or ``None``.

    ``None`` for anything without a ``"bitmap"`` sidecar to read: debris, a
    box-only envelope, ``encoding: "full"`` (there IS no sidecar — the shard
    id is the exact MOC; :func:`bitmap_and` short-circuits on it), or a
    missing sidecar object. A present-but-corrupt sidecar raises (decoder's
    posture). The shard id comes from the leaf basename via the frozen
    first-``_`` split; ``cell_order`` from the envelope. Pass an
    already-read ``coverage`` envelope to skip the stamp GET.
    """
    import obstore
    from obstore.exceptions import NotFoundError

    if coverage is None:
        coverage = read_leaf_coverage(store_root, leaf, **store_kwargs)
    if not coverage or coverage.get("encoding") != "bitmap" or not coverage.get("sidecar"):
        return None
    shard, _window = split_leaf_name(leaf.rstrip("/").rsplit("/", 1)[-1])
    store = open_object_store(store_root, **store_kwargs)
    try:
        data = obstore.get(store, f"{leaf.strip('/')}/{coverage['sidecar']}").bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    return decode_bitmap(bytes(data), shard, int(coverage["cell_order"]))


def bitmap_and(store_root: str, leaf: str, aoi, **store_kwargs: Any) -> np.ndarray | None:
    """Exact cell-level intersection via a leaf's coverage (bitmap or full).

    Reads the stamp once. ``encoding: "full"`` short-circuits to MOC
    membership against the shard's own id — no sidecar GET, no expansion.
    The ``"bitmap"`` path pays the one sidecar GET. ``None`` when the leaf
    carries neither (box-only, debris, absent) — the caller falls back to
    the box verdict. An empty array is a definitive miss: both encodings
    are exact, not conservative.
    """
    from mortie import moc_and

    coverage = read_leaf_coverage(store_root, leaf, **store_kwargs)
    if not coverage:
        return None
    if coverage.get("encoding") == "full":
        word = morton_word(split_leaf_name(leaf.rstrip("/").rsplit("/", 1)[-1])[0])
        return moc_and(np.asarray([word], dtype=np.uint64), np.asarray(aoi, dtype=np.uint64))
    occupied = read_coverage_bitmap(store_root, leaf, coverage=coverage, **store_kwargs)
    if occupied is None:
        return None
    return moc_and(occupied, np.asarray(aoi, dtype=np.uint64))


def walk_leaves(store_root: str, **store_kwargs: Any) -> Iterator[str]:
    """Yield the store-relative path of every leaf zarr — the discovery walk.

    The fallback/verification path (never the hot path): one delimiter-LIST
    per digit node — root children must be ``{sign+base}``-shaped, deeper
    children a single ``1..4`` digit; a ``*.zarr`` child is a leaf at that
    node; no digit children means nothing finer exists (LIST is strongly
    consistent and object stores have no empty prefixes, so absence is
    definitive). Yields stamped and debris leaves alike — completeness is
    the caller's check (:func:`read_commit`), matching the tiered postures.
    Non-conforming names below the root are ignored (the node invariant says
    they are not ours to interpret).
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    stack = [""]
    while stack:
        prefix = stack.pop()
        listing = obstore.list_with_delimiter(store, prefix or None)
        for child in listing["common_prefixes"]:
            rel = child.rstrip("/")
            name = rel.split("/")[-1]
            if name.endswith(".zarr"):
                yield rel
                continue
            is_digit_node = (
                is_base_component(name) if prefix == "" else len(name) == 1 and name in "1234"
            )
            if is_digit_node:
                stack.append(rel + "/")


def warn_if_stale(store_root: str, shard: str | int, envelope: dict | None) -> bool:
    """O7 lazy staleness detection for one opened, commit-stamped leaf.

    Call with POSITIVE evidence of a committed shard (an opened leaf with a
    stamp) that the root MOC does not list. Usually benign — a run in
    progress writes the root MOC only at end of run — so this warns ONCE per
    store per process with the context, returns whether stale, and never
    auto-walks. ``envelope`` may be ``None`` (no root MOC at all): that is
    absence, not staleness, and reads ``False``. A malformed envelope cannot
    vouch for the shard and counts as stale.
    """
    if envelope is None:
        return False
    from moczarr.convention import morton_decimal

    decimal = morton_decimal(shard)
    try:
        if ranges_contain(envelope, decimal):
            return False
    except (KeyError, TypeError, ValueError):
        pass  # malformed envelope cannot vouch for the shard -> stale
    key = store_root.rstrip("/")
    if key not in _stale_warned:
        _stale_warned.add(key)
        warnings.warn(
            f"commit-stamped shard {decimal} is not listed by {store_root}/"
            f"{ROOT_COVERAGE_NAME} — the root MOC lags the leaves. Usually benign "
            f"(a run in progress writes the root MOC at end of run); otherwise a "
            f"crashed run, a concurrent-run union race, or out-of-band writes. "
            f"The store's writer can regenerate it (zagg's refresh, or the sweep).",
            stacklevel=2,
        )
    return True


__all__ = [
    "bitmap_and",
    "load_root_coverage",
    "open_object_store",
    "read_commit",
    "read_coverage_bitmap",
    "read_json",
    "read_leaf_coverage",
    "read_manifest",
    "walk_leaves",
    "warn_if_stale",
]
