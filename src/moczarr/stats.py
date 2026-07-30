"""D20 stats sidecars + D22 stats rollups (read side), and O11 verification.

zagg's writer (englacial/zagg PR #307) PUTs one small JSON **stats sidecar**
per successful shard as a SIBLING object next to the leaf ``.zarr`` (never
inside it — the leaf stays vanilla zarr v3 and the D4 commit stamp stays its
final write): timings, counts, memory, cost, catalog identity
(``granules_sha256``), the D19 ``semantic_hash``, and — when the writer
records them — the O11 per-array content hashes. The unified second-pass
sweep (englacial/zagg#300, D22) folds those sidecars up-tree into
``stats.rollup.json`` objects at digit nodes.

Everything here is read-side and telemetry-class: sidecars and rollups are
**never load-bearing** (D9 — deleting every one leaves leaf reads intact), so
the readers are tolerant — absent or malformed objects read as ``None`` with
a debug log, never a raise. The one loud surface is name arithmetic
(:func:`stats_sidecar_key`), which mirrors zagg's spec-keyed seam exactly: a
writer/reader spec mismatch must fail, not silently key the wrong name.

O11 (englacial/zagg design registry, resolved 2026-07-20): the logical
content hash of a leaf is **per-array sha256 over decoded values** — each
named zarr array's full decoded contents as raw C-order little-endian bytes
at the declared dtype, after decompression — never stored object bytes,
which churn on codec/library upgrades. :func:`hash_arrays` recomputes it;
:func:`verify_arrays` compares against the sidecar's recorded hashes
("intended identical" — the semantic hash — vs "actually byte-identical").
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from moczarr.convention import (
    HIVE_SPEC,
    HIVE_SPEC_V2,
    HIVE_SPEC_V3,
    decimal_base,
    group_digits,
    leaf_path,
    manifest_path_grouping,
    morton_decimal,
    morton_word,
    split_leaf_name,
    validate_label,
)
from moczarr.store import _resolve_store, read_json, read_manifest

logger = logging.getLogger(__name__)

#: Bare-leaf sidecar object name (legacy ``/1``–``/2`` grammar, frozen).
STATS_SIDECAR_NAME = "stats.json"
#: The D22 sweep's per-node stats rollup object name (distinct from the leaf
#: sidecar names by design: under D24 a node can be leaf and interior at once).
STATS_ROLLUP_NAME = "stats.rollup.json"
#: Envelope version of every rollup object the zagg sweep writes.
SWEEP_SPEC = "zagg-sweep/1"

#: Specs keying the frozen legacy sidecar names. An absent spec (``None``) is
#: a ``morton-hive/1`` store by definition.
_LEGACY_SPECS = (None, HIVE_SPEC, HIVE_SPEC_V2)


def stats_sidecar_key(leaf_name: str, spec: str | None = None) -> str:
    """Sidecar object name for a leaf zarr basename, keyed by store spec.

    Mirrors zagg's writer seam (``zagg.telemetry.sidecar_key``, PR #307) so
    the two sides cannot drift silently:

    - Legacy (``spec`` absent / ``morton-hive/1`` / ``/2``): bare leaves get
      ``stats.json``; windowed leaves get ``stats_{window}.json`` (a hive
      node holds every window's leaf of its one shard, so a bare name would
      self-clobber across windows). Frozen grammar.
    - ``morton-hive/3`` (D23 window-basename naming): the leaf stem +
      ``.stats.json`` — ``{window}.stats.json``, and ``all.stats.json`` for
      the ``schedule: none`` reserved token (``all`` satisfies the label
      grammar by design and is excluded from explicit window labels
      spec-side, so it needs no special case here).

    An unrecognized spec RAISES rather than defaulting to the legacy
    grammar: a versioned naming bump must be a loud, deliberate change, or a
    writer/reader spec mismatch would key the wrong name and read as absent
    instead of failing.
    """
    if spec == HIVE_SPEC_V3:
        stem = leaf_name.removesuffix(".zarr")
        if not stem or stem == leaf_name:
            raise ValueError(f"{leaf_name!r} is not a leaf zarr name")
        # Same strictness as the legacy branch: a malformed stem (path
        # escape, forbidden ``_``) raises, never composes a traversing key.
        validate_label(stem)
        return f"{stem}.stats.json"
    if spec not in _LEGACY_SPECS:
        raise ValueError(
            f"unknown store spec {spec!r} (one of {_LEGACY_SPECS} for legacy names "
            f"or {HIVE_SPEC_V3!r} for D23 window-only naming)"
        )
    _full_id, window = split_leaf_name(leaf_name)
    if window is None:
        return STATS_SIDECAR_NAME
    stem, ext = STATS_SIDECAR_NAME.rsplit(".", 1)
    return f"{stem}_{window}.{ext}"


def stats_sidecar_path(leaf: str, spec: str | None = None) -> str:
    """Store-relative path of a leaf's stats sidecar (sibling of the ``.zarr``)."""
    prefix, _, name = leaf.rstrip("/").rpartition("/")
    return f"{prefix}/{stats_sidecar_key(name, spec)}"


def _read_tolerant(handle: Any, key: str, what: str) -> dict | None:
    """GET+parse one telemetry JSON object; anything unusable reads ``None``.

    The D9 telemetry posture: sidecars/rollups are regenerable, never
    load-bearing, so a malformed object degrades to absent with a debug log —
    the opposite of the manifest's loud bootstrap posture.
    """
    try:
        payload = read_json(handle, key)
    except ValueError as e:
        logger.debug(f"unparsable {what} at {key} ({e}); reading as absent")
        return None
    if payload is None:
        return None
    if not isinstance(payload, dict):
        logger.debug(f"{what} at {key} is not a mapping; reading as absent")
        return None
    return payload


def read_stats(
    store_root: str,
    shard,
    *,
    window: str | None = None,
    store: Any = None,
    **store_kwargs: Any,
) -> dict | None:
    """One leaf's D20 stats record, or ``None`` when absent/unusable.

    ``shard`` is the leaf's shard id (packed ``uint64`` word or decimal
    string) at the store's shard order; ``window`` selects the windowed
    leaf's sidecar (``stats_{window}.json``) exactly as it selects the leaf.
    The sidecar name and hive path are arithmetic off the manifest (spec,
    ``path_grouping``) — no LIST. Sidecars exist on success only and their
    PUT is fail-open writer-side, so absence means "no telemetry", never
    "no data" — completeness stays the commit stamp's job (D4).

    For folded stats above the leaf level see :func:`read_stats_rollup`.
    """
    handle = _resolve_store(store_root, store, store_kwargs)
    manifest = read_manifest(store_root, store=handle)
    if manifest is None:
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    rel = leaf_path(
        morton_word(shard), window=window, path_grouping=manifest_path_grouping(manifest)
    )
    return _read_tolerant(handle, stats_sidecar_path(rel, manifest["spec"]), "stats sidecar")


def _node_rel(decimal: str, path_grouping: int) -> str:
    """A node decimal's relative digit path, chunked per the manifest (D21)."""
    base = decimal_base(decimal)
    return "/".join([base, *group_digits(decimal[len(base) :], path_grouping)])


def read_stats_rollup(
    store_root: str,
    node,
    *,
    store: Any = None,
    **store_kwargs: Any,
) -> dict | None:
    """The D22 stats rollup envelope at a digit node, or ``None``.

    ``node`` is any digit-tree prefix — a base component, an interior
    prefix, or a shard id (packed word or decimal string; shard-node rollups
    fold that shard's windows). Returns the sweep envelope as written
    (``spec``/``family``/``node``/``order``/``generation``/``payload``,
    plus ``windows`` at shard nodes); ``envelope["payload"]`` is a merged
    D20 stats record — the same shape :func:`read_stats` returns, folded
    over every leaf beneath the node.

    Cache posture (matching the sweep's own reader): missing, unparsable,
    wrong-spec/family, or stamp-less objects all read as ``None`` with a
    debug log — a rollup is a regenerable derived artifact (D9), and its
    generation stamp makes staleness *detectable, not prevented*; consumers
    needing exactness fold leaf records themselves.
    """
    handle = _resolve_store(store_root, store, store_kwargs)
    manifest = read_manifest(store_root, store=handle)
    if manifest is None:
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    decimal = morton_decimal(node) if not isinstance(node, str) else node
    key = f"{_node_rel(decimal, manifest_path_grouping(manifest))}/{STATS_ROLLUP_NAME}"
    envelope = _read_tolerant(handle, key, "stats rollup")
    if envelope is None:
        return None
    usable = (
        envelope.get("spec") == SWEEP_SPEC
        and envelope.get("family") == "stats"
        and isinstance(envelope.get("generation"), dict)
        and isinstance(envelope.get("payload"), dict)
    )
    if not usable:
        logger.debug(f"stats rollup at {key} has an unknown spec/shape; reading as absent")
        return None
    return envelope


def hash_arrays(
    store_root: str,
    leaf: str,
    *,
    store: Any = None,
    **store_kwargs: Any,
) -> dict[str, str]:
    """O11 per-array content hashes of one leaf, recomputed from decoded values.

    Opens the leaf as vanilla zarr v3 and hashes EVERY array beneath it
    (data fields, ``morton``, every coordinate — the resolved O11 scope),
    keyed by the array's path relative to the leaf root (e.g.
    ``"8/morton"``). Each hash is sha256 over the array's full decoded
    contents as raw **C-order little-endian** bytes at the declared dtype —
    decoded values, never stored object bytes, so codec/packaging changes
    (ShardingCodec inner chunks, compressor upgrades) are invisible by
    construction while any value change flips the hash (exact bytes, no
    float tolerance — the PR #282 class is meant to flip it).
    """
    import zarr
    from zarr.storage import ObjectStore

    handle = _resolve_store(store_root, store, store_kwargs)
    group = zarr.open_group(
        ObjectStore(handle, read_only=True), path=leaf.strip("/"), mode="r", zarr_format=3
    )
    hashes = {}
    for key, node in group.members(max_depth=None):
        if not isinstance(node, zarr.Array):
            continue
        values = np.ascontiguousarray(node[...])
        if values.dtype.byteorder == ">":  # canonical form is little-endian
            values = values.astype(values.dtype.newbyteorder("<"))
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
    return hashes


def combined_hash(hashes: dict[str, str]) -> str:
    """The O11 combined hash: sha256 of the sorted per-array hex digests.

    Serialization pinned here (and by the committed fixture's golden): the
    digests sorted lexically and joined with ``"\\n"``, hashed as ASCII —
    array *names* deliberately excluded, per O11's definition ("hash of the
    sorted per-array hashes").
    """
    return hashlib.sha256("\n".join(sorted(hashes.values())).encode()).hexdigest()


def _recorded_hashes(sidecar: dict | None) -> tuple[dict[str, str] | None, str | None]:
    """``(per-array, combined)`` recorded in a sidecar's ``content_hashes``.

    Accepts both shapes the D20/O11 wording admits — the nested
    ``{"arrays": {name: hash}, "combined": hash}`` envelope (what the
    committed fixture pins) and a flat ``{name: hash}`` mapping — so the
    verifier keeps working whichever zagg's writer lands.
    """
    content = (sidecar or {}).get("content_hashes")
    if not isinstance(content, dict) or not content:
        return None, None
    if isinstance(content.get("arrays"), dict):
        return dict(content["arrays"]), content.get("combined")
    return dict(content), None


def verify_arrays(
    store_root: str,
    shard,
    *,
    window: str | None = None,
    store: Any = None,
    **store_kwargs: Any,
) -> dict:
    """Verify one leaf's decoded contents against its sidecar's O11 hashes.

    The verification half of the D19 identity story: the ``semantic_hash``
    says two leaves were *intended* identical; the O11 content hashes say
    they *are* byte-identical — and localize a mismatch to the array. Also
    the detection mechanism for stamped-but-torn leaves under the
    concurrency contract's out-of-contract case.

    Returns::

        {
            "leaf": ...,              # store-relative leaf path
            "computed": {name: hash}, # recomputed per-array hashes
            "combined": ...,          # recomputed combined hash
            "recorded": ...,          # sidecar's per-array hashes (or None)
            "recorded_combined": .,   # sidecar's combined hash (or None)
            "match": ...,             # True/False; None = nothing recorded
            "mismatched": [...],      # array names differing (either side)
        }

    ``match`` is ``None`` when the sidecar is absent or records no
    ``content_hashes`` — "nothing to verify against" is a distinct answer
    from "verified" (the conservative posture zagg's dedup takes: an
    unverifiable leaf is never a hit).
    """
    handle = _resolve_store(store_root, store, store_kwargs)
    manifest = read_manifest(store_root, store=handle)
    if manifest is None:
        raise ValueError(f"no morton_hive.json at {store_root} — not a hive store root")
    rel = leaf_path(
        morton_word(shard), window=window, path_grouping=manifest_path_grouping(manifest)
    )
    sidecar = _read_tolerant(handle, stats_sidecar_path(rel, manifest["spec"]), "stats sidecar")
    recorded, recorded_combined = _recorded_hashes(sidecar)
    computed = hash_arrays(store_root, rel, store=handle)
    mismatched: list[str] = []
    match: bool | None = None
    if recorded is not None:
        names = sorted(set(computed) | set(recorded))
        mismatched = [n for n in names if computed.get(n) != recorded.get(n)]
        match = not mismatched
    return {
        "leaf": rel,
        "computed": computed,
        "combined": combined_hash(computed),
        "recorded": recorded,
        "recorded_combined": recorded_combined,
        "match": match,
        "mismatched": mismatched,
    }


__all__ = [
    "STATS_ROLLUP_NAME",
    "STATS_SIDECAR_NAME",
    "SWEEP_SPEC",
    "combined_hash",
    "hash_arrays",
    "read_stats",
    "read_stats_rollup",
    "stats_sidecar_key",
    "stats_sidecar_path",
    "verify_arrays",
]
