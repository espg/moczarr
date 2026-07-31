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
O11's scope includes the ragged (vlen-bytes) arrays, whose decoded elements
are Python objects rather than a flat buffer — see :func:`hash_arrays` for
the length-prefixed vlen recipe this reader defines (there is no zagg O11
writer yet, so the recipe is pinned here and by the committed fixture).
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

    Cache posture (mirroring the sweep's own reader, ``zagg.sweep._read_rollup``):
    missing, unparsable, wrong-spec/family, or stamp-less objects all read as
    ``None`` with a debug log — a rollup is a regenerable derived artifact
    (D9), and its generation stamp makes staleness *detectable, not
    prevented*; consumers needing exactness fold leaf records themselves.
    "Stamp-less" is checked to the same depth as the sweep (a ``generation``
    dict with an integer ``n_leaves``), with one deliberate divergence: the
    sweep requires only that ``payload`` be *present*, this requires it to be
    a mapping — the documented ``envelope["payload"]`` contract, and cheaper
    to enforce here than to guard at every consumer.
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
    generation = envelope.get("generation")
    usable = (
        envelope.get("spec") == SWEEP_SPEC
        and envelope.get("family") == "stats"
        and isinstance(generation, dict)
        and isinstance(generation.get("n_leaves"), int)
        and isinstance(envelope.get("payload"), dict)
    )
    if not usable:
        logger.debug(f"stats rollup at {key} has an unknown spec/shape; reading as absent")
        return None
    return envelope


#: Width of the O11 vlen recipe's per-element length prefix (uint64, LE).
VLEN_LENGTH_PREFIX = 8


def _element_bytes(element: object) -> bytes:
    """One vlen cell's decoded payload as raw little-endian bytes.

    ``variable_length_bytes`` cells decode to :class:`bytes` (zagg's ragged
    digest payloads and their ``{field}_locations`` sibling are both this
    dtype — the location words are raw little-endian ``uint64`` bytes). The
    other object-dtype element kinds are covered for the typed
    ``vlen-array<T>`` / vlen-utf8 futures; anything else RAISES rather than
    hashing a ``repr`` — a digest that is silently wrong is worse here than
    no digest at all.
    """
    if element is None:
        return b""  # an unwritten vlen cell can decode as None, not b""
    if isinstance(element, bytes | bytearray | memoryview):
        return bytes(element)
    if isinstance(element, str):
        return element.encode()
    if isinstance(element, np.ndarray):
        values = np.ascontiguousarray(element)
        if values.dtype.byteorder == ">":
            values = values.astype(values.dtype.newbyteorder("<"))
        return values.tobytes()
    raise ValueError(
        f"vlen element of type {type(element).__name__} has no O11 byte recipe "
        f"(expected bytes, str, ndarray, or None)"
    )


def hash_arrays(
    store_root: str,
    leaf: str,
    *,
    store: Any = None,
    **store_kwargs: Any,
) -> dict[str, str]:
    """O11 per-array content hashes of one leaf, recomputed from decoded values.

    Opens the leaf as vanilla zarr v3 and hashes EVERY array beneath it
    (data fields, the ragged vlen arrays, ``morton``, every coordinate — the
    resolved O11 scope), keyed by the array's path relative to the leaf root
    (e.g. ``"8/morton"``). Each hash is sha256 over the array's full decoded
    contents as raw **C-order little-endian** bytes at the declared dtype —
    decoded values, never stored object bytes, so codec/packaging changes
    (ShardingCodec inner chunks, compressor upgrades) are invisible by
    construction while any value change flips the hash (exact bytes, no
    float tolerance — the PR #282 class is meant to flip it).

    **Vlen (ragged) recipe.** An object-dtype array (zagg's
    ``variable_length_bytes`` ragged payloads, issue #209, and their
    ``{field}_locations`` sibling, issue #87) has no flat buffer to hash —
    ``ndarray.tobytes()`` on it serializes *pointer addresses*, which are
    per-process garbage. Such an array is instead hashed as, in flat C order
    over cells::

        sha256( for each cell:  uint64_le(len(payload)) || payload )

    where ``payload`` is that cell's decoded bytes (:func:`_element_bytes`)
    — for zagg that is exactly the little-endian buffer
    ``write._ragged_payload_bytes`` stores. The length prefix is what makes
    the digest injective (without it ``[b"ab", b"c"]`` and ``[b"a", b"bc"]``
    collide), and an empty cell contributes its zero length, so a ``b""``
    fill is distinguishable from a missing cell only by position — which is
    the intent: the digest covers the cell grid, not just the payloads.

    zagg has **no O11 writer yet**, so this recipe (and the committed
    fixture's golden) is the only artifact of the choice: zagg's writer must
    adopt it verbatim when it lands (spec home: englacial/zagg#340).
    """
    import zarr
    from zarr.core.sync import sync
    from zarr.storage import ObjectStore

    async def _metadata_keys(gen: Any) -> list[str]:
        # Filter in the stream: the listing covers every object under the leaf
        # (chunk payloads included), and only the metadata keys are wanted.
        return [key async for key in gen if key.endswith("/zarr.json")]

    handle = _resolve_store(store_root, store, store_kwargs)
    zstore = ObjectStore(handle, read_only=True)
    prefix = leaf.strip("/")
    # Walk the leaf by its ``zarr.json`` keys rather than ``Group.members``:
    # a zagg leaf carries non-zarr sidecar objects (the in-leaf
    # ``coverage.moc`` occupancy bitmap), which the recursive member probe
    # trips over — they are simply not arrays, so they are not in O11 scope.
    # The probe's failure is also backend-dependent (zarr's own LocalStore
    # warns and skips the sidecar; over obstore the same probe raises), so
    # this walk is the one behavior BOTH backends share. Each node's metadata
    # is read twice — once here, once by ``zarr.open_array`` — which is the
    # price of the tolerant ``node_type`` probe: opening first would raise on
    # a metadata object that is not a node at all.
    keys = sync(_metadata_keys(zstore.list_prefix(f"{prefix}/")))
    hashes = {}
    for meta_key in sorted(keys):
        node_path = meta_key[: -len("/zarr.json")]
        meta = read_json(handle, meta_key)
        if not isinstance(meta, dict) or meta.get("node_type") != "array":
            continue
        node = zarr.open_array(zstore, path=node_path, mode="r", zarr_format=3)
        key = node_path[len(prefix) + 1 :]
        values = np.ascontiguousarray(node[...])
        if values.dtype.kind == "O":  # vlen: length-prefixed payloads, C order
            digest = hashlib.sha256()
            for element in values.ravel(order="C"):
                payload = _element_bytes(element)
                digest.update(len(payload).to_bytes(VLEN_LENGTH_PREFIX, "little"))
                digest.update(payload)
            hashes[key] = digest.hexdigest()
            continue
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


#: Reserved (non-array) key of a FLAT ``content_hashes`` mapping. O11's
#: wording is "``{array_name: hash}`` plus one combined hash", so the flat
#: encoding a writer reaches for is arrays and combined in one mapping;
#: ``combined`` is not a legal zarr array name in any zagg output, so
#: reserving it costs nothing.
_COMBINED_KEY = "combined"


def _recorded_hashes(sidecar: dict | None) -> tuple[dict[str, str] | None, str | None]:
    """``(per-array, combined)`` recorded in a sidecar's ``content_hashes``.

    Accepts both shapes the D20/O11 wording admits — the nested
    ``{"arrays": {name: hash}, "combined": hash}`` envelope (what the
    committed fixture pins) and a flat ``{name: hash, "combined": hash}``
    mapping — so the verifier keeps working whichever zagg's writer lands.
    In the flat shape the reserved :data:`_COMBINED_KEY` is popped rather
    than read as a phantom array name (which would report an intact leaf as
    mismatched on ``combined``); a flat record carrying ONLY that key has no
    per-array hashes, so it reads as unverifiable (``None``), not tampered.
    An EMPTY ``arrays`` mapping reads the same way, for the same reason: the
    docstring's own framing is that unverifiable is distinct from verified,
    so "nothing was recorded" must not report as "tampered".
    """
    content = (sidecar or {}).get("content_hashes")
    if not isinstance(content, dict) or not content:
        return None, None
    if isinstance(content.get("arrays"), dict):
        return (dict(content["arrays"]) or None), content.get(_COMBINED_KEY)
    flat = dict(content)
    combined = flat.pop(_COMBINED_KEY, None)
    return (flat or None), combined


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
            "combined_match": ...,    # True/False; None = none recorded
            "mismatched": [...],      # array names differing (either side)
        }

    ``match`` is ``None`` when the sidecar is absent or records no
    ``content_hashes`` — "nothing to verify against" is a distinct answer
    from "verified" (the conservative posture zagg's dedup takes: an
    unverifiable leaf is never a hit).

    ``combined_match`` is the recorded-vs-recomputed combined-hash verdict,
    kept as its own greppable signal because it fails for a DIFFERENT reason
    than a per-array mismatch: the combined hash is derived from the
    per-array digests, so "every array matches but combined differs" is not
    a data problem at all — it is proof the writer's combined serialization
    diverges from :func:`combined_hash` (the choice this PR asks zagg to
    freeze). A ``False`` there also forces ``match`` to ``False``: a leaf
    whose recorded combined hash disagrees is never reported verified.
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
    combined = combined_hash(computed)
    mismatched: list[str] = []
    match: bool | None = None
    if recorded is not None:
        names = sorted(set(computed) | set(recorded))
        mismatched = [n for n in names if computed.get(n) != recorded.get(n)]
        match = not mismatched
    combined_match = None if recorded_combined is None else recorded_combined == combined
    if match and combined_match is False:
        match = False  # a disagreeing combined hash is never "verified"
    return {
        "leaf": rel,
        "computed": computed,
        "combined": combined,
        "recorded": recorded,
        "recorded_combined": recorded_combined,
        "match": match,
        "combined_match": combined_match,
        "mismatched": mismatched,
    }


__all__ = [
    "STATS_ROLLUP_NAME",
    "STATS_SIDECAR_NAME",
    "SWEEP_SPEC",
    "VLEN_LENGTH_PREFIX",
    "combined_hash",
    "hash_arrays",
    "read_stats",
    "read_stats_rollup",
    "stats_sidecar_key",
    "stats_sidecar_path",
    "verify_arrays",
]
