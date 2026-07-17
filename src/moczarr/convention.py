"""The morton-hive layout convention, read side (pure functions, no I/O).

The layout is owned by the mortie spec (espg/mortie#62); zagg writes it
(``zagg.hive``); moczarr reads it. Summary::

    {store_root}/
      morton_hive.json               <- static manifest; root-only exception
      coverage.moc                   <- root ranges MOC; root-only exception
      {sign+base}/{d1}/.../{d_n}/    <- one decimal digit per level
        {full_id}.zarr/              <- vanilla zarr v3 leaf
        {full_id}_{window}.zarr/     <- time-windowed leaf (morton-hive/2)

- Ids are morton decimal strings: sign + base digit (``1..6``), then one
  digit ``1..4`` per order. A string prefix is a spatial ancestor.
- Below the root a node holds only digit children and ``*.zarr`` objects
  (the node invariant); the manifest and root ``coverage.moc`` are the two
  root-only exceptions.
- A leaf is complete iff its root zarr attrs carry the commit stamp
  (``morton_hive_commit``); an unstamped ``.zarr/`` prefix is debris.
- Windowed leaf names split on the FIRST ``_`` (morton ids and window labels
  never contain one); labels use the frozen charset ``[0-9A-Za-z-]{1,32}``.

Everything here is arithmetic on ids and dict validation — the store layer
(phase 2) supplies the bytes. Golden vectors in ``tests/`` pin this
implementation against zagg's writer so the two cannot drift silently.
"""

from __future__ import annotations

import re

import numpy as np

#: Manifest convention versions (``spec`` field). A ``/1`` store is a ``/2``
#: store with ``schedule: none``; ``/2`` adds the temporal block.
HIVE_SPEC = "morton-hive/1"
HIVE_SPEC_V2 = "morton-hive/2"
#: Root manifest object name.
MANIFEST_NAME = "morton_hive.json"
#: Root-group attrs key carrying the commit stamp.
COMMIT_ATTR = "morton_hive_commit"
#: In-leaf occupancy-bitmap sidecar object name; same name at the store root
#: holds the shard-order ranges MOC (different location, different encoding).
COVERAGE_SIDECAR = "coverage.moc"
ROOT_COVERAGE_NAME = "coverage.moc"

#: Frozen window-label charset (no ``_``, so leaf names split unambiguously).
_LABEL_RE = re.compile(r"^[0-9A-Za-z-]{1,32}$")


def morton_word(label: str | int) -> int:
    """Packed ``uint64`` morton word of a decimal id (pass-through for ints).

    Rides mortie's private-but-documented ``_decimal_to_word`` (numpy-only;
    the public array classes require pandas — upstream ask for a public
    export stands, same note as zagg's boundary helper).
    """
    if isinstance(label, (int, np.integer)):
        return int(label)
    from mortie.morton_index import _decimal_to_word

    return int(_decimal_to_word(str(label)))


def morton_decimal(word: str | int) -> str:
    """Decimal morton string of a packed word (pass-through for strings)."""
    if isinstance(word, str):
        return word
    from mortie import MortonIndexArray

    return MortonIndexArray.from_words(np.asarray([int(word)], dtype=np.uint64)).decimal_repr()[0]


def decimal_order(decimal: str) -> int:
    """HEALPix order of a decimal id (one digit per level past the base)."""
    return len(decimal) - (2 if decimal.startswith("-") else 1)


def decimal_base(decimal: str) -> str:
    """The ``{sign+base}`` component of a decimal id."""
    return decimal[:2] if decimal.startswith("-") else decimal[:1]


def decimal_rank(decimal: str) -> int:
    """Base-4 value of a decimal id's digit tail (digits ``1..4`` -> ``0..3``).

    The bit/rank convention of the coverage encodings: ascending packed-word
    (Z-)order within one base cell at a fixed order.
    """
    rank = 0
    for ch in decimal[len(decimal_base(decimal)) :]:
        rank = rank * 4 + (int(ch) - 1)
    return rank


def rank_tail(rank: int, depth: int) -> str:
    """Inverse of :func:`decimal_rank`: the width-``depth`` digit tail."""
    digits = []
    for _ in range(depth):
        digits.append(str(rank % 4 + 1))
        rank //= 4
    return "".join(reversed(digits))


def is_base_component(name: str) -> bool:
    """Whether ``name`` is a ``{sign+base}``-shaped hive root child."""
    base = name[1:] if name.startswith("-") else name
    return len(base) == 1 and base in "123456"


def validate_label(label: str) -> str:
    """Validate a window label against the frozen charset; returns it."""
    if not isinstance(label, str) or not _LABEL_RE.match(label):
        raise ValueError(
            f"window label {label!r} does not match the frozen grammar "
            f"({_LABEL_RE.pattern}; morton-hive/2, mortie#62)"
        )
    return label


def leaf_name(full_id: str, window: str | None = None) -> str:
    """The leaf zarr basename: ``{full_id}_{window}.zarr``, or bare."""
    if window is None:
        return f"{full_id}.zarr"
    validate_label(window)
    return f"{full_id}_{window}.zarr"


def split_leaf_name(name: str) -> tuple[str, str | None]:
    """``(full_id, window-or-None)`` from a leaf basename — split on the FIRST ``_``.

    Morton decimal ids never contain ``_`` and window labels cannot
    (charset), so the first underscore is the one separator. Raises on a
    non-``.zarr`` name or a malformed window label.
    """
    if not name.endswith(".zarr"):
        raise ValueError(f"{name!r} is not a leaf zarr name")
    stem = name.removesuffix(".zarr")
    if "_" not in stem:
        return stem, None
    full_id, window = stem.split("_", 1)
    validate_label(window)
    return full_id, window


def leaf_path(shard: str | int, window: str | None = None) -> str:
    """Store-relative hive path of a shard's leaf zarr.

    Computed by mortie's ``hive_path`` (the convention owner) and re-checked
    against the node invariant so drift on either side fails loudly.
    ``window`` selects the time-windowed leaf at the same node.
    """
    from mortie import MortonIndexArray

    word = morton_word(shard)
    rel = MortonIndexArray.from_words(np.asarray([word], dtype=np.uint64)).hive_path()[0]
    if window is not None:
        node, _sep, bare = rel.rpartition("/")
        rel = f"{node}/{leaf_name(bare.removesuffix('.zarr'), window)}"
    check_node_invariant(rel)
    return rel


def check_node_invariant(rel_path: str) -> None:
    """Raise unless ``rel_path`` is a legal hive leaf path.

    Below the root only digit components are allowed — ``{sign+base}``
    (optional ``-``, one digit ``1..6``) at the first level, one ``1..4``
    digit per level after — terminating in ``{full_id}.zarr`` (or the
    windowed ``{full_id}_{window}.zarr``) whose id equals the concatenated
    components. This is the walker's contract: any other name under the root
    (bar the manifest and the root ``coverage.moc``) breaks child
    classification.
    """
    parts = rel_path.strip("/").split("/")
    leaf = parts[-1]
    ok = len(parts) >= 2 and leaf.endswith(".zarr")
    if ok:
        head, digits = parts[0], parts[1:-1]
        try:
            full_id, _window = split_leaf_name(leaf)
        except ValueError:
            full_id = None  # malformed window label -> not a legal leaf
        ok = is_base_component(head)
        ok = ok and all(len(d) == 1 and d in "1234" for d in digits)
        ok = ok and full_id == head + "".join(digits)
    if not ok:
        raise ValueError(f"path {rel_path!r} violates the hive node invariant")


def parse_manifest(payload: object) -> dict:
    """Validate a ``morton_hive.json`` payload; returns it as a dict.

    Loud on malformed input (a manifest is the reader's bootstrap — there is
    no degraded mode without it): unknown ``spec``, missing/non-integer
    orders, a ``/2`` manifest without its temporal block, or a temporal block
    without a schedule all raise ``ValueError``.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a mapping: {type(payload).__name__}")
    spec = payload.get("spec")
    if spec not in (HIVE_SPEC, HIVE_SPEC_V2):
        raise ValueError(f"unknown manifest spec {spec!r} (expected {HIVE_SPEC} or {HIVE_SPEC_V2})")
    for key in ("cell_order", "shard_order"):
        value = payload.get(key)
        if not isinstance(value, int):
            raise ValueError(f"manifest {key} must be an integer (got {value!r})")
    if payload["cell_order"] < payload["shard_order"]:
        raise ValueError(
            f"manifest cell_order {payload['cell_order']} is above shard_order "
            f"{payload['shard_order']} (cells nest inside shards)"
        )
    temporal = payload.get("temporal")
    if spec == HIVE_SPEC_V2:
        if not isinstance(temporal, dict) or not temporal.get("schedule"):
            raise ValueError(f"a {HIVE_SPEC_V2} manifest requires a temporal block with a schedule")
    elif temporal is not None:
        raise ValueError(f"a {HIVE_SPEC} manifest must not carry a temporal block")
    return payload
