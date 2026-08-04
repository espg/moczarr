"""The morton-hive layout convention, read side (pure functions, no I/O).

The layout is owned by the mortie spec (espg/mortie#62); zagg writes it
(``zagg.hive``); moczarr reads it. Summary::

    {store_root}/
      morton_hive.json               <- static manifest; root-only exception
      coverage.moc                   <- root ranges MOC; root-only exception
      {sign+base}/{d1}/.../{d_n}/    <- digit component(s) per level
        {full_id}.zarr/              <- vanilla zarr v3 leaf
        {full_id}_{window}.zarr/     <- time-windowed leaf (morton-hive/2)

- Ids are morton decimal strings: sign + base digit (``1..6``), then one
  digit ``1..4`` per order. A string prefix is a spatial ancestor.
- Path components chunk the digit tail per the manifest's ``path_grouping``
  (spec §6.1; zagg D21): ``path_grouping`` digits per component, the LAST
  component carrying the remainder when the order does not divide evenly
  (leading components stay full-width, so component boundaries are ancestor
  prefixes shared by deeper shards regardless of their order). ``1`` — the
  default, and every existing store retroactively — is a value of the one
  generic chunking, never a separate code path; readers chunk per the
  manifest, never by assumption.
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
import warnings

import numpy as np

#: Manifest convention versions (``spec`` field). A ``/1`` store is a ``/2``
#: store with ``schedule: none``; ``/2`` adds the temporal block.
HIVE_SPEC = "morton-hive/1"
HIVE_SPEC_V2 = "morton-hive/2"
#: The D23 window-basename grammar (spec §6.4): leaf basename = time window
#: (``{window}.zarr``; reserved token ``all`` for ``schedule: none``), the
#: morton id recoverable from the digit path / stamp ``shard_key``. moczarr
#: consumes this spec string for SIDECAR-NAME arithmetic only so far
#: (:func:`moczarr.stats.stats_sidecar_key`); ``parse_manifest`` still
#: rejects it — the ``/3`` store opener lands with zagg's ``/3`` writer.
HIVE_SPEC_V3 = "morton-hive/3"
#: Root manifest object name.
MANIFEST_NAME = "morton_hive.json"
#: Root-group attrs key carrying the commit stamp.
COMMIT_ATTR = "morton_hive_commit"
#: In-leaf occupancy-bitmap sidecar object name; same name at the store root
#: holds the shard-order ranges MOC (different location, different encoding).
COVERAGE_SIDECAR = "coverage.moc"
ROOT_COVERAGE_NAME = "coverage.moc"

#: Spec §5: the permanent, self-declared zarr-convention UUID for
#: morton-declared stores (minted once; readers may key on it).
MORTON_CONVENTION_UUID = "3e22156d-ea9e-4e01-95fe-e3809a4b41e7"
#: Spec §5: the self-declared ``zarr_conventions`` entry, verbatim. The list
#: may carry other entries alongside (e.g. the generic dggs registry one).
MORTON_CONVENTION_ENTRY = {
    "schema_url": "https://github.com/espg/mortie/blob/main/docs/specification.md#dggs-attrs",
    "spec_url": "https://github.com/espg/mortie/blob/main/docs/specification.md",
    "uuid": MORTON_CONVENTION_UUID,
    "name": "morton-dggs",
    "description": "Packed-u64 morton (HEALPix) DGGS convention",
}

#: Frozen window-label charset (no ``_``, so leaf names split unambiguously).
_LABEL_RE = re.compile(r"^[0-9A-Za-z-]{1,32}$")
#: The reserved all-time token (zagg spec §4.2): the BASENAME of the
#: unwindowed (``schedule: none``) leaf and of the sweep's all-time overview
#: fold. It satisfies the label charset by design — it is a name, and names
#: share one grammar — but it is excluded from explicit window labels
#: forever, so it is never a legal ``window=`` argument
#: (:func:`validate_window`).
ALL_TOKEN = "all"

#: Spec §1 suffix bands: ``0..=27`` area (order == suffix), ``28..=47``
#: order-28/29 area preorder, ``48..=63`` order-29 POINT (no area claim).
_POINT_SUFFIX_MIN = 48
_SUFFIX_MASK = np.uint64(0x3F)


def is_point_word(word) -> bool | np.ndarray:
    """Whether packed word(s) encode an order-29 POINT (spec §1/§4).

    Kind is carried by the word's 6-bit suffix — ``48..=63`` is the point
    band; everything below (``0..=47``) is an AREA element, exact at its
    encoded order — never by store or array metadata (§4). Suffix-mask
    implementation, golden-tested against the spec §1 table. mortie's public
    predicate (``mortie.is_point``) DID land in the new floor — espg/mortie#116
    shipped it in 0.9.1 — and it is bit-identical across all 64 suffix values,
    pinned in ``tests/test_convention.py``. The local mask stays anyway: it is
    moczarr's independent read of the §1 table, and delegating would leave the
    golden test asserting mortie against itself rather than against the spec.
    Scalar in -> bool; array in -> bool array.
    """
    words = np.asarray(word, dtype=np.uint64)
    mask = (words & _SUFFIX_MASK) >= np.uint64(_POINT_SUFFIX_MIN)
    return bool(mask) if words.ndim == 0 else mask


def point_to_area29(word):
    """Order-29 AREA twin(s) of point word(s); area words pass through.

    Spec §1 suffix arithmetic on the same body: point ``48 + t28*4 + t29``
    -> area ``28 + t28*5 + (t29 + 1)``. The twin shares the point's full
    path, so containment arithmetic (§4: membership of a point at a coarser
    level is ordinary truncation) runs uniformly in area space — the
    normalization :func:`moczarr.coverage.aoi_mask` applies. Scalar in ->
    int; array in -> ``uint64`` array.
    """
    words = np.asarray(word, dtype=np.uint64)
    # Suffix arithmetic in int64 (values <= 63): negative intermediates for
    # area words are discarded by the where, without uint64 wrap warnings.
    suffix = (words & _SUFFIX_MASK).astype(np.int64)
    r2 = suffix - _POINT_SUFFIX_MIN
    area_suffix = (29 + (r2 >> 2) * 5 + (r2 & 3)).astype(np.uint64)
    twins = np.where(suffix >= _POINT_SUFFIX_MIN, (words & ~_SUFFIX_MASK) | area_suffix, words)
    return int(twins) if twins.ndim == 0 else twins.astype(np.uint64)


def area29_to_point(word):
    """Max-encoded POINT twin(s) of order-29 AREA word(s) — the ``p`` parse.

    Inverse of :func:`point_to_area29`: area ``28 + t28*5 + (t29 + 1)`` ->
    point ``48 + t28*4 + t29`` (spec §1). Raises on any word that is not an
    order-29 area encoding — points exist only at order 29 (§2/§4).
    """
    words = np.asarray(word, dtype=np.uint64)
    suffix = (words & _SUFFIX_MASK).astype(np.int64)  # int64: no wrap on invalid input
    r = suffix - 28
    # Order-29 area suffixes are 28 + t28*5 + (t29+1), t29 in 0..3 -> r % 5 != 0.
    ok = (suffix > 27) & (suffix < _POINT_SUFFIX_MIN) & (r % 5 != 0)
    if not np.all(ok):
        raise ValueError(
            "not an order-29 area word: the point twin exists only at order 29 (spec §1/§4)"
        )
    point_suffix = (_POINT_SUFFIX_MIN + (r // 5) * 4 + (r % 5 - 1)).astype(np.uint64)
    points = (words & ~_SUFFIX_MASK) | point_suffix
    return int(points) if points.ndim == 0 else points.astype(np.uint64)


def morton_word(label: str | int) -> int:
    """Packed ``uint64`` morton word of a decimal id (pass-through for ints).

    Accepts the §2 ``p`` kind-suffix on full order-29 POINT ids: the stem
    parses as the area word and the §1 point suffix is restored moczarr-side
    (:func:`area29_to_point`) — mortie 0.9.0 predates the ``p`` grammar
    (espg/mortie#120/#121). An UNMARKED order-29 string parses as the AREA
    word, the normative §4 tie-break. Rides mortie's private-but-documented
    ``_decimal_to_word`` (numpy-only; the public array classes require
    pandas — upstream ask for a public export stands, same note as zagg's
    boundary helper).
    """
    if isinstance(label, (int, np.integer)):
        return int(label)
    from mortie.morton_index import _decimal_to_word

    text = str(label)
    if text.endswith("p"):
        stem = text[:-1]
        if decimal_order(stem) != 29:
            raise ValueError(
                f"{label!r}: the 'p' kind-suffix is legal only on a full order-29 "
                f"POINT id (points exist only at order 29; spec §2)"
            )
        return int(area29_to_point(int(_decimal_to_word(stem))))
    return int(_decimal_to_word(text))


def morton_decimal(word: str | int) -> str:
    """Decimal morton string of a packed word (pass-through for strings).

    POINT words (§1 suffix ``48..=63``) render with the terminal ``p`` kind
    marker — the §2 render/interchange form — via the order-29 area twin's
    digits (same path; kind restored by the marker, so the round-trip is
    lossless for BOTH kinds, pinned by goldens).
    """
    if isinstance(word, str):
        return word
    from mortie import MortonIndexArray

    value, marker = int(word), ""
    if is_point_word(value):
        value, marker = point_to_area29(value), "p"
    rendered = MortonIndexArray.from_words(np.asarray([value], dtype=np.uint64)).decimal_repr()[0]
    return rendered + marker


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


def normalize_subtree(subtree) -> tuple[int, str, int]:
    """Normalize a ``subtree=`` argument to ``(word, decimal, order)``.

    Both currencies of the readers' ``subtree=`` keyword (issue #29, mirroring
    zagg's ``readers._layout`` pair): a packed morton AREA word (``int``) or a
    decimal morton string (``str``). An UNMARKED string always yields the area
    word (the §4 tie-break); a ``p``-marked order-29 id parses as the POINT
    word and is refused below, so no separate kind flag is needed.
    Raises ``ValueError``
    on a malformed member of either currency — an unparsable string, an int
    outside the uint64 range (not a packed word; parse a decimal id by
    passing it as a string), an invalid packed word, or an order-29 POINT
    word (kind rides
    the §1 suffix; a subtree names an AREA ancestor, and points have no
    descendants).
    """
    word = morton_word(subtree) if isinstance(subtree, str) else int(subtree)
    if not 0 <= word < 2**64:
        raise ValueError(
            f"subtree {subtree!r} is outside the uint64 range, not a packed morton "
            f"word; parse a decimal id by passing it as a string instead"
        )
    if is_point_word(word):
        raise ValueError(
            f"subtree {subtree!r} is an order-29 POINT word: a subtree names an "
            f"AREA ancestor (spec §1/§4 — points have no descendants)"
        )
    decimal = morton_decimal(word)  # raises on an invalid packed word
    return word, decimal, decimal_order(decimal)


def subtree_cell_span(
    subtree,
    anchor: int,
    anchor_index: int,
    cell_order: int,
    n_cells: int,
    field: str,
    *,
    stacklevel: int = 2,
) -> tuple[int, int]:
    """Cell-index span ``[start, stop)`` of ``subtree`` on a nested cells axis.

    The zagg spec §1.5 subtree-span identity, resolved by pure arithmetic
    (issue #29; logic-parallel with zagg ``readers/_layout.subtree_cell_span``):
    a cell's axis position is its HEALPix NESTED id, so the order-``k``
    subtree below a word with nested id ``h`` occupies ``[h·4^d, (h+1)·4^d)``
    (``d = cell_order - k``) in fullsphere coordinates. A single-root
    power-of-four axis (a hive leaf's shard subtree) subtracts its root's own
    span start, with the root derived from ``anchor`` — any WRITTEN cell word
    (the morton anchor slice the caller already read); no store bytes are
    touched here.

    That identity is CHECKED, not assumed: ``anchor`` sits at ``anchor_index``
    on the axis, so ``nested(anchor) - root_start`` must equal
    ``anchor_index`` or the axis is not in canonical nested placement and
    this arithmetic would hand back plausible-but-wrong cells silently — a
    mismatch raises ``ValueError``.

    The span is returned intersected with the axis: a word at or above the
    axis root clips to the whole axis (every stored cell is its descendant),
    and a well-formed word DISJOINT from the axis warns — naming the word and
    the axis root — and returns the empty span ``(0, 0)``. An empty read is
    therefore ambiguous between "in-domain, nothing stored" and "outside the
    domain" unless the warning is observed; DELIVERY is the caller's
    ``warnings`` filter's call (the default action dedups a repeat of the
    same warning in the same process), so the guarantee is "at most once per
    call". ``stacklevel`` is the caller's depth from here — the readers pass
    the depth that attributes the warning to THEIR caller rather than to a
    moczarr frame. Raises ``ValueError`` for a malformed word, or one deeper
    than the cells-axis order (grammatically impossible — nothing stored
    could be its descendant).
    """
    from mortie import clip2order, mort2healpix

    word, decimal, order = normalize_subtree(subtree)
    if order > cell_order:
        raise ValueError(
            f"subtree {decimal} is order {order} — deeper than {field!r}'s "
            f"order-{cell_order} cells axis, so nothing stored can be its descendant "
            f"(a subtree names an ancestor, at any order up to the cell order)"
        )
    d = cell_order - order
    (h,), _o = mort2healpix(np.asarray([word], dtype=np.uint64))
    lo = int(h) * 4**d
    n = int(n_cells)
    depth = (n.bit_length() - 1) // 2
    # A fullsphere axis (12·4^c cells) starts at 0 — the nested id IS the axis
    # position, and every well-formed word's span lies inside it. A single-root
    # power-of-four axis starts at its root's own span start.
    single_root = 4**depth == n
    root_order, root, root_start = cell_order - depth, 0, 0
    if single_root:
        root = int(clip2order(root_order, np.asarray([anchor], dtype=np.uint64))[0])
        (rh,), _ro = mort2healpix(np.asarray([root], dtype=np.uint64))
        root_start = int(rh) * 4**depth
    (ah,), _ao = mort2healpix(np.asarray([anchor], dtype=np.uint64))
    if int(ah) - root_start != int(anchor_index):
        raise ValueError(
            f"{field!r}'s cells axis is not in canonical nested placement: cell "
            f"{int(anchor_index)} carries morton {morton_decimal(int(anchor))}, whose "
            f"nested id puts it at axis position {int(ah) - root_start}. A subtree "
            f"span is derived arithmetically (cell position == nested id minus the "
            f"axis root's start), so this axis would yield the wrong cells"
        )
    if not single_root:
        return lo, lo + 4**d
    lo -= root_start
    start, stop = max(lo, 0), min(lo + 4**d, n)
    if start >= stop:
        warnings.warn(
            f"subtree {decimal} is outside this axis' order-{root_order} root "
            f"{morton_decimal(root)} — yielding nothing",
            stacklevel=stacklevel,
        )
        return 0, 0
    return start, stop


def is_base_component(name: str) -> bool:
    """Whether ``name`` is a ``{sign+base}``-shaped hive root child."""
    base = name[1:] if name.startswith("-") else name
    return len(base) == 1 and base in "123456"


def group_digits(digits: str, path_grouping: int) -> list[str]:
    """Chunk a digit tail into hive path components (spec §6.1, zagg D21).

    Left to right, ``path_grouping`` digits per component; the LAST component
    carries the remainder when ``len(digits) % path_grouping != 0``. Leading
    components stay full-width so every component boundary is a spatial
    ancestor prefix (§2) shared by deeper shards regardless of their order —
    mixed-order shards share ancestor nodes, the property the digit tree
    exists for.
    """
    return [digits[i : i + path_grouping] for i in range(0, len(digits), path_grouping)]


def manifest_path_grouping(manifest: dict) -> int:
    """The manifest's ``path_grouping`` (D21: absent reads as ``1``).

    Assumes a :func:`parse_manifest`-validated manifest; the single accessor
    keeps the absent->1 normalization in one place.
    """
    return int(manifest.get("path_grouping", 1))


def validate_label(label: str) -> str:
    """Validate a window label against the frozen charset; returns it."""
    if not isinstance(label, str) or not _LABEL_RE.match(label):
        raise ValueError(
            f"window label {label!r} does not match the frozen grammar "
            f"({_LABEL_RE.pattern}; morton-hive/2, mortie#62)"
        )
    return label


def validate_window(label: str, *, where: str | None = None) -> str:
    """Validate an explicit ``window=`` label; returns it.

    :func:`validate_label` plus the one exclusion the charset cannot carry —
    the reserved all-time token :data:`ALL_TOKEN`. ``all`` satisfies the
    grammar because it is a *basename* (the unwindowed leaf's, and the
    sweep's all-time overview fold's), so a reader entry point that only
    validates the charset resolves it to objects that are not that window's:
    ``open_hive`` names ``{shard}_all.zarr`` leaves that do not exist and
    quietly empties, while an overview order would open the all-time folds
    beside a 0-cell source node. Three behaviors for one reserved token is an
    API trap (espg/moczarr#30), so every ``window=`` seam refuses it here,
    with one message.

    Name arithmetic that legitimately spells ``all`` — the overview sidecar
    stem in :func:`moczarr.stats.stats_sidecar_key`, and the telemetry
    readers that address an ``all.zarr`` object by name — keeps calling
    :func:`validate_label`: reading one record cannot mislead about coverage
    the way an all-time dataset node would.
    """
    validate_label(label)
    if label == ALL_TOKEN:
        at = f" at {where}" if where else ""
        raise ValueError(
            f"window={label!r}{at}: {ALL_TOKEN!r} is the reserved all-time token (zagg "
            f"spec §4.2 — excluded from the window grammar forever), not a window label. "
            f"The all-time folds (pyramid.overview.all_time, §4.5) exist on disk but are "
            f"not yet a reader surface: the source axis has no all-time leaf to pair them "
            f"with, so opening them alone would mislead. Pass a declared window label "
            f"(espg/moczarr#31 tracks the opt-in surface)"
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


def leaf_path(shard: str | int, window: str | None = None, *, path_grouping: int = 1) -> str:
    """Store-relative hive path of a shard's leaf zarr.

    ``path_grouping`` is the manifest's digit-chunking (spec §6.1; default
    ``1`` — every pre-D21 store). At ``1`` the path is computed by mortie's
    ``hive_path`` (the convention owner) and re-checked against the node
    invariant so drift on either side fails loudly; the grouped form chunks
    the decimal here (:func:`group_digits`) until mortie grows the grouped
    ``hive_path`` (espg/mortie#62). ``window`` selects the time-windowed
    leaf at the same node.
    """
    from mortie import MortonIndexArray

    word = morton_word(shard)
    if word < 0:
        raise ValueError(
            f"shard {shard!r} is a negative int, not a packed morton word; a "
            f"packed morton word is required. Parse a decimal id by passing it "
            f"as a string (e.g. morton_word('-5112333')) instead."
        )
    if is_point_word(word):
        raise ValueError(
            f"shard {shard!r} is an order-29 POINT word: points never live in "
            f"hive paths (spec §2/§6.6)"
        )
    if path_grouping == 1:
        rel = MortonIndexArray.from_words(np.asarray([word], dtype=np.uint64)).hive_path()[0]
        if window is not None:
            node, _sep, bare = rel.rpartition("/")
            rel = f"{node}/{leaf_name(bare.removesuffix('.zarr'), window)}"
    else:
        decimal = morton_decimal(word)
        base = decimal_base(decimal)
        components = [base, *group_digits(decimal[len(base) :], path_grouping)]
        rel = f"{'/'.join(components)}/{leaf_name(decimal, window)}"
    check_node_invariant(rel, path_grouping=path_grouping)
    return rel


def check_node_invariant(rel_path: str, *, path_grouping: int = 1) -> None:
    """Raise unless ``rel_path`` is a legal hive leaf path.

    Below the root only digit components are allowed — ``{sign+base}``
    (optional ``-``, one digit ``1..6``) at the first level, then digit
    (``1..4``) components chunked per ``path_grouping`` (spec §6.1): every
    component full-width except the last, which carries the remainder —
    terminating in ``{full_id}.zarr`` (or the windowed
    ``{full_id}_{window}.zarr``) whose id equals the concatenated
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
        if full_id is not None and full_id.endswith("p"):
            raise ValueError(
                f"path {rel_path!r} carries the 'p' point kind-suffix: paths never "
                f"do (spec §2/§6.6 — an order-29 path component reads as AREA per "
                f"the §4 tie-break; kind rides the packed words)"
            )
        ok = is_base_component(head)
        ok = ok and all(1 <= len(d) <= path_grouping and set(d) <= set("1234") for d in digits)
        # Only the LAST component may be short (the remainder rides last).
        ok = ok and all(len(d) == path_grouping for d in digits[:-1])
        ok = ok and full_id == head + "".join(digits)
    if not ok:
        raise ValueError(
            f"path {rel_path!r} violates the hive node invariant (path_grouping={path_grouping})"
        )


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
    grouping = payload.get("path_grouping", 1)
    if isinstance(grouping, bool) or not isinstance(grouping, int) or grouping < 1:
        # Spec §6.1 defines path_grouping as an integer digit count (absent
        # reads as 1, D21); no list form exists in the spec grammar.
        raise ValueError(f"manifest path_grouping must be an integer >= 1 (got {grouping!r})")
    temporal = payload.get("temporal")
    if spec == HIVE_SPEC_V2:
        if not isinstance(temporal, dict) or not temporal.get("schedule"):
            raise ValueError(f"a {HIVE_SPEC_V2} manifest requires a temporal block with a schedule")
    elif temporal is not None:
        raise ValueError(f"a {HIVE_SPEC} manifest must not carry a temporal block")
    return payload
