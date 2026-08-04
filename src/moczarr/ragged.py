"""Generic ``zagg-ragged/1`` vlen decode layer — product-agnostic (issue #19).

The read side of zagg's ragged store spec (``docs/specification.md`` §1 in
englacial/zagg — normative on ``main``, re-checked at ``9e11e65``, the
englacial/zagg#346 rebase merge): a ``kind: ragged`` field is ONE
``variable_length_bytes`` zarr v3 array on the cells axis. Each populated
cell holds the raw **little-endian** bytes of its ``(n, *inner_shape)``
payload; empty cells keep the ``b""`` fill; an all-empty inner chunk is
omitted from disk entirely. The element interpretation is self-describing in
the payload array's attrs under the versioned ``ragged`` block (§1.2), and a
**located** field declares its row-aligned ``uint64`` sibling there too —
readers bind the sibling by metadata, never by reconstructing the
``{field}_locations`` naming convention.

Zero product knowledge lives here: this module decodes any conforming
element declaration (a t-digest's ``float32 (n, 2)``, a locations sibling's
``uint64 (n,)``, or anything else a future writer declares) and never
imports zagg. The genericity is over the ELEMENT, and the scope is bounded
on the indexing side: :func:`read_ragged` and the profiles read a 1-D cells
axis with a sibling ``morton`` coordinate — every conforming ragged field of
a **morton-hive (HEALPix) product**, which is moczarr's whole domain. zagg's
rectilinear grid writes ``kind: ragged`` fields too, but on a 2-D ``(y, x)``
cell grid with no per-cell morton coordinate (its shard keys are packed
``(rb, cb)`` tile ints), so a rect ragged field is out of scope for this
layer and would need its own cell-identity binding. The element gates
(:func:`parse_ragged_attrs`, :func:`decode_cell`) carry no such assumption.
The HHDC tensor profile (:mod:`moczarr.hhdc`) composes on top.

The sweeps accept ``subtree=`` (issue #29): the nested cells axis makes
"everything below one morton node" a contiguous index span (zagg spec §1.5),
so a restricted read fetches only the stored objects covering that span —
never a whole-array sweep. Span grammar in
:func:`moczarr.convention.subtree_cell_span`.

Postures, inherited from the spec's conformance rules:

- **Strict attrs gate.** ``spec`` is strict-checked: a missing block, a
  foreign marker, or a newer revision raises — a reader must never
  half-parse under a guessed layout (a mis-guessed element dtype
  misinterprets every payload). Contrast the coverage envelopes' tolerant
  ``None``: those are caches that degrade to the walk; ragged attrs are
  truth. The one marker-absence carve-out is the ``/2`` revision, whose
  typed dtype IS the signal (§1.6/§6.1): :func:`open_ragged` names it from
  the declared data type *before* the attrs gate, so a ``/2`` array is
  refused as a newer revision rather than mis-diagnosed as unsignaled.
- **One code path for both storage geometries** (§1.5). Sharded
  (``sharding_indexed``; every hive leaf) and per-inner-chunk (regular
  array; the unsharded flat path) layouts are self-describing in the
  array's own metadata, so the sweep derives the stored span from
  ``arr.shards`` when present, else ``arr.chunks``.
- **Debris-tolerant listing.** The sweep LISTs the array's ``c/`` prefix,
  so unlike zagg's own open-by-name readers it *does* meet the in-leaf
  debris the spec's §5.1 posture names (a foreign array prefix, a
  ``.zarr.status/`` prefix, editor/OS files). A key under ``c/`` that is not
  a chunk ordinal is skipped with a warning — never coerced to an int.
- **Store-scoped.** The readers never traverse the hive digit tree — leaf
  discovery stays with :func:`moczarr.open_hive` / the coverage MOC. Open
  the leaf store and pass the in-leaf array path (e.g. ``"6/h_tdigest"``).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import zarr
from zarr.abc.store import Store

from moczarr.convention import (
    decimal_order,
    morton_decimal,
    normalize_subtree,
    subtree_cell_span,
)

__all__ = [
    "RAGGED2_DATA_TYPE",
    "RAGGED2_SPEC",
    "RAGGED_ATTR",
    "RAGGED_SPEC",
    "RaggedElement",
    "decode_cell",
    "iter_populated_chunks",
    "open_ragged",
    "parse_ragged_attrs",
    "read_cell",
    "read_ragged",
    "stored_chunk_spans",
]

#: Attrs key of the versioned element declaration (spec §1.2).
RAGGED_ATTR = "ragged"
#: The one convention revision this layer decodes.
RAGGED_SPEC = "zagg-ragged/1"
#: The successor revision (spec §6), signaled by the DATA TYPE rather than by
#: an attrs marker: on a ``/2`` array the declaration lives in the typed
#: :data:`RAGGED2_DATA_TYPE` dtype and the ``ragged`` marker is deliberately
#: retired (§1.6/§6.1/§6.3), so marker-absence there is a newer revision — not
#: an unsignaled array. :func:`open_ragged` names it before the attrs gate.
RAGGED2_SPEC = "zagg-ragged/2"
#: The registered zarr v3 extension data type that IS the ``/2`` signal.
RAGGED2_DATA_TYPE = "vlen-ndarray"


@dataclass(frozen=True)
class RaggedElement:
    """One array's parsed §1.2 element declaration.

    Attributes
    ----------
    dtype : numpy.dtype
        Element dtype, coerced little-endian (the wire byte order,
        independent of the producing machine).
    inner_shape : tuple of int
        Trailing element shape; a cell decodes as ``(n, *inner_shape)``.
    locations : str or None
        Sibling array name carrying the per-row ``uint64`` location words
        (located fields only, spec §1.1/§2.2); ``None`` when unlocated.
    """

    dtype: np.dtype
    inner_shape: tuple[int, ...]
    locations: str | None = None


def parse_ragged_attrs(attrs: Mapping | None, *, field: str = "<array>") -> RaggedElement:
    """The strict §1.2 gate: attrs → :class:`RaggedElement`, or raise.

    ``field`` names the array in errors. Raises ``ValueError`` when the
    ``ragged`` block is missing (not a ``zagg-ragged/1`` array — pre-spec CSR
    stores are a hard break), when ``spec`` is foreign or a future revision
    (never half-parse), or when the element declaration is malformed
    (``shape`` must be ``[-1, *inner_shape]``). The missing-block branch
    assumes the caller has already ruled out :data:`RAGGED2_SPEC`, whose
    marker is retired by design — :func:`open_ragged` does that by dtype.
    """
    block = attrs.get(RAGGED_ATTR) if isinstance(attrs, Mapping) else None
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{field!r} carries no ragged element declaration "
            f'(attrs["{RAGGED_ATTR}"]); it is not a {RAGGED_SPEC!r} array — refusing '
            f"to decode under a guessed layout (pre-spec CSR stores are a hard break)"
        )
    if block.get("spec") != RAGGED_SPEC:
        raise ValueError(
            f"{field!r} declares ragged spec {block.get('spec')!r}; this reader "
            f"understands {RAGGED_SPEC!r} only — an unknown or future revision must "
            f"be adopted deliberately, never half-parsed"
        )
    element = block.get("element")
    if not isinstance(element, Mapping) or "dtype" not in element or "shape" not in element:
        raise ValueError(
            f"{field!r} has a malformed element declaration "
            f'(attrs["{RAGGED_ATTR}"]["element"] needs "dtype" and "shape")'
        )
    try:
        dtype = np.dtype(str(element["dtype"])).newbyteorder("<")
    except TypeError as exc:
        raise ValueError(
            f"{field!r} declares an unreadable element dtype {element['dtype']!r}"
        ) from exc
    shape = [int(s) for s in element["shape"]]
    if not shape or shape[0] != -1 or any(s < 0 for s in shape[1:]):
        raise ValueError(
            f"{field!r} declares element shape {element['shape']!r}; the spec §1.2 "
            f"form is [-1, *inner_shape] (the -1 marks the per-cell varying count)"
        )
    locations = block.get("locations")
    return RaggedElement(
        dtype=dtype,
        inner_shape=tuple(shape[1:]),
        locations=str(locations) if locations else None,
    )


def decode_cell(raw: object, element: RaggedElement) -> np.ndarray:
    """One cell's ``(n, *inner_shape)`` payload from its raw vlen bytes.

    The §1.2 reconstruction: ``np.frombuffer(raw, dtype).reshape(-1,
    *inner_shape)`` with the dtype read little-endian. ``None`` and ``b""``
    (an absent/empty cell) decode to the zero-length ``(0, *inner_shape)``
    array.

    The result is a read-only, non-owning VIEW of those bytes
    (``flags.writeable`` and ``flags.owndata`` both False — parity-faithful
    with zagg's ``_decode_cell``): in-place work on a decoded payload raises,
    and keeping one alive pins its span's buffer. Copy if you need either.
    """
    data = b"" if raw is None else bytes(raw)  # type: ignore[call-overload]
    return np.frombuffer(data, dtype=element.dtype).reshape((-1, *element.inner_shape))


def _data_type_name(declared: object) -> str | None:
    """The NAME of a declared zarr ``data_type`` (a bare string or named config)."""
    if isinstance(declared, Mapping):
        declared = declared.get("name")
    return declared if isinstance(declared, str) else None


def _declared_data_type(store: Store, field: str) -> str | None:
    """The array's declared ``data_type`` name, read from its raw metadata.

    The fallback diagnosis path for an array this zarr stack cannot open at
    all: a ``/2`` typed dtype without the extension installed makes
    ``zarr.open_array`` raise before any attrs or dtype are visible, so the
    revision has to come from the metadata object itself.
    """
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    key = f"{field}/zarr.json" if field else "zarr.json"
    buf = sync(store.get(key, prototype=default_buffer_prototype()))
    if buf is None:
        return None
    try:
        meta = json.loads(buf.to_bytes())
    except ValueError:
        return None
    return _data_type_name(meta.get("data_type") if isinstance(meta, Mapping) else None)


def _ragged2_error(field: str) -> ValueError:
    """The pointed ``/2`` refusal (spec §6.3: actionable, never a mis-decode)."""
    return ValueError(
        f"{field!r} is a {RAGGED2_SPEC!r} array: its declared data type is the typed "
        f"{RAGGED2_DATA_TYPE!r} extension, which carries the element declaration and "
        f"retires the {RAGGED_ATTR!r} attrs marker (spec §1.6/§6.1). This reader "
        f"decodes {RAGGED_SPEC!r} only — reading a {RAGGED2_SPEC!r} store needs the "
        f"'zarr-vlen-ndarray' dtype extension and a moczarr revision that dispatches "
        f"on it (englacial/zagg#210); {RAGGED_SPEC!r} stores stay valid indefinitely"
    )


def _is_empty(raw: object) -> bool:
    """Whether a vlen cell carries no payload — ``b""`` **or** ``None``.

    The §5.2 normalization :func:`decode_cell` documents (an unwritten vlen
    cell may decode as ``None``, not ``b""``) as a predicate, so every
    populated-cell test in this module honours the same contract instead of
    calling ``len()`` on a value the decoder tolerates.
    """
    return raw is None or not len(raw)  # type: ignore[arg-type]


def open_ragged(
    store: Store,
    field: str,
    *,
    zarr_format: Literal[2, 3] = 3,
) -> tuple[zarr.Array, RaggedElement]:
    """Open a ragged vlen array and its parsed element declaration, strictly.

    Raises ``ValueError`` when the array is not a 1-D vlen-bytes cells axis,
    fails the :func:`parse_ragged_attrs` gate, or is the :data:`RAGGED2_SPEC`
    revision — which is ruled out BEFORE marker absence is blamed on a
    pre-spec store, since a ``/2`` array retires the marker by design and
    deserves the actionable error §6.3 mandates instead. The declared data
    type is read from the metadata object only on those two paths (an
    unopenable dtype, or a missing marker), never on the happy one.
    """
    try:
        arr = zarr.open_array(store, path=field, mode="r", zarr_format=zarr_format)
    except ValueError as exc:
        # The typed ``/2`` dtype is unknown to a zarr stack without the
        # extension, so zarr refuses the array before any attrs are visible.
        if _declared_data_type(store, field) == RAGGED2_DATA_TYPE:
            raise _ragged2_error(field) from exc
        raise
    attrs = dict(arr.attrs)
    if RAGGED_ATTR not in attrs and _declared_data_type(store, field) == RAGGED2_DATA_TYPE:
        raise _ragged2_error(field)  # marker absence is legal on ``/2`` only
    element = parse_ragged_attrs(attrs, field=field)
    if arr.dtype.kind != "O":
        raise ValueError(
            f"{field!r} carries a ragged declaration but its zarr dtype is "
            f"{arr.dtype!s}, not variable-length bytes — not a {RAGGED_SPEC!r} array"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{field!r} has {arr.ndim} dimensions; this layer decodes one vlen array "
            f"on the 1-D morton cells axis (a rect-grid {RAGGED_SPEC!r} field is 2-D "
            f"and out of scope — it carries no per-cell morton coordinate)"
        )
    return arr, element


def stored_chunk_spans(arr: zarr.Array) -> list[tuple[int, int]]:
    """Cell spans ``(start, stop)`` of the array's STORED objects, ascending.

    The whole-store read plan: one LIST of the array's ``c/<ordinal>`` data
    keys instead of probing every chunk of a mostly-fill array. Under the
    ``sharding_indexed`` codec an object spans a whole shard
    (``arr.shards``); on the per-inner-chunk flat layout it is one read
    chunk (``arr.chunks``) — both self-describing in the array's own
    metadata, so the two §1.5 geometries share this path.
    ``zarr.core.sync.sync`` runs the store's async listing on zarr's own
    event loop (zarr 3 exposes no public sync listing).

    Because this path LISTs, it meets in-leaf debris the spec's §5.1 posture
    names as expected (a foreign array prefix, a ``.zarr.status/`` prefix,
    editor/OS files): a key under ``c/`` that is not a chunk ordinal is
    **skipped with a warning**, never coerced to an int. Debris is inert for
    zagg's own readers because they open arrays by name; a LISTing reader has
    to state the same tolerance in code.
    """
    from zarr.core.sync import sync

    async def _collect(gen) -> list:
        return [key async for key in gen]

    span = int((arr.shards or arr.chunks)[0])
    prefix = f"{arr.path}/c/" if arr.path else "c/"
    keys = sync(_collect(arr.store_path.store.list_prefix(prefix)))
    ordinals = sorted(int(k[len(prefix) :]) for k in keys if k[len(prefix) :].isdigit())
    debris = sorted(k for k in keys if not k[len(prefix) :].isdigit())
    if debris:
        warnings.warn(
            f"skipping {len(debris)} object(s) under {prefix!r} that are not chunk "
            f"ordinals ({', '.join(debris[:3])}): in-leaf debris (a foreign array "
            f"prefix, a '.zarr.status/' prefix, editor/OS files) is not chunk data. "
            f"Note a RENAMED data object reads as an absent chunk, i.e. dropped cells",
            UserWarning,
            stacklevel=2,
        )
    return [(o * span, min((o + 1) * span, int(arr.shape[0]))) for o in ordinals]


def iter_populated_chunks(
    arr: zarr.Array,
    span: tuple[int, int] | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> Iterator[tuple[int, list[tuple[int, object]]]]:
    """Yield ``(chunk_start, [(rank, raw_bytes), ...])`` per populated read chunk.

    The bulk-sweep primitive shared by :func:`read_ragged` and the profile
    readers (:mod:`moczarr.hhdc`). Restricted to the stored objects
    (:func:`stored_chunk_spans`), each read in ONE slice: slicing per inner
    chunk would re-fetch a sharded object's index suffix on every
    ``__getitem__`` (~K redundant GETs per shard), while the full-span slice
    fetches the stored object once (zarr reads a whole outer chunk in a
    single GET and splits it locally). The held cost is one span's decoded
    payload. Chunks whose cells are all absent are skipped — absence is
    ``b""`` or ``None`` (:func:`_is_empty`, the §5.2 normalization), and
    includes the §1.5 shard-index absence sentinel. ``rank`` is the cell's
    chunk-local NESTED rank — the same index the writer placed it at on the
    nested-ordered cells axis, never a row-major position.

    ``span`` (issue #29) restricts the sweep to the stored objects
    overlapping the ``[start, stop)`` cell span, clipped to whole read
    chunks — the spec §1.5 contiguous-slice read plan: only the covering
    portion of each overlapping object is sliced, and non-overlapping
    stored objects are skipped without a fetch. ``spans`` threads in an
    ALREADY-LISTED :func:`stored_chunk_spans` result (the one
    :func:`_subtree_span` resolved against) so a restricted read LISTs the
    array's data keys once, not twice; ``None`` lists here.
    """
    cells_per_chunk = int(arr.chunks[0])
    for span_start, span_stop in stored_chunk_spans(arr) if spans is None else spans:
        if span is not None:
            span_start = max(span_start, span[0] - span[0] % cells_per_chunk)
            span_stop = min(span_stop, span[1] + -span[1] % cells_per_chunk)
            if span_start >= span_stop:
                continue
        data = cast(np.ndarray, arr[span_start:span_stop])
        for offset in range(0, span_stop - span_start, cells_per_chunk):
            block = data[offset : offset + cells_per_chunk]
            populated = [
                (pos, block[pos]) for pos in range(len(block)) if not _is_empty(block[pos])
            ]
            if populated:
                yield span_start + offset, populated


def _cells_order(words: np.ndarray, field: str, start: int) -> int:
    """HEALPix order of a chunk's written cell words (the cells-axis order).

    ``words`` are per-cell morton coordinates (packed uint64 area words at
    the cell order; ``0`` is the unwritten fill).
    """
    written = words[words != 0]
    if written.size == 0:
        raise ValueError(
            f"chunk at cell {start} of {field!r} has ragged payloads but no written "
            f"'morton' coordinate — the dense coordinate write did not cover it"
        )
    return decimal_order(morton_decimal(int(written[0])))


def _subtree_span(arr, morton, field: str, subtree) -> tuple[tuple[int, int] | None, list | None]:
    """Resolve the readers' ``subtree=`` to ``(span, spans)`` on this axis.

    ``span`` is ``None`` = no restriction (no ``subtree`` given); ``(0, 0)``
    = visit nothing (a disjoint word —
    :func:`~moczarr.convention.subtree_cell_span` already warned — or an
    empty store, which has nothing below any word). The anchor is the first
    WRITTEN word of the ``morton`` coordinate over the whole first stored
    span — one coordinate slice, no payload bytes. The window is the whole
    span, not its first read chunk, because a stored span's leading inner
    chunks may be absent and the coordinate holds its FILL across them
    (zagg spec §7: "a reader MUST NOT assume the coordinate is dense across
    a shard"); the wider window is free, since :class:`_MortonWords` reads
    that stored coordinate object either way. The anchor's own axis index is
    passed alongside so
    :func:`~moczarr.convention.subtree_cell_span` can CHECK the
    nested-placement identity it derives from. A malformed ``subtree``
    raises even on an empty store.

    ``spans`` is the :func:`stored_chunk_spans` listing the span was
    resolved against (``None`` when there was no ``subtree`` and nothing was
    listed): the caller threads it back into its sweep so the restricted
    read LISTs the array once, not twice.
    """
    if subtree is None:
        return None, None
    spans = stored_chunk_spans(arr)
    if not spans:
        normalize_subtree(subtree)  # malformed still raises; nothing to visit
        return (0, 0), spans
    words = morton[spans[0][0] : spans[0][1]]
    cell_order = _cells_order(words, field, spans[0][0])
    written = int(np.flatnonzero(words)[0])
    return (
        subtree_cell_span(
            subtree,
            int(words[written]),
            spans[0][0] + written,
            cell_order,
            int(arr.shape[0]),
            field,
            # The disjoint-word warning belongs to the READER's caller: from
            # subtree_cell_span that is here (2), the reader's own frame (3)
            # — a generator, so it runs on the consumer's first next() — and
            # the consumer (4). Anything less points inside moczarr.
            stacklevel=4,
        ),
        spans,
    )


def _refuse_sub_chunk(span, subtree, field: str, cells_per_chunk: int) -> None:
    """The ratified v1 refusal: a finer-than-chunk ``subtree`` raises.

    The store's smallest fetch unit is the inner chunk (no I/O win below
    it), and the chunk-granular sweeps yield whole read chunks — serving a
    sub-chunk word would either over-yield cells outside the span or grow a
    rank filter the contract keeps out of v1 (issue #29, mirroring zagg's
    ``read_tensors`` refusal). :func:`read_cell` addresses any single cell
    in 2 GETs.
    """
    if span is not None and span != (0, 0) and span[1] - span[0] < cells_per_chunk:
        raise ValueError(
            f"subtree {subtree!r} is finer than {field!r}'s read chunks "
            f"({cells_per_chunk} cells): the sweep readers yield whole read "
            f"chunks only; use read_cell, which addresses any single cell of "
            f"the span in 2 GETs"
        )


def _open_morton(store: Store, field: str, zarr_format: Literal[2, 3]) -> zarr.Array:
    """The sibling per-cell ``morton`` coordinate array (cell identity source)."""
    parent, _, _name = field.rpartition("/")
    path = f"{parent}/morton" if parent else "morton"
    try:
        return zarr.open_array(store, path=path, mode="r", zarr_format=zarr_format)
    except (FileNotFoundError, KeyError) as exc:
        raise ValueError(
            f"{field!r} has no sibling 'morton' coordinate array at {path!r}; the "
            f"ragged readers derive cell identity from the per-cell morton "
            f"coordinate (spec §1.1)"
        ) from exc


class _MortonWords:
    """The sibling ``morton`` coordinate array, read ONE stored object at a time.

    Cell identity comes from the dense ``morton`` coordinate, but the readers
    walk the *payload* array's spans (and :mod:`moczarr.hhdc` its tensor
    blocks) — and one coordinate object covers many of them, so slicing the
    coordinate per span re-fetches the same object every time, plus a
    shard-index suffix on every ``__getitem__`` when the coordinate is
    sharded (measured at ~2 GETs per block). This serves the readers'
    ascending spans out of one cached coordinate window, so the coordinate
    array obeys the same "one GET per stored object" posture as the payload.
    The held cost is that window — 8 bytes per cell over one stored object,
    never the whole cells axis.
    """

    def __init__(self, arr: zarr.Array) -> None:
        self._arr = arr
        self._span = int((arr.shards or arr.chunks)[0])
        self._cells = int(arr.shape[0])
        self._lo = self._hi = 0
        self._window = np.zeros(0, dtype=np.uint64)

    def __getitem__(self, span: slice) -> np.ndarray:
        start, stop = int(span.start), int(span.stop)
        if not self._lo <= start < stop <= self._hi:
            lo = start // self._span * self._span
            hi = min(-(-stop // self._span) * self._span, self._cells)
            self._window = np.asarray(self._arr[lo:hi])
            self._lo, self._hi = lo, hi
        return self._window[start - self._lo : stop - self._lo]


def _morton_words(store: Store, field: str, zarr_format: Literal[2, 3]) -> _MortonWords:
    """The field's sibling ``morton`` coordinate, as a span-cached reader."""
    return _MortonWords(_open_morton(store, field, zarr_format))


def _sibling_path(field: str, sibling: str) -> str:
    """A sibling array's path next to ``field`` (same group)."""
    parent, _, _name = field.rpartition("/")
    return f"{parent}/{sibling}" if parent else sibling


def read_ragged(
    store: Store,
    field: str,
    *,
    locations: bool = False,
    subtree: int | str | None = None,
    zarr_format: Literal[2, 3] = 3,
) -> Iterator[tuple]:
    """Yield every populated cell of a ragged field, decoded per its attrs.

    The generic whole-store sweep: one LIST of the array's stored data
    objects, then one slice per stored span (sharded and flat §1.5
    geometries through the same path — no per-inner-chunk re-fetch), with
    the sibling ``morton`` coordinate served from one cached window per
    stored coordinate object (:class:`_MortonWords`) rather than re-sliced
    per span. Yields
    ``(morton_word, values)`` per populated cell — or ``(morton_word,
    values, location_words)`` with ``locations=True`` — where ``morton_word``
    is the cell's own packed ``uint64`` coordinate from the sibling
    ``morton`` array and ``values`` is the cell's decoded ``(n,
    *inner_shape)`` payload.

    Parameters
    ----------
    store : Store
        Zarr store holding the ragged array (a hive leaf store, or a flat
        store root).
    field : str
        The payload array's path (e.g. ``"6/h_tdigest_signal"``).
    locations : bool, optional
        Also decode the located sibling (spec §1.1/§2.2), bound by the
        payload array's ``locations`` attrs declaration — never by naming
        convention. Raises ``ValueError`` on an unlocated field.
    subtree : int or str, optional
        Restrict the sweep to the cells below this morton ancestor — a
        packed area word (``int``) or its decimal string (``str``), the
        spec §1.5 subtree-span identity (issue #29; zagg's issue #351
        contract). Only stored objects overlapping the subtree's contiguous
        cell span are fetched: on a sharded store the index suffix plus the
        covering inner chunks, on the flat layout the covering chunk
        objects — the :func:`read_cell` 2-GET pattern generalized. The
        subtree must be at or coarser than the READ-CHUNK order — a finer
        word raises pointing at :func:`read_cell` (the ratified v1
        refusal). A well-formed word DISJOINT from this axis warns at most
        once per call (naming the word and the axis root) and yields
        nothing — so an **empty yield is ambiguous** between "in-domain,
        nothing stored" and "outside the domain"; the warning is the only
        discriminator, and its DELIVERY is the caller's ``warnings``
        filter's call (the default action dedups a repeat of the same
        warning in the same process). Malformed / too-deep words raise
        ``ValueError``.
    zarr_format : int, optional
        Zarr format version (default 3).

    Raises
    ------
    ValueError
        On the strict attrs gate (:func:`parse_ragged_attrs`), a missing
        ``morton`` sibling, a populated cell with no written morton word, a
        located sibling whose row count disagrees with the payload (the
        §1.1 row-alignment MUST), ``locations=True`` on an unlocated
        field, or a malformed / too-deep / finer-than-chunk ``subtree``.
    """
    arr, element = open_ragged(store, field, zarr_format=zarr_format)
    morton = _morton_words(store, field, zarr_format)
    span, spans = _subtree_span(arr, morton, field, subtree)
    _refuse_sub_chunk(span, subtree, field, int(arr.chunks[0]))
    loc_arr = loc_element = None
    if locations:
        if element.locations is None:
            raise ValueError(
                f"{field!r} declares no locations sibling "
                f'(attrs["{RAGGED_ATTR}"]["locations"]); it was not written as a '
                f"located ragged field"
            )
        loc_arr, loc_element = open_ragged(
            store, _sibling_path(field, element.locations), zarr_format=zarr_format
        )
    cells_per_chunk = int(arr.chunks[0])
    for span_start, span_stop in stored_chunk_spans(arr) if spans is None else spans:
        if span is not None:
            # Clip to the whole read chunks OVERLAPPING the subtree span —
            # an I/O optimization only. It is exact just when the read chunk
            # divides the span's 4^d length (any power-of-two chunk); this
            # generic layer requires no such chunk shape, so membership is
            # enforced per cell below rather than assumed from alignment.
            span_start = max(span_start, span[0] - span[0] % cells_per_chunk)
            span_stop = min(span_stop, span[1] + -span[1] % cells_per_chunk)
            if span_start >= span_stop:
                continue
        data = cast(np.ndarray, arr[span_start:span_stop])
        words = morton[span_start:span_stop]
        loc_span = cast(np.ndarray, loc_arr[span_start:span_stop]) if loc_arr is not None else None
        for pos in range(span_stop - span_start):
            raw = data[pos]
            if _is_empty(raw):
                continue
            # Per-cell membership: the widened chunks carry non-descendants
            # unless the chunk divides the span (zagg's per-cell readers keep
            # this guard beside the identical clip — readers/tdigest_tensor.py
            # read_raw_values). It belongs here only: iter_populated_chunks
            # yields WHOLE chunks by contract, and its consumer read_tensors
            # refuses sub-chunk spans on a square power-of-four chunk, where
            # the clip is exact by construction.
            if span is not None and not span[0] <= span_start + pos < span[1]:
                continue
            word = int(words[pos])
            if word == 0:
                raise ValueError(
                    f"cell {span_start + pos} of {field!r} holds a ragged payload "
                    f"but no written 'morton' coordinate — the dense coordinate "
                    f"write did not cover it"
                )
            values = decode_cell(raw, element)
            if loc_span is None:
                yield word, values
                continue
            assert loc_element is not None
            cell_locations = decode_cell(loc_span[pos], loc_element)
            if len(cell_locations) != len(values):
                raise ValueError(
                    f"cell {span_start + pos} of {field!r} has {len(values)} payload "
                    f"rows but {len(cell_locations)} location words — the located "
                    f"sibling is not row-aligned (spec §1.1)"
                )
            yield word, values, cell_locations


def read_cell(
    store: Store,
    field: str,
    cell: int,
    *,
    zarr_format: Literal[2, 3] = 3,
) -> np.ndarray:
    """Random-access ONE cell's ragged payload, decoded per the element attrs.

    Indexing the vlen array reads exactly two ranged GETs on a sharded store
    — the shard-index suffix, then the one inner chunk holding the cell —
    never the whole shard object (spec §1.5); one GET on the flat layout.
    ``cell`` is the cell's global position on the array's cells axis (no
    negative-index wrap). An absent/empty cell returns the zero-length
    ``(0, *inner_shape)`` array; an out-of-range index raises ``IndexError``
    naming the valid range (zarr basic selection would silently clamp the
    slice). Works on any conforming array — a located field's sibling
    included (its elements decode as ``(n,)`` uint64 words).
    """
    arr, element = open_ragged(store, field, zarr_format=zarr_format)
    cell = int(cell)
    if not 0 <= cell < int(arr.shape[0]):
        raise IndexError(
            f"cell {cell} out of range for {field!r} with {int(arr.shape[0])} cells "
            f"(valid: 0..{int(arr.shape[0]) - 1}; negative indices do not wrap)"
        )
    raw = cast(np.ndarray, arr[cell : cell + 1])[0]
    return decode_cell(raw, element)
