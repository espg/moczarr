"""Generic ``zagg-ragged/1`` vlen decode layer — product-agnostic (issue #19).

The read side of zagg's ragged store spec (``docs/specification.md`` §1 in
englacial/zagg — currently normative on the ``claude/340-store-spec`` branch,
confirmed at englacial/zagg#346's merge): a ``kind: ragged`` field is ONE
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
imports zagg. The HHDC tensor profile (:mod:`moczarr.hhdc`) composes on top.

Postures, inherited from the spec's conformance rules:

- **Strict attrs gate.** ``spec`` is strict-checked: a missing block, a
  foreign marker, or a newer revision raises — a reader must never
  half-parse under a guessed layout (a mis-guessed element dtype
  misinterprets every payload). Contrast the coverage envelopes' tolerant
  ``None``: those are caches that degrade to the walk; ragged attrs are
  truth.
- **One code path for both storage geometries** (§1.5). Sharded
  (``sharding_indexed``; every hive leaf) and per-inner-chunk (regular
  array; the unsharded flat path) layouts are self-describing in the
  array's own metadata, so the sweep derives the stored span from
  ``arr.shards`` when present, else ``arr.chunks``.
- **Store-scoped.** The readers never traverse the hive digit tree — leaf
  discovery stays with :func:`moczarr.open_hive` / the coverage MOC. Open
  the leaf store and pass the in-leaf array path (e.g. ``"6/h_tdigest"``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import zarr
from zarr.abc.store import Store

__all__ = [
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
#: The one convention revision this layer decodes. ``zagg-ragged/2`` moves
#: the declaration into a typed ``vlen-ndarray`` dtype and retires the attrs
#: marker (spec §6) — a ``/2`` array never reaches this gate.
RAGGED_SPEC = "zagg-ragged/1"


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
    (``shape`` must be ``[-1, *inner_shape]``).
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
    """
    data = b"" if raw is None else bytes(raw)  # type: ignore[call-overload]
    return np.frombuffer(data, dtype=element.dtype).reshape((-1, *element.inner_shape))


def open_ragged(
    store: Store,
    field: str,
    *,
    zarr_format: Literal[2, 3] = 3,
) -> tuple[zarr.Array, RaggedElement]:
    """Open a ragged vlen array and its parsed element declaration, strictly.

    Raises ``ValueError`` when the array is not a 1-D vlen-bytes cells axis
    or fails the :func:`parse_ragged_attrs` gate.
    """
    arr = zarr.open_array(store, path=field, mode="r", zarr_format=zarr_format)
    element = parse_ragged_attrs(dict(arr.attrs), field=field)
    if arr.dtype.kind != "O":
        raise ValueError(
            f"{field!r} carries a ragged declaration but its zarr dtype is "
            f"{arr.dtype!s}, not variable-length bytes — not a {RAGGED_SPEC!r} array"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{field!r} has {arr.ndim} dimensions; a {RAGGED_SPEC!r} field is one "
            f"vlen array on the 1-D cells axis"
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
    """
    from zarr.core.sync import sync

    async def _collect(gen) -> list:
        return [key async for key in gen]

    span = int((arr.shards or arr.chunks)[0])
    prefix = f"{arr.path}/c/" if arr.path else "c/"
    keys = sync(_collect(arr.store_path.store.list_prefix(prefix)))
    ordinals = sorted(int(k.rsplit("/", 1)[-1]) for k in keys)
    return [(o * span, min((o + 1) * span, int(arr.shape[0]))) for o in ordinals]


def iter_populated_chunks(arr: zarr.Array) -> Iterator[tuple[int, list[tuple[int, object]]]]:
    """Yield ``(chunk_start, [(rank, raw_bytes), ...])`` per populated read chunk.

    The bulk-sweep primitive shared by :func:`read_ragged` and the profile
    readers (:mod:`moczarr.hhdc`). Restricted to the stored objects
    (:func:`stored_chunk_spans`), each read in ONE slice: slicing per inner
    chunk would re-fetch a sharded object's index suffix on every
    ``__getitem__`` (~K redundant GETs per shard), while the full-span slice
    fetches the stored object once (zarr reads a whole outer chunk in a
    single GET and splits it locally). The held cost is one span's decoded
    payload. Chunks whose cells are all absent (the ``b""`` fill — including
    the §1.5 shard-index absence sentinel) are skipped. ``rank`` is the
    cell's chunk-local NESTED rank — the same index the writer placed it at
    on the nested-ordered cells axis, never a row-major position.
    """
    cells_per_chunk = int(arr.chunks[0])
    for span_start, span_stop in stored_chunk_spans(arr):
        span = cast(np.ndarray, arr[span_start:span_stop])
        for offset in range(0, span_stop - span_start, cells_per_chunk):
            block = span[offset : offset + cells_per_chunk]
            populated = [(pos, block[pos]) for pos in range(len(block)) if len(block[pos])]
            if populated:
                yield span_start + offset, populated


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
    zarr_format : int, optional
        Zarr format version (default 3).

    Raises
    ------
    ValueError
        On the strict attrs gate (:func:`parse_ragged_attrs`), a missing
        ``morton`` sibling, a populated cell with no written morton word, a
        located sibling whose row count disagrees with the payload (the
        §1.1 row-alignment MUST), or ``locations=True`` on an unlocated
        field.
    """
    arr, element = open_ragged(store, field, zarr_format=zarr_format)
    morton = _morton_words(store, field, zarr_format)
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
    for span_start, span_stop in stored_chunk_spans(arr):
        span = cast(np.ndarray, arr[span_start:span_stop])
        words = morton[span_start:span_stop]
        loc_span = cast(np.ndarray, loc_arr[span_start:span_stop]) if loc_arr is not None else None
        for pos in range(span_stop - span_start):
            raw = span[pos]
            if not len(raw):
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
