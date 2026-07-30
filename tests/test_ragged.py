"""Tests for the generic ``zagg-ragged/1`` decode layer (issue #19).

Fixtures are built **spec-text-only**: every byte is written by hand from
the spec's §1.3/§1.4/§1.5 recipes (vlen framing, zstd chain, shard index
with the absence sentinel) with no zarr write machinery — so decoding them
through :mod:`moczarr.ragged` pins that the published byte recipes are
sufficient on their own. The element is deliberately **non-HHDC**
(``int16 (n, 3)``) to pin that the layer has zero product knowledge; the
zagg-written conformance fixtures live in ``test_spec_conformance.py``.
"""

import json
import struct

import google_crc32c  # zarr's own crc32c provider — not a new dependency
import numpy as np
import pytest
from numcodecs import Zstd
from zarr.storage import LocalStore

import moczarr.ragged
from moczarr.convention import morton_word
from moczarr.ragged import (
    RAGGED_SPEC,
    RaggedElement,
    decode_cell,
    iter_populated_chunks,
    open_ragged,
    parse_ragged_attrs,
    read_cell,
    read_ragged,
)

#: Order-6 shard whose 16 order-8 cells are the fixture's cells axis.
SHARD = "-5112333"
#: Nested-ascending 2-digit tails of the depth-2 subtree.
TAILS = [a + b for a in "1234" for b in "1234"]
#: The non-HHDC element declaration: int16 rows of 3.
ELEMENT = {"dtype": "int16", "shape": [-1, 3]}

# ------------------------------------------------------------------ builders


def _inner_chunk(cells):
    """One inner chunk's §1.4 wire framing, zstd-compressed (§1.3)."""
    framed = struct.pack("<I", len(cells)) + b"".join(struct.pack("<I", len(c)) + c for c in cells)
    return bytes(Zstd(level=3).encode(framed))


def _shard_object(chunks):
    """A §1.5 sharded object: data + u64-pair index + crc32c, index at end.

    ``chunks`` maps each inner-chunk ordinal to its cell payload list, or
    ``None`` for an absent chunk (the ``2^64 - 1`` sentinel in both index
    fields).
    """
    data, index = b"", []
    for cells in chunks:
        if cells is None:
            index.append((2**64 - 1, 2**64 - 1))
        else:
            enc = _inner_chunk(cells)
            index.append((len(data), len(enc)))
            data += enc
    idx = b"".join(struct.pack("<QQ", o, n) for o, n in index)
    return data + idx + struct.pack("<I", google_crc32c.value(idx))


def _vlen_meta(n, chunk, *, sharded, attrs):
    inner = [
        {"name": "vlen-bytes", "configuration": {}},
        {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
    ]
    if sharded:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [chunk],
                    "codecs": inner,
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ]
        outer = n
    else:
        codecs, outer = inner, chunk
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [n],
        "data_type": "variable_length_bytes",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [outer]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "",
        "codecs": codecs,
        "dimension_names": ["cells"],
        "attributes": attrs,
    }


def _uint64_meta(n):
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [n],
        "data_type": "uint64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [n]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": ["cells"],
        "attributes": {},
    }


def _write(root, rel, payload):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload))
    else:
        path.write_bytes(payload)


#: Populated cells: global index -> (n_rows) of the int16 (n, 3) payload.
CELLS = {0: 2, 3: 1, 13: 4}


def _payloads():
    """Deterministic per-cell ``(values, raw_bytes)`` in the declared element."""
    rng = np.random.default_rng(19)
    out = {}
    for cell, rows in CELLS.items():
        values = rng.integers(-500, 500, (rows, 3)).astype("<i2")
        out[cell] = (values, values.tobytes())
    return out


def build_store(root, *, sharded, located=False, morton_words=None, loc_counts=None):
    """A spec-text-only store: 16 cells, 4 inner chunks; chunks 1-2 empty.

    Chunk 0 holds cells 0 and 3 (cells 1-2 keep the ``b""`` fill), chunk 3
    holds cell 13. On the sharded layout chunks 1-2 are the index absence
    sentinel; on the flat layout their objects are simply missing. Returns
    ``(store_root_path, {cell: expected_values})``.
    """
    payloads = _payloads()
    attrs = {"ragged": {"spec": RAGGED_SPEC, "element": dict(ELEMENT)}}
    if located:
        attrs["ragged"]["locations"] = "geo_words"  # deliberately NOT {field}_locations
    grid = root / "store"

    def chunk_cells(chunk, source):
        return [source.get(chunk * 4 + i, b"") for i in range(4)]

    raw = {c: b for c, (v, b) in payloads.items()}
    chunks = [chunk_cells(0, raw), None, None, chunk_cells(3, raw)]
    _write(grid, "g/field/zarr.json", _vlen_meta(16, 4, sharded=sharded, attrs=attrs))
    if sharded:
        _write(grid, "g/field/c/0", _shard_object(chunks))
    else:
        for ordinal, cells in enumerate(chunks):
            if cells is not None:
                _write(grid, f"g/field/c/{ordinal}", _inner_chunk(cells))

    if located:
        counts = loc_counts or {c: len(v) for c, (v, _b) in payloads.items()}
        loc_words = {c: np.arange(1, counts[c] + 1, dtype="<u8") * 7 for c in payloads}
        loc_raw = {c: w.tobytes() for c, w in loc_words.items()}
        loc_chunks = [chunk_cells(0, loc_raw), None, None, chunk_cells(3, loc_raw)]
        loc_attrs = {"ragged": {"spec": RAGGED_SPEC, "element": {"dtype": "uint64", "shape": [-1]}}}
        _write(grid, "g/geo_words/zarr.json", _vlen_meta(16, 4, sharded=sharded, attrs=loc_attrs))
        if sharded:
            _write(grid, "g/geo_words/c/0", _shard_object(loc_chunks))
        else:
            for ordinal, cells in enumerate(loc_chunks):
                if cells is not None:
                    _write(grid, f"g/geo_words/c/{ordinal}", _inner_chunk(cells))

    words = (
        morton_words
        if morton_words is not None
        else np.array([morton_word(SHARD + t) for t in TAILS], dtype="<u8")
    )
    _write(grid, "g/morton/zarr.json", _uint64_meta(16))
    _write(grid, "g/morton/c/0", np.asarray(words, dtype="<u8").tobytes())
    expected = {c: v for c, (v, _b) in payloads.items()}
    return grid, expected


class CountingStore(LocalStore):
    """LocalStore recording every satisfied GET as ``(key, byte_range, nbytes)``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gets: list = []

    def with_read_only(self, read_only=True):
        s = CountingStore(self.root, read_only=read_only)
        s.gets = self.gets
        return s

    async def get(self, key, prototype, byte_range=None):
        r = await super().get(key, prototype, byte_range)
        if r is not None:
            self.gets.append((key, byte_range, len(r)))
        return r


# --------------------------------------------------------------------- tests


class TestParseRaggedAttrs:
    """The strict §1.2 gate: raise on missing/foreign/newer, never guess."""

    def test_parses_declaration(self):
        element = parse_ragged_attrs(
            {"ragged": {"spec": RAGGED_SPEC, "element": ELEMENT, "locations": "geo_words"}}
        )
        assert element == RaggedElement(np.dtype("<i2"), (3,), "geo_words")
        assert element.dtype.byteorder in ("<", "=")  # little-endian wire order

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="no ragged element declaration"):
            parse_ragged_attrs({}, field="g/field")

    def test_foreign_spec_raises(self):
        with pytest.raises(ValueError, match="understands 'zagg-ragged/1' only"):
            parse_ragged_attrs({"ragged": {"spec": "someone-else/9", "element": ELEMENT}})

    def test_newer_revision_raises(self):
        """A future revision must be adopted deliberately, never half-parsed
        (spec conformance rule; ``/2`` retires the attrs marker anyway)."""
        with pytest.raises(ValueError, match="never half-parsed"):
            parse_ragged_attrs({"ragged": {"spec": "zagg-ragged/3", "element": ELEMENT}})

    def test_malformed_element_raises(self):
        with pytest.raises(ValueError, match="malformed element declaration"):
            parse_ragged_attrs({"ragged": {"spec": RAGGED_SPEC, "element": {"dtype": "f4"}}})

    def test_shape_without_varying_count_raises(self):
        with pytest.raises(ValueError, match=r"\[-1, \*inner_shape\]"):
            parse_ragged_attrs(
                {"ragged": {"spec": RAGGED_SPEC, "element": {"dtype": "f4", "shape": [2, 2]}}}
            )


class TestDecodeCell:
    def test_decodes_declared_element(self):
        element = RaggedElement(np.dtype("<i2"), (3,))
        values = np.arange(6, dtype="<i2").reshape(2, 3)
        np.testing.assert_array_equal(decode_cell(values.tobytes(), element), values)

    def test_empty_and_none_decode_to_zero_length(self):
        element = RaggedElement(np.dtype("<u8"), ())
        assert decode_cell(b"", element).shape == (0,)
        assert decode_cell(None, element).shape == (0,)
        assert decode_cell(b"", RaggedElement(np.dtype("<f4"), (2,))).shape == (0, 2)


@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "flat"])
class TestReadRagged:
    """Sharded and flat §1.5 geometries through ONE code path."""

    def test_yields_populated_cells_with_morton_identity(self, tmp_path, sharded):
        grid, expected = build_store(tmp_path, sharded=sharded)
        out = dict(read_ragged(LocalStore(grid), "g/field"))
        words = {morton_word(SHARD + TAILS[c]): c for c in CELLS}
        assert set(out) == set(words)
        for word, values in out.items():
            assert values.dtype == np.dtype("<i2")
            np.testing.assert_array_equal(values, expected[words[word]])

    def test_locations_bound_by_metadata_not_naming(self, tmp_path, sharded):
        """The sibling is named ``geo_words`` — nothing like
        ``{field}_locations`` — so only the attrs declaration can find it."""
        grid, expected = build_store(tmp_path, sharded=sharded, located=True)
        out = list(read_ragged(LocalStore(grid), "g/field", locations=True))
        assert len(out) == len(CELLS)
        for _word, values, locations in out:
            assert locations.dtype == np.dtype("<u8")
            assert locations.shape == (len(values),)  # §1.1 row alignment
            np.testing.assert_array_equal(locations, np.arange(1, len(values) + 1, dtype="<u8") * 7)

    def test_misaligned_locations_raise(self, tmp_path, sharded):
        grid, _ = build_store(
            tmp_path, sharded=sharded, located=True, loc_counts={0: 1, 3: 1, 13: 4}
        )
        with pytest.raises(ValueError, match="not row-aligned"):
            list(read_ragged(LocalStore(grid), "g/field", locations=True))

    def test_locations_on_unlocated_field_raises(self, tmp_path, sharded):
        grid, _ = build_store(tmp_path, sharded=sharded)
        with pytest.raises(ValueError, match="declares no locations sibling"):
            list(read_ragged(LocalStore(grid), "g/field", locations=True))

    def test_payload_without_morton_word_raises(self, tmp_path, sharded):
        words = np.array([morton_word(SHARD + t) for t in TAILS], dtype="<u8")
        words[13] = 0  # cell 13 holds a payload but no written coordinate
        grid, _ = build_store(tmp_path, sharded=sharded, morton_words=words)
        with pytest.raises(ValueError, match="no written 'morton' coordinate"):
            list(read_ragged(LocalStore(grid), "g/field"))

    def test_in_leaf_debris_is_skipped_with_a_warning(self, tmp_path, sharded):
        """A LISTing reader meets the spec's §5.1 in-leaf debris classes: a
        key under ``c/`` that is not a chunk ordinal must be skipped, not
        coerced to an int (which took the whole sweep down)."""
        grid, expected = build_store(tmp_path, sharded=sharded)
        _write(grid, "g/field/c/.DS_Store", b"junk")  # an OS file
        _write(grid, "g/field/c/foreign/zarr.json", {"node_type": "group"})  # a prefix
        with pytest.warns(UserWarning, match="not chunk ordinals"):
            out = dict(read_ragged(LocalStore(grid), "g/field"))
        assert len(out) == len(CELLS)  # every payload still decoded
        for word, values in out.items():
            np.testing.assert_array_equal(
                values, expected[{morton_word(SHARD + TAILS[c]): c for c in CELLS}[word]]
            )

    def test_non_ragged_array_rejected(self, tmp_path, sharded):
        """The dense morton array carries no ragged block — hard refusal."""
        grid, _ = build_store(tmp_path, sharded=sharded)
        with pytest.raises(ValueError, match="no ragged element declaration"):
            list(read_ragged(LocalStore(grid), "g/morton"))

    @pytest.mark.parametrize(
        "data_type",
        [
            "vlen-ndarray",
            {"name": "vlen-ndarray", "configuration": {"element_dtype": "int16"}},
        ],
        ids=["bare", "configured"],
    )
    def test_ragged2_array_named_by_revision(self, tmp_path, sharded, data_type):
        """A ``/2`` array retires the ``ragged`` marker by design (spec
        §1.6/§6.1), so the attrs gate would call it an unsignaled pre-spec
        store. The dtype is the signal: name the revision, not a guess.
        The typed dtype is unknown to this zarr stack, so this also covers
        §6.3's "actionable failure without the extension installed"."""
        grid, _ = build_store(tmp_path, sharded=sharded)
        meta = _vlen_meta(16, 4, sharded=sharded, attrs={})
        meta["data_type"] = data_type
        _write(grid, "g/typed/zarr.json", meta)
        with pytest.raises(ValueError, match="zagg-ragged/2"):
            list(read_ragged(LocalStore(grid), "g/typed"))

    def test_ragged_attrs_on_fixed_dtype_rejected(self, tmp_path, sharded):
        grid, _ = build_store(tmp_path, sharded=sharded)
        meta = _uint64_meta(16)
        meta["attributes"] = {"ragged": {"spec": RAGGED_SPEC, "element": ELEMENT}}
        _write(grid, "g/bogus/zarr.json", meta)
        _write(grid, "g/bogus/c/0", np.zeros(16, dtype="<u8").tobytes())
        with pytest.raises(ValueError, match="not variable-length bytes"):
            list(read_ragged(LocalStore(grid), "g/bogus"))


class _NoneFillArray:
    """Proxy of a real vlen array whose EMPTY cells read back as ``None``.

    zarr's ``variable_length_bytes`` decode hands out ``b""`` for an
    unwritten cell, so the ``None`` shape spec §5.2 admits (and
    :func:`decode_cell` documents) has no store-level fixture. Wrapping the
    real array is how the sweeps get to meet it.
    """

    def __init__(self, arr):
        self._arr = arr

    def __getattr__(self, name):
        return getattr(self._arr, name)

    def __getitem__(self, key):
        out = np.array(self._arr[key], dtype=object)
        out[[i for i, value in enumerate(out) if value == b""]] = None
        return out


class TestNoneCells:
    """§5.2: an unwritten vlen cell may decode as ``None``, not ``b""`` — so
    the sweeps must not ``len()`` it (``decode_cell`` already handles both)."""

    def test_sweep_tolerates_none_cells(self, tmp_path, monkeypatch):
        grid, expected = build_store(tmp_path, sharded=True)
        real = moczarr.ragged.open_ragged

        def patched(*args, **kwargs):
            arr, element = real(*args, **kwargs)
            return _NoneFillArray(arr), element

        monkeypatch.setattr(moczarr.ragged, "open_ragged", patched)
        out = dict(read_ragged(LocalStore(grid), "g/field"))
        words = {morton_word(SHARD + TAILS[c]): c for c in CELLS}
        assert set(out) == set(words)
        for word, values in out.items():
            np.testing.assert_array_equal(values, expected[words[word]])

    def test_iter_populated_chunks_tolerates_none_cells(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        arr, _element = open_ragged(LocalStore(grid), "g/field")
        chunks = dict(iter_populated_chunks(_NoneFillArray(arr)))
        assert sorted(chunks) == [0, 12]  # cells 0/3 in chunk 0, cell 13 in chunk 3
        assert [rank for rank, _raw in chunks[0]] == [0, 3]


class TestReadCell:
    def test_roundtrip_and_empty_fill(self, tmp_path):
        grid, expected = build_store(tmp_path, sharded=True)
        store = LocalStore(grid)
        np.testing.assert_array_equal(read_cell(store, "g/field", 13), expected[13])
        assert read_cell(store, "g/field", 1).shape == (0, 3)  # b"" fill
        assert read_cell(store, "g/field", 5).shape == (0, 3)  # absent chunk

    def test_out_of_range_raises_pointed(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        store = LocalStore(grid)
        with pytest.raises(IndexError, match="cell 16 out of range"):
            read_cell(store, "g/field", 16)
        with pytest.raises(IndexError, match="negative indices do not wrap"):
            read_cell(store, "g/field", -1)

    def test_two_ranged_gets_on_sharded_store(self, tmp_path):
        """The §1.5 random-access recipe: index suffix + ONE inner chunk,
        never the whole shard object."""
        grid, expected = build_store(tmp_path, sharded=True)
        store = CountingStore(grid)
        store.gets.clear()
        values = read_cell(store, "g/field", 13)
        np.testing.assert_array_equal(values, expected[13])
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == 2
        (key, _r0, n0), (_k1, _r1, n1) = data_gets
        obj_size = (grid / "g/field/c/0").stat().st_size
        assert n0 == 16 * 4 + 4  # the K=4 shard-index suffix
        assert n1 < obj_size  # one ranged inner chunk, not the object

    def test_one_get_on_flat_store(self, tmp_path):
        grid, expected = build_store(tmp_path, sharded=False)
        store = CountingStore(grid)
        store.gets.clear()
        np.testing.assert_array_equal(read_cell(store, "g/field", 0), expected[0])
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == 1  # the one inner-chunk object


class TestSweepGetPosture:
    def test_sweep_reads_each_stored_object_once(self, tmp_path):
        """The whole-store sweep slices each stored span ONCE — never a
        per-inner-chunk re-fetch of a sharded object's index suffix."""
        grid, _ = build_store(tmp_path, sharded=True)
        store = CountingStore(grid)
        store.gets.clear()
        out = list(read_ragged(store, "g/field"))
        assert len(out) == len(CELLS)
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == 1  # one stored object, fetched once

    def test_sweep_reads_the_coordinate_once(self, tmp_path):
        """The posture covers the sibling ``morton`` coordinate too: the
        cached window is read once per stored coordinate object, never
        re-sliced per payload span (which re-fetches the same object, plus a
        shard-index suffix when the coordinate is sharded)."""
        grid, _ = build_store(tmp_path, sharded=True)
        store = CountingStore(grid)
        store.gets.clear()
        assert len(list(read_ragged(store, "g/field"))) == len(CELLS)
        assert len([g for g in store.gets if "morton/c/" in g[0]]) == 1
