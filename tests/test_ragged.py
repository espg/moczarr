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
    LOCATED_SPEC,
    MORTON_GRAMMAR,
    RAGGED_SPEC,
    TOC_GRAMMAR,
    TOC_SPEC,
    CompanionDeclaration,
    RaggedElement,
    decode_cell,
    iter_populated_chunks,
    open_ragged,
    parse_companion_attrs,
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


def _payloads(cells=None):
    """Deterministic per-cell ``(values, raw_bytes)`` in the declared element."""
    rng = np.random.default_rng(19)
    out = {}
    for cell, rows in (cells or CELLS).items():
        values = rng.integers(-500, 500, (rows, 3)).astype("<i2")
        out[cell] = (values, values.tobytes())
    return out


#: The default §8.3 declaration :func:`build_store` stamps on ``t_words``.
TEMPORAL_DECLARATION = {"spec": TOC_SPEC, "shape": "per-centroid", "grammar": TOC_GRAMMAR}


def build_store(
    root,
    *,
    sharded,
    located=False,
    timed=False,
    morton_words=None,
    loc_counts=None,
    times_counts=None,
    temporal_declaration=TEMPORAL_DECLARATION,
    cells=None,
):
    """A spec-text-only store: 16 cells, 4 inner chunks; chunks 1-2 empty.

    Chunk 0 holds cells 0 and 3 (cells 1-2 keep the ``b""`` fill), chunk 3
    holds cell 13. On the sharded layout chunks 1-2 are the index absence
    sentinel; on the flat layout their objects are simply missing. ``cells``
    overrides :data:`CELLS` (``{cell: n_rows}``); an inner chunk with no
    populated cell is absent either way. ``timed`` writes a §8.3 temporal
    sibling ``t_words`` bound from the payload's top-level ``times`` key,
    stamped with ``temporal_declaration`` (``None`` omits the block — the
    non-conformant bound-but-undeclared case). Returns
    ``(store_root_path, {cell: expected_values})``.
    """
    payloads = _payloads(cells)
    attrs = {"ragged": {"spec": RAGGED_SPEC, "element": dict(ELEMENT)}}
    if located:
        attrs["ragged"]["locations"] = "geo_words"  # deliberately NOT {field}_locations
    if timed:
        attrs["times"] = "t_words"  # §8.3: beside the block, deliberately NOT inside it
    grid = root / "store"

    def chunk_cells(chunk, source):
        """One inner chunk's 4 cells, or ``None`` when none is populated."""
        out = [source.get(chunk * 4 + i, b"") for i in range(4)]
        return out if any(out) else None

    raw = {c: b for c, (v, b) in payloads.items()}
    chunks = [chunk_cells(c, raw) for c in range(4)]
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
        loc_chunks = [chunk_cells(c, loc_raw) for c in range(4)]
        loc_attrs = {"ragged": {"spec": RAGGED_SPEC, "element": {"dtype": "uint64", "shape": [-1]}}}
        _write(grid, "g/geo_words/zarr.json", _vlen_meta(16, 4, sharded=sharded, attrs=loc_attrs))
        if sharded:
            _write(grid, "g/geo_words/c/0", _shard_object(loc_chunks))
        else:
            for ordinal, cells in enumerate(loc_chunks):
                if cells is not None:
                    _write(grid, f"g/geo_words/c/{ordinal}", _inner_chunk(cells))

    if timed:
        counts = times_counts or {c: len(v) for c, (v, _b) in payloads.items()}
        t_words = {c: np.arange(1, counts[c] + 1, dtype="<u8") * 11 for c in payloads}
        t_raw = {c: w.tobytes() for c, w in t_words.items()}
        t_chunks = [chunk_cells(c, t_raw) for c in range(4)]
        t_attrs: dict = {
            "ragged": {"spec": RAGGED_SPEC, "element": {"dtype": "uint64", "shape": [-1]}}
        }
        if temporal_declaration is not None:
            t_attrs["temporal"] = dict(temporal_declaration)
        _write(grid, "g/t_words/zarr.json", _vlen_meta(16, 4, sharded=sharded, attrs=t_attrs))
        if sharded:
            _write(grid, "g/t_words/c/0", _shard_object(t_chunks))
        else:
            for ordinal, cells in enumerate(t_chunks):
                if cells is not None:
                    _write(grid, f"g/t_words/c/{ordinal}", _inner_chunk(cells))

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

    def test_times_binding_is_beside_the_block(self):
        """§8.3: the ``times`` key is a spec-owned SIBLING of the ``ragged``
        block — read from the top level, and a same-named key inside the
        block is not the binding (the /1 block grammar is unchanged)."""
        element = parse_ragged_attrs(
            {"ragged": {"spec": RAGGED_SPEC, "element": ELEMENT}, "times": "t_words"}
        )
        assert element.times == "t_words"
        assert element.locations is None
        inside_only = parse_ragged_attrs(
            {"ragged": {"spec": RAGGED_SPEC, "element": ELEMENT, "times": "t_words"}}
        )
        assert inside_only.times is None


class TestParseCompanionAttrs:
    """The §8/§9 declaration gate: strict when present, silent when absent."""

    def test_absent_block_is_never_a_refusal(self):
        """§9's absent-``located`` = §2.2 verbatim; §8's absent-``temporal``
        = the legacy encoding — both parse as ``None``, no raise."""
        assert parse_companion_attrs({"ragged": {}}, domain="located") is None
        assert parse_companion_attrs({}, domain="temporal") is None
        assert parse_companion_attrs(None, domain="temporal") is None

    def test_parses_both_domains(self):
        located = parse_companion_attrs(
            {"located": {"spec": LOCATED_SPEC, "shape": "per-centroid", "grammar": MORTON_GRAMMAR}},
            domain="located",
        )
        assert located == CompanionDeclaration(LOCATED_SPEC, "per-centroid", MORTON_GRAMMAR)
        temporal = parse_companion_attrs(
            {"temporal": {"spec": TOC_SPEC, "shape": "per-centroid", "grammar": TOC_GRAMMAR}},
            domain="temporal",
        )
        assert temporal == CompanionDeclaration(TOC_SPEC, "per-centroid", TOC_GRAMMAR)

    def test_informative_extra_keys_are_ignored(self):
        """§9: "informative extra keys ignored rather than refused"."""
        decl = parse_companion_attrs(
            {
                "temporal": {
                    "spec": TOC_SPEC,
                    "shape": "per-centroid",
                    "grammar": TOC_GRAMMAR,
                    "docs": "https://example.invalid",
                }
            },
            domain="temporal",
        )
        assert decl is not None

    def test_future_spec_revision_raises(self):
        with pytest.raises(ValueError, match="never half-parsed"):
            parse_companion_attrs(
                {
                    "temporal": {
                        "spec": "zagg-toc/2",
                        "shape": "per-centroid",
                        "grammar": TOC_GRAMMAR,
                    }
                },
                domain="temporal",
            )

    def test_unimplemented_shape_raises(self):
        """§8: a reader MUST refuse a shape it does not implement — the
        ragged sibling path decodes ``per-centroid`` only."""
        with pytest.raises(ValueError, match="MUST refuse a"):
            parse_companion_attrs(
                {"temporal": {"spec": TOC_SPEC, "shape": "per-cell", "grammar": TOC_GRAMMAR}},
                domain="temporal",
            )

    def test_unimplemented_grammar_raises(self):
        with pytest.raises(ValueError, match="grammar"):
            parse_companion_attrs(
                {"located": {"spec": LOCATED_SPEC, "shape": "per-centroid", "grammar": "acme/1"}},
                domain="located",
            )

    def test_malformed_block_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_companion_attrs({"located": "yes"}, domain="located")


class TestDecodeCell:
    def test_decodes_declared_element(self):
        element = RaggedElement(np.dtype("<i2"), (3,))
        values = np.arange(6, dtype="<i2").reshape(2, 3)
        np.testing.assert_array_equal(decode_cell(values.tobytes(), element), values)

    def test_payload_is_a_read_only_view(self):
        """Documented posture (and zagg's): a decoded cell is a non-owning
        ``frombuffer`` view — in-place work on it raises."""
        values = decode_cell(
            np.arange(6, dtype="<i2").tobytes(), RaggedElement(np.dtype("<i2"), (3,))
        )
        assert not values.flags.writeable and not values.flags.owndata
        with pytest.raises(ValueError, match="read-only"):
            values[0, 0] = 1

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

    def test_times_bound_by_metadata_not_naming(self, tmp_path, sharded):
        """The temporal sibling is named ``t_words`` — nothing like
        ``{field}_times`` — so only the top-level ``times`` key can find it
        (spec §8.3)."""
        grid, expected = build_store(tmp_path, sharded=sharded, timed=True)
        out = list(read_ragged(LocalStore(grid), "g/field", times=True))
        assert len(out) == len(CELLS)
        for _word, values, times in out:
            assert times.dtype == np.dtype("<u8")
            assert times.shape == (len(values),)  # §8.3 row alignment
            np.testing.assert_array_equal(times, np.arange(1, len(values) + 1, dtype="<u8") * 11)

    def test_both_channels_yield_in_fixed_order(self, tmp_path, sharded):
        """``locations`` before ``times`` — the documented 4-tuple."""
        grid, expected = build_store(tmp_path, sharded=sharded, located=True, timed=True)
        out = list(read_ragged(LocalStore(grid), "g/field", locations=True, times=True))
        assert len(out) == len(CELLS)
        for _word, values, locations, times in out:
            np.testing.assert_array_equal(locations, np.arange(1, len(values) + 1, dtype="<u8") * 7)
            np.testing.assert_array_equal(times, np.arange(1, len(values) + 1, dtype="<u8") * 11)

    def test_misaligned_times_raise(self, tmp_path, sharded):
        grid, _ = build_store(
            tmp_path, sharded=sharded, timed=True, times_counts={0: 1, 3: 1, 13: 4}
        )
        with pytest.raises(ValueError, match=r"not row-aligned \(spec §8.3\)"):
            list(read_ragged(LocalStore(grid), "g/field", times=True))

    def test_times_on_untimed_field_raises(self, tmp_path, sharded):
        grid, _ = build_store(tmp_path, sharded=sharded)
        with pytest.raises(ValueError, match="declares no temporal sibling"):
            list(read_ragged(LocalStore(grid), "g/field", times=True))

    def test_bound_times_sibling_without_declaration_raises(self, tmp_path, sharded):
        """§8.3: the payload carries the binding, the sibling the
        declaration — a bound sibling missing it is non-conformant, not the
        absent-key legacy case."""
        grid, _ = build_store(tmp_path, sharded=sharded, timed=True, temporal_declaration=None)
        with pytest.raises(ValueError, match="carries no attrs"):
            list(read_ragged(LocalStore(grid), "g/field", times=True))

    def test_bound_times_sibling_with_foreign_shape_raises(self, tmp_path, sharded):
        grid, _ = build_store(
            tmp_path,
            sharded=sharded,
            timed=True,
            temporal_declaration={"spec": TOC_SPEC, "shape": "per-cell", "grammar": TOC_GRAMMAR},
        )
        with pytest.raises(ValueError, match="per-cell"):
            list(read_ragged(LocalStore(grid), "g/field", times=True))

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

    def test_span_and_spans_are_keyword_only(self, tmp_path):
        """``span`` and ``spans`` are adjacent, same-shaped and easy to swap,
        and a swapped positional call reads as valid — so the public surface
        (this name is in ``__all__``) takes them by keyword only."""
        grid, _ = build_store(tmp_path, sharded=True)
        arr, _element = open_ragged(LocalStore(grid), "g/field")
        with pytest.raises(TypeError, match="positional"):
            list(iter_populated_chunks(arr, (0, 4)))
        assert [start for start, _cells in iter_populated_chunks(arr, span=(0, 4))] == [0]


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
        (_k0, range0, n0), (_k1, range1, n1) = data_gets
        obj_size = (grid / "g/field/c/0").stat().st_size
        # Ranged-ness is the posture; the byte counts are only its symptom.
        assert range0 is not None and range1 is not None
        assert n0 == 16 * 4 + 4  # the K=4 shard-index suffix
        assert n1 < obj_size  # one ranged inner chunk, not the object

    def test_one_get_on_flat_store(self, tmp_path):
        grid, expected = build_store(tmp_path, sharded=False)
        store = CountingStore(grid)
        store.gets.clear()
        np.testing.assert_array_equal(read_cell(store, "g/field", 0), expected[0])
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == 1  # the one inner-chunk object


@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "flat"])
class TestSweepGetPosture:
    """One GET per stored object — on BOTH §1.5 geometries.

    The sharded fixture stores one object, so it pins "no per-inner-chunk
    re-fetch"; the flat fixture stores TWO, which is where
    one-GET-per-object is a multiplicity claim rather than a count of one.
    """

    #: Stored payload objects per geometry (the flat layout omits chunks 1-2).
    STORED = {True: 1, False: 2}

    def test_sweep_reads_each_stored_object_once(self, tmp_path, sharded):
        grid, _ = build_store(tmp_path, sharded=sharded)
        store = CountingStore(grid)
        store.gets.clear()
        out = list(read_ragged(store, "g/field"))
        assert len(out) == len(CELLS)
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == self.STORED[sharded]
        assert len(set(key for key, _r, _n in data_gets)) == len(data_gets)  # distinct objects

    def test_sweep_reads_the_coordinate_once(self, tmp_path, sharded):
        """The posture covers the sibling ``morton`` coordinate too: it is read
        once per stored coordinate object, never re-sliced per payload span
        (which re-fetched the same object — twice on the flat geometry, and
        with a shard-index suffix each time when the coordinate is sharded)."""
        grid, _ = build_store(tmp_path, sharded=sharded)
        store = CountingStore(grid)
        store.gets.clear()
        assert len(list(read_ragged(store, "g/field"))) == len(CELLS)
        assert len([g for g in store.gets if "morton/c/" in g[0]]) == 1


class TestSubtreeReadRagged:
    """Issue #29: ``subtree=`` on the generic sweep (zagg's #351 contract).

    Golden contract: ``read_ragged(..., subtree=w)`` equals the whole-store
    sweep filtered to the cells that are descendants of ``w`` — the filter
    is INDEPENDENT of the reader's span arithmetic (each yielded cell's own
    morton word coarsened with ``mortie.clip2order``), so the two paths
    cannot share a bug.
    """

    @staticmethod
    def _filtered(out, subtree_key):
        """Whole-sweep entries whose CELL word descends from ``subtree_key``."""
        from mortie import clip2order

        from moczarr.convention import decimal_order, morton_decimal

        word = morton_word(subtree_key)
        order = decimal_order(morton_decimal(word))
        return [
            entry
            for entry in out
            if int(clip2order(order, np.asarray([entry[0]], dtype=np.uint64))[0]) == word
        ]

    @staticmethod
    def _assert_same(got, expected):
        assert [entry[0] for entry in got] == [entry[0] for entry in expected]
        for g, e in zip(got, expected):
            for a, b in zip(g[1:], e[1:]):
                np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "flat"])
    @pytest.mark.parametrize("key", [SHARD + "1", SHARD + "4"])
    def test_equals_filtered_sweep_both_currencies(self, tmp_path, sharded, key):
        grid, _ = build_store(tmp_path, sharded=sharded)
        store = LocalStore(grid)
        sweep = list(read_ragged(store, "g/field"))
        expected = self._filtered(sweep, key)
        assert len(expected) > 0
        # Both ratified currencies: packed area word (int) and decimal string.
        self._assert_same(list(read_ragged(store, "g/field", subtree=key)), expected)
        self._assert_same(list(read_ragged(store, "g/field", subtree=morton_word(key))), expected)

    def test_locations_ride_the_restricted_sweep(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True, located=True)
        store = LocalStore(grid)
        sweep = list(read_ragged(store, "g/field", locations=True))
        got = list(read_ragged(store, "g/field", locations=True, subtree=SHARD + "1"))
        self._assert_same(got, self._filtered(sweep, SHARD + "1"))
        assert len(got) == 2  # cells 0 and 3; each entry (word, values, locations)
        assert all(len(entry) == 3 for entry in got)

    def test_flat_get_accounting_skips_disjoint_objects(self, tmp_path):
        """The §1.5 read plan on the flat layout: only the covering chunk
        objects are fetched — the stored-but-disjoint chunk 3 never is."""
        grid, _ = build_store(tmp_path, sharded=False)
        store = CountingStore(grid)
        store.gets.clear()
        out = list(read_ragged(store, "g/field", subtree=SHARD + "1"))
        assert [entry[0] for entry in out] == [morton_word(SHARD + t) for t in ("11", "14")]
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert len(data_gets) == 1 and data_gets[0][0].endswith("c/0")

    def test_sharded_get_accounting_fetches_only_the_covering_span(self, tmp_path):
        """Sharded: the shard-index suffix plus ONLY the covering inner
        chunks — ranged GETs that never reach the disjoint chunk 3's bytes.

        The ``morton`` sibling is pinned alongside, because the payload
        objects are not the whole cost: resolving the span anchors on the
        coordinate's first stored object. Here that is the ONE object the
        fixture's coordinate has, served to the sweep from the same cached
        window — so the restricted read costs exactly one coordinate GET,
        the same as the unrestricted sweep."""
        grid, _ = build_store(tmp_path, sharded=True)
        store = CountingStore(grid)
        store.gets.clear()
        out = list(read_ragged(store, "g/field", subtree=SHARD + "1"))
        assert len(out) == 2
        assert len([g for g in store.gets if "morton/c/" in g[0]]) == 1
        data_gets = [g for g in store.gets if "field/c/" in g[0]]
        assert all(rng is not None for _k, rng, _n in data_gets)
        (obj_key, _r0, n0), *chunk_gets = data_gets
        assert n0 == 16 * 4 + 4  # the K=4 shard-index suffix
        assert len(chunk_gets) >= 1
        # Decode the shard index: chunk 3's payload starts after chunk 0's,
        # so every fetched range must end before it.
        obj = (grid / "g/field/c/0").read_bytes()
        idx = np.frombuffer(obj[-n0:-4], dtype="<u8").reshape(-1, 2)
        chunk3_start = int(idx[3][0])
        # Bound it inside the object: the §1.5 absent sentinel (2**64 - 1)
        # would leave the range assertion below vacuously true.
        assert 0 < chunk3_start < len(obj)
        assert all(int(r.end) <= chunk3_start for _k, r, _n in chunk_gets)

    def test_subtree_resolution_shares_the_sweep_listing(self, tmp_path):
        """The span resolves against the same ``stored_chunk_spans`` listing
        the sweep consumes — one LIST of the data keys, not two."""
        grid, _ = build_store(tmp_path, sharded=True)

        class ListCountingStore(CountingStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lists: list = []

            def with_read_only(self, read_only=True):
                s = ListCountingStore(self.root, read_only=read_only)
                s.gets, s.lists = self.gets, self.lists
                return s

            def list_prefix(self, prefix):
                self.lists.append(prefix)
                return super().list_prefix(prefix)

        store = ListCountingStore(grid)
        store.lists.clear()
        assert len(list(read_ragged(store, "g/field", subtree=SHARD))) == len(CELLS)
        assert store.lists.count("g/field/c/") == 1

    def test_ancestor_of_the_axis_root_clips_to_the_whole_axis(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        store = LocalStore(grid)
        got = list(read_ragged(store, "g/field", subtree=SHARD[:-1]))
        self._assert_same(got, list(read_ragged(store, "g/field")))
        assert len(got) == len(CELLS)

    def test_out_of_domain_word_warns_once_then_yields_nothing(self, tmp_path):
        import warnings

        grid, _ = build_store(tmp_path, sharded=True)
        disjoint = SHARD.lstrip("-")  # the northern mirror: well-formed, disjoint
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = list(read_ragged(LocalStore(grid), "g/field", subtree=disjoint))
        assert out == []
        msgs = [str(w.message) for w in rec if "outside this axis" in str(w.message)]
        assert msgs == [
            f"subtree {disjoint} is outside this axis' order-6 root {SHARD} — yielding nothing"
        ]

    def test_out_of_domain_warning_is_attributed_to_the_caller(self, tmp_path):
        import warnings

        grid, _ = build_store(tmp_path, sharded=True)
        # The reader is a generator, so the warning fires on the consumer's
        # first next() — the stacklevel must reach THIS frame, not a moczarr one.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD.lstrip("-")))
        (w,) = [r for r in rec if "outside this axis" in str(r.message)]
        assert w.filename == __file__

    def test_in_domain_empty_subtree_yields_nothing_silently(self, tmp_path):
        import warnings

        grid, _ = build_store(tmp_path, sharded=True)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD + "2")) == []
        assert [w for w in rec if "outside this axis" in str(w.message)] == []

    def test_sub_chunk_subtree_refused_pointing_at_read_cell(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        with pytest.raises(ValueError, match="read_cell"):
            list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD + "11"))

    @pytest.mark.parametrize("bad", ["", "abc", "913", 3, -5])
    def test_malformed_subtree_raises(self, tmp_path, bad):
        grid, _ = build_store(tmp_path, sharded=True)
        with pytest.raises(ValueError):
            list(read_ragged(LocalStore(grid), "g/field", subtree=bad))

    def test_deeper_than_the_cells_axis_raises(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        with pytest.raises(ValueError, match="deeper than"):
            list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD + "111"))

    def test_misplaced_morton_coordinate_raises(self, tmp_path):
        """The identity the span arithmetic rests on — axis position ==
        nested id minus the root's start — is checked, not assumed: a rolled
        coordinate still shares its chunk ancestor, so only the anchor check
        catches it before the arithmetic names plausible-but-wrong cells."""
        words = np.roll(np.array([morton_word(SHARD + t) for t in TAILS], dtype="<u8"), 1)
        grid, _ = build_store(tmp_path, sharded=True, morton_words=words)
        with pytest.raises(ValueError, match="not in canonical nested placement"):
            list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD))

    def test_empty_store_yields_nothing_but_malformed_still_raises(self, tmp_path):
        grid = tmp_path / "empty"
        attrs = {"ragged": {"spec": RAGGED_SPEC, "element": dict(ELEMENT)}}
        _write(grid, "g/field/zarr.json", _vlen_meta(16, 4, sharded=True, attrs=attrs))
        _write(grid, "g/morton/zarr.json", _uint64_meta(16))
        assert list(read_ragged(LocalStore(grid), "g/field", subtree=SHARD)) == []
        with pytest.raises(ValueError):
            list(read_ragged(LocalStore(grid), "g/field", subtree="abc"))

    def test_absent_leading_inner_chunk_still_anchors(self, tmp_path):
        """The anchor must window the whole first stored SPAN, not its first
        read chunk: a shard's leading inner chunks may be absent, and zagg
        spec §7 is explicit that the ``morton`` coordinate holds its fill
        across them ("a reader MUST NOT assume the coordinate is dense
        across a shard"). Windowing chunk 0 made such a leaf read fine
        unrestricted but raise the payload-flavoured "no written 'morton'
        coordinate" error under ``subtree=``."""
        words = np.zeros(16, dtype="<u8")
        words[12:] = [morton_word(SHARD + t) for t in TAILS[12:]]  # only chunk 3 written
        grid, expected = build_store(tmp_path, sharded=True, cells={13: 4}, morton_words=words)
        store = LocalStore(grid)
        sweep = list(read_ragged(store, "g/field"))
        assert [entry[0] for entry in sweep] == [morton_word(SHARD + TAILS[13])]
        self._assert_same(list(read_ragged(store, "g/field", subtree=SHARD)), sweep)
        got = list(read_ragged(store, "g/field", subtree=SHARD + "4"))
        self._assert_same(got, self._filtered(sweep, SHARD + "4"))
        np.testing.assert_array_equal(got[0][1], expected[13])

    def test_undividing_read_chunk_yields_only_descendants(self, tmp_path):
        """The whole-chunk clip is exact only when the read chunk DIVIDES the
        span's ``4^d`` length, and nothing in this generic layer requires a
        power-of-four (or -two) cells chunk — the tensor profile states that
        for itself. With 6 cells per chunk on a 64-cell axis the clip widens
        ``[16, 32)`` to ``[12, 36)``, so membership has to be checked per
        cell or the sweep hands back cells from sibling subtrees."""
        tails = [a + b + c for a in "1234" for b in "1234" for c in "1234"]
        grid = tmp_path / "store"
        rng = np.random.default_rng(29)
        values = {
            c: rng.integers(-500, 500, (n, 3)).astype("<i2")
            for c, n in {12: 2, 17: 1, 34: 3}.items()
        }
        blocks: dict[int, list] = {}
        for cell, v in values.items():
            blocks.setdefault(cell // 6, [b""] * 6)[cell % 6] = v.tobytes()
        attrs = {"ragged": {"spec": RAGGED_SPEC, "element": dict(ELEMENT)}}
        _write(grid, "g/field/zarr.json", _vlen_meta(64, 6, sharded=False, attrs=attrs))
        for ordinal, cells in blocks.items():
            _write(grid, f"g/field/c/{ordinal}", _inner_chunk(cells))
        _write(grid, "g/morton/zarr.json", _uint64_meta(64))
        _write(
            grid,
            "g/morton/c/0",
            np.array([morton_word(SHARD + t) for t in tails], dtype="<u8").tobytes(),
        )

        store = LocalStore(grid)
        sweep = list(read_ragged(store, "g/field"))
        assert [entry[0] for entry in sweep] == [
            morton_word(SHARD + tails[c]) for c in (12, 17, 34)
        ]
        # Cells 12 and 34 sit in the clip's widened chunks but outside the span.
        got = list(read_ragged(store, "g/field", subtree=SHARD + "2"))
        self._assert_same(got, self._filtered(sweep, SHARD + "2"))
        assert [entry[0] for entry in got] == [morton_word(SHARD + tails[17])]

    def test_subtree_none_is_the_default_sweep(self, tmp_path):
        grid, _ = build_store(tmp_path, sharded=True)
        store = LocalStore(grid)
        self._assert_same(
            list(read_ragged(store, "g/field", subtree=None)),
            list(read_ragged(store, "g/field")),
        )
