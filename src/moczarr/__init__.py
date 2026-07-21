"""moczarr: sparse-DGGS xarray reader for morton-hive zarr stores."""

from moczarr.convention import (
    COMMIT_ATTR,
    COVERAGE_SIDECAR,
    HIVE_SPEC,
    HIVE_SPEC_V2,
    MANIFEST_NAME,
    MORTON_CONVENTION_ENTRY,
    MORTON_CONVENTION_UUID,
    ROOT_COVERAGE_NAME,
    check_node_invariant,
    is_point_word,
    leaf_path,
    morton_decimal,
    morton_word,
    parse_manifest,
    split_leaf_name,
)
from moczarr.coverage import (
    COVERAGE_SPEC,
    aoi_mask,
    box_and,
    box_words,
    decode_bitmap,
    parse_leaf_coverage,
    parse_root_coverage,
    ranges_contain,
    ranges_words,
    root_coverage_and,
)
from moczarr.exceptions import NoCoverageError
from moczarr.fabricate import FLOAT64_EXACT_MAX_ORDER, fabricate_cell_ids
from moczarr.join import join_coarse, parent_cells
from moczarr.open import open_hive
from moczarr.ranges import MortonRanges

# moczarr.moc_index (MortonMocIndex) is imported by module path, not here:
# the package root stays xarray-import-free (the repo's lazy-import posture),
# and the index reaches most users through open_hive(index_kind="moc").
from moczarr.store import (
    bitmap_and,
    load_root_coverage,
    open_object_store,
    read_commit,
    read_coverage_bitmap,
    read_leaf_coverage,
    read_manifest,
    walk_leaves,
    warn_if_stale,
)

try:
    from moczarr._version import __version__
except ImportError:  # pragma: no cover - version file is generated at build time
    __version__ = "0.0.0+unknown"

__all__ = [
    "COMMIT_ATTR",
    "COVERAGE_SIDECAR",
    "COVERAGE_SPEC",
    "FLOAT64_EXACT_MAX_ORDER",
    "HIVE_SPEC",
    "HIVE_SPEC_V2",
    "MANIFEST_NAME",
    "MORTON_CONVENTION_ENTRY",
    "MORTON_CONVENTION_UUID",
    "MortonRanges",
    "NoCoverageError",
    "ROOT_COVERAGE_NAME",
    "__version__",
    "aoi_mask",
    "bitmap_and",
    "box_and",
    "box_words",
    "check_node_invariant",
    "decode_bitmap",
    "fabricate_cell_ids",
    "is_point_word",
    "join_coarse",
    "leaf_path",
    "load_root_coverage",
    "morton_decimal",
    "morton_word",
    "open_hive",
    "open_object_store",
    "parent_cells",
    "parse_leaf_coverage",
    "parse_manifest",
    "parse_root_coverage",
    "ranges_contain",
    "ranges_words",
    "read_commit",
    "read_coverage_bitmap",
    "read_leaf_coverage",
    "read_manifest",
    "root_coverage_and",
    "split_leaf_name",
    "walk_leaves",
    "warn_if_stale",
]
