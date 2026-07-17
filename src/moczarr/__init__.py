"""moczarr: sparse-DGGS xarray reader for morton-hive zarr stores."""

from moczarr.convention import (
    COMMIT_ATTR,
    COVERAGE_SIDECAR,
    HIVE_SPEC,
    HIVE_SPEC_V2,
    MANIFEST_NAME,
    ROOT_COVERAGE_NAME,
    check_node_invariant,
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
from moczarr.open import open_hive
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
    "HIVE_SPEC",
    "HIVE_SPEC_V2",
    "MANIFEST_NAME",
    "ROOT_COVERAGE_NAME",
    "__version__",
    "aoi_mask",
    "bitmap_and",
    "box_and",
    "box_words",
    "check_node_invariant",
    "decode_bitmap",
    "leaf_path",
    "load_root_coverage",
    "morton_decimal",
    "morton_word",
    "open_hive",
    "open_object_store",
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
