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
    box_words,
    decode_bitmap,
    parse_leaf_coverage,
    parse_root_coverage,
    ranges_contain,
    ranges_words,
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
    "box_words",
    "check_node_invariant",
    "decode_bitmap",
    "leaf_path",
    "morton_decimal",
    "morton_word",
    "parse_leaf_coverage",
    "parse_manifest",
    "parse_root_coverage",
    "ranges_contain",
    "ranges_words",
    "split_leaf_name",
]
