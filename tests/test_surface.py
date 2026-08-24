"""The package surface: every public-workflow name is one ``mz.<name>`` away.

Issue #49 item 4 — ``open_ragged`` was importable only by module path
(``moczarr.ragged.open_ragged``) while every sibling it is used alongside
sat on the package root, so a notebook mixing the two spellings was forced
by the library, not chosen. The roster test pins the invariant so the
asymmetry cannot recur: a function that appears in a public workflow (the
quickstart, the demo-notebook selection loop, the per-leaf readers) must be
reachable as ``mz.<name>``, and every ``__all__`` name must resolve.
"""

import moczarr as mz

#: The public-workflow roster: what a reader following the documented flows
#: (select -> open -> read; coverage ask; per-leaf ragged/HHDC reads) calls.
#: Additions to those flows belong here in the same PR that documents them.
WORKFLOW = [
    # selection: coverage ask + candidate roster (issues #39/#45/#49)
    "candidate_leaves",
    "candidate_shards",
    "coverage_moc",
    "coverage_toc",
    "load_root_coverage",
    "read_manifest",
    # open: whole store, one product, one leaf
    "open_hive",
    "open_leaf",
    "open_object_store",
    "open_store",
    # per-leaf readers the leaf handle feeds
    "open_ragged",
    "read_cell",
    "read_ragged",
    "read_tensors",
    # cross-store composition and stats sidecars
    "iter_occupancy_and",
    "occupancy_and",
    "read_stats",
    "walk_leaves",
]


class TestPackageSurface:
    def test_workflow_names_are_package_level(self):
        for name in WORKFLOW:
            assert hasattr(mz, name), f"mz.{name} missing from the package root"
            assert name in mz.__all__, f"{name} reachable but not declared in __all__"

    def test_all_names_resolve_and_stay_sorted(self):
        # __all__ is the surface contract: no dangling names, and kept
        # sorted so additions land in one obvious place.
        for name in mz.__all__:
            assert getattr(mz, name, None) is not None, f"__all__ names unresolvable {name}"
        assert list(mz.__all__) == sorted(mz.__all__)

    def test_open_ragged_is_the_module_function(self):
        # The export is the same object, not a wrapper — module-path callers
        # and package-root callers share one seam.
        from moczarr import ragged

        assert mz.open_ragged is ragged.open_ragged
