# moczarr.coverage

::: moczarr.coverage
    options:
      filters:
        - "!^_"
        # The two AOI/window boundary NORMALIZERS are internal by espg
        # ruling (issue #45, ruling 3): importable, but off both the package
        # surface (`moczarr.__all__`) and the docs surface. Everything else
        # this module defines is public and documented — the temporal decoder
        # and its spec marker included, as the exact twins of `ranges_words`
        # and `COVERAGE_SPEC` (issue #45; see the PR body's note).
        - "!^as_moc_words$"
        - "!^as_toc_words$"
