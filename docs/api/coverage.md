# moczarr.coverage

::: moczarr.coverage
    options:
      filters:
        - "!^_"
        - "!^as_moc_words$"
        # Not on the package surface (no `moczarr.__all__` entry), so not on
        # the docs surface either — the two agree. The planned public
        # temporal seam is phase 5's `coverage_toc`, which wraps this
        # decoder; promotion rides with it, not ahead of it (issue #45).
        - "!^TEMPORAL_SPEC$"
        - "!^temporal_shard_words$"
